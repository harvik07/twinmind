"""The optimisation benchmark: every method on every held-out goal.

    python scripts/run_optim_benchmark.py --workers 6          # the full run
    python scripts/run_optim_benchmark.py --smoke              # tiny, 2 dev goals

RESUMABLE BY CONSTRUCTION. One atomic JSON file per ``(goal, method, seed)``
cell plus a manifest. A cell whose file exists and parses is skipped, so an
interrupted multi-hour run restarts where it stopped. The manifest records the
twin config hash, the surrogate hash and every declared assumption; resuming
against a changed twin is refused, because mixing results from two different
simulators into one table is worse than having no table.

THE METHODS, AND WHAT EACH IS FOR.

* ``sf_random`` / ``sf_tpe`` / ``sf_cmaes`` -- surrogate-first with twin
  verification. The thing being tested.
* ``surrogate_only`` -- the ablation. No verification, so the gap between what
  it predicted and what the twin found is the winner's curse, isolated.
* ``twin_tpe`` / ``twin_cmaes`` -- the honest competitor: the same optimisers
  driving the twin directly at 5 replications per evaluation, with anytime
  checkpoints at 50 / 150 / 300 evaluations so the cost-quality frontier can be
  drawn rather than asserted.
* ``best_known`` -- NOT an oracle. It is the best twin-verified plan anyone
  found, including a 500k-evaluation surrogate sweep whose top 100 are verified.
  Every reported value is re-evaluated on the "reeval" stream, so the reference
  itself does not suffer the winner's curse.

Nothing here is tuned on a held-out goal. ``--smoke`` runs on DEV goals only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twinmind.eval.registry import results_dir
from twinmind.models.surrogate import KPI_ORDER
from twinmind.optim.contracts import ObjectiveSpec, Plan
from twinmind.optim.evaluators import EvalCost, SurrogateEvaluator, TwinEvaluator, feasible_robust
from twinmind.optim.goals import Goal, load_goals
from twinmind.optim.novelty import NoveltyScorer
from twinmind.optim.optimisers import OPTIMISERS, free_dimensions
from twinmind.optim.surrogate_first import VerifyConfig, surrogate_first_verify, surrogate_only
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("run_optim_benchmark")

RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "optim_benchmark"
ROLLOUTS = PROJECT_ROOT / "data" / "processed" / "rollouts"

#: Anytime checkpoints for the twin-only arms, in twin evaluations.
ANYTIME_CHECKPOINTS = (50, 150, 300)

#: Budgets. Surrogate-side budgets are large because surrogate evaluations are
#: free by comparison; twin-only gets 300 evaluations, which at 5 reps each is
#: 1,500 twin runs -- roughly what a plant engineer could afford to wait for.
#: The surrogate-side TPE budget, shared by ``sf_tpe`` and its ablation.
#:
#: ONE constant, referenced twice, because the two arms must stay matched.
#: ``surrogate_only`` isolates what twin verification is worth, so it has to
#: differ from ``surrogate_first_verify`` in the verification step alone -- K=1
#: and no verification against K=20 verified -- and in nothing else. It was
#: 20,000 against this arm's 3,000, which made the ablation the BEST-SEARCHED
#: method in the benchmark and would have left H3 uninterpretable: a win for
#: ``surrogate_only`` could have been the extra search rather than the absence
#: of verification. See Amendment 1 in reports/preregistration_optim.md.
#:
#: Two literals would drift apart the moment one was tuned, which is exactly how
#: they came to differ in the first place.
SURROGATE_TPE_BUDGET = 3_000

BUDGETS = {
    "sf_random": 50_000,
    "sf_tpe": SURROGATE_TPE_BUDGET,
    "sf_cmaes": 20_000,
    "surrogate_only": SURROGATE_TPE_BUDGET,
    "twin_tpe": 300,
    "twin_cmaes": 300,
    "best_known": 500_000,
}

METHODS = tuple(BUDGETS)


@dataclass(frozen=True)
class Cell:
    """One (goal, method, seed) unit of work."""

    goal: str
    method: str
    seed: int

    @property
    def filename(self) -> str:
        """Deterministic result filename."""
        return f"{self.goal}__{self.method}__seed{self.seed}.json"


def build_manifest(resources: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Everything that invalidates existing cells if it changes."""
    config = resources.config
    surrogate_dir = PROJECT_ROOT / "models" / "surrogate"
    digest = hashlib.sha256()
    for path in sorted(surrogate_dir.glob("member_*.pt")):
        digest.update(path.read_bytes())
    return {
        "twin_config_hash": hashlib.sha256(
            json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "surrogate_hash": digest.hexdigest()[:16],
        "assumptions": {k: float(v.value) for k, v in config.assumptions.items()},
        "demand_scale": float(resources.demand_scale),
        "budgets": dict(BUDGETS),
        "budgets_hash": budgets_hash(),
        "methods": list(METHODS),
        "seeds": list(args.seeds),
        "kpi_order": list(KPI_ORDER),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def budgets_hash() -> str:
    """A stable hash of every method's budget.

    Recorded in the manifest AND in each result file. The manifest check refuses
    a resume across a budget change, which protects the run as a whole; the
    per-cell copy is finer-grained, so a cell produced under an old budget is
    simply retried instead of the whole run being refused. That matters here
    because ``surrogate_only``'s budget changed from 20,000 to 3,000 while cells
    from the old value could still be sitting in a results tree.
    """
    payload = json.dumps(dict(sorted(BUDGETS.items())), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


INVALIDATING_KEYS = (
    "twin_config_hash",
    "surrogate_hash",
    "assumptions",
    "demand_scale",
    "budgets",
    "budgets_hash",
)


def manifest_conflicts(existing: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Keys on which a stored manifest disagrees with the current one."""
    return [k for k in INVALIDATING_KEYS if existing.get(k) != current.get(k)]


def write_atomic(payload: dict[str, Any], path: Path) -> Path:
    """Write JSON atomically: temp file then ``os.replace``.

    A multi-hour run interrupted mid-write would otherwise leave a truncated
    file that the resume logic counts as done.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temp, path)
    return path


def cell_done(path: Path) -> bool:
    """True when a cell exists, parses, AND was produced under the current budgets.

    The budget check is what stops a resumed run silently mixing cells from two
    different configurations. A cell written under the old 20,000-trial
    ``surrogate_only`` budget would otherwise be counted as finished and its
    number averaged in beside 3,000-trial cells, producing a table whose rows
    were not comparable and giving no sign that anything was wrong.

    A cell with no recorded hash predates the check and is treated as stale, so
    it is regenerated rather than trusted.
    """
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("budgets_hash") == budgets_hash())


# --------------------------------------------------------------------------- #
# The methods
# --------------------------------------------------------------------------- #
def run_twin_only(
    spec: ObjectiveSpec,
    context: Any,
    resources: Any,
    seed: int,
    optimiser: str,
    budget: int,
    n_reps: int = 5,
) -> tuple[list[Plan], dict[str, Any]]:
    """The honest competitor: an optimiser driving the twin directly.

    Anytime checkpoints are recorded so the cost-quality frontier shows what
    twin-only achieves *per twin-second*, rather than only its final number.
    """
    started = time.perf_counter()
    twin_eval = TwinEvaluator(
        resources=resources, context=context, goal_seed=seed, n_reps=n_reps, stream="search"
    )
    history: list[dict[str, Any]] = []

    class Recording:
        """Wraps the evaluator to snapshot the best-so-far at each checkpoint."""

        def __init__(self, inner: TwinEvaluator) -> None:
            self.inner = inner
            self.seen = 0
            self.best = -np.inf
            self.best_feasible = -np.inf

        def evaluate(self, scenarios: Any) -> list[dict[str, float]]:
            rows = self.inner.evaluate(scenarios)
            for kpis in rows:
                self.seen += 1
                score = spec.score(kpis)
                self.best = max(self.best, score)
                if spec.feasible(kpis):
                    self.best_feasible = max(self.best_feasible, score)
                if self.seen in ANYTIME_CHECKPOINTS:
                    history.append(
                        {
                            "evaluations": self.seen,
                            "twin_runs": self.inner.cost.twin_runs,
                            "cpu_seconds": self.inner.cost.cpu_seconds,
                            "best_score": float(self.best),
                            "best_feasible_score": float(self.best_feasible)
                            if np.isfinite(self.best_feasible)
                            else None,
                        }
                    )
            return rows

        @property
        def cost(self) -> EvalCost:
            """Passthrough."""
            return self.inner.cost

    recording = Recording(twin_eval)
    candidates = OPTIMISERS[optimiser](spec, recording, budget, seed)
    if not candidates:
        return [], {"status": "no_candidate", "anytime": history}

    best = candidates[0]
    # Independent re-evaluation, same as every other method.
    final_eval = TwinEvaluator(
        resources=resources, context=context, goal_seed=seed, n_reps=30, stream="reeval"
    )
    mean, sem = final_eval.evaluate_with_spread([best.scenario])[0]
    cost = EvalCost(
        twin_runs=twin_eval.cost.twin_runs + 30,
        cpu_seconds=twin_eval.cost.cpu_seconds + final_eval.cost.cpu_seconds,
        wall_seconds=time.perf_counter() - started,
    )
    plan = Plan(
        scenario=best.scenario,
        surrogate_pred={},
        twin_verified={"mean": mean, "sem": sem},
        feasible_point=spec.feasible(mean),
        feasible_robust=feasible_robust(spec, mean, sem),
        constraint_margins={
            f"{c.kpi}{c.op}{c.threshold}": c.margin(float(mean[c.kpi]))
            for c in spec.constraints
            if c.hard
        },
        cost=cost.to_dict(),
        status="ok" if spec.feasible(mean) else "infeasible_unverified",
        objective_value=spec.score(mean),
    )
    return [plan], {"status": plan.status, "anytime": history}


def run_best_known(
    spec: ObjectiveSpec,
    context: Any,
    surrogate: Any,
    resources: Any,
    seed: int,
    budget: int,
    top_n: int = 100,
) -> tuple[list[Plan], dict[str, Any]]:
    """The reference: a huge surrogate sweep, top-N twin-verified at 30 reps.

    Called ``best_known`` and not ``oracle`` on purpose. It is the best plan
    anyone has found, not the true optimum, and the benchmark's "gap to
    best_known" can therefore be negative if a method beats it. That is
    reported rather than clipped.
    """
    started = time.perf_counter()
    surrogate_eval = SurrogateEvaluator(surrogate=surrogate, context=context)
    candidates = OPTIMISERS["random"](spec, surrogate_eval, budget, seed)
    shortlist = candidates[:top_n]

    final_eval = TwinEvaluator(
        resources=resources, context=context, goal_seed=seed, n_reps=30, stream="reeval"
    )
    verified = final_eval.evaluate_with_spread([c.scenario for c in shortlist])

    plans: list[Plan] = []
    for candidate, (mean, sem) in zip(shortlist, verified, strict=True):
        plans.append(
            Plan(
                scenario=candidate.scenario,
                surrogate_pred=candidate.kpis,
                twin_verified={"mean": mean, "sem": sem},
                feasible_point=spec.feasible(mean),
                feasible_robust=feasible_robust(spec, mean, sem),
                constraint_margins={
                    f"{c.kpi}{c.op}{c.threshold}": c.margin(float(mean[c.kpi]))
                    for c in spec.constraints
                    if c.hard
                },
                surrogate_twin_gap={kpi: candidate.kpis[kpi] - mean[kpi] for kpi in KPI_ORDER},
                status="ok" if spec.feasible(mean) else "infeasible_unverified",
                objective_value=spec.score(mean),
            )
        )
    from twinmind.optim.contracts import rank_plans

    plans = rank_plans(plans)
    cost = EvalCost(
        surrogate_evals=surrogate_eval.cost.surrogate_evals,
        twin_runs=final_eval.cost.twin_runs,
        cpu_seconds=final_eval.cost.cpu_seconds,
        wall_seconds=time.perf_counter() - started,
    )
    if plans:
        plans[0] = Plan(**{**plans[0].__dict__, "cost": cost.to_dict()})
    return plans[:10], {"status": plans[0].status if plans else "no_candidate"}


def run_cell(
    cell: Cell,
    goal: Goal,
    context: Any,
    surrogate: Any,
    resources: Any,
    novelty: NoveltyScorer | None,
    smoke: bool,
) -> dict[str, Any]:
    """Run one (goal, method, seed) and return its serialisable result."""
    spec = goal.spec
    budget = BUDGETS[cell.method] if not smoke else max(BUDGETS[cell.method] // 500, 40)
    verify = VerifyConfig(
        budget=budget,
        k=20 if not smoke else 4,
        verify_reps=10 if not smoke else 2,
        final_reps=30 if not smoke else 3,
        novelty_active=novelty is not None,
    )
    started = time.perf_counter()

    if cell.method.startswith("sf_"):
        verify.optimiser = cell.method.removeprefix("sf_")
        plans, trace = surrogate_first_verify(
            spec, context, surrogate, resources, cell.seed, verify, novelty
        )
    elif cell.method == "surrogate_only":
        verify.optimiser = "tpe"
        plans, trace = surrogate_only(spec, context, surrogate, resources, cell.seed, verify)
    elif cell.method.startswith("twin_"):
        plans, trace = run_twin_only(
            spec,
            context,
            resources,
            cell.seed,
            cell.method.removeprefix("twin_"),
            budget if not smoke else 12,
            n_reps=5 if not smoke else 2,
        )
    elif cell.method == "best_known":
        plans, trace = run_best_known(
            spec, context, surrogate, resources, cell.seed, budget, top_n=100 if not smoke else 4
        )
    else:  # pragma: no cover - METHODS is closed
        raise ValueError(f"unknown method {cell.method!r}")

    return {
        "goal": cell.goal,
        "split": goal.split,
        "method": cell.method,
        "seed": cell.seed,
        "budget": budget,
        "budgets_hash": budgets_hash(),
        "infeasible_by_design": spec.infeasible_by_design,
        "n_free_dims": len(free_dimensions(spec)),
        "plans": [p.to_dict() for p in plans[:5]],
        "trace": trace,
        "wall_seconds": time.perf_counter() - started,
        "smoke": smoke,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Run the optimisation benchmark.")
    parser.add_argument("--workers", type=int, default=6, help="Max 6, spawn start method.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--out", default=str(RESULTS_DIR))
    parser.add_argument("--smoke", action="store_true", help="Tiny budgets on 2 dev goals.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument(
        "--goals",
        nargs="+",
        default=None,
        help="Run only these goals by name. Used for the production-budget dry run, "
        "which must target a DEV goal so no held-out result is ever produced by it.",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "held_out"),
        default=None,
        help="Override the split. Defaults to held_out, or dev with --smoke.",
    )
    parser.add_argument("--force-new-manifest", action="store_true")
    return parser


_WORKER: dict[str, Any] = {}


def _init_worker() -> None:
    """Worker set-up: ignore Ctrl-C, and pin torch to a single thread.

    Ctrl-C is ignored so the parent can shut the pool down cleanly.

    The single thread matters more than it looks. Torch defaults to one thread
    per physical core -- 6 here -- so a 6-worker pool launches 36 threads onto 6
    cores, a 6x oversubscription where the threads spend their time fighting
    over the cores rather than working. Measured in the dry run: sf_tpe took
    652 s against about 105 s of actual work. Inference is deterministic
    regardless of thread count, so this costs nothing but contention.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def _worker_context() -> dict[str, Any]:
    """Build the shared, read-only objects once per worker process."""
    if _WORKER:
        return _WORKER
    from twinmind.models.surrogate_data import load_rollouts
    from twinmind.models.surrogate_mlp import SurrogateEnsemble
    from twinmind.twin.demand_proc import CachedDemandModel
    from twinmind.twin.runner import build_resources

    cache_path = ROLLOUTS / "demand_grids.npz"
    if not cache_path.exists():
        raise SystemExit(
            f"no demand grid cache at {cache_path}. Build it first:\n"
            "    python scripts/build_demand_cache.py"
        )
    model = CachedDemandModel.load(cache_path)
    _WORKER["resources"] = build_resources(demand_model=model, series_id=model.series_ids[0])
    _WORKER["surrogate"] = SurrogateEnsemble.load(PROJECT_ROOT / "models" / "surrogate")
    train = load_rollouts(ROLLOUTS / "train")
    _WORKER["novelty"] = NoveltyScorer.from_scenarios(train.scenario_matrix())
    _WORKER["t1"] = load_rollouts(ROLLOUTS / "test_t1", allow_test=True)
    return _WORKER


def _context_for_goal(goal: Goal, state: dict[str, Any]) -> Any:
    """Rebuild the T1 context a goal names, from its block id."""
    from twinmind.twin.context import sample_context

    t1 = state["t1"]
    resources = state["resources"]
    blocks = sorted(set(t1.context_blocks.tolist()))
    block = blocks[goal.spec.context_id % len(blocks)]
    origin = t1.frame.loc[t1.context_blocks == block, "origin"].iloc[0]
    return sample_context(
        10_000 + int(block),
        cache=getattr(resources.lookup, "cache", None),
        series_id=resources.series_id,
        origin=origin,
    )


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point. Module-level and plain-data in/out, for spawn.

    **Never raises.** A failure is returned as a result with ``ok=False`` and a
    traceback, because an exception escaping here propagates out of
    ``imap_unordered`` and kills the whole pool -- which is exactly what a Sobol
    ValueError in one cell did to the first production run, taking the other 251
    cells with it. A three-hour job must not be destroyed by one bad cell.
    """
    import traceback

    started = time.perf_counter()
    try:
        state = _worker_context()
        goal = next(g for g in load_goals() if g.name == task["goal"])
        cell = Cell(goal=task["goal"], method=task["method"], seed=task["seed"])
        context = _context_for_goal(goal, state)
        novelty = state["novelty"] if task["novelty_active"] else None
        result = run_cell(
            cell, goal, context, state["surrogate"], state["resources"], novelty, task["smoke"]
        )
        result["ok"] = True
    except BaseException as error:  # noqa: BLE001 - every failure must be reported, not raised
        result = {
            "ok": False,
            "goal": task["goal"],
            "method": task["method"],
            "seed": task["seed"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - started,
            "smoke": task["smoke"],
        }
    try:
        import psutil

        result["worker_rss_gb"] = float(psutil.Process(os.getpid()).memory_info().rss / 1e9)
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        result["worker_rss_gb"] = float("nan")
    return result


def main() -> int:
    """Entry point."""
    import multiprocessing as mp

    args = build_parser().parse_args()
    workers = max(1, min(args.workers, 6))
    out = Path(args.out) / ("smoke" if args.smoke else "full")

    from twinmind.twin.demand_proc import CachedDemandModel
    from twinmind.twin.runner import build_resources

    cache_path = ROLLOUTS / "demand_grids.npz"
    if not cache_path.exists():
        raise SystemExit(
            f"no demand grid cache at {cache_path}. Build it:\n"
            "    python scripts/build_demand_cache.py"
        )
    model = CachedDemandModel.load(cache_path)
    resources = build_resources(demand_model=model, series_id=model.series_ids[0])

    manifest = build_manifest(resources, args)
    manifest["smoke"] = args.smoke
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force_new_manifest:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        conflicts = manifest_conflicts(existing, manifest)
        if conflicts:
            raise SystemExit(
                f"refusing to resume: the manifest in {out} disagrees on {conflicts}.\n"
                "The twin, its assumptions or the surrogate changed since those cells were "
                "written, so mixing them would put results from two different systems in one "
                "table.\nPass --force-new-manifest to overwrite, after moving the old cells."
            )
    else:
        write_atomic(manifest, manifest_path)

    split = args.split or ("dev" if args.smoke else "held_out")
    goals = load_goals(split=split)
    if args.goals:
        wanted = set(args.goals)
        goals = [g for g in goals if g.name in wanted]
        missing = wanted - {g.name for g in goals}
        if missing:
            raise SystemExit(
                f"no such goal(s) in the {split} split: {sorted(missing)}. "
                f"Available: {', '.join(g.name for g in load_goals(split=split))}"
            )
    elif args.smoke:
        goals = goals[:2]
    logger.info("split=%s, %d goal(s): %s", split, len(goals), ", ".join(g.name for g in goals))
    methods = [m for m in args.methods if m in METHODS]

    tasks = []
    for goal in goals:
        for method in methods:
            for seed in args.seeds:
                cell = Cell(goal=goal.name, method=method, seed=seed)
                if cell_done(out / cell.filename):
                    continue
                tasks.append(
                    {
                        "goal": goal.name,
                        "method": method,
                        "seed": seed,
                        "smoke": args.smoke,
                        "novelty_active": True,
                    }
                )

    total = len(goals) * len(methods) * len(args.seeds)
    logger.info(
        "%d of %d cells outstanding (%d goals x %d methods x %d seeds) on %d workers",
        len(tasks),
        total,
        len(goals),
        len(methods),
        len(args.seeds),
        workers,
    )
    if not tasks:
        logger.info("nothing to do; every cell is present")
        if args.smoke:
            # Still print the projection: a resumed smoke run is exactly when
            # someone wants to know how long the real one will take.
            _estimate_full_run(out, workers, len(args.seeds))
        return 0

    started = time.perf_counter()
    done = 0
    rss: list[float] = []
    context = mp.get_context("spawn")
    pool = context.Pool(workers, initializer=_init_worker)
    failures: list[dict[str, Any]] = []
    failure_dir = out / "failures"
    try:
        for result in pool.imap_unordered(_run_one, tasks):
            cell = Cell(goal=result["goal"], method=result["method"], seed=result["seed"])
            done += 1
            if result.get("ok", True):
                # Only a SUCCESSFUL cell is written to the results tree, so a
                # rerun retries the failure and skips the completed work.
                write_atomic(result, out / cell.filename)
            else:
                failures.append(result)
                write_atomic(result, failure_dir / cell.filename)
                logger.error(
                    "[%3d/%3d] FAILED %-28s %-15s seed %d  %s",
                    done,
                    len(tasks),
                    result["goal"],
                    result["method"],
                    result["seed"],
                    result["error"],
                )
                continue
            if np.isfinite(result.get("worker_rss_gb", np.nan)):
                rss.append(result["worker_rss_gb"])
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            logger.info(
                "[%3d/%3d] %-32s %-15s seed %d  %5.1fs  eta %s",
                done,
                len(tasks),
                result["goal"],
                result["method"],
                result["seed"],
                result["wall_seconds"],
                _hms((len(tasks) - done) / max(rate, 1e-9)),
            )
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.warning("interrupted; %d cells written and valid, rerun to resume", done)
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    elapsed = time.perf_counter() - started
    logger.info(
        "finished %d cells in %.1f min (%d succeeded, %d FAILED)",
        done,
        elapsed / 60.0,
        done - len(failures),
        len(failures),
    )
    if rss:
        logger.info(
            "worker memory: %.2f GB mean, %.2f GB peak; ~%.1f GB across %d workers",
            float(np.mean(rss)),
            float(np.max(rss)),
            float(np.max(rss)) * workers,
            workers,
        )
    if args.smoke:
        _estimate_full_run(out, workers, len(args.seeds))

    if failures:
        # Loud, itemised, and a non-zero exit. A silent partial run is how a
        # results table ends up quietly missing a method.
        logger.error("=== %d CELL(S) FAILED ===", len(failures))
        by_error: dict[str, list[str]] = {}
        for failure in failures:
            key = failure["error"][:120]
            by_error.setdefault(key, []).append(
                f"{failure['goal']}/{failure['method']}/seed{failure['seed']}"
            )
        for error, cells in sorted(by_error.items(), key=lambda kv: -len(kv[1])):
            logger.error("  %d x %s", len(cells), error)
            for name in cells[:6]:
                logger.error("      %s", name)
            if len(cells) > 6:
                logger.error("      ... and %d more", len(cells) - 6)
        logger.error("tracebacks are in %s", failure_dir)
        logger.error("rerun the same command to retry failures; completed cells are skipped")

    summary = {
        "cells": done,
        "succeeded": done - len(failures),
        "failed": len(failures),
        "failures": [{k: f[k] for k in ("goal", "method", "seed", "error")} for f in failures],
        "elapsed_s": elapsed,
        "out": str(out),
        "manifest": manifest,
    }
    # The summary ALWAYS goes beside the cells it describes.
    write_atomic(summary, out / "summary.json")
    # Only a run against the DEFAULT results tree also publishes the canonical
    # copy in reports/results. An earlier version wrote the canonical file
    # unconditionally, so a production-budget dry run pointed at a scratch
    # directory overwrote reports/results/optim_benchmark_run.json with a
    # one-cell dev-goal summary -- at the very path the real held-out run owns.
    if Path(args.out).resolve() == RESULTS_DIR.resolve():
        write_atomic(
            summary,
            results_dir()
            / ("optim_benchmark_smoke.json" if args.smoke else "optim_benchmark_run.json"),
        )
    else:
        logger.info(
            "non-default --out: summary written to %s only, not to reports/results",
            out / "summary.json",
        )
    return 1 if failures else 0


def _estimate_full_run(out: Path, workers: int, n_seeds: int) -> None:
    """Project the full run from MEASURED per-method timings.

    Every figure below comes from a production-budget run on one dev goal, one
    seed, with the real twin and the real surrogate, on 6 workers with torch
    pinned to one thread each. Nothing is extrapolated any more:
    ``surrogate_only`` was the one estimated entry and has been re-measured at
    its matched 3,000-trial budget (377.1 s, down from about 10,850 s at the old
    20,000 -- see Amendment 1 in reports/preregistration_optim.md).

    The arithmetic, stated so it can be checked: each method runs
    ``n_seeds x 12`` cells, the cells are spread over ``workers``, so

        hours = seconds_per_cell x (n_seeds x 12) / workers / 3600

    which at 3 seeds and 6 workers is simply ``seconds_per_cell / 600``.
    """
    measured = {
        "best_known": 744.7,
        "sf_tpe": 499.6,
        "surrogate_only": 377.1,
        "twin_tpe": 351.5,
        "twin_cmaes": 330.9,
        "sf_cmaes": 89.5,
        "sf_random": 66.9,
    }

    n_cells_per_method = n_seeds * 12
    divisor = n_cells_per_method / workers / 3600.0

    logger.info("=== duration estimate for the full run ===")
    logger.info(
        "  %d cells (12 held-out goals x %d methods x %d seeds) on %d workers",
        n_cells_per_method * len(measured),
        len(measured),
        n_seeds,
        workers,
    )
    logger.info("  %-15s %9s %9s", "method", "s/cell", "hours")
    for name, seconds in sorted(measured.items(), key=lambda kv: -kv[1]):
        logger.info("  %-15s %9.1f %9.2f", name, seconds, seconds * divisor)
    serial = sum(measured.values())
    logger.info("  %-15s %9.1f %9.2f", "SUM", serial, serial * divisor)
    logger.info(
        "  = %.1f s/cell x %d cells / %d workers / 3600 = %.2f hours",
        serial,
        n_cells_per_method,
        workers,
        serial * divisor,
    )
    del out


def _hms(seconds: float) -> str:
    """Seconds as H:MM:SS."""
    seconds = max(int(seconds), 0)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


if __name__ == "__main__":
    sys.exit(main())
