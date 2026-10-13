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
BUDGETS = {
    "sf_random": 50_000,
    "sf_tpe": 3_000,
    "sf_cmaes": 20_000,
    "surrogate_only": 20_000,
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
        "methods": list(METHODS),
        "seeds": list(args.seeds),
        "kpi_order": list(KPI_ORDER),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


INVALIDATING_KEYS = (
    "twin_config_hash",
    "surrogate_hash",
    "assumptions",
    "demand_scale",
    "budgets",
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
    """True when a cell's file exists and parses."""
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return True


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
    parser.add_argument("--force-new-manifest", action="store_true")
    return parser


_WORKER: dict[str, Any] = {}


def _init_worker() -> None:
    """Workers ignore Ctrl-C so the parent can shut the pool down cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


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
    """Worker entry point. Module-level and plain-data in/out, for spawn."""
    state = _worker_context()
    goal = next(g for g in load_goals() if g.name == task["goal"])
    cell = Cell(goal=task["goal"], method=task["method"], seed=task["seed"])
    context = _context_for_goal(goal, state)
    novelty = state["novelty"] if task["novelty_active"] else None
    result = run_cell(
        cell, goal, context, state["surrogate"], state["resources"], novelty, task["smoke"]
    )
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

    goals = load_goals(split="dev" if args.smoke else "held_out")
    if args.smoke:
        goals = goals[:2]
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
    try:
        for result in pool.imap_unordered(_run_one, tasks):
            cell = Cell(goal=result["goal"], method=result["method"], seed=result["seed"])
            write_atomic(result, out / cell.filename)
            done += 1
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
    logger.info("wrote %d cells in %.1f min", done, elapsed / 60.0)
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

    summary = {"cells": done, "elapsed_s": elapsed, "out": str(out), "manifest": manifest}
    write_atomic(
        summary,
        results_dir()
        / ("optim_benchmark_smoke.json" if args.smoke else "optim_benchmark_run.json"),
    )
    return 0


def _estimate_full_run(out: Path, workers: int, n_seeds: int) -> None:
    """Project the full run from the configured budgets, not from smoke wall time.

    Scaling the smoke wall time by a guessed factor spans 7.5 to 30 hours, which
    is too wide to plan around. The full run is dominated by TWIN runs, and the
    number of those is known exactly from the budgets rather than guessed, so
    the projection is built from them and a measured throughput.
    """
    n_cells = n_seeds * 12  # 12 held-out goals
    # Twin runs per cell, from the configured budgets.
    per_cell_twin = {
        "sf_random": 20 * 10 + 30,
        "sf_tpe": 20 * 10 + 30,
        "sf_cmaes": 20 * 10 + 30,
        "surrogate_only": 30,
        "twin_tpe": 300 * 5 + 30,
        "twin_cmaes": 300 * 5 + 30,
        "best_known": 100 * 30,
    }
    per_cell_surrogate = {
        "sf_random": 50_000,
        "sf_tpe": 3_000,
        "sf_cmaes": 20_000,
        "surrogate_only": 20_000,
        "twin_tpe": 0,
        "twin_cmaes": 0,
        "best_known": 500_000,
    }
    total_twin = sum(v * n_cells for v in per_cell_twin.values())
    total_surrogate = sum(v * n_cells for v in per_cell_surrogate.values())

    # Measured on this machine: 27.5 twin runs/s on 6 workers with the grid
    # cache (reports/data_cards/rollouts.md section 4).
    twin_hours = total_twin / 27.5 / 3600.0
    # The surrogate's batched rate is 0.079 ms/scenario, but TPE and CMA-ES
    # evaluate in small batches where per-call overhead dominates, so the range
    # spans both.
    best_case = total_surrogate * 0.079e-3 / workers / 3600.0
    worst_case = total_surrogate * 2.0e-3 / workers / 3600.0

    logger.info("=== duration estimate for the full run ===")
    logger.info(
        "  %d cells (12 held-out goals x %d methods x %d seeds)",
        n_cells * len(per_cell_twin),
        len(per_cell_twin),
        n_seeds,
    )
    logger.info("  %s twin runs -> %.1f h at a measured 27.5 runs/s", f"{total_twin:,}", twin_hours)
    logger.info(
        "  %s surrogate evals -> %.1f to %.1f h depending on batch size",
        f"{total_surrogate:,}",
        best_case,
        worst_case,
    )
    logger.info(
        "  TOTAL: %.1f to %.1f hours on %d workers (twin-dominated)",
        twin_hours + best_case,
        twin_hours + worst_case,
        workers,
    )
    del out


def _hms(seconds: float) -> str:
    """Seconds as H:MM:SS."""
    seconds = max(int(seconds), 0)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


if __name__ == "__main__":
    sys.exit(main())
