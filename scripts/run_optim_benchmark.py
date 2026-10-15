"""The optimisation benchmark: every method on the design's held-out goals.

    python scripts/run_optim_benchmark.py --workers 3          # the full run
    python scripts/run_optim_benchmark.py --smoke              # tiny, 2 dev goals

THE DESIGN (Amendment 2). Which held-out cells exist is set by
configs/benchmark_design.yaml: 8 core goals, ``best_known`` on seed 1, every
other method on seeds 1 and 2 -- 104 cells. Reserve goals and seed 3 are
refused on the held-out split, and the analysis reads design cells only
(:func:`twinmind.optim.design.load_design_cells`). ``--max-hours N`` stops
starting cells after N hours and exits cleanly; rerunning resumes.

RESUMABLE BY CONSTRUCTION. One atomic JSON file per ``(goal, method, seed)``
cell plus a manifest. A cell whose file exists and parses is skipped, so an
interrupted multi-hour run restarts where it stopped. The manifest records the
twin config hash, the surrogate hash, every declared assumption, the budgets,
the goal library and the seeds -- but not the git commit; resuming against a
change to any of them is refused, because mixing results from two different
experiments into one table is worse than having no table.

BOUNDED MEMORY. At most 3 workers by default and at most ONE ``best_known``
cell at a time (see :func:`dispatch`), workers recycled every
:data:`MAX_TASKS_PER_CHILD` cells, and every cell records its own peak memory.

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
from twinmind.optim.design import DESIGN_PATH, BenchmarkDesign
from twinmind.optim.evaluators import (
    OPTIM_SEED_ROOT,
    STREAM_NAMES,
    EvalCost,
    SurrogateEvaluator,
    TwinEvaluator,
    feasible_robust,
)
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

#: The memory-heavy method, and how many of its cells may run at once.
#:
#: The first full run queued goal -> method -> seed, so the first six tasks
#: handed to six workers were all ``best_known`` cells, which then each tried to
#: allocate their 500,000-point Sobol block at the same moment on an 8 GB
#: machine. ``random_search`` is now bounded-memory, but ``best_known`` is still
#: the longest cell and the one that verifies 100 plans on the twin, so it gets
#: one lane: the scheduler in :func:`dispatch` never runs more than this many
#: concurrently, whatever the worker count.
HEAVY_METHOD = "best_known"
MAX_CONCURRENT_HEAVY = 1

#: Recycle a worker after this many cells. A fresh worker costs about 20 s
#: (imports plus the surrogate, demand grid and rollout loads), so recycling
#: after every cell would add ~25 min to a 230-cell run on 3 workers; every
#: 4 cells bounds any fragmentation or slow growth for about 6 min.
MAX_TASKS_PER_CHILD = 4

#: Seconds one worker takes to start: imports plus the surrogate, demand-grid
#: and rollout loads. Measured, cold, in a single process.
WORKER_STARTUP_S = 20.0

#: Measured seconds per cell at PRODUCTION budget: one dev goal, one seed, real
#: twin and surrogate, torch pinned to one thread per worker, 6 workers. Used
#: for the scheduling priority and for the duration estimate.
#:
#: A SPEED CAVEAT, measured. Re-run on 2026-10-14 at 3 workers (hot machine,
#: ~1.3 GB RAM free at idle), the two cells that finished inside a 10-minute
#: cap took 1.75x (sf_cmaes 156.9 s) and 1.76x (sf_random 117.9 s) these
#: figures, and twin_cmaes had not finished at 600 s (>= 1.81x). Every cell is
#: CPU-bound (~98% of one core), so the slowdown is the machine's clock, not
#: memory. Treat the estimate as a cool-machine lower bound and expect up to
#: ~1.8x on a hot one.
MEASURED_S_PER_CELL = {
    "best_known": 744.7,
    "sf_tpe": 499.6,
    "surrogate_only": 377.1,
    "twin_tpe": 351.5,
    "twin_cmaes": 330.9,
    "sf_cmaes": 89.5,
    "sf_random": 66.9,
}


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


def build_manifest(
    resources: Any,
    args: argparse.Namespace,
    design: BenchmarkDesign | None = None,
    plan: list[tuple[str, str, int]] | None = None,
) -> dict[str, Any]:
    """Everything that invalidates existing cells if it changes."""
    design = design if design is not None else BenchmarkDesign.load()
    seeds = sorted({s for _, _, s in plan}) if plan else list(args.seeds or [])
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
        "goals_hash": goals_hash(),
        "seed_scheme": seed_scheme(),
        "design_hash": design.hash(),
        "design": design.to_dict(),
        "methods": list(METHODS),
        "seeds": seeds,
        "kpi_order": list(KPI_ORDER),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def goals_hash() -> str:
    """A stable hash of every goal in the library, both splits.

    A goal is defined by its spec -- objective, constraints, free and fixed
    dimensions, context -- so an edited constraint threshold changes what every
    cell of that goal means. Hashed over the parsed specs rather than the YAML
    bytes, so a comment or whitespace edit does not invalidate a run.
    """
    payload = json.dumps(
        [
            {"name": g.name, "split": g.split, "spec": g.spec.to_dict()}
            for g in sorted(load_goals(), key=lambda g: g.name)
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def seed_scheme() -> dict[str, Any]:
    """What a cell's seed MEANS: the root and the named streams it derives from.

    The seed list alone is not enough. Seed 1 under a different root, or with
    the streams renamed or reordered, draws different twin replications, so a
    cell written under one scheme is not the same experiment as one under
    another.
    """
    return {"root": int(OPTIM_SEED_ROOT), "streams": list(STREAM_NAMES)}


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


#: What a results tree's cells depend on. Deliberately NOT the git commit: a
#: commit that changes neither the twin, the surrogate, the goals, the budgets,
#: the seed scheme nor the run design -- a docs edit, or the bounded-memory
#: rewrite of ``random_search``, which is bitwise identical -- must not orphan
#: hours of completed cells.
#:
#: The seed LIST is covered by ``design_hash``: since Amendment 2 the seeds a
#: run uses come from configs/benchmark_design.yaml (seed 1 for best_known,
#: 1 and 2 for the rest), so the design hash is what pins them. The bare
#: ``seeds`` entry is still recorded, for reading, but not compared: a
#: pre-amendment manifest says [1, 2, 3] and the cells it covers are valid.
INVALIDATING_KEYS = (
    "twin_config_hash",
    "surrogate_hash",
    "assumptions",
    "demand_scale",
    "budgets",
    "budgets_hash",
    "goals_hash",
    "seed_scheme",
    "design_hash",
)

#: Keys added to the manifest after a run had already started. A manifest that
#: predates them is UPGRADED rather than refused, provided every key it does
#: record matches. This is a closed list: any other missing key is a conflict.
#:
#: Why upgrading is safe:
#:
#: * ``goals_hash``, ``seed_scheme`` -- the goal library and the seed scheme are
#:   unchanged since the pre-registration commit (4f54ac8), which predates
#:   every manifest written by this script; checked with ``git diff 4f54ac8 --
#:   configs/goals src/twinmind/optim/goals.py src/twinmind/optim/evaluators.py``.
#: * ``design_hash`` -- a cell's result does not depend on the design; the
#:   design only selects WHICH cells are run and analysed. A pre-amendment
#:   results tree is a superset of the reduced design's cells, and the cells
#:   outside the design are ignored by both the runner and the analysis.
#:
#: From the upgrade on, they are enforced like any other key, so a later run
#: under a DIFFERENT design is refused.
ADDED_MANIFEST_KEYS = ("goals_hash", "seed_scheme", "design_hash")


def manifest_conflicts(existing: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Keys on which a stored manifest disagrees with the current one.

    A key in :data:`ADDED_MANIFEST_KEYS` that the stored manifest lacks is not
    a conflict; see :func:`upgrade_manifest`.
    """
    return [
        k
        for k in INVALIDATING_KEYS
        if not (k in ADDED_MANIFEST_KEYS and k not in existing)
        and existing.get(k) != current.get(k)
    ]


def upgrade_manifest(
    existing: dict[str, Any], current: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Fill keys a pre-upgrade manifest lacks, keeping everything it recorded.

    Only reached once :func:`manifest_conflicts` is empty, so every invalidating
    key the stored manifest lacks is one of :data:`ADDED_MANIFEST_KEYS`; any
    other missing key is informational (``design``) and is filled too.
    """
    added = [k for k in current if k not in existing]
    return {**existing, **{k: current[k] for k in added}}, added


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
    logger.info(
        "best_known %s seed %d: %d surrogate points in %.0f s; verifying the top %d at 30 reps",
        spec.name,
        seed,
        budget,
        time.perf_counter() - started,
        len(shortlist),
    )

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
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Max 6, spawn start method. Default 3, from measured peak memory and heat.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Held-out: only NARROWS the design's seeds. Dev/smoke: the seeds (default 1 2 3).",
    )
    parser.add_argument(
        "--design",
        default=str(DESIGN_PATH),
        help="Run design: which held-out goals and seeds (Amendment 2).",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Stop STARTING cells after this many hours; in-flight cells finish, the "
        "summary is written, and a rerun resumes from there.",
    )
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


def plan_cells(
    args: argparse.Namespace, design: BenchmarkDesign
) -> tuple[str, list[tuple[str, str, int]]]:
    """The (goal, method, seed) cells this invocation is responsible for.

    HELD-OUT: exactly the design's cells. ``--goals`` and ``--seeds`` may only
    narrow it; asking for a reserve goal or a seed outside the design is an
    error, because running one "just to see" is how a reserve turns into a
    pool to cherry-pick from.

    DEV / SMOKE: unchanged -- the requested goals (two with ``--smoke``) on the
    requested seeds, default 1 2 3. The design governs held-out goals only.
    """
    split = args.split or ("dev" if args.smoke else "held_out")
    methods = [m for m in args.methods if m in METHODS]
    library = load_goals(split=split)
    names = [g.name for g in library]
    if args.goals:
        unknown = sorted(set(args.goals) - set(names))
        if unknown:
            raise SystemExit(
                f"no such goal(s) in the {split} split: {unknown}. Available: {', '.join(names)}"
            )

    if split != "held_out":
        goals = [n for n in names if n in set(args.goals)] if args.goals else names
        if args.smoke and not args.goals:
            goals = goals[:2]
        seeds = args.seeds or [1, 2, 3]
        return split, [(g, m, s) for g in goals for m in methods for s in seeds]

    design.check_against(names, METHODS)
    if args.goals:
        reserve = sorted(set(args.goals) & set(design.reserve_goals))
        if reserve:
            raise SystemExit(
                f"{reserve} are RESERVE goals under Amendment 2: they stay in the library "
                "and are never run. See configs/benchmark_design.yaml."
            )
    plan = []
    for goal, method, seed in design.cells(methods):
        if args.goals and goal not in args.goals:
            continue
        if args.seeds and seed not in args.seeds:
            continue
        plan.append((goal, method, seed))
    if args.seeds:
        allowed = {s for m in methods for s in design.seeds_for(m)}
        outside = sorted(set(args.seeds) - allowed)
        if outside:
            raise SystemExit(
                f"seed(s) {outside} are outside the design (seeds per method: "
                f"{design.to_dict()['seeds']}). Seed 3 is excluded by rule under Amendment 2."
            )
    return split, plan


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
    traceback, because an exception escaping here propagated out of
    ``imap_unordered`` and killed the whole pool -- which is exactly what a Sobol
    ValueError in one cell did to the first production run, taking the other 251
    cells with it. A multi-hour job must not be destroyed by one bad cell.
    (:func:`dispatch` now also turns an escaped exception into a failure
    result, as a second line of defence.)

    Each cell records its own peak memory; see :class:`_PeakMemory`.
    """
    import traceback

    started = time.perf_counter()
    sampler = _PeakMemory()
    sampler.start()
    logger.info(
        "start %s / %s / seed %d in worker pid %d",
        task["goal"],
        task["method"],
        task["seed"],
        os.getpid(),
    )
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
    result.update(sampler.stop())
    return result


class _PeakMemory:
    """Peak memory of THIS process over one cell, sampled on a thread.

    Two figures, because on Windows they diverge and the second is the one
    that fails. ``rss`` is the working set, what is resident. ``private`` is
    committed memory (``memory_info().private`` on Windows, USS elsewhere), and
    a ``MemoryError`` on Windows means the commit limit was hit, not RAM: a
    freshly loaded worker measured 0.39 GB resident against 1.19 GB committed.

    Sampled every 0.25 s, so a spike shorter than that can be missed; the
    lifetime peak working set the OS records (``peak_wset``) is reported too,
    as a check that the sampler is not under-reading.
    """

    def __init__(self, interval: float = 0.25) -> None:
        import threading

        self.interval = interval
        self.peak_rss = 0.0
        self.peak_private = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        self.peak_rss = max(self.peak_rss, float(info.rss))
        private = getattr(info, "private", None)
        if private is None:
            private = psutil.Process(os.getpid()).memory_full_info().uss
        self.peak_private = max(self.peak_private, float(private))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:  # noqa: BLE001 - diagnostics must never break the run
                return
            self._stop.wait(self.interval)

    def start(self) -> None:
        """Begin sampling."""
        self._thread.start()

    def stop(self) -> dict[str, float]:
        """Stop, take a last sample, and return the figures in GB."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        out = {
            "worker_rss_gb": float("nan"),
            "cell_peak_rss_gb": float("nan"),
            "cell_peak_private_gb": float("nan"),
            "worker_lifetime_peak_rss_gb": float("nan"),
        }
        try:
            import psutil

            self._sample()
            info = psutil.Process(os.getpid()).memory_info()
            out["worker_rss_gb"] = float(info.rss) / 1e9
            out["cell_peak_rss_gb"] = self.peak_rss / 1e9
            out["cell_peak_private_gb"] = self.peak_private / 1e9
            lifetime = getattr(info, "peak_wset", None)
            if lifetime is not None:
                out["worker_lifetime_peak_rss_gb"] = float(lifetime) / 1e9
        except Exception:  # noqa: BLE001 - diagnostics must never break the run
            pass
        return out


def order_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Priority order: the heavy method first, then longest expected cell first.

    The heavy cells form a serial lane (see :data:`MAX_CONCURRENT_HEAVY`) and
    are the longest, so that lane is the critical path and must never sit idle.
    The rest is longest-processing-time first, the standard greedy for keeping
    the last worker from finishing long after the others.
    """
    return sorted(
        tasks,
        key=lambda t: (
            t["method"] != HEAVY_METHOD,
            -MEASURED_S_PER_CELL.get(t["method"], 0.0),
            t["goal"],
            t["seed"],
        ),
    )


def _next_task(
    pending: list[dict[str, Any]], heavy_running: int, max_heavy: int
) -> dict[str, Any] | None:
    """Pop the first pending task allowed to start now, or None."""
    for index, task in enumerate(pending):
        if task["method"] != HEAVY_METHOD or heavy_running < max_heavy:
            return pending.pop(index)
    return None


def dispatch(
    pool: Any,
    func: Any,
    tasks: list[dict[str, Any]],
    workers: int,
    max_heavy: int = MAX_CONCURRENT_HEAVY,
    deadline: float | None = None,
    clock: Any = time.monotonic,
) -> Any:
    """Run ``func`` over ``tasks`` on ``pool``; yield results as they finish.

    Unlike ``imap_unordered``, which hands tasks out in list order, this keeps
    at most ``workers`` tasks in flight and at most ``max_heavy`` of the heavy
    method, so no queue order can put two ``best_known`` cells side by side.

    ``deadline`` (a ``clock()`` value, from ``--max-hours``) stops new cells
    from STARTING once it has passed. Cells already running are allowed to
    finish and are yielded as usual, because killing one mid-cell throws away
    up to ~25 minutes of work. The caller sees fewer results than tasks and
    reports the difference as not started.
    """
    import queue

    pending = order_tasks(tasks)
    finished: queue.Queue[dict[str, Any]] = queue.Queue()
    in_flight = heavy = 0
    while pending or in_flight:
        if deadline is not None and pending and clock() >= deadline:
            logger.warning(
                "--max-hours reached: starting no more cells; %d in flight will finish, "
                "%d not started",
                in_flight,
                len(pending),
            )
            pending = []
            if not in_flight:
                break
        while in_flight < workers:
            task = _next_task(pending, heavy, max_heavy)
            if task is None:
                break
            heavy += task["method"] == HEAVY_METHOD
            in_flight += 1
            pool.apply_async(
                func,
                (task,),
                callback=finished.put,
                error_callback=lambda error, task=task: finished.put(
                    _dispatch_failure(task, error)
                ),
            )
        while True:
            # A timeout, so Ctrl-C is not stuck behind an uninterruptible wait.
            try:
                result = finished.get(timeout=1.0)
                break
            except queue.Empty:
                continue
        in_flight -= 1
        heavy -= result["method"] == HEAVY_METHOD
        yield result


def _dispatch_failure(task: dict[str, Any], error: BaseException) -> dict[str, Any]:
    """A failure result for a task whose worker could not even return one."""
    return {
        "ok": False,
        "goal": task["goal"],
        "method": task["method"],
        "seed": task["seed"],
        "error": f"{type(error).__name__}: {error}",
        "traceback": "",
        "wall_seconds": float("nan"),
        "smoke": task["smoke"],
    }


def simulate_hours(
    counts: dict[str, int],
    workers: int,
    seconds: dict[str, float] | None = None,
    max_heavy: int = MAX_CONCURRENT_HEAVY,
    startup_s: float = WORKER_STARTUP_S,
) -> float:
    """Wall-clock hours for ``counts`` cells under exactly :func:`dispatch`'s policy.

    A discrete-event replay of the scheduler with each cell taking its measured
    seconds, plus a worker start-up every :data:`MAX_TASKS_PER_CHILD` cells. It
    replaces "total work / workers", which ignores that the heavy lane is
    serial: with one ``best_known`` at a time, the run can never finish before
    ``n_best_known x seconds_per_best_known``, however many workers there are.
    """
    import heapq

    seconds = MEASURED_S_PER_CELL if seconds is None else seconds
    pending = order_tasks(
        [
            {"method": method, "goal": "", "seed": index}
            for method, n in counts.items()
            for index in range(n)
        ]
    )
    clock = 0.0
    running: list[tuple[float, int, str]] = []  # (finish time, tiebreak, method)
    heavy = started = 0
    while pending or running:
        while len(running) < workers:
            task = _next_task(pending, heavy, max_heavy)
            if task is None:
                break
            heavy += task["method"] == HEAVY_METHOD
            # Each worker starts fresh once per MAX_TASKS_PER_CHILD cells.
            cost = seconds[task["method"]] + startup_s / MAX_TASKS_PER_CHILD
            heapq.heappush(running, (clock + cost, started, task["method"]))
            started += 1
        clock, _, method = heapq.heappop(running)
        heavy -= method == HEAVY_METHOD
    return clock / 3600.0


def main() -> int:
    """Entry point."""
    import multiprocessing as mp

    args = build_parser().parse_args()
    if args.max_hours is not None and args.max_hours <= 0:
        raise SystemExit(f"--max-hours must be positive, got {args.max_hours}")
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

    design = BenchmarkDesign.load(Path(args.design))
    split, plan = plan_cells(args, design)
    logger.info(
        "split=%s, design amendment %d (%s), %d cell(s) in the plan",
        split,
        design.amendment,
        design.hash(),
        len(plan),
    )

    manifest = build_manifest(resources, args, design, plan)
    manifest["smoke"] = args.smoke
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force_new_manifest:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        conflicts = manifest_conflicts(existing, manifest)
        if conflicts:
            raise SystemExit(
                f"refusing to resume: the manifest in {out} disagrees on {conflicts}.\n"
                "The twin, its assumptions, the surrogate, the goals, the budgets or the seeds "
                "changed since those cells were written, so mixing them would put results from "
                "two different experiments in one table.\n"
                "Pass --force-new-manifest to overwrite, after moving the old cells."
            )
        upgraded, added = upgrade_manifest(existing, manifest)
        if added:
            write_atomic(upgraded, manifest_path)
            logger.warning(
                "manifest in %s predates %s; every key it records matches, so it was "
                "upgraded in place and its completed cells are kept",
                out,
                added,
            )
    else:
        write_atomic(manifest, manifest_path)

    tasks = []
    for goal_name, method, seed in plan:
        if cell_done(out / Cell(goal=goal_name, method=method, seed=seed).filename):
            continue
        tasks.append(
            {
                "goal": goal_name,
                "method": method,
                "seed": seed,
                "smoke": args.smoke,
                "novelty_active": True,
            }
        )

    logger.info(
        "%d of %d cells outstanding (%d goals, %d methods) on %d workers",
        len(tasks),
        len(plan),
        len({g for g, _, _ in plan}),
        len({m for _, m, _ in plan}),
        workers,
    )
    if tasks and not args.smoke:
        outstanding: dict[str, int] = {}
        for task in tasks:
            outstanding[task["method"]] = outstanding.get(task["method"], 0) + 1
        _estimate_full_run(outstanding, workers)
    if not tasks:
        logger.info("nothing to do; every cell is present")
        if args.smoke:
            # Still print the projection: a resumed smoke run is exactly when
            # someone wants to know how long the real one will take.
            _estimate_full_run(design_counts(design), workers)
        return 0

    started = time.perf_counter()
    deadline = time.monotonic() + args.max_hours * 3600.0 if args.max_hours is not None else None
    if args.max_hours is not None:
        logger.info(
            "--max-hours %.2f: no new cell starts after %s from now",
            args.max_hours,
            _hms(args.max_hours * 3600.0),
        )
    done = 0
    rss: dict[str, list[tuple[float, float]]] = {}
    context = mp.get_context("spawn")
    pool = context.Pool(workers, initializer=_init_worker, maxtasksperchild=MAX_TASKS_PER_CHILD)
    failures: list[dict[str, Any]] = []
    failure_dir = out / "failures"
    try:
        for result in dispatch(pool, _run_one, tasks, workers, deadline=deadline):
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
            peak = result.get("cell_peak_rss_gb", np.nan)
            if np.isfinite(peak):
                rss.setdefault(result["method"], []).append(
                    (peak, result.get("cell_peak_private_gb", np.nan))
                )
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
    not_started = len(tasks) - done
    logger.info(
        "finished %d cells in %.1f min (%d succeeded, %d FAILED)",
        done,
        elapsed / 60.0,
        done - len(failures),
        len(failures),
    )
    if not_started:
        logger.warning(
            "stopped by --max-hours: %d cell(s) not started. Every finished cell is written; "
            "rerun the same command to continue.",
            not_started,
        )
    for method, peaks in sorted(rss.items()):
        logger.info(
            "peak memory %-15s rss %.2f GB, committed %.2f GB (max over %d cells)",
            method,
            max(p[0] for p in peaks),
            max(p[1] for p in peaks),
            len(peaks),
        )
    if args.smoke:
        _estimate_full_run(design_counts(design), workers)

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
        "planned": len(plan),
        "outstanding_at_start": len(tasks),
        "not_started": not_started,
        "stopped_by_max_hours": bool(not_started),
        "max_hours": args.max_hours,
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


def design_counts(design: BenchmarkDesign) -> dict[str, int]:
    """Cells per method in the full design: core goals x that method's seeds."""
    return {m: len(design.core_goals) * len(design.seeds_for(m)) for m in METHODS}


def _estimate_full_run(counts: dict[str, int], workers: int) -> None:
    """Project a run's wall-clock time from MEASURED per-method timings.

    ``counts`` is cells per method -- the full grid, or just what is still
    outstanding on a resume. The seconds per cell are
    :data:`MEASURED_S_PER_CELL`; the hours come from :func:`simulate_hours`,
    which replays the real scheduling policy. That matters: with at most one
    ``best_known`` at a time, the run cannot finish before the serial
    ``best_known`` lane does, so "total work / workers" understates it.

    The bound stated so it can be checked by hand:

        hours >= max(sum(seconds x cells) / workers,
                     best_known seconds x best_known cells) / 3600
    """
    seconds = MEASURED_S_PER_CELL
    total = sum(seconds[m] * n for m, n in counts.items())
    lane = seconds[HEAVY_METHOD] * counts.get(HEAVY_METHOD, 0)
    logger.info("=== duration estimate: %d cells ===", sum(counts.values()))
    logger.info("  %-15s %6s %9s %11s", "method", "cells", "s/cell", "cell-hours")
    for name, n in sorted(counts.items(), key=lambda kv: -seconds[kv[0]]):
        logger.info("  %-15s %6d %9.1f %11.2f", name, n, seconds[name], seconds[name] * n / 3600)
    logger.info("  %-15s %6d %9s %11.2f", "SUM", sum(counts.values()), "", total / 3600)
    logger.info(
        "  serial %s lane: %.2f h, a floor whatever the worker count", HEAVY_METHOD, lane / 3600
    )
    for n_workers in sorted({2, 3, 4, workers}):
        logger.info(
            "  %d workers: %.2f h simulated (lower bound %.2f h)%s",
            n_workers,
            simulate_hours(counts, n_workers),
            max(total / n_workers, lane) / 3600,
            "  <- this run" if n_workers == workers else "",
        )


def _hms(seconds: float) -> str:
    """Seconds as H:MM:SS."""
    seconds = max(int(seconds), 0)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


if __name__ == "__main__":
    sys.exit(main())
