"""Twin-component ablation of the agent benchmark (EXPLORATORY, pre-declared).

    python scripts/ablate_agent_bench.py --workers 3

Re-evaluates every agent final plan and its baseline at 30 paired reps on the
reeval stream (goal seed = optimiser seed) with the twin's
``health_replay.hazard_source`` switched from ``learned`` to ``fixed_mtbf``
(a configuration value, no code change), and compares with the learned-hazard
evaluation already in each agent cell: KPI levels, feasibility, and whether
the plan still beats the baseline.

Point demand is NOT ablated: the twin has no point-demand mode. Its only lever,
``demand_spread_multiplier`` = 0, collapses each randomly sampled path to a
constant at that path's own median, which is not the forecast's median path, so
producing point demand would need a code change.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from functools import lru_cache
from typing import Any

from twinmind.eval.registry import results_dir
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("ablate_agent_bench")
CELLS = PROJECT_ROOT / "data" / "processed" / "agent_bench" / "full"
OUT = results_dir() / "agent_bench_ablation.json"


@lru_cache(maxsize=1)
def _fixed_mtbf_toolkit() -> Any:
    from twinmind.agents.graph import _shared_state, twin_toolkit
    from twinmind.twin.demand_proc import CachedDemandModel
    from twinmind.twin.runner import build_resources
    from twinmind.utils.config import load_config

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass
    twin: Any = load_config("twin")
    fixed = twin.model_copy(
        update={
            "health_replay": twin.health_replay.model_copy(update={"hazard_source": "fixed_mtbf"})
        }
    )
    learned = _shared_state()["resources"]
    model = CachedDemandModel.load(
        PROJECT_ROOT / "data" / "processed" / "rollouts" / "demand_grids.npz"
    )
    resources = build_resources(
        config=fixed,
        demand_model=model,
        series_id=model.series_ids[0],
        demand_scale=learned.demand_scale,  # same demand calibration as the benchmark
    )
    agent: Any = load_config("agent")
    toolkit = twin_toolkit(agent)
    toolkit.resources = resources
    return toolkit


def _cell(name: str) -> dict[str, Any]:
    from twinmind.optim.contracts import ObjectiveSpec

    cell = json.loads((CELLS / name).read_text(encoding="utf-8"))
    result = cell["result"]
    spec = ObjectiveSpec.from_dict(result["spec"])
    toolkit = _fixed_mtbf_toolkit()
    toolkit.goal_seed = cell["seed"]
    toolkit._baseline_cache = {}
    fixed = toolkit.evaluate(spec, result["plan"]["scenario"])
    learned = result["baseline_comparison"]

    def summary(ev: dict[str, Any]) -> dict[str, Any]:
        diff = ev["objective"]["diff"]
        return {
            "plan_objective": ev["objective"]["plan"]["mean"],
            "baseline_objective": ev["objective"]["baseline"]["mean"],
            "diff": diff["mean"],
            "diff_ci95": [diff["ci_low"], diff["ci_high"]],
            "beats": bool(
                diff["ci_low"] > 0 or (ev["feasible_point"] and not ev["baseline_feasible_point"])
            ),
            "feasible": bool(ev["feasible_point"]),
            "baseline_feasible": bool(ev["baseline_feasible_point"]),
            "kpis": {k: v["plan"]["mean"] for k, v in ev["kpis"].items()},
            "baseline_kpis": {k: v["baseline"]["mean"] for k, v in ev["kpis"].items()},
        }

    return {
        "goal": cell["goal"],
        "seed": cell["seed"],
        "category": cell["category"],
        "learned": summary(learned),
        "fixed_mtbf": summary(fixed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hazard-source ablation (exploratory).")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    names = sorted(p.name for p in CELLS.glob("*__agent__seed*.json"))
    with mp.get_context("spawn").Pool(max(1, min(args.workers, 3))) as pool:
        rows = pool.map(_cell, names)
    flips = [r for r in rows if r["learned"]["beats"] != r["fixed_mtbf"]["beats"]]
    feas_flips = [r for r in rows if r["learned"]["feasible"] != r["fixed_mtbf"]["feasible"]]
    OUT.write_text(
        json.dumps(
            {
                "label": "EXPLORATORY: hazard_source learned vs fixed_mtbf; point demand not "
                "supported without code changes",
                "cells": rows,
                "beats_baseline_flips": [(r["goal"], r["seed"]) for r in flips],
                "feasibility_flips": [(r["goal"], r["seed"]) for r in feas_flips],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info(
        "%d cells; 'beats baseline' changed in %d, plan feasibility changed in %d",
        len(rows),
        len(flips),
        len(feas_flips),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
