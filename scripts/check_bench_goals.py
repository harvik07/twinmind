"""Check each benchmark goal's label on the twin, at a small budget, BEFORE
pre-registration.

    python scripts/check_bench_goals.py --ids G01 G02 ...

Per goal: a cheap surrogate-first search (Sobol over the surrogate, 4,000
points; shortlist of 10 verified at 5 reps on the search stream; up to two
rounds), then the chosen plan and the current policy re-evaluated at 30 paired
reps on the reeval stream (seed 1).

* feasible / ambiguous_answerable -- expects a point-feasible re-evaluated plan.
* infeasible -- expects none.
* no_room -- expects the paired objective gain's 95% CI NOT to lie above 0.

A goal whose check disagrees with its label is relabelled or redesigned BEFORE
the pre-registration is written; never after. One worker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from twinmind.agents.bench_goals import load_bench_goals
from twinmind.agents.graph import twin_toolkit
from twinmind.eval.registry import results_dir
from twinmind.utils.config import load_config
from twinmind.utils.logging import get_logger

logger = get_logger("check_bench_goals")
OUT = results_dir() / "agent_bench_goal_check.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check benchmark goal labels on the twin.")
    parser.add_argument("--ids", nargs="+", default=None)
    args = parser.parse_args()

    config: Any = load_config("agent")
    explorer = config.explorer.model_copy(
        update={"optimiser": "random", "budget": 4000, "k": 10, "verify_reps": 5, "max_rounds": 1}
    )
    toolkit = twin_toolkit(config.model_copy(update={"explorer": explorer}))
    goals = [g for g in load_bench_goals() if args.ids is None or g.id in args.ids]
    results: dict[str, Any] = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for goal in goals:
        started = time.perf_counter()
        spec = goal.spec
        exploration = toolkit.explore(spec, 1.0, 1)
        evaluation = toolkit.evaluate(spec, exploration.chosen["scenario"])
        diff = evaluation["objective"]["diff"]
        feasible = bool(evaluation["feasible_point"])
        improves = diff["ci_low"] > 0
        if goal.category in ("feasible", "ambiguous_answerable"):
            agrees = feasible
        elif goal.category == "infeasible":
            agrees = not feasible
        else:
            agrees = not improves
        results[goal.id] = {
            "category": goal.category,
            "context_id": goal.context_id,
            "exploration_status": exploration.status,
            "feasible_point": feasible,
            "feasible_robust": bool(evaluation["feasible_robust"]),
            "baseline_feasible_point": bool(evaluation["baseline_feasible_point"]),
            "objective_diff": diff["mean"],
            "objective_diff_ci95": [diff["ci_low"], diff["ci_high"]],
            "constraint_margins": evaluation["constraint_margins"],
            "label_confirmed": agrees,
            "seconds": time.perf_counter() - started,
        }
        logger.info(
            "%-4s %-21s ctx %2d  explore=%-21s feasible=%-5s robust=%-5s base_feasible=%-5s "
            "gain %.4g [%.4g, %.4g]  -> %s  (%.0fs)",
            goal.id,
            goal.category,
            goal.context_id,
            exploration.status,
            feasible,
            evaluation["feasible_robust"],
            evaluation["baseline_feasible_point"],
            diff["mean"],
            diff["ci_low"],
            diff["ci_high"],
            "CONFIRMED" if agrees else "DISAGREES",
            time.perf_counter() - started,
        )
        OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
