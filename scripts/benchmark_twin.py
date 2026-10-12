"""Measure ms/run and the rollout count it allows.

    python scripts/benchmark_twin.py --runs 200

Week 4 needs about 650,000 rollouts on 8 cores in about 8 hours, which is a
budget of 0.25 s per run on one core. This script is the thing that decides
whether that holds, so it reports the p95 as well as the mean: the surrogate's
training set is built from a fixed wall-clock allowance, and a long tail costs
rollouts even when the mean looks fine.

Several scenarios are timed, not just the baseline, because run time is not
scenario-independent here. More overtime means more staffed hours and more
lots; a higher speed setpoint means shorter lots but faster wear and more
maintenance events. The slowest scenario is what the budget has to survive.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from typing import Any

import numpy as np

from twinmind.twin.runner import Scenario, build_resources, run_twin
from twinmind.utils.logging import get_logger

logger = get_logger("benchmark_twin")

#: One run must fit in this, on one core, for the week-4 rollout budget to hold.
BUDGET_SECONDS = 0.25
TARGET_ROLLOUTS = 650_000
BUDGET_CORES = 8
BUDGET_HOURS = 8.0

SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "high_overtime": {"overtime_hours": 4.0, "shift_count": 3},
    "high_speed": {f"speed_setpoint_M{i}": 1.2 for i in range(1, 6)},
    "short_pm": {f"pm_interval_M{i}": 24.0 for i in range(1, 6)},
    "offshore_supply": {"supplier_choice": 2, "reorder_point": 9000.0},
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Benchmark the digital twin.")
    parser.add_argument("--runs", type=int, default=200, help="Timed runs per scenario.")
    parser.add_argument("--pool", type=int, default=2, help="Workers for the spawn check.")
    return parser


def _time_scenario(
    name: str, overrides: dict[str, Any], runs: int, resources: Any
) -> dict[str, float]:
    """Time one scenario, discarding a warm-up run."""
    scenario = Scenario(**overrides)
    run_twin(scenario, 0, resources=resources)  # warm up lazy caches
    durations = np.empty(runs, dtype="float64")
    for index in range(runs):
        started = time.perf_counter()
        run_twin(scenario, index, resources=resources)
        durations[index] = time.perf_counter() - started
    return {
        "mean_s": float(durations.mean()),
        "p50_s": float(np.percentile(durations, 50)),
        "p95_s": float(np.percentile(durations, 95)),
        "max_s": float(durations.max()),
    }


def _worker(seed: int) -> float:
    """Run once in a worker process. Must be module-level to be picklable."""
    return float(run_twin(Scenario(), seed).profit)


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    started = time.perf_counter()
    resources = build_resources()
    logger.info("resources built in %.1f s", time.perf_counter() - started)

    results: dict[str, dict[str, float]] = {}
    for name, overrides in SCENARIOS.items():
        measured = _time_scenario(name, overrides, args.runs, resources)
        results[name] = measured
        logger.info(
            "%-16s mean %6.1f ms   p50 %6.1f ms   p95 %6.1f ms   max %6.1f ms",
            name,
            1000 * measured["mean_s"],
            1000 * measured["p50_s"],
            1000 * measured["p95_s"],
            1000 * measured["max_s"],
        )

    worst = max(results.values(), key=lambda m: m["p95_s"])
    core_seconds = BUDGET_CORES * BUDGET_HOURS * 3600.0
    affordable_mean = int(core_seconds / max(worst["mean_s"], 1e-9))
    affordable_p95 = int(core_seconds / max(worst["p95_s"], 1e-9))

    logger.info("=== budget ===")
    logger.info(
        "  worst-case mean %.1f ms, p95 %.1f ms", 1000 * worst["mean_s"], 1000 * worst["p95_s"]
    )
    logger.info("  budget is %.0f ms/run on one core", 1000 * BUDGET_SECONDS)
    logger.info(
        "  %s: %d rollouts affordable at the mean, %d at the p95 (target %d)",
        "MET" if worst["p95_s"] <= BUDGET_SECONDS else "MISSED",
        affordable_mean,
        affordable_p95,
        TARGET_ROLLOUTS,
    )

    # Multiprocessing under "spawn" is the real deployment path on Windows, and
    # a lazily-built module-level cache is exactly the thing that breaks under
    # it, so it is checked here rather than assumed.
    context = mp.get_context("spawn")
    with context.Pool(args.pool) as pool:
        profits = pool.map(_worker, [1, 2, 3, 4])
    logger.info(
        "spawn Pool(%d) returned %d finite results", args.pool, int(np.isfinite(profits).sum())
    )

    from twinmind.eval.registry import results_dir

    payload = {
        "per_scenario": results,
        "budget_seconds": BUDGET_SECONDS,
        "worst_mean_s": worst["mean_s"],
        "worst_p95_s": worst["p95_s"],
        "rollouts_affordable_mean": affordable_mean,
        "rollouts_affordable_p95": affordable_p95,
        "target_rollouts": TARGET_ROLLOUTS,
        "budget_met": bool(worst["p95_s"] <= BUDGET_SECONDS),
        "lot_size": resources.config.flow.lot_size,
    }
    (results_dir() / "twin_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
