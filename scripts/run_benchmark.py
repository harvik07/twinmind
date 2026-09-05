"""Benchmark the agent against random / grid / Optuna baselines and the oracle."""

from __future__ import annotations

import argparse

from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description='Run the optimisation benchmark.')
    parser.add_argument(
        "--method",
        choices=["agent", "random", "grid", "tpe", "cmaes", "all"],
        default="all",
        help="Which search method to benchmark.",
    )
    parser.add_argument("--budget", type=int, default=200, help="Twin evaluations allowed.")
    parser.add_argument("--out", default="reports/benchmark.json", help="Results file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    logger.info("run_benchmark.py starting with %s", vars(args))
    raise NotImplementedError("run_benchmark.py is not implemented yet")


if __name__ == "__main__":
    main()
