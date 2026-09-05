"""Sample scenarios, run the twin, and write rollouts for surrogate training."""

from __future__ import annotations

import argparse

from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Generate simulator rollouts for Model C.")
    parser.add_argument("--n", type=int, default=65000, help="Number of rollouts.")
    parser.add_argument("--reps", type=int, default=5, help="Replications per scenario.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processes.")
    parser.add_argument("--out", default="data/processed/rollouts", help="Output dir.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    logger.info("generate_rollouts.py starting with %s", vars(args))
    raise NotImplementedError("generate_rollouts.py is not implemented yet")


if __name__ == "__main__":
    main()
