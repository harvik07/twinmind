"""Train Model C (heteroscedastic residual MLP) on the generated rollouts."""

from __future__ import annotations

import argparse

from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description='Train the twin surrogate.')
    parser.add_argument("--rollouts", default="data/processed/rollouts", help="Input dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    logger.info("train_surrogate.py starting with %s", vars(args))
    raise NotImplementedError("train_surrogate.py is not implemented yet")


if __name__ == "__main__":
    main()
