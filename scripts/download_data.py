"""Download C-MAPSS, AI4I 2020 and Favorita into data/raw via the Kaggle API."""

from __future__ import annotations

import argparse

from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description='Download the raw datasets.')
    parser.add_argument(
        "--dataset",
        choices=["cmapss", "ai4i", "favorita", "all"],
        default="all",
        help="Which dataset to fetch.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download if present.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    logger.info("download_data.py starting with %s", vars(args))
    raise NotImplementedError("download_data.py is not implemented yet")


if __name__ == "__main__":
    main()
