"""Precompute the demand quantile grids the rollout workers need.

    python scripts/build_demand_cache.py

Measured motivation, not a guess. A rollout worker was holding **1.33 GB**, of
which **0.71 GB was the featurised Favorita frame** and 0.15 GB the LightGBM
boosters. On a 7.3 GB machine that caps the pool at about four workers, and an
eleven-worker pool died outright producing no shards.

The twin never needs that frame at run time. It asks the sampler for one
calibrated quantile grid per ``(series, origin)``, and the grid cannot change --
the forecast depends on neither the scenario nor the seed. Computing every grid
once, offline, lets a worker carry about 0.47 GB instead of 1.33 GB.

The grids are tiny: 28 steps x 7 quantiles per origin, plus one 28x28 copula
correlation shared by all of them. A few hundred kilobytes for the whole
training design.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from twinmind.data.rollouts import ROLLOUT_DIR, origin_for_block
from twinmind.twin.demand_proc import CachedDemandModel
from twinmind.twin.runner import HORIZON_DAYS, build_resources
from twinmind.utils.logging import get_logger

logger = get_logger("build_demand_cache")

DEFAULT_PATH = ROLLOUT_DIR / "demand_grids.npz"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Precompute demand grids for rollouts.")
    parser.add_argument("--contexts", type=int, default=512)
    parser.add_argument("--t1-contexts", type=int, default=40)
    parser.add_argument("--ood-blocks", type=int, default=12)
    parser.add_argument("--out", default=str(DEFAULT_PATH))
    return parser


def design_origins(args: argparse.Namespace) -> list[pd.Timestamp]:
    """Every origin any split in the design will ask for.

    Built from the same ``origin_for_block`` the generator uses, so the cache
    cannot cover a different set of days from the one the job requests. A
    missing origin raises loudly at run time rather than silently falling back
    to a different forecast.
    """
    origins: set[pd.Timestamp] = set()
    for block in range(args.contexts):
        origins.add(origin_for_block(block, args.contexts, test=False))
    for block in range(args.t1_contexts):
        origins.add(origin_for_block(block, args.t1_contexts, test=True))
    for n_blocks in (1, args.ood_blocks):
        for block in range(n_blocks):
            origins.add(origin_for_block(block, n_blocks, test=True))
    return sorted(origins)


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    resources = build_resources()
    origins = design_origins(args)
    logger.info(
        "caching %d distinct origins, %s to %s",
        len(origins),
        origins[0].date(),
        origins[-1].date(),
    )

    cache = CachedDemandModel.build(
        resources.demand_model, resources.series_id, origins, HORIZON_DAYS
    )
    path = cache.save(args.out)
    size_kb = path.stat().st_size / 1024
    logger.info(
        "wrote %d grids for series %s in %.0f KB; workers can now skip the Favorita frame entirely",
        len(cache.grids),
        resources.series_id,
        size_kb,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
