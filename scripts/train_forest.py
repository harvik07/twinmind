"""Fit the production hazard model: a random forest behind the HealthModel Protocol.

The stopping rule in the remediation task selected this. At FD001 scale a tuned
tree ensemble on explicit degradation-trend features matches or beats a
738k-parameter sequence model, so the twin consumes the forest and A1 stays as
the deep-learning comparison. Evidence is in reports/model_cards/a1_health.md.

    python scripts/train_forest.py --subset FD001 --seeds 3
    python scripts/train_forest.py --subset all --seeds 3

Checkpoints land in models/forest/ and metrics append to the run registry, so
the forest appears in summary.md beside A1 and the other baselines.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np

from twinmind.data.cmapss_prep import prepare_cmapss
from twinmind.eval.metrics import evaluate_cmapss_test, format_mean_std
from twinmind.eval.registry import (
    RunRecord,
    config_hash,
    current_device,
    log_run,
    write_summary,
)
from twinmind.models.forest_health import ForestConfig, ForestHealthModel
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import models_dir
from twinmind.utils.seed import set_seed

logger = get_logger("train_forest")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Fit the production forest hazard model.")
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="FD001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=3, help="Independent fits.")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--spread", choices=["tree_std", "quantile"], default="tree_std")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    config = cast(DataConfig, load_config("data"))
    subsets = SUBSETS if args.subset == "all" else [args.subset]
    seeds = [args.seed + i for i in range(args.seeds)]
    output_dir = Path(args.output_dir) if args.output_dir else models_dir() / "forest"

    forest_config = ForestConfig(
        n_estimators=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        horizon_h=args.horizon,
        rul_cap=config.cmapss.rul_cap,
        spread=args.spread,  # type: ignore[arg-type]
    )

    for subset in subsets:
        per_metric: dict[str, list[float]] = {}
        for index, seed in enumerate(seeds, start=1):
            logger.info(
                "=== forest %s seed %d (independent fit %d of %d) ===",
                subset,
                seed,
                index,
                len(seeds),
            )
            set_seed(seed)
            data = prepare_cmapss(subset, config.cmapss, seed=seed)  # type: ignore[arg-type]

            started = time.perf_counter()
            model = ForestHealthModel(forest_config).fit(data, seed=seed)
            mean, std, probability = model.predict_arrays(data.x_test)
            duration = time.perf_counter() - started

            metrics = evaluate_cmapss_test(data.y_test, mean, data.units_test)
            metrics["mean_rul_std"] = float(np.mean(std))
            metrics["mean_p_fail"] = float(np.mean(probability))

            output_dir.mkdir(parents=True, exist_ok=True)
            model.save(output_dir / f"forest_{subset}_seed{seed}.joblib")

            log_run(
                RunRecord(
                    model="forest_health",
                    dataset="cmapss",
                    subset=subset,
                    seed=seed,
                    metrics=metrics,
                    config_hash=config_hash(
                        {
                            "trees": args.trees,
                            "min_samples_leaf": args.min_samples_leaf,
                            "horizon": args.horizon,
                            "spread": args.spread,
                            "window": config.cmapss.window_length,
                        }
                    ),
                    duration_s=round(duration, 2),
                    device=current_device(),
                    notes=(
                        f"{args.trees} trees; per-tree {args.spread} uncertainty; "
                        "last-window-per-unit protocol; production hazard model"
                    ),
                )
            )
            for name, value in metrics.items():
                per_metric.setdefault(name, []).append(value)
            logger.info(
                "  seed %d: RMSE %.3f  NASA %.1f  mean std %.2f  (%.1fs)",
                seed,
                metrics["rmse"],
                metrics["nasa_score"],
                metrics["mean_rul_std"],
                duration,
            )

        logger.info("--- forest %s across %d seeds ---", subset, len(seeds))
        for name, values in per_metric.items():
            logger.info("    %-18s %s", name, format_mean_std(values))

    write_summary()
    logger.info("checkpoints in %s; summary refreshed", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
