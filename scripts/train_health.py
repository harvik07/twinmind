"""Train Model A1 (CNN-BiLSTM RUL + failure head) on C-MAPSS.

Trains three independent seeds by default and reports mean +/- std for every
metric, because a single seed is not a result. The three runs are three separate
fits -- fresh weights, fresh data shuffling -- not three evaluations of one
checkpoint, and the log says so per run.

Runs unchanged on CPU and CUDA; AMP is enabled only under CUDA.

    python scripts/train_health.py --subset FD001 --epochs 3
    python scripts/train_health.py --subset all --seeds 3
    python scripts/train_health.py --subset FD001 --tune-lambda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import torch

from twinmind.data.cmapss_prep import prepare_cmapss
from twinmind.eval.metrics import evaluate_cmapss_test, format_mean_std, summarise
from twinmind.eval.registry import (
    RunRecord,
    config_hash,
    current_device,
    log_run,
    results_dir,
    write_summary,
)
from twinmind.models.a1_train import TrainConfig, ensemble_predictions, train_a1
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT, models_dir
from twinmind.utils.seed import set_seed

logger = get_logger("train_health")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def resolve_device(requested: str) -> str:
    """Map ``auto`` onto whatever is actually available."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Train the A1 machine-health model.")
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="FD001")
    parser.add_argument("--seed", type=int, default=42, help="First seed; later ones increment.")
    parser.add_argument("--seeds", type=int, default=3, help="Independent training runs.")
    parser.add_argument("--config", default="data", help="Config name for C-MAPSS settings.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    # Default None, not a literal: an argparse default silently shadows the
    # dataclass default, so TrainConfig stays the single source of truth.
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--horizon", type=int, default=30, help="Failure horizon h, in cycles.")
    parser.add_argument(
        "--lambda-fail",
        type=float,
        default=0.5,
        help="Weight on the focal-BCE term. Ignored when --tune-lambda is set.",
    )
    parser.add_argument(
        "--tune-lambda",
        action="store_true",
        help="Search lambda over {0.1, 0.5, 1.0} on validation RMSE.",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=None, help="Checkpoint directory.")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads.")
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    config = cast(DataConfig, load_config(args.config))
    device = resolve_device(args.device)
    learning_rate = args.lr if args.lr is not None else TrainConfig().learning_rate
    subsets = SUBSETS if args.subset == "all" else [args.subset]
    seeds = [args.seed + i for i in range(args.seeds)]
    output_dir = Path(args.output_dir) if args.output_dir else models_dir() / "a1"

    settings = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        patience=args.patience,
        mc_passes=args.mc_passes,
        lambda_candidates=(0.1, 0.5, 1.0) if args.tune_lambda else (args.lambda_fail,),
    )

    logger.info(
        "A1 on %s | %d independent seeds %s | device=%s | epochs<=%d | lambda=%s",
        subsets,
        len(seeds),
        seeds,
        device,
        args.epochs,
        "tuned {0.1,0.5,1.0}" if args.tune_lambda else args.lambda_fail,
    )

    summary: dict[str, dict[str, list[float]]] = {}

    for subset in subsets:
        per_metric: dict[str, list[float]] = {}
        for index, seed in enumerate(seeds, start=1):
            logger.info(
                "=== %s seed %d (independent training run %d of %d) ===",
                subset,
                seed,
                index,
                len(seeds),
            )
            set_seed(seed)
            data = prepare_cmapss(subset, config.cmapss, seed=seed)  # type: ignore[arg-type]

            model, result = train_a1(
                data,
                seed=seed,
                horizon_h=args.horizon,
                dropout=args.dropout,
                lambda_fail=None if args.tune_lambda else args.lambda_fail,
                train_config=settings,
                device=device,
                checkpoint_dir=output_dir,
            )

            log_run(
                RunRecord(
                    model="a1_cnn_bilstm",
                    dataset="cmapss",
                    subset=subset,
                    seed=seed,
                    metrics=result.metrics,
                    config_hash=config_hash(
                        {
                            "dropout": args.dropout,
                            "horizon": args.horizon,
                            "lambda": result.lambda_fail,
                            "epochs": args.epochs,
                            "batch": args.batch_size,
                            "lr": learning_rate,
                            "point_estimate": "deterministic",
                            "window": config.cmapss.window_length,
                        }
                    ),
                    duration_s=round(result.duration_s, 2),
                    device=current_device(),
                    notes=(
                        f"lambda={result.lambda_fail}; best epoch {result.best_epoch}; "
                        f"{result.n_parameters} params; last-window-per-unit protocol"
                    ),
                )
            )

            history_path = results_dir() / f"history_a1_{subset}_seed{seed}.json"
            history_path.write_text(
                json.dumps(result.history, indent=1) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            for name, value in result.metrics.items():
                per_metric.setdefault(name, []).append(value)
            logger.info(
                "  seed %d: RMSE %.3f  NASA %.1f  (%.0fs)",
                seed,
                result.metrics["rmse"],
                result.metrics["nasa_score"],
                result.duration_s,
            )

        # The three seeds are three independent models, so averaging them is a
        # fourth model. Logged as its own row -- never folded into the per-seed
        # statistics, which would overstate what one training run buys.
        checkpoints = [output_dir / f"a1_{subset}_seed{seed}.pt" for seed in seeds]
        existing = [path for path in checkpoints if path.exists()]
        if len(existing) > 1:
            ensemble = ensemble_predictions(existing, data, device=torch.device(device))
            ensemble_metrics = evaluate_cmapss_test(data.y_test, ensemble, data.units_test)
            log_run(
                RunRecord(
                    model="a1_ensemble",
                    dataset="cmapss",
                    subset=subset,
                    seed=-1,
                    metrics=ensemble_metrics,
                    config_hash=config_hash(
                        {
                            "members": len(existing),
                            "kind": "seed-mean",
                            # The members' own config must be part of the
                            # ensemble's identity, or two ensembles built from
                            # differently-trained seeds hash alike and get averaged.
                            "lr": learning_rate,
                            "dropout": args.dropout,
                            "point_estimate": "deterministic",
                        }
                    ),
                    duration_s=0.0,
                    device=current_device(),
                    notes=(
                        f"mean of {len(existing)} independently trained seeds; "
                        "last-window-per-unit protocol"
                    ),
                )
            )
            logger.info(
                "  %d-seed ensemble: RMSE %.3f  NASA %.1f",
                len(existing),
                ensemble_metrics["rmse"],
                ensemble_metrics["nasa_score"],
            )

        summary[subset] = per_metric
        logger.info("--- %s across %d seeds ---", subset, len(seeds))
        for name, values in per_metric.items():
            if name.startswith("n_"):
                continue
            logger.info("    %-14s %s", name, format_mean_std(values))

    write_summary()
    logger.info("wrote reports/results/summary.md")

    for subset, metrics in summary.items():
        if "rmse" in metrics:
            stats = summarise(metrics["rmse"])
            logger.info(
                "FINAL %s: RMSE %.3f +/- %.3f over %d seeds",
                subset,
                stats["mean"],
                stats["std"],
                int(stats["n"]),
            )
    logger.info("checkpoints in %s", output_dir.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
