"""Test the A1 remediation hypotheses one at a time on FD001 and record each effect.

Each hypothesis is measured against the same baseline and the same protocol
(last-window-per-unit), changing exactly one thing, so the reported delta is
attributable. Results append to reports/results/hypotheses.jsonl.

    python scripts/run_hypotheses.py --list
    python scripts/run_hypotheses.py --only H1 H2
    python scripts/run_hypotheses.py                 # all, in order

Stopping rule: the search exists to beat the random forest at RMSE 12.72. The
runner prints whether each variant clears that line, and --stop-when-beaten ends
the sweep at the first one that does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, cast

from twinmind.data.cmapss_prep import prepare_cmapss
from twinmind.eval.registry import results_dir
from twinmind.models.a1_cv import train_a1_cv
from twinmind.models.a1_train import TrainConfig, train_a1
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger("hypotheses")

#: The number to beat. Random forest on handcrafted window features, FD001,
#: last-window-per-unit, mean of three seeds.
FOREST_RMSE = 12.72
SUBSET = "FD001"


@dataclass
class Hypothesis:
    """One variant: a label, what it changes, and why it might help."""

    key: str
    title: str
    rationale: str
    arch: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    scaling: str = "zscore"
    cross_validate: bool = False
    lambda_fail: float | None = 0.5


HYPOTHESES: list[Hypothesis] = [
    Hypothesis(
        "H0",
        "baseline (v0.3-health)",
        "The shipped configuration, re-measured here so every delta shares a protocol.",
    ),
    Hypothesis(
        "H1",
        "5-fold GroupKFold CV + fold ensemble",
        "A 20-engine validation set is too noisy for patience-15 early stopping; "
        "averaging five folds should pick the epoch on signal rather than noise.",
        cross_validate=True,
    ),
    Hypothesis(
        "H2",
        "smaller trunk (32/64 conv, 64 LSTM x1)",
        "738k parameters against 14.5k windows is ~50x over-parameterised; published "
        "C-MAPSS CNN-LSTMs sit at 100-250k.",
        arch={"conv1_channels": 32, "conv2_channels": 64, "lstm_hidden": 64, "lstm_layers": 1},
    ),
    Hypothesis(
        "H3a",
        "noise sigma 0.01",
        "Mild input noise as a regulariser.",
        train={"noise_sigma": 0.01},
    ),
    Hypothesis(
        "H3b",
        "noise sigma 0.03",
        "Moderate input noise.",
        train={"noise_sigma": 0.03},
    ),
    Hypothesis(
        "H3c",
        "noise sigma 0.05",
        "Strong input noise.",
        train={"noise_sigma": 0.05},
    ),
    Hypothesis(
        "H3d",
        "noise 0.03 + channel mask 0.1 + jitter 0.05",
        "All three augmentations together.",
        train={"noise_sigma": 0.03, "mask_prob": 0.1, "jitter": 0.05},
    ),
    Hypothesis(
        "H4a",
        "LayerNorm in the conv block",
        "BatchNorm immediately followed by Dropout is a known bad pairing: dropout "
        "shifts the variance BN just normalised.",
        arch={"conv_norm": "layer"},
    ),
    Hypothesis(
        "H4b",
        "no conv normalisation",
        "Removes the interaction entirely.",
        arch={"conv_norm": "none"},
    ),
    Hypothesis(
        "H5a",
        "lambda = 0 (RUL head only)",
        "Tests whether the auxiliary failure head is degrading the trunk at all.",
        lambda_fail=0.0,
    ),
    Hypothesis(
        "H5b",
        "lambda = 0.1",
        "A lighter auxiliary weight.",
        lambda_fail=0.1,
    ),
    Hypothesis(
        "H5c",
        "lambda = 0.5, failure head detached",
        "Keeps the head but stops its gradient reshaping the shared trunk.",
        arch={"detach_fail_head": True},
    ),
    Hypothesis(
        "H6",
        "min-max scaling to [-1, 1]",
        "The most common C-MAPSS convention, against the current z-score.",
        scaling="minmax",
    ),
    Hypothesis(
        "H7",
        "trend-feature branch (last, slope, delta)",
        "Hands the network the degradation trend the random forest wins with, "
        "instead of making it infer the slope from raw sequence.",
        arch={"trend_branch": True},
    ),
    Hypothesis(
        "H7cv",
        "trend branch + 5-fold CV ensemble",
        "The two most promising changes together.",
        arch={"trend_branch": True},
        cross_validate=True,
    ),
]


def run_one(
    hypothesis: Hypothesis, config: DataConfig, seed: int, device: str, epochs: int, folds: int
) -> dict[str, Any]:
    """Train one variant and return its record."""
    set_seed(seed)
    data = prepare_cmapss(
        SUBSET,  # type: ignore[arg-type]
        config.cmapss,
        seed=seed,
        scaling=hypothesis.scaling,  # type: ignore[arg-type]
    )
    settings = TrainConfig(
        epochs=epochs,
        lambda_candidates=(hypothesis.lambda_fail if hypothesis.lambda_fail is not None else 0.5,),
        **hypothesis.train,
    )
    started = time.perf_counter()

    if hypothesis.cross_validate:
        models, cv = train_a1_cv(
            data,
            n_splits=folds,
            seed=seed,
            lambda_fail=hypothesis.lambda_fail,
            train_config=settings,
            device=device,
            arch_overrides=hypothesis.arch or None,
        )
        record = {
            "rmse": cv.ensemble_metrics["rmse"],
            "nasa_score": cv.ensemble_metrics["nasa_score"],
            "single_fold_rmse": cv.mean_fold_metrics["rmse"],
            "single_fold_nasa": cv.mean_fold_metrics["nasa_score"],
            "mean_best_epoch": cv.mean_best_epoch,
            "n_parameters": cv.n_parameters,
        }
        del models
    else:
        model, result = train_a1(
            data,
            seed=seed,
            lambda_fail=hypothesis.lambda_fail,
            train_config=settings,
            device=device,
            checkpoint_dir=None,
            quiet=True,
            arch_overrides=hypothesis.arch or None,
        )
        record = {
            "rmse": result.metrics["rmse"],
            "nasa_score": result.metrics["nasa_score"],
            "single_fold_rmse": result.metrics["rmse"],
            "single_fold_nasa": result.metrics["nasa_score"],
            "mean_best_epoch": float(result.best_epoch),
            "n_parameters": result.n_parameters,
        }
        del model

    record.update(
        key=hypothesis.key,
        title=hypothesis.title,
        seed=seed,
        duration_s=round(time.perf_counter() - started, 1),
        beats_forest=bool(record["rmse"] < FOREST_RMSE),
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Run the A1 remediation hypotheses.")
    parser.add_argument("--only", nargs="*", default=None, help="Keys to run, e.g. H1 H7.")
    parser.add_argument("--list", action="store_true", help="List the hypotheses and exit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stop-when-beaten",
        action="store_true",
        help=f"Stop at the first variant below RMSE {FOREST_RMSE}.",
    )
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    if args.list:
        for h in HYPOTHESES:
            print(f"{h.key:<6} {h.title}")
        return 0

    config = cast(DataConfig, load_config("data"))
    selected = [h for h in HYPOTHESES if args.only is None or h.key in args.only]
    if not selected:
        logger.error("no hypothesis matched %s", args.only)
        return 1

    path = results_dir() / "hypotheses.jsonl"
    logger.info(
        "running %d variants on %s; target is RMSE < %.2f", len(selected), SUBSET, FOREST_RMSE
    )

    for hypothesis in selected:
        logger.info("=== %s: %s ===", hypothesis.key, hypothesis.title)
        try:
            record = run_one(hypothesis, config, args.seed, args.device, args.epochs, args.folds)
        except Exception as exc:  # noqa: BLE001 - one bad variant must not end the sweep
            logger.error("%s failed: %s", hypothesis.key, str(exc)[:300])
            continue

        with path.open("a", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info(
            "%s  RMSE %.3f  NASA %.0f  params %s  epoch %.1f  %s",
            hypothesis.key,
            record["rmse"],
            record["nasa_score"],
            f"{record['n_parameters']:,}",
            record["mean_best_epoch"],
            "BEATS FOREST" if record["beats_forest"] else f"(forest {FOREST_RMSE})",
        )
        if record["beats_forest"] and args.stop_when_beaten:
            logger.info("stopping rule: %s cleared the baseline", hypothesis.key)
            break

    logger.info("records in %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
