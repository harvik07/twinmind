"""The single neural comparison: N-HiTS with MQLoss, on the identical folds.

    python scripts/train_demand_nhits.py --top-n 200 --max-steps 300

Needs the optional ``forecast`` extra. The non-data CI suite must not require
it, so every test that touches this module uses ``pytest.importorskip``.

API note for the installed version. neuralforecast 1.7.7 takes
``MQLoss(quantiles=[...])`` directly (older releases only took ``level=[...]``
as percentages), and the exogenous arguments are ``futr_exog_list`` and
``hist_exog_list``. Both were read off the installed signature rather than
assumed, because they moved between releases.

Only genuinely known-future covariates go into ``futr_exog_list``: the calendar
and holiday terms plus onpromotion. Oil price and lagged sales are not knowable
28 days ahead and are deliberately withheld, which is the same classification
the LightGBM path uses, so the comparison is on the model and not on who was
allowed to peek.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import cast

import numpy as np
import pandas as pd

from twinmind.data.features import build_demand_features, make_rolling_origin_splits
from twinmind.data.loaders import load_favorita
from twinmind.eval.demand_metrics import summarise_fold
from twinmind.eval.registry import (
    RunRecord,
    config_hash,
    current_device,
    log_run,
    results_dir,
    write_summary,
)
from twinmind.models.demand_features import KNOWN_FUTURE
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger("train_demand_nhits")
HORIZON = 28


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="N-HiTS demand comparison.")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--input-size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _pick(merged: pd.DataFrame, columns: list[str], suffix: str) -> np.ndarray:
    """One quantile column back on the original scale.

    The names MQLoss emits vary between neuralforecast releases -- 1.7.7 writes
    ``NHITS-lo-80.0`` with a trailing ``.0`` that earlier releases omit -- so
    this matches a substring and fails loudly rather than assuming a spelling.
    """
    matches = [c for c in columns if suffix in c]
    if not matches:
        raise KeyError(f"no prediction column ending {suffix!r} in {columns}")
    return np.clip(np.expm1(merged[matches[0]].to_numpy()), 0.0, None)


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    set_seed(args.seed)

    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS
    except ImportError:
        logger.error('neuralforecast is not installed; pip install -e ".[forecast]"')
        return 1

    import importlib.metadata as metadata

    logger.info("neuralforecast %s", metadata.version("neuralforecast"))

    config = cast(DataConfig, load_config("data"))
    features = build_demand_features(load_favorita(), config.favorita, top_n=args.top_n)
    folds = make_rolling_origin_splits(features, n_splits=args.folds, horizon=HORIZON)

    # Only the covariates that are genuinely known ahead, matching the LightGBM
    # classification exactly. Anything numeric and present; the categoricals the
    # tree handled natively are dropped rather than one-hot expanded here.
    future_exog = [
        c
        for c in KNOWN_FUTURE
        if c in features.columns and pd.api.types.is_numeric_dtype(features[c])
    ]
    logger.info("futr_exog_list (%d): %s", len(future_exog), future_exog)

    per_fold = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        train = features.iloc[train_idx]
        val = features.iloc[val_idx]
        if train["ds"].max() >= val["ds"].min():
            raise AssertionError(f"fold {fold} is not chronological")

        columns = ["unique_id", "ds", "y", *future_exog]
        train_df = train[columns].copy()
        future_df = val[["unique_id", "ds", *future_exog]].copy()

        # Same log1p transform as every other model, so the comparison is fair.
        train_df["y"] = np.log1p(np.clip(train_df["y"], 0.0, None))

        logger.info(
            "=== fold %d: train to %s (%d rows), validate %s..%s ===",
            fold,
            train["ds"].max().date(),
            len(train_df),
            val["ds"].min().date(),
            val["ds"].max().date(),
        )

        model = NHITS(
            h=HORIZON,
            input_size=args.input_size,
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
            futr_exog_list=future_exog,
            max_steps=args.max_steps,
            early_stop_patience_steps=3,
            val_check_steps=50,
            scaler_type="robust",
            random_seed=args.seed,
            # Lightning defaults to a TensorBoard logger, which on this machine
            # imports a tensorflow/tensorboard pair that does not agree with
            # itself and dies on the first hparam write. Nothing here needs a
            # logger -- the metrics are computed below and go to the registry.
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        forecaster = NeuralForecast(models=[model], freq="D")

        started = time.perf_counter()
        forecaster.fit(train_df, val_size=HORIZON, verbose=False)
        predicted = forecaster.predict(futr_df=future_df)
        duration = time.perf_counter() - started

        merged = predicted.reset_index().merge(
            val[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner"
        )
        quantile_columns = [c for c in merged.columns if c.startswith("NHITS")]
        logger.info("prediction columns: %s", quantile_columns)

        q10 = _pick(merged, quantile_columns, "lo-80")
        q50 = _pick(merged, quantile_columns, "median")
        q90 = _pick(merged, quantile_columns, "hi-80")
        steps = (merged["ds"] - train["ds"].max()).dt.days.to_numpy()

        metrics = summarise_fold(merged["y"].to_numpy(), q10, q50, q90, steps)
        metrics["fold"] = float(fold)
        metrics["duration_s"] = round(duration, 1)
        per_fold.append(metrics)
        logger.info(
            "  fold %d: RMSLE %.4f  cov80 %.3f  pinball %.4f  (%.0fs)",
            fold,
            metrics["rmsle"],
            metrics["coverage_80"],
            metrics["pinball_mean"],
            duration,
        )

    if not per_fold:
        return 1

    aggregate = {
        key: float(np.mean([f[key] for f in per_fold])) for key in per_fold[0] if key != "fold"
    }
    logger.info("=== N-HiTS across %d folds ===", len(per_fold))
    for key, value in sorted(aggregate.items()):
        logger.info("    %-24s %.4f", key, value)

    (results_dir() / "demand_nhits_folds.json").write_text(
        json.dumps(per_fold, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    log_run(
        RunRecord(
            model="demand_nhits",
            dataset="favorita",
            subset=f"top{args.top_n}",
            seed=args.seed,
            metrics=aggregate,
            config_hash=config_hash(
                {
                    "model": "nhits",
                    "max_steps": args.max_steps,
                    "input_size": args.input_size,
                    "top_n": args.top_n,
                    "horizon": HORIZON,
                }
            ),
            duration_s=float(sum(f["duration_s"] for f in per_fold)),
            device=current_device(),
            notes=f"MQLoss q10/q50/q90; {args.folds} rolling-origin folds; log1p",
        )
    )
    write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
