"""Train and evaluate Model B on the rolling-origin folds.

    python scripts/train_demand.py --top-n 200
    python scripts/train_demand.py --top-n 20 --origins 8   # fast smoke run

Every model sees the identical folds from ``make_rolling_origin_splits``, the
identical direct multi-horizon feature set, and the identical ``log1p``
transform, so the comparison is about the model rather than about who got the
better preprocessing.

Models: seasonal naive at lag 7 and lag 364, a LightGBM point forecast, and the
LightGBM quantile model that is the production candidate. N-HiTS is a separate
script because it needs the optional ``forecast`` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from twinmind.data.features import build_demand_features, make_rolling_origin_splits
from twinmind.data.loaders import load_favorita
from twinmind.eval.demand_metrics import (
    mae,
    mase,
    pinball_loss,
    rmsle,
    seasonal_naive_forecast,
    seasonal_naive_scale,
    summarise_fold,
)
from twinmind.eval.registry import (
    RunRecord,
    config_hash,
    current_device,
    log_run,
    results_dir,
    write_summary,
)
from twinmind.models.demand import DemandData
from twinmind.models.demand_features import build_training_rows, weekly_origins
from twinmind.models.demand_lgbm import (
    SAMPLER_QUANTILES,
    LGBMParams,
    LightGBMQuantileModel,
    fit_conformal,
    fit_copula,
    fit_quantile_models,
)
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import models_dir
from twinmind.utils.seed import set_seed

logger = get_logger("train_demand")

HORIZON = 28
TARGETS = {"relative_mae": 0.85, "coverage_low": 0.76, "coverage_high": 0.84}


def _fold_frames(
    features: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = features.iloc[train_idx].copy()
    val = features.iloc[val_idx].copy()
    if train["ds"].max() >= val["ds"].min():
        raise AssertionError(
            f"fold is not chronological: train ends {train['ds'].max()}, "
            f"val starts {val['ds'].min()}"
        )
    return train, val


def _evaluate_naive(
    features: pd.DataFrame, origin: pd.Timestamp, val: pd.DataFrame, season: int
) -> pd.DataFrame:
    """Seasonal-naive predictions aligned to the validation rows."""
    out = []
    for series_id, block in val.groupby("unique_id"):
        history = features[(features["unique_id"] == series_id) & (features["ds"] <= origin)]
        forecast = seasonal_naive_forecast(history, origin, HORIZON, season)
        for _, row in block.iterrows():
            step = int((row["ds"] - origin).days)
            if 1 <= step <= HORIZON:
                out.append(
                    {
                        "unique_id": series_id,
                        "ds": row["ds"],
                        "k": step,
                        "y": float(row["y"]),
                        "pred": float(forecast[step - 1]),
                    }
                )
    return pd.DataFrame(out)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Train and evaluate the demand model.")
    parser.add_argument("--top-n", type=int, default=200, help="Series by volume.")
    parser.add_argument("--origins", type=int, default=40, help="Training origins per fold.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-conformal", action="store_true")
    return parser


def main() -> int:  # noqa: PLR0915 - a linear experiment script reads better whole
    """Entry point."""
    args = build_parser().parse_args()
    set_seed(args.seed)
    config = cast(DataConfig, load_config("data"))
    output_dir = Path(args.output_dir) if args.output_dir else models_dir() / "demand"

    features = build_demand_features(load_favorita(), config.favorita, top_n=args.top_n)
    logger.info(
        "features: %d rows, %d series, %s to %s",
        len(features),
        features["unique_id"].nunique(),
        features["ds"].min().date(),
        features["ds"].max().date(),
    )

    folds = make_rolling_origin_splits(features, n_splits=args.folds, horizon=HORIZON)
    per_fold: list[dict[str, Any]] = []
    fold_predictions: list[pd.DataFrame] = []
    residual_paths: list[np.ndarray] = []
    last_model: LightGBMQuantileModel | None = None
    last_spec = None

    for fold, (train_idx, val_idx) in enumerate(folds):
        train, val = _fold_frames(features, train_idx, val_idx)
        origin = train["ds"].max()
        logger.info(
            "=== fold %d: train to %s, validate %s..%s ===",
            fold,
            origin.date(),
            val["ds"].min().date(),
            val["ds"].max().date(),
        )

        origins = weekly_origins(train, origin, args.origins)
        rows = build_training_rows(train, origins, horizon=HORIZON)
        if rows.empty:
            logger.error("fold %d produced no training rows", fold)
            continue

        started = time.perf_counter()
        boosters, spec = fit_quantile_models(rows, SAMPLER_QUANTILES, seed=args.seed)
        point_booster, _ = fit_quantile_models(rows, (0.5,), seed=args.seed)
        last_spec = spec

        snapshot = DemandData(features)
        model = LightGBMQuantileModel(boosters, spec, snapshot, horizon=HORIZON)
        point_model = LightGBMQuantileModel(point_booster, spec, snapshot, horizon=HORIZON)

        # Quantile predictions over the validation window, one call per series.
        records = []
        for series_id in sorted(val["unique_id"].unique()):
            forecast = model.quantiles(series_id, origin, HORIZON)
            point = point_model.quantiles(series_id, origin, HORIZON)
            actual = val[val["unique_id"] == series_id].set_index("ds")["y"]
            for step in range(1, HORIZON + 1):
                day = origin + pd.Timedelta(days=step)
                if day not in actual.index:
                    continue
                records.append(
                    {
                        "unique_id": series_id,
                        "ds": day,
                        "k": step,
                        "y": float(actual.loc[day]),
                        "q10": forecast.q10[step - 1],
                        "q50": forecast.q50[step - 1],
                        "q90": forecast.q90[step - 1],
                        "point": point.q50[step - 1],
                    }
                )
        predictions = pd.DataFrame(records)
        duration = time.perf_counter() - started

        metrics = summarise_fold(
            predictions["y"].to_numpy(),
            predictions["q10"].to_numpy(),
            predictions["q50"].to_numpy(),
            predictions["q90"].to_numpy(),
            predictions["k"].to_numpy(),
        )
        metrics["point_rmsle"] = rmsle(predictions["y"].to_numpy(), predictions["point"].to_numpy())
        metrics["point_median_pinball"] = pinball_loss(
            predictions["y"].to_numpy(), predictions["point"].to_numpy(), 0.5
        )

        # MASE against both seasonal naives, scaled on training data only.
        naive_mae: dict[int, float] = {}
        for season in (7, 364):
            scale = seasonal_naive_scale(train, season)
            metrics[f"mase_lag{season}"] = mase(predictions.assign(pred=predictions["q50"]), scale)
            metrics[f"mase_lag{season}_excluded"] = float(scale.n_excluded)
            naive = _evaluate_naive(features, origin, val, season)
            naive_mae[season] = mae(naive["y"].to_numpy(), naive["pred"].to_numpy())
            metrics[f"naive_lag{season}_mae"] = naive_mae[season]

        best_naive = min(naive_mae.values())
        metrics["relative_mae"] = metrics["mae"] / best_naive if best_naive > 0 else float("nan")
        metrics["fold"] = float(fold)
        metrics["duration_s"] = round(duration, 1)
        per_fold.append(metrics)
        fold_predictions.append(predictions.assign(fold=fold))

        logger.info(
            "  fold %d: RMSLE %.4f  relMAE %.3f  cov80 %.3f  pinball %.4f  (%.0fs)",
            fold,
            metrics["rmsle"],
            metrics["relative_mae"],
            metrics["coverage_80"],
            metrics["pinball_mean"],
            duration,
        )

        # Residuals from THIS fold feed the copula used on LATER folds only.
        wide = predictions.pivot_table(
            index="unique_id", columns="k", values=["y", "q50"], aggfunc="first"
        )
        if ("y", 1) in wide.columns:
            resid = (wide["y"] - wide["q50"]).to_numpy(dtype="float64")
            if resid.shape[1] == HORIZON:
                residual_paths.append(resid)
        last_model = model

    if not per_fold:
        logger.error("no folds produced results")
        return 1

    # ------------------------------------------------------ conformal step --
    # Calibrate on the EARLIEST fold and apply to the later ones, so nothing is
    # calibrated on the data it is then scored against. That costs the first
    # fold as an evaluation point, which is the honest price.
    conformal: dict[str, np.ndarray] = {}
    coverage_before = float(np.mean([f["coverage_80"] for f in per_fold]))
    coverage_reported = coverage_before
    if (
        not args.skip_conformal
        and len(fold_predictions) >= 2
        and not TARGETS["coverage_low"] <= coverage_before <= TARGETS["coverage_high"]
    ):
        calibration = fold_predictions[0]
        conformal = fit_conformal(
            calibration["y"].to_numpy(),
            calibration["q10"].to_numpy(),
            calibration["q90"].to_numpy(),
            calibration["k"].to_numpy(),
            HORIZON,
        )
        logger.info(
            "conformal calibrated on fold 0 (%s..%s); scoring folds 1..%d",
            calibration["ds"].min().date(),
            calibration["ds"].max().date(),
            len(fold_predictions) - 1,
        )

        adjusted: list[dict[str, Any]] = []
        for frame in fold_predictions[1:]:
            steps = frame["k"].to_numpy() - 1
            low = np.clip(frame["q10"].to_numpy() - conformal["lower"][steps], 0.0, None)
            high = frame["q90"].to_numpy() + conformal["upper"][steps]
            calibrated = summarise_fold(
                frame["y"].to_numpy(), low, frame["q50"].to_numpy(), high, frame["k"].to_numpy()
            )
            calibrated["fold"] = float(frame["fold"].iloc[0])
            adjusted.append(calibrated)

        coverage_after = float(np.mean([f["coverage_80"] for f in adjusted]))
        coverage_reported = coverage_after
        logger.info(
            "coverage 80%%: %.3f before conformal -> %.3f after (target %.2f-%.2f)",
            coverage_before,
            coverage_after,
            TARGETS["coverage_low"],
            TARGETS["coverage_high"],
        )
        for bucket in ("coverage_d1_7", "coverage_d8_14", "coverage_d15_28"):
            before = float(np.mean([f[bucket] for f in per_fold]))
            after = float(np.mean([f[bucket] for f in adjusted]))
            logger.info("    %-18s %.3f -> %.3f", bucket, before, after)

        (results_dir() / "demand_conformal.json").write_text(
            json.dumps(
                {
                    "calibration_fold": 0,
                    "calibration_window": [
                        str(calibration["ds"].min().date()),
                        str(calibration["ds"].max().date()),
                    ],
                    "coverage_before": coverage_before,
                    "coverage_after": coverage_after,
                    "scored_folds": [int(f["fold"]) for f in adjusted],
                    "lower_offsets": conformal["lower"].tolist(),
                    "upper_offsets": conformal["upper"].tolist(),
                    "before": per_fold,
                    "after": adjusted,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    # ------------------------------------------------------------- summary --
    def _mean_sd(key: str) -> tuple[float, float]:
        values = [f[key] for f in per_fold]
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return float(np.mean(values)), sd

    aggregate = {key: _mean_sd(key) for key in per_fold[0] if key != "fold"}
    logger.info("=== across %d folds ===", len(per_fold))
    for key, (mean, sd) in sorted(aggregate.items()):
        logger.info("    %-26s %8.4f +/- %.4f", key, mean, sd)

    for name, value in (
        ("relative_mae <= 0.85", aggregate["relative_mae"][0] <= TARGETS["relative_mae"]),
        (
            # Judged on the post-conformal number when calibration ran, since
            # that is the interval the twin would actually consume.
            "coverage_80 in [0.76, 0.84]",
            TARGETS["coverage_low"] <= coverage_reported <= TARGETS["coverage_high"],
        ),
        (
            "quantile q50 pinball beats point model",
            aggregate["pinball_q50"][0] <= aggregate["point_median_pinball"][0],
        ),
    ):
        logger.info("  target %-42s %s", name, "HIT" if value else "MISS")

    # ----------------------------------------------------------- artifacts --
    if last_model is not None and last_spec is not None:
        copula = None
        if residual_paths:
            pooled = (
                np.vstack(residual_paths[:-1]) if len(residual_paths) > 1 else residual_paths[0]
            )
            try:
                copula = fit_copula(pooled, HORIZON)
            except ValueError as exc:
                logger.warning("copula not fitted: %s", exc)
        last_model.copula = copula
        last_model.conformal = conformal
        last_model.save(
            output_dir,
            metadata={
                "git_sha": __import__("subprocess")
                .run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
                .stdout.strip(),
                "series_ids": sorted(features["unique_id"].unique().tolist()),
                "top_n": args.top_n,
                "origins_per_fold": args.origins,
                "folds": args.folds,
                "training_window": [
                    str(features["ds"].min().date()),
                    str(features["ds"].max().date()),
                ],
                "lgbm_params": asdict(LGBMParams()),
                "copula_source": "pooled residuals from all but the final fold",
            },
        )

    (results_dir() / "demand_folds.json").write_text(
        json.dumps(per_fold, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )

    log_run(
        RunRecord(
            model="demand_lgbm_quantile",
            dataset="favorita",
            subset=f"top{args.top_n}",
            seed=args.seed,
            metrics={k: v[0] for k, v in aggregate.items()},
            config_hash=config_hash(
                {
                    "model": "lgbm_quantile",
                    "quantiles": list(SAMPLER_QUANTILES),
                    "origins": args.origins,
                    "top_n": args.top_n,
                    "horizon": HORIZON,
                }
            ),
            duration_s=float(sum(f["duration_s"] for f in per_fold)),
            device=current_device(),
            notes=f"{args.folds} rolling-origin folds; direct multi-horizon; log1p",
        )
    )
    write_summary()
    logger.info("artifacts in %s; summary refreshed", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
