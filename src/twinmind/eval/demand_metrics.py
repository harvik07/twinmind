"""Forecast metrics for Model B, evaluated on the rolling-origin folds.

Three choices here are worth stating, because each is a place the number can be
made to look better than it is.

**MASE scaling comes from training data only.** The denominator is the in-sample
MAE of a seasonal-naive forecast over the *training* period of that fold, per
series. Computing it on validation data -- which is easy to do by accident when
the frame holds both -- makes the metric partly a function of what is being
scored, and the ratio stops meaning "how many times better than naive".

**Zero denominators are excluded and counted, not patched.** A series whose
seasonal-naive forecast is perfect in-sample (constant demand, usually a dead
SKU) gives a zero denominator. Replacing it with an epsilon manufactures a
spectacular MASE; dropping it silently changes the population being averaged.
Both functions here return the exclusion count so it can be reported.

**Pinball for a point model is reported at q50 only, in its own column.** A
point forecast has no q10 or q90, so averaging "its pinball" against a quantile
model's three-level average compares different quantities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)


def rmsle(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Root mean squared log error, the Favorita competition's own metric."""
    true = np.clip(np.asarray(y_true, dtype="float64"), 0.0, None)
    pred = np.clip(np.asarray(y_pred, dtype="float64"), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def pinball_loss(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], quantile: float
) -> float:
    """Quantile (pinball) loss at one level."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    error = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def interval_coverage(
    y_true: NDArray[np.float64], lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> float:
    """Fraction of outcomes inside ``[lower, upper]``."""
    true = np.asarray(y_true, dtype="float64")
    return float(np.mean((true >= np.asarray(lower)) & (true <= np.asarray(upper))))


def coverage_by_bucket(
    y_true: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    steps: NDArray[np.int64],
    buckets: tuple[tuple[int, int], ...] = ((1, 7), (8, 14), (15, 28)),
) -> dict[str, float]:
    """Coverage split by how far ahead the forecast was.

    Intervals routinely look well calibrated overall while undercovering badly at
    long horizons, because the short steps dominate the average. Splitting is the
    only way to see it.
    """
    out: dict[str, float] = {}
    for low, high in buckets:
        mask = (steps >= low) & (steps <= high)
        key = f"coverage_d{low}_{high}"
        out[key] = (
            interval_coverage(y_true[mask], lower[mask], upper[mask])
            if mask.any()
            else float("nan")
        )
    return out


@dataclass
class SeasonalNaiveScale:
    """Per-series MASE denominators, with the series that had to be dropped."""

    scale: dict[str, float]
    excluded: list[str]
    season: int

    @property
    def n_excluded(self) -> int:
        """How many series had a zero in-sample denominator."""
        return len(self.excluded)


def seasonal_naive_scale(
    train: pd.DataFrame,
    season: int,
    id_column: str = "unique_id",
    target: str = "y",
    date_column: str = "ds",
) -> SeasonalNaiveScale:
    """In-sample MAE of a lag-``season`` naive forecast, per series.

    Computed on the training frame only. A series with fewer than ``season + 1``
    observations, or a zero denominator, is excluded and named.
    """
    scale: dict[str, float] = {}
    excluded: list[str] = []
    for series_id, block in train.sort_values([id_column, date_column]).groupby(id_column):
        values = block[target].to_numpy(dtype="float64")
        if len(values) <= season:
            excluded.append(str(series_id))
            continue
        denominator = float(np.mean(np.abs(values[season:] - values[:-season])))
        if denominator <= 0 or not np.isfinite(denominator):
            excluded.append(str(series_id))
            continue
        scale[str(series_id)] = denominator

    if excluded:
        logger.info(
            "MASE(season=%d): excluded %d of %d series with a zero or undefined denominator",
            season,
            len(excluded),
            len(excluded) + len(scale),
        )
    return SeasonalNaiveScale(scale=scale, excluded=excluded, season=season)


def mase(
    predictions: pd.DataFrame,
    scale: SeasonalNaiveScale,
    id_column: str = "unique_id",
    target: str = "y",
    prediction_column: str = "pred",
) -> float:
    """Mean absolute scaled error against a seasonal-naive denominator.

    Series absent from ``scale`` (zero denominator) are skipped; the count is on
    ``scale.n_excluded``.
    """
    errors: list[float] = []
    for series_id, block in predictions.groupby(id_column):
        denominator = scale.scale.get(str(series_id))
        if denominator is None:
            continue
        absolute = np.abs(
            block[prediction_column].to_numpy(dtype="float64")
            - block[target].to_numpy(dtype="float64")
        )
        errors.extend((absolute / denominator).tolist())
    return float(np.mean(errors)) if errors else float("nan")


def seasonal_naive_forecast(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    season: int,
    target: str = "y",
    date_column: str = "ds",
) -> NDArray[np.float64]:
    """Lag-``season`` naive forecast for ``origin+1 .. origin+horizon``.

    Step ``k`` repeats the value observed ``season`` days before the target day,
    cycling back through history when the horizon exceeds the season.
    """
    past = history[history[date_column] <= pd.Timestamp(origin)]
    values = past[target].to_numpy(dtype="float64")
    if len(values) == 0:
        return np.zeros(horizon, dtype="float64")

    out = np.empty(horizon, dtype="float64")
    for step in range(1, horizon + 1):
        back = season - (step % season or season) + 1
        index = len(values) - back
        out[step - 1] = values[index] if 0 <= index < len(values) else values[-1]
    return np.clip(out, 0.0, None)


def summarise_fold(
    y_true: NDArray[np.float64],
    q10: NDArray[np.float64],
    q50: NDArray[np.float64],
    q90: NDArray[np.float64],
    steps: NDArray[np.int64],
) -> dict[str, float]:
    """Every quantile-model metric for one fold."""
    metrics = {
        "rmsle": rmsle(y_true, q50),
        "mae": mae(y_true, q50),
        "pinball_q10": pinball_loss(y_true, q10, 0.1),
        "pinball_q50": pinball_loss(y_true, q50, 0.5),
        "pinball_q90": pinball_loss(y_true, q90, 0.9),
        "coverage_80": interval_coverage(y_true, q10, q90),
    }
    metrics["pinball_mean"] = float(
        np.mean([metrics["pinball_q10"], metrics["pinball_q50"], metrics["pinball_q90"]])
    )
    metrics.update(coverage_by_bucket(y_true, q10, q90, steps))
    return metrics
