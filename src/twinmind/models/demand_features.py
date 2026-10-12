"""Direct multi-horizon features for demand, built so nothing after the origin leaks.

THE TRAP THIS EXISTS TO AVOID.

``twinmind.data.features`` builds lags {1, 7, 14, 28} shifted by one day, which
is correct for *one-step-ahead* forecasting and silently wrong for 28. At a
28-day horizon, standing at the origin, ``lag_1`` for target day ``origin+28``
would be the sales on day ``origin+27`` -- a value that does not exist yet. A
single multi-horizon model trained on those columns validates beautifully and is
unusable in the twin, because at run time there is nothing to put in the column.

**The strategy here is direct multi-horizon with a horizon-step feature.** One
training row is ``(series, origin, k)`` for ``k = 1..28``. Every history feature
is computed **as of the origin** and is therefore identical across all 28 rows
of that origin; ``k`` tells the model how far ahead it is predicting. Known-future
covariates are read from the target day, because those genuinely are known.

This costs 28x the rows of a one-step model, which is why origins are subsampled
weekly by default.

COVARIATE CLASSIFICATION, which is the whole safety argument:

* **Known at the origin for future days** -- calendar terms, Fourier terms,
  holiday flags (with the transferred-holiday handling already in the data
  layer), store metadata. A calendar is knowable years ahead.
* **onpromotion -- treated as known future.** Promotions are planned, so a
  retailer does know them in advance. This is an assumption, stated here and in
  the model card: in the twin it becomes a scenario lever, which is the point.
* **Not known in the future** -- oil price, transactions, and lagged sales.
  These enter only at their origin value (``oil_price_at_origin``) or not at
  all. Using their target-day values would be reading tomorrow's newspaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Calendar and holiday columns that are knowable arbitrarily far ahead, so they
#: may be read from the *target* day rather than the origin.
KNOWN_FUTURE: tuple[str, ...] = (
    "dow",
    "dom",
    "month",
    "week_of_year",
    "is_weekend",
    "is_payday",
    "is_holiday",
    "is_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "is_work_day",
    "fourier_weekly_sin_1",
    "fourier_weekly_cos_1",
    "fourier_weekly_sin_2",
    "fourier_weekly_cos_2",
    "fourier_weekly_sin_3",
    "fourier_weekly_cos_3",
    "fourier_yearly_sin_1",
    "fourier_yearly_cos_1",
    "fourier_yearly_sin_2",
    "fourier_yearly_cos_2",
    "fourier_yearly_sin_3",
    "fourier_yearly_cos_3",
    "fourier_yearly_sin_4",
    "fourier_yearly_cos_4",
    "fourier_yearly_sin_5",
    "fourier_yearly_cos_5",
    # Planned, therefore known. See the module docstring.
    "onpromotion",
)

#: Static per-series attributes. Constant in time, so trivially known future.
STATIC: tuple[str, ...] = ("store_nbr", "cluster")

#: Read at the origin only. Their future values are not knowable.
#:
#: ``transactions`` is listed even though the current featurised frame does not
#: carry it: it is one of the six Favorita tables and an obvious thing to merge
#: in later, and store footfall is emphatically not known 28 days ahead.
#: Classifying it now means adding the column cannot silently make it a leak.
ORIGIN_ONLY: tuple[str, ...] = ("oil_price", "transactions")

#: Lags of the target, measured backwards from the origin.
ORIGIN_LAGS: tuple[int, ...] = (1, 7, 14, 28, 364)

#: Rolling windows ending at the origin.
ORIGIN_WINDOWS: tuple[int, ...] = (7, 28, 91)


@dataclass(frozen=True)
class FeatureSpec:
    """The exact feature set, recorded so training and inference cannot diverge."""

    columns: list[str]
    known_future: list[str] = field(default_factory=lambda: list(KNOWN_FUTURE))
    static: list[str] = field(default_factory=lambda: list(STATIC))
    origin_only: list[str] = field(default_factory=lambda: list(ORIGIN_ONLY))
    lags: list[int] = field(default_factory=lambda: list(ORIGIN_LAGS))
    windows: list[int] = field(default_factory=lambda: list(ORIGIN_WINDOWS))
    transform: str = "log1p"


def origin_features(history: pd.DataFrame, target: str = "y") -> dict[str, float]:
    """Summarise one series' history **as of its last row**.

    ``history`` must already end at the origin -- the caller slices it, this
    function never looks past what it is given, which is what makes the
    truncation-invariance test pass by construction rather than by care.
    """
    values = history[target].to_numpy(dtype="float64")
    out: dict[str, float] = {}

    for lag in ORIGIN_LAGS:
        out[f"origin_lag_{lag}"] = float(values[-lag]) if len(values) >= lag else np.nan
    for window in ORIGIN_WINDOWS:
        tail = values[-window:] if len(values) >= 1 else values
        out[f"origin_mean_{window}"] = float(np.mean(tail)) if len(tail) else np.nan
        out[f"origin_std_{window}"] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
    # A crude level and trend, which is what the tree has to work with when the
    # seasonal lags are missing early in a series' life.
    recent, previous = values[-28:], values[-56:-28]
    out["origin_trend_28"] = float(np.mean(recent) - np.mean(previous)) if len(previous) else 0.0
    out["origin_nonzero_rate_28"] = float(np.mean(recent > 0)) if len(recent) else 0.0

    for column in ORIGIN_ONLY:
        if column in history.columns:
            out[f"{column}_at_origin"] = float(history[column].to_numpy()[-1])
    return out


def build_training_rows(
    frame: pd.DataFrame,
    origins: list[pd.Timestamp],
    horizon: int = 28,
    target: str = "y",
    id_column: str = "unique_id",
    date_column: str = "ds",
    min_history: int = 60,
) -> pd.DataFrame:
    """Expand ``(series, origin)`` pairs into ``(series, origin, k)`` rows.

    Returns a frame with the origin-side features, the known-future covariates of
    the target day, ``k``, and the target ``y``. Rows whose target day is absent
    from ``frame`` are dropped rather than imputed.
    """
    frame = frame.sort_values([id_column, date_column])
    by_series = {sid: block.reset_index(drop=True) for sid, block in frame.groupby(id_column)}
    known = [c for c in KNOWN_FUTURE if c in frame.columns]
    static = [c for c in STATIC if c in frame.columns]

    rows: list[dict[str, object]] = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        for series_id, block in by_series.items():
            history = block[block[date_column] <= origin]
            if len(history) < min_history:
                continue
            base = origin_features(history, target)
            statics = {c: history[c].to_numpy()[-1] for c in static}

            window = block[
                (block[date_column] > origin)
                & (block[date_column] <= origin + pd.Timedelta(days=horizon))
            ]
            for _, future_row in window.iterrows():
                step = int((future_row[date_column] - origin).days)
                rows.append(
                    {
                        id_column: series_id,
                        "origin": origin,
                        date_column: future_row[date_column],
                        "k": step,
                        target: float(future_row[target]),
                        **base,
                        **statics,
                        **{c: future_row[c] for c in known},
                    }
                )

    built = pd.DataFrame(rows)
    logger.info(
        "built %d training rows from %d origins x %d series (horizon %d)",
        len(built),
        len(origins),
        len(by_series),
        horizon,
    )
    return built


def build_inference_rows(
    history: pd.DataFrame,
    future: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    target: str = "y",
    date_column: str = "ds",
) -> pd.DataFrame:
    """The ``horizon`` rows to predict for one series at one origin.

    ``history`` ends at the origin; ``future`` holds the target days. Only the
    known-future columns of ``future`` are read, so a caller passing a frame that
    still contains realised sales cannot leak them.
    """
    base = origin_features(history, target)
    known = [c for c in KNOWN_FUTURE if c in future.columns]
    static = [c for c in STATIC if c in history.columns]
    statics = {c: history[c].to_numpy()[-1] for c in static}

    rows = []
    for step in range(1, horizon + 1):
        day = pd.Timestamp(origin) + pd.Timedelta(days=step)
        match = future[future[date_column] == day]
        covariates = (
            {c: match.iloc[0][c] for c in known}
            if len(match)
            # Beyond the snapshot the calendar is still derivable; anything else
            # falls back to zero, which for holiday and promotion flags means
            # "nothing special planned".
            else _calendar_fallback(day, known)
        )
        rows.append({date_column: day, "k": step, **base, **statics, **covariates})
    return pd.DataFrame(rows)


def _calendar_fallback(day: pd.Timestamp, known: list[str]) -> dict[str, float]:
    """Derive the calendar part of the known-future set for a date beyond the snapshot."""
    epoch = (day - pd.Timestamp("2013-01-01")).days
    values: dict[str, float] = {}
    for column in known:
        if column == "dow":
            values[column] = float(day.dayofweek)
        elif column == "dom":
            values[column] = float(day.day)
        elif column == "month":
            values[column] = float(day.month)
        elif column == "week_of_year":
            values[column] = float(day.isocalendar().week)
        elif column == "is_weekend":
            values[column] = float(day.dayofweek >= 5)
        elif column == "is_payday":
            values[column] = float(day.day == 15 or day.is_month_end)
        elif column.startswith("fourier_"):
            parts = column.split("_")
            period = 7.0 if parts[1] == "weekly" else 365.25
            order = int(parts[-1])
            angle = 2 * np.pi * order * epoch / period
            values[column] = float(np.sin(angle) if parts[2] == "sin" else np.cos(angle))
        else:
            values[column] = 0.0
    return values


def feature_columns(rows: pd.DataFrame, target: str = "y") -> list[str]:
    """Model inputs: everything except the identifiers, the date and the target."""
    excluded = {"unique_id", "origin", "ds", target, "family", "city", "state", "type", "year"}
    return [c for c in rows.columns if c not in excluded]


def weekly_origins(
    frame: pd.DataFrame,
    last_origin: pd.Timestamp,
    n_origins: int,
    step_days: int = 7,
    date_column: str = "ds",
) -> list[pd.Timestamp]:
    """``n_origins`` origins walking backwards from ``last_origin``, weekly by default.

    Subsampling is what keeps the direct multi-horizon expansion affordable:
    every origin contributes 28 rows per series.
    """
    first = pd.to_datetime(frame[date_column]).min()
    origins = [
        pd.Timestamp(last_origin) - pd.Timedelta(days=step_days * i) for i in range(n_origins)
    ]
    return sorted(o for o in origins if o > first)
