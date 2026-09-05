"""Favorita demand feature pipeline, producing a neuralforecast-ready long frame.

Output contract: one row per (series, date) with

    unique_id   str            "<store_nbr>_<family>"
    ds          datetime64[ns] the date
    y           float64        sales

plus exogenous columns. Nixtla's ``NeuralForecast`` consumes exactly this layout.

Every derived feature in here is non-anticipative: a row dated ``t`` may only use
information available at ``t``. Lags and rolling statistics are shifted before
aggregation, which is enforced by ``test_features_are_non_anticipative``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from twinmind.utils.config import FavoritaConfig
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def add_calendar_features(frame: pd.DataFrame, date_column: str = "ds") -> pd.DataFrame:
    """Add plain calendar columns: day of week, day of month, month, week of year."""
    out = frame.copy()
    dates = pd.to_datetime(out[date_column])
    out["dow"] = dates.dt.dayofweek.astype("int16")
    out["dom"] = dates.dt.day.astype("int16")
    out["month"] = dates.dt.month.astype("int16")
    out["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    out["year"] = dates.dt.year.astype("int16")
    out["is_weekend"] = dates.dt.dayofweek.ge(5).astype("int8")
    # Ecuadorian public-sector wages are paid on the 15th and the last day of the
    # month, and Favorita sales visibly spike around both.
    out["is_payday"] = (
        dates.dt.day.eq(15) | dates.dt.is_month_end
    ).astype("int8")
    return out


def add_fourier_terms(
    frame: pd.DataFrame,
    date_column: str = "ds",
    weekly_k: int = 3,
    yearly_k: int = 5,
) -> pd.DataFrame:
    """Add sin/cos Fourier pairs for weekly (period 7) and yearly (365.25) cycles.

    Fourier terms let a model represent smooth seasonality with a handful of
    columns instead of 7 + 365 dummies, and they extrapolate to dates never seen
    in training -- which one-hot day-of-year cannot do.
    """
    out = frame.copy()
    dates = pd.to_datetime(out[date_column])
    # Day index from a fixed epoch so the phase is identical across series and folds.
    t = (dates - pd.Timestamp("2013-01-01")).dt.days.to_numpy(dtype="float64")

    for k in range(1, weekly_k + 1):
        out[f"fourier_weekly_sin_{k}"] = np.sin(2 * np.pi * k * t / 7.0)
        out[f"fourier_weekly_cos_{k}"] = np.cos(2 * np.pi * k * t / 7.0)
    for k in range(1, yearly_k + 1):
        out[f"fourier_yearly_sin_{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        out[f"fourier_yearly_cos_{k}"] = np.cos(2 * np.pi * k * t / 365.25)
    return out


# --------------------------------------------------------------------------- #
# Holidays
# --------------------------------------------------------------------------- #
def build_holiday_flags(
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve holidays_events.csv into per (date, store_nbr) flags.

    The transferred/bridge logic, which is the trap in this file:

    * ``transferred == True`` on a ``Holiday`` row means the holiday was moved.
      It did **not** happen on the listed date -- that date was a normal working
      day. Treating it as a holiday is the classic error and misaligns some of
      the largest demand spikes in the dataset.
    * A ``Transfer`` row carries the date the holiday was *actually* observed.
      Those rows are real holidays.
    * ``Bridge`` days are extra days off, granted to join a holiday to a weekend.
    * ``Work Day`` rows are the *opposite* of a holiday: ordinary workdays
      (usually Saturdays) that repay a Bridge. They are excluded, and flagged
      separately so a model can learn the depressed demand.
    * ``Additional`` rows are extra official holiday days and count as holidays.

    Locale matters too: a ``Local`` holiday applies only to stores in that city
    and a ``Regional`` one only to that state, so the flags are resolved per
    store rather than nationally.
    """
    events = holidays.copy()
    events["date"] = pd.to_datetime(events["date"])
    for column in ("type", "locale", "locale_name"):
        events[column] = events[column].astype(str)
    if "transferred" in events.columns:
        events["transferred"] = events["transferred"].astype(bool)
    else:  # pragma: no cover - the real file always has it
        events["transferred"] = False

    store_meta = stores[["store_nbr", "city", "state"]].copy()
    store_meta["store_nbr"] = store_meta["store_nbr"].astype("int64")
    all_stores = store_meta[["store_nbr"]].drop_duplicates()

    # A moved holiday did not happen; a "Transfer" row says when it actually did.
    real = events[~(events["type"].eq("Holiday") & events["transferred"])]
    real = real[~real["type"].eq("Work Day")]

    pieces: list[pd.DataFrame] = []

    def _national(rows: pd.DataFrame, column: str) -> None:
        if rows.empty:
            return
        cross = rows[["date"]].drop_duplicates().merge(all_stores, how="cross")
        cross[column] = 1
        pieces.append(cross)

    def _by_locale(rows: pd.DataFrame, key: str, column: str) -> None:
        if rows.empty:
            return
        merged = rows[["date", "locale_name"]].drop_duplicates().merge(
            store_meta, left_on="locale_name", right_on=key, how="inner"
        )[["date", "store_nbr"]]
        if merged.empty:
            return
        merged[column] = 1
        pieces.append(merged)

    _national(real[real["locale"].eq("National")], "is_national_holiday")
    _by_locale(real[real["locale"].eq("Regional")], "state", "is_regional_holiday")
    _by_locale(real[real["locale"].eq("Local")], "city", "is_local_holiday")
    # Work Days are national in this dataset, so they cross-join like one.
    _national(events[events["type"].eq("Work Day")], "is_work_day")

    columns = [
        "is_national_holiday", "is_regional_holiday", "is_local_holiday", "is_work_day",
    ]
    if pieces:
        flags = pieces[0]
        for piece in pieces[1:]:
            flags = flags.merge(piece, on=["date", "store_nbr"], how="outer")
    else:  # pragma: no cover - only when the events file is empty
        flags = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                              "store_nbr": pd.Series(dtype="int64")})

    # Fill only after every merge; an earlier fillna would be undone by the next
    # outer join and leave NaNs that break the int8 cast.
    for column in columns:
        if column not in flags.columns:
            flags[column] = 0
        flags[column] = flags[column].fillna(0).astype("int8")

    flags["store_nbr"] = flags["store_nbr"].astype("int64")
    flags["is_holiday"] = flags[columns[:3]].max(axis=1).astype("int8")

    n_moved = int((events["type"].eq("Holiday") & events["transferred"]).sum())
    logger.info("holiday flags built; %d transferred holidays excluded from their listed date",
                n_moved)
    return flags[["date", "store_nbr", "is_holiday", *columns]]


# --------------------------------------------------------------------------- #
# Oil
# --------------------------------------------------------------------------- #
def prepare_oil(oil: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Reindex oil prices onto a daily calendar and forward-fill.

    ``oil.csv`` only has entries for trading days, so weekends and holidays are
    missing entirely. Forward-filling carries the last close forward, which is
    what the price *was* on those days. A backfill would be anticipative, so the
    single leading gap is filled with the first observed price instead.
    """
    frame = oil.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    price_column = "dcoilwtico" if "dcoilwtico" in frame.columns else frame.columns[-1]

    calendar = pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq="D")})
    merged = calendar.merge(frame[["date", price_column]], on="date", how="left")
    merged[price_column] = merged[price_column].ffill().bfill()
    return merged.rename(columns={price_column: "oil_price"})


# --------------------------------------------------------------------------- #
# Lags and rolling statistics
# --------------------------------------------------------------------------- #
def add_lag_features(
    frame: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    group_column: str = "unique_id",
    target_column: str = "y",
    date_column: str = "ds",
) -> pd.DataFrame:
    """Add per-series lags and shifted rolling mean/std.

    Non-anticipative by construction:
        ``lag_k`` is ``groupby(id)[y].shift(k)`` with ``k >= 1``.
        Rolling statistics are computed on the **already-shifted** series, so a
        ``roll_mean_7`` at date ``t`` averages ``t-7 .. t-1`` and never includes
        ``y`` at ``t``. Using ``.rolling(7).mean()`` without the shift would leak
        the current target into its own feature -- the mistake that makes a
        validation score look excellent and a live forecast fall apart.
    """
    out = frame.sort_values([group_column, date_column]).copy()
    grouped = out.groupby(group_column, sort=False)[target_column]

    for lag in lags:
        if lag < 1:
            raise ValueError(f"lag must be >= 1 to stay non-anticipative, got {lag}")
        out[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in rolling_windows:
        if window < 1:
            raise ValueError(f"rolling window must be >= 1, got {window}")
        roll = shifted.groupby(out[group_column], sort=False).rolling(window, min_periods=1)
        out[f"roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"roll_std_{window}"] = (
            roll.std().reset_index(level=0, drop=True).fillna(0.0)
        )

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
def select_top_series(frame: pd.DataFrame, top_n: int, target_column: str = "y") -> pd.DataFrame:
    """Keep the ``top_n`` series with the largest total volume.

    The full dataset is 1,782 store x family series. Most are near-zero sellers
    whose forecast error is irrelevant to the twin's order book, and training on
    all of them costs a Kaggle session for no benefit.
    """
    totals = frame.groupby("unique_id")[target_column].sum().sort_values(ascending=False)
    keep = set(totals.head(top_n).index)
    selected = frame[frame["unique_id"].isin(keep)].reset_index(drop=True)
    logger.info(
        "kept top %d of %d series (%.1f%% of total volume)",
        len(keep),
        len(totals),
        100.0 * totals.head(top_n).sum() / max(totals.sum(), 1e-9),
    )
    return selected


def build_demand_features(
    frames: dict[str, pd.DataFrame],
    config: FavoritaConfig,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Assemble the neuralforecast-ready long frame from the raw Favorita tables.

    Args:
        frames: output of :func:`twinmind.data.loaders.load_favorita`.
        config: the ``favorita`` section of configs/data.yaml.
        top_n: series cap; defaults to ``config.top_series``.

    Returns a frame sorted by ``(unique_id, ds)``.
    """
    top_n = top_n if top_n is not None else config.top_series

    train = frames["train"].copy()
    train["date"] = pd.to_datetime(train["date"])
    train["store_nbr"] = train["store_nbr"].astype("int64")

    long = train.rename(
        columns={"date": "ds", config.target_column: "y"}
    )
    long["unique_id"] = (
        long["store_nbr"].astype(str) + "_" + long["family"].astype(str)
    )
    long["y"] = long["y"].astype("float64")
    long = long[["unique_id", "ds", "y", "store_nbr", "family", "onpromotion"]]
    long["onpromotion"] = long["onpromotion"].fillna(0).astype("float64")

    long = select_top_series(long, top_n)

    # Store metadata as categoricals.
    stores = frames["stores"].copy()
    stores["store_nbr"] = stores["store_nbr"].astype("int64")
    long = long.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    for column in ("city", "state", "type", "family"):
        if column in long.columns:
            long[column] = long[column].astype("category")
    if "cluster" in long.columns:
        long["cluster"] = long["cluster"].astype("int16")

    # Calendar and seasonality.
    long = add_calendar_features(long, "ds")
    long = add_fourier_terms(
        long, "ds", weekly_k=config.fourier_weekly_k, yearly_k=config.fourier_yearly_k
    )

    # Holidays, resolved per store.
    flags = build_holiday_flags(frames["holidays"], stores)
    long = long.merge(
        flags.rename(columns={"date": "ds"}),
        on=["ds", "store_nbr"],
        how="left",
        validate="many_to_one",
    )
    for column in (
        "is_holiday", "is_national_holiday", "is_regional_holiday",
        "is_local_holiday", "is_work_day",
    ):
        long[column] = long[column].fillna(0).astype("int8")

    # Oil, forward-filled onto the daily calendar.
    oil = prepare_oil(frames["oil"], long["ds"])
    long = long.merge(
        oil.rename(columns={"date": "ds"}), on="ds", how="left", validate="many_to_one"
    )
    long["oil_price"] = long["oil_price"].ffill().bfill()

    # Lags and rolling statistics, last so they see the final target column.
    long = add_lag_features(long, config.lags, config.rolling_windows)

    long = long.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    logger.info(
        "built demand features: %d rows, %d series, %d columns, %s to %s",
        len(long),
        long["unique_id"].nunique(),
        long.shape[1],
        long["ds"].min().date(),
        long["ds"].max().date(),
    )
    return long


# --------------------------------------------------------------------------- #
# Rolling-origin splits
# --------------------------------------------------------------------------- #
def make_rolling_origin_splits(
    frame: pd.DataFrame,
    n_splits: int = 3,
    horizon: int = 28,
    date_column: str = "ds",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, val_idx)`` pairs split by time, never at random.

    Fold ``i`` validates on the ``horizon`` days ending at
    ``max_date - (n_splits - 1 - i) * horizon``, and trains on everything strictly
    before that window. So the folds walk forward:

        fold 0:  train [start, T-3h)   val [T-3h, T-2h)
        fold 1:  train [start, T-2h)   val [T-2h, T-1h)
        fold 2:  train [start, T-1h)   val [T-1h, T)

    A random K-fold on a time series trains on the future to predict the past and
    reports an accuracy the model will never achieve in production.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    dates = pd.to_datetime(frame[date_column])
    end = dates.max()
    span = (end - dates.min()).days
    if span < n_splits * horizon:
        raise ValueError(
            f"need at least {n_splits * horizon} days of history for "
            f"{n_splits} folds of horizon {horizon}, but the frame spans {span}"
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    positions = np.arange(len(frame))
    for i in range(n_splits):
        val_end = end - pd.Timedelta(days=(n_splits - 1 - i) * horizon)
        val_start = val_end - pd.Timedelta(days=horizon - 1)
        val_mask = (dates >= val_start) & (dates <= val_end)
        train_mask = dates < val_start
        splits.append((positions[train_mask.to_numpy()], positions[val_mask.to_numpy()]))
        logger.debug(
            "fold %d: train < %s, val %s..%s", i, val_start.date(), val_start.date(),
            val_end.date()
        )
    return splits
