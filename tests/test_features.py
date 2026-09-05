"""Favorita feature pipeline: calendar, holidays, oil, lags and rolling-origin splits.

The non-anticipative tests are the important ones here. A leaked feature does not
raise; it just makes validation look good and production look broken.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from twinmind.data.features import (
    add_calendar_features,
    add_fourier_terms,
    add_lag_features,
    build_demand_features,
    build_holiday_flags,
    make_rolling_origin_splits,
    prepare_oil,
    select_top_series,
)
from twinmind.utils.config import FavoritaConfig

from .synthetic import make_favorita


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return make_favorita()


@pytest.fixture(scope="module")
def features(frames: dict[str, pd.DataFrame], favorita_config: FavoritaConfig) -> pd.DataFrame:
    return build_demand_features(frames, favorita_config, top_n=8)


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def test_output_has_the_neuralforecast_contract(features: pd.DataFrame) -> None:
    for column in ("unique_id", "ds", "y"):
        assert column in features.columns
    assert pd.api.types.is_datetime64_any_dtype(features["ds"])
    assert pd.api.types.is_float_dtype(features["y"])
    assert features["unique_id"].dtype == object


def test_output_is_sorted_and_unique_per_series_date(features: pd.DataFrame) -> None:
    assert not features.duplicated(["unique_id", "ds"]).any()
    for _, block in features.groupby("unique_id"):
        assert block["ds"].is_monotonic_increasing


def test_top_n_caps_the_series_count(
    frames: dict[str, pd.DataFrame], favorita_config: FavoritaConfig
) -> None:
    small = build_demand_features(frames, favorita_config, top_n=3)
    assert small["unique_id"].nunique() == 3


def test_select_top_series_keeps_the_largest_by_volume() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "ds": pd.date_range("2020-01-01", periods=3).tolist() * 3,
            "y": [100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0],
        }
    )
    kept = select_top_series(frame, top_n=2)
    assert set(kept["unique_id"]) == {"a", "c"}


# --------------------------------------------------------------------------- #
# Calendar and Fourier
# --------------------------------------------------------------------------- #
def test_calendar_features_are_correct() -> None:
    frame = pd.DataFrame({"ds": pd.to_datetime(["2015-01-15", "2015-03-29", "2016-02-29"])})
    out = add_calendar_features(frame)

    assert out["dow"].tolist() == [3, 6, 0]
    assert out["dom"].tolist() == [15, 29, 29]
    assert out["month"].tolist() == [1, 3, 2]
    assert out["is_weekend"].tolist() == [0, 1, 0]
    assert out["is_payday"].tolist() == [1, 0, 1]


def test_fourier_terms_are_bounded_and_periodic() -> None:
    frame = pd.DataFrame({"ds": pd.date_range("2015-01-01", periods=400, freq="D")})
    out = add_fourier_terms(frame, weekly_k=3, yearly_k=5)

    fourier = [c for c in out.columns if c.startswith("fourier_")]
    assert len(fourier) == 2 * 3 + 2 * 5
    assert out[fourier].abs().max().max() <= 1.0 + 1e-9

    # The weekly terms must repeat exactly every 7 days.
    weekly = [c for c in fourier if "weekly" in c]
    np.testing.assert_allclose(
        out[weekly].iloc[0].to_numpy(), out[weekly].iloc[7].to_numpy(), atol=1e-9
    )


# --------------------------------------------------------------------------- #
# Holidays -- the transferred/bridge trap
# --------------------------------------------------------------------------- #
def test_transferred_holiday_is_not_flagged_on_its_listed_date(
    frames: dict[str, pd.DataFrame], features: pd.DataFrame
) -> None:
    """A transferred holiday did NOT happen on the date it is listed against."""
    holidays = frames["holidays"]
    moved = holidays[holidays["type"].eq("Holiday") & holidays["transferred"]]["date"].iloc[0]

    rows = features[features["ds"] == moved]
    assert not rows.empty, "the moved date should still be present as a normal day"
    assert rows["is_holiday"].sum() == 0, "a transferred holiday was wrongly flagged"


def test_transfer_row_marks_the_date_the_holiday_actually_happened(
    frames: dict[str, pd.DataFrame], features: pd.DataFrame
) -> None:
    holidays = frames["holidays"]
    actual = holidays[holidays["type"].eq("Transfer")]["date"].iloc[0]
    rows = features[features["ds"] == actual]
    assert rows["is_holiday"].sum() == len(rows), "the Transfer date must be a holiday"


def test_work_day_is_not_a_holiday(
    frames: dict[str, pd.DataFrame], features: pd.DataFrame
) -> None:
    """A Work Day repays a Bridge -- it is the opposite of a day off."""
    holidays = frames["holidays"]
    work = holidays[holidays["type"].eq("Work Day")]["date"].iloc[0]
    rows = features[features["ds"] == work]

    assert rows["is_work_day"].sum() == len(rows)
    assert rows["is_holiday"].sum() == 0


def test_bridge_and_additional_days_count_as_holidays(
    frames: dict[str, pd.DataFrame], features: pd.DataFrame
) -> None:
    holidays = frames["holidays"]
    for kind in ("Bridge", "Additional"):
        date = holidays[holidays["type"].eq(kind)]["date"].iloc[0]
        rows = features[features["ds"] == date]
        assert rows["is_holiday"].sum() == len(rows), f"{kind} should be a holiday"


def test_local_and_regional_holidays_apply_only_to_matching_stores(
    frames: dict[str, pd.DataFrame], features: pd.DataFrame
) -> None:
    holidays, stores = frames["holidays"], frames["stores"]
    local = holidays[holidays["locale"].eq("Local")].iloc[0]
    expected = set(stores[stores["city"].eq(local["locale_name"])]["store_nbr"])

    flagged = set(
        features[(features["ds"] == local["date"]) & features["is_local_holiday"].eq(1)][
            "store_nbr"
        ]
    )
    assert flagged
    assert flagged <= expected, "a local holiday leaked to stores in other cities"


def test_holiday_flags_are_one_row_per_date_store(
    frames: dict[str, pd.DataFrame]
) -> None:
    flags = build_holiday_flags(frames["holidays"], frames["stores"])
    assert not flags.duplicated(["date", "store_nbr"]).any()
    assert flags["store_nbr"].notna().all()


# --------------------------------------------------------------------------- #
# Oil
# --------------------------------------------------------------------------- #
def test_oil_is_forward_filled_across_weekend_gaps(frames: dict[str, pd.DataFrame]) -> None:
    dates = pd.Series(pd.date_range("2015-01-01", periods=100, freq="D"))
    out = prepare_oil(frames["oil"], dates)

    assert out["oil_price"].isna().sum() == 0
    assert len(out) == (out["date"].max() - out["date"].min()).days + 1

    # A Sunday must carry the previous Friday's price, not a future one.
    sunday = out[out["date"].dt.dayofweek.eq(6)].iloc[0]
    friday = out[(out["date"] < sunday["date"]) & out["date"].dt.dayofweek.eq(4)].iloc[-1]
    assert sunday["oil_price"] == friday["oil_price"]


def test_features_have_no_missing_oil(features: pd.DataFrame) -> None:
    assert features["oil_price"].isna().sum() == 0


# --------------------------------------------------------------------------- #
# Lags and rolling statistics -- non-anticipative
# --------------------------------------------------------------------------- #
def _toy_series(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["s1"] * n + ["s2"] * n,
            "ds": pd.date_range("2020-01-01", periods=n).tolist() * 2,
            "y": np.arange(n, dtype="float64").tolist()
            + (100 + np.arange(n, dtype="float64")).tolist(),
        }
    )


def test_lags_are_shifted_by_exactly_k() -> None:
    out = add_lag_features(_toy_series(), lags=[1, 7], rolling_windows=[7])
    s1 = out[out["unique_id"] == "s1"].reset_index(drop=True)

    assert pd.isna(s1.loc[0, "lag_1"])
    assert s1.loc[10, "lag_1"] == s1.loc[9, "y"]
    assert s1.loc[10, "lag_7"] == s1.loc[3, "y"]


def test_lags_do_not_bleed_across_series() -> None:
    out = add_lag_features(_toy_series(), lags=[1], rolling_windows=[7])
    first_of_s2 = out[out["unique_id"] == "s2"].iloc[0]
    assert pd.isna(first_of_s2["lag_1"]), "the first row of a series has no lag"


def test_rolling_statistics_exclude_the_current_row() -> None:
    """roll_mean_7 at t must average t-7..t-1, never including y at t."""
    out = add_lag_features(_toy_series(), lags=[1], rolling_windows=[7])
    s1 = out[out["unique_id"] == "s1"].reset_index(drop=True)

    row = 20
    expected = s1.loc[row - 7 : row - 1, "y"].mean()
    assert s1.loc[row, "roll_mean_7"] == pytest.approx(expected)
    assert s1.loc[row, "roll_mean_7"] != pytest.approx(s1.loc[row - 6 : row, "y"].mean())


def test_rolling_features_reject_a_zero_lag() -> None:
    with pytest.raises(ValueError, match="non-anticipative"):
        add_lag_features(_toy_series(), lags=[0], rolling_windows=[7])


def test_features_are_non_anticipative_under_truncation(
    frames: dict[str, pd.DataFrame], favorita_config: FavoritaConfig
) -> None:
    """Build features, chop off the future, rebuild -- overlapping rows must be identical.

    This is the general test for leakage: if any feature at time t depended on data
    after t, removing that future data would change the value at t.
    """
    full = build_demand_features(frames, favorita_config, top_n=4)

    cutoff = full["ds"].max() - pd.Timedelta(days=60)
    truncated_raw = {k: v.copy() for k, v in frames.items()}
    train = truncated_raw["train"]
    truncated_raw["train"] = train[pd.to_datetime(train["date"]) <= cutoff]

    partial = build_demand_features(truncated_raw, favorita_config, top_n=4)

    derived = [
        c for c in partial.columns
        if c.startswith(("lag_", "roll_", "fourier_"))
        or c in {"oil_price", "is_holiday", "is_work_day", "dow", "month", "is_payday"}
    ]
    assert derived, "no derived columns found to check"

    merged = full.merge(
        partial, on=["unique_id", "ds"], suffixes=("_full", "_part"), how="inner"
    )
    assert len(merged) > 0

    for column in derived:
        a = merged[f"{column}_full"]
        b = merged[f"{column}_part"]
        if pd.api.types.is_numeric_dtype(a):
            mismatch = ~np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            mismatch = a.ne(b)
        assert not mismatch.any(), (
            f"{column} changed when future data was removed -- it is anticipative "
            f"({int(mismatch.sum())} of {len(merged)} overlapping rows differ)"
        )


# --------------------------------------------------------------------------- #
# Rolling-origin splits
# --------------------------------------------------------------------------- #
def test_every_fold_is_strictly_ordered_in_time(features: pd.DataFrame) -> None:
    folds = make_rolling_origin_splits(features, n_splits=3, horizon=28)
    assert len(folds) == 3

    for i, (train_idx, val_idx) in enumerate(folds):
        assert len(train_idx) > 0 and len(val_idx) > 0
        train_max = features["ds"].iloc[train_idx].max()
        val_min = features["ds"].iloc[val_idx].min()
        assert train_max < val_min, f"fold {i} trains on data at or after its validation window"


def test_validation_windows_are_the_requested_horizon(features: pd.DataFrame) -> None:
    for _, val_idx in make_rolling_origin_splits(features, n_splits=3, horizon=28):
        dates = features["ds"].iloc[val_idx]
        assert (dates.max() - dates.min()).days == 27


def test_folds_walk_forward_without_overlapping_validation(features: pd.DataFrame) -> None:
    folds = make_rolling_origin_splits(features, n_splits=3, horizon=28)
    windows = [
        (features["ds"].iloc[v].min(), features["ds"].iloc[v].max()) for _, v in folds
    ]
    for earlier, later in itertools.pairwise(windows):
        assert earlier[1] < later[0], "validation windows overlap"
    assert all(len(folds[i][0]) < len(folds[i + 1][0]) for i in range(len(folds) - 1))


def test_splits_are_never_random(features: pd.DataFrame) -> None:
    """Consecutive calls must give identical folds -- there is no RNG involved."""
    a = make_rolling_origin_splits(features, n_splits=3, horizon=28)
    b = make_rolling_origin_splits(features, n_splits=3, horizon=28)
    for (ta, va), (tb, vb) in zip(a, b, strict=True):
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(va, vb)


def test_splits_reject_an_impossible_request(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="need at least"):
        make_rolling_origin_splits(features, n_splits=50, horizon=28)
    with pytest.raises(ValueError, match="n_splits must be"):
        make_rolling_origin_splits(features, n_splits=0, horizon=28)
