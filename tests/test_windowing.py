"""C-MAPSS preprocessing: RUL targets, sensor selection, regime normalisation,
windowing and unit-wise splits.

Tests marked ``data`` need data/raw populated; everything else runs on the
synthetic fixtures so a clean checkout stays green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twinmind.data.windowing import (
    OperatingConditionNormaliser,
    compute_rul,
    compute_test_rul,
    drop_constant_sensors,
    fit_operating_condition_normaliser,
    make_last_windows,
    make_windows,
    split_by_unit,
)
from twinmind.utils.config import CMAPSSConfig

from .synthetic import make_cmapss, make_cmapss_multi_regime

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]


# --------------------------------------------------------------------------- #
# (a) RUL
# --------------------------------------------------------------------------- #
def test_rul_is_capped_at_exactly_125() -> None:
    frame = make_cmapss(n_units=4, min_cycles=200, max_cycles=260, seed=1)
    rul = compute_rul(frame, cap=125)

    assert rul.max() == 125, "the cap must bind on long trajectories"
    assert rul.min() == 0, "RUL reaches zero at the failure cycle"
    assert (rul <= 125).all()
    assert rul.dtype == np.int64


def test_rul_is_monotonically_non_increasing_within_each_unit() -> None:
    frame = make_cmapss(n_units=6, seed=2)
    frame["rul"] = compute_rul(frame, cap=125)

    for unit, block in frame.groupby("unit"):
        series = block.sort_values("cycle")["rul"]
        assert series.is_monotonic_decreasing, f"unit {unit} RUL increases somewhere"


def test_rul_last_cycle_is_zero_for_every_unit() -> None:
    frame = make_cmapss(n_units=5, seed=3)
    frame["rul"] = compute_rul(frame, cap=125)
    last = frame.sort_values("cycle").groupby("unit").tail(1)
    assert (last["rul"] == 0).all()


def test_rul_is_piecewise_linear_not_plain_linear() -> None:
    """Below the cap the target decreases one-for-one with cycle; above it is flat."""
    frame = make_cmapss(n_units=1, min_cycles=300, max_cycles=300, seed=4)
    rul = compute_rul(frame, cap=125).to_numpy()

    assert (rul[:175] == 125).all(), "early cycles must sit flat at the cap"
    tail_diff = np.diff(rul[175:])
    assert (tail_diff == -1).all(), "past the knee RUL must fall linearly"


def test_rul_cap_is_configurable() -> None:
    frame = make_cmapss(n_units=2, min_cycles=200, max_cycles=200, seed=5)
    assert compute_rul(frame, cap=60).max() == 60


def test_rul_rejects_a_non_positive_cap() -> None:
    with pytest.raises(ValueError, match="cap must be positive"):
        compute_rul(make_cmapss(n_units=1), cap=0)


def test_test_rul_accounts_for_truncation() -> None:
    """Test trajectories stop before failure, so the last cycle's RUL is not 0."""
    frame = make_cmapss(n_units=2, min_cycles=50, max_cycles=50, seed=6)
    frame["rul_at_last_cycle"] = 30

    rul = compute_test_rul(frame, cap=125)
    last = frame.assign(rul=rul).sort_values("cycle").groupby("unit").tail(1)
    assert (last["rul"] == 30).all()

    first = frame.assign(rul=rul).sort_values("cycle").groupby("unit").head(1)
    assert (first["rul"] == min(49 + 30, 125)).all()


def test_test_rul_requires_the_ground_truth_column() -> None:
    with pytest.raises(ValueError, match="rul_at_last_cycle"):
        compute_test_rul(make_cmapss(n_units=1))


# --------------------------------------------------------------------------- #
# (b) Constant sensors
# --------------------------------------------------------------------------- #
def test_constant_sensors_are_detected_not_assumed() -> None:
    frame = make_cmapss(n_units=4, n_constant=3, seed=7)
    kept, dropped = drop_constant_sensors(frame, SENSORS)

    assert set(dropped) == {"sensor_1", "sensor_2", "sensor_3"}
    assert set(kept) == set(SENSORS) - set(dropped)
    assert kept + dropped != [] and not set(kept) & set(dropped)


def test_constant_sensor_detection_preserves_input_order() -> None:
    frame = make_cmapss(n_units=3, n_constant=2, seed=8)
    kept, _ = drop_constant_sensors(frame, SENSORS)
    assert kept == [c for c in SENSORS if c in set(kept)]


def test_drop_constant_sensors_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError, match="not in frame"):
        drop_constant_sensors(make_cmapss(n_units=2), ["sensor_1", "nope"])


@pytest.mark.data
def test_fd001_constant_sensors_match_the_known_set(cmapss_config: CMAPSSConfig) -> None:
    """Sanity check against the documented FD001 result -- an assertion, not the implementation.

    The detector is a variance threshold; this pins that it reproduces the set
    every published C-MAPSS paper drops on FD001.
    """
    from twinmind.data.loaders import load_cmapss

    train = load_cmapss("FD001", "train")
    _, dropped = drop_constant_sensors(
        train, cmapss_config.sensor_columns, cmapss_config.constant_variance_threshold
    )
    indices = {int(c.split("_")[1]) for c in dropped}
    assert indices == {1, 5, 6, 10, 16, 18, 19}


@pytest.mark.data
def test_multi_regime_subsets_drop_fewer_sensors(cmapss_config: CMAPSSConfig) -> None:
    """FD002/FD004 vary across regimes, so sensors flat on FD001 carry signal there."""
    from twinmind.data.loaders import load_cmapss

    dropped_by_subset = {}
    for subset in ("FD001", "FD002"):
        train = load_cmapss(subset, "train")  # type: ignore[arg-type]
        _, dropped = drop_constant_sensors(
            train, cmapss_config.sensor_columns, cmapss_config.constant_variance_threshold
        )
        dropped_by_subset[subset] = set(dropped)

    assert len(dropped_by_subset["FD002"]) < len(dropped_by_subset["FD001"])


# --------------------------------------------------------------------------- #
# (c) Operating-condition normalisation
# --------------------------------------------------------------------------- #
def test_normaliser_standardises_the_training_split() -> None:
    frame = make_cmapss(n_units=6, n_constant=0, seed=9)
    norm = fit_operating_condition_normaliser(frame, SENSORS, SETTINGS, n_conditions=1)
    out = norm.transform(frame)

    assert out[SENSORS].mean().abs().max() < 1e-6
    assert abs(out[SENSORS].std().mean() - 1.0) < 1e-6


def test_normaliser_does_not_read_statistics_from_the_frame_it_transforms() -> None:
    """The leak check: transforming a shifted frame must NOT recentre it on zero.

    If ``transform`` recomputed the mean from its argument, the output would come
    out centred whatever we fed it, and fitting on train would be pointless.
    """
    train = make_cmapss(n_units=6, n_constant=0, seed=10)
    norm = fit_operating_condition_normaliser(train, SENSORS, SETTINGS, n_conditions=1)

    shifted = train.copy()
    shifted[SENSORS] = shifted[SENSORS] + 500.0

    out = norm.transform(shifted)
    observed = out[SENSORS].mean().abs().min()
    assert observed > 1.0, (
        "transform() recentred a shifted frame, so it is reading statistics from "
        "its input instead of using the fitted ones"
    )


def test_normaliser_refuses_to_transform_before_fitting() -> None:
    norm = OperatingConditionNormaliser(n_conditions=1, sensor_columns=SENSORS)
    with pytest.raises(RuntimeError, match="not fitted"):
        norm.transform(make_cmapss(n_units=2))


def test_multi_regime_normaliser_recovers_the_trend_global_scaling_hides() -> None:
    """Within-regime z-scoring should leave far less regime variance than global."""
    frame = make_cmapss_multi_regime(n_units=6, n_regimes=6, seed=11)

    per_regime = fit_operating_condition_normaliser(
        frame, SENSORS, SETTINGS, n_conditions=6
    ).transform(frame)
    global_z = fit_operating_condition_normaliser(
        frame, SENSORS, SETTINGS, n_conditions=1
    ).transform(frame)

    truth = frame["true_regime"]
    spread_regime = per_regime.groupby(truth)["sensor_5"].mean().std()
    spread_global = global_z.groupby(truth)["sensor_5"].mean().std()

    assert spread_regime < spread_global / 10, (
        f"within-regime scaling left spread {spread_regime:.3f} vs global "
        f"{spread_global:.3f}; the regime effect was not removed"
    )


def test_normaliser_round_trips_through_a_dict() -> None:
    frame = make_cmapss_multi_regime(n_units=4, n_regimes=6, seed=12)
    original = fit_operating_condition_normaliser(frame, SENSORS, SETTINGS, n_conditions=6)

    restored = OperatingConditionNormaliser.from_dict(original.to_dict())
    pd.testing.assert_frame_equal(original.transform(frame), restored.transform(frame))


def test_unfitted_normaliser_cannot_be_serialised() -> None:
    with pytest.raises(RuntimeError, match="unfitted"):
        OperatingConditionNormaliser(sensor_columns=SENSORS).to_dict()


# --------------------------------------------------------------------------- #
# (d) Windowing
# --------------------------------------------------------------------------- #
def test_window_shapes() -> None:
    frame = make_cmapss(n_units=5, min_cycles=60, max_cycles=90, seed=13)
    frame["rul"] = compute_rul(frame)
    features = SENSORS[:8]

    x, y, units = make_windows(frame, features, window=30, stride=1)

    assert x.ndim == 3
    assert x.shape[1:] == (30, len(features))
    assert x.shape[0] == y.shape[0] == units.shape[0]
    assert x.dtype == np.float32


def test_window_count_matches_the_stride_arithmetic() -> None:
    frame = make_cmapss(n_units=1, min_cycles=100, max_cycles=100, seed=14)
    frame["rul"] = compute_rul(frame)

    x, _, _ = make_windows(frame, SENSORS[:4], window=30, stride=1)
    assert len(x) == 100 - 30 + 1

    x2, _, _ = make_windows(frame, SENSORS[:4], window=30, stride=7)
    assert len(x2) == len(range(30, 101, 7))


def test_short_units_are_left_padded_with_their_first_observation() -> None:
    """A unit shorter than the window must still yield exactly one window."""
    frame = make_cmapss(n_units=3, min_cycles=50, max_cycles=60, seed=15, short_unit_cycles=12)
    frame["rul"] = compute_rul(frame)
    features = SENSORS[:5]

    x, _, units = make_windows(frame, features, window=30, stride=1)

    short_id = frame["unit"].max()
    short_windows = x[units == short_id]
    assert len(short_windows) == 1, "the short unit must not be dropped"

    short_block = frame[frame["unit"] == short_id].sort_values("cycle")
    first_row = short_block[features].to_numpy(dtype="float32")[0]
    window = short_windows[0]

    n_pad = 30 - 12
    assert np.allclose(window[:n_pad], first_row), "padding must repeat the first observation"
    assert np.allclose(window[n_pad:], short_block[features].to_numpy(dtype="float32"))


def test_window_label_is_the_target_at_the_final_cycle() -> None:
    frame = make_cmapss(n_units=1, min_cycles=40, max_cycles=40, seed=16)
    frame["rul"] = compute_rul(frame)
    _, y, _ = make_windows(frame, SENSORS[:3], window=30, stride=1)

    expected = frame.sort_values("cycle")["rul"].to_numpy()[29:]
    assert np.array_equal(y, expected)


def test_make_last_windows_gives_one_window_per_unit() -> None:
    frame = make_cmapss(n_units=7, min_cycles=45, max_cycles=80, seed=17)
    frame["rul"] = compute_rul(frame)

    x, y, units = make_windows(frame, SENSORS[:4], window=30)
    last_x, last_y, last_units = make_last_windows(frame, SENSORS[:4], window=30)

    assert len(last_x) == frame["unit"].nunique()
    assert len(np.unique(last_units)) == len(last_units)
    assert len(last_x) < len(x)
    assert last_y.min() >= 0


def test_windowing_rejects_bad_arguments() -> None:
    frame = make_cmapss(n_units=2)
    frame["rul"] = compute_rul(frame)
    with pytest.raises(ValueError, match="must be positive"):
        make_windows(frame, SENSORS[:2], window=0)
    with pytest.raises(ValueError, match="feature columns not in frame"):
        make_windows(frame, ["nope"], window=5)
    with pytest.raises(ValueError, match="target column"):
        make_windows(frame, SENSORS[:2], window=5, target_column="missing")


# --------------------------------------------------------------------------- #
# (e) Splitting
# --------------------------------------------------------------------------- #
def test_split_by_unit_produces_disjoint_unit_sets() -> None:
    frame = make_cmapss(n_units=20, seed=18)
    train, val = split_by_unit(frame, val_fraction=0.25, seed=42)

    train_units, val_units = set(train["unit"]), set(val["unit"])
    assert not train_units & val_units, "a unit appeared in both splits"
    assert train_units | val_units == set(frame["unit"])
    assert len(train) + len(val) == len(frame)


def test_split_by_unit_never_splits_a_trajectory() -> None:
    """Every cycle of a given unit must land on the same side of the split."""
    frame = make_cmapss(n_units=15, seed=19)
    train, val = split_by_unit(frame, val_fraction=0.2, seed=7)

    for unit in frame["unit"].unique():
        total = int((frame["unit"] == unit).sum())
        in_train = int((train["unit"] == unit).sum())
        in_val = int((val["unit"] == unit).sum())
        assert {in_train, in_val} == {0, total}, f"unit {unit} was split across folds"


def test_split_by_unit_is_deterministic_for_a_seed() -> None:
    frame = make_cmapss(n_units=20, seed=20)
    a, _ = split_by_unit(frame, 0.2, seed=123)
    b, _ = split_by_unit(frame, 0.2, seed=123)
    c, _ = split_by_unit(frame, 0.2, seed=456)

    assert set(a["unit"]) == set(b["unit"])
    assert set(a["unit"]) != set(c["unit"])


def test_split_by_unit_respects_the_requested_fraction() -> None:
    frame = make_cmapss(n_units=20, seed=21)
    _, val = split_by_unit(frame, val_fraction=0.25, seed=42)
    assert val["unit"].nunique() == 5


def test_split_by_unit_rejects_a_degenerate_fraction() -> None:
    frame = make_cmapss(n_units=5)
    with pytest.raises(ValueError, match="must be in"):
        split_by_unit(frame, val_fraction=0.0)
    with pytest.raises(ValueError, match="must be in"):
        split_by_unit(frame, val_fraction=1.0)


@pytest.mark.data
def test_padding_is_needed_by_the_real_test_splits(cmapss_config: CMAPSSConfig) -> None:
    """FD002/FD004 test units really are shorter than the window.

    Without left-padding these engines would be silently dropped, and the C-MAPSS
    protocol requires exactly one prediction per test engine.
    """
    from twinmind.data.loaders import load_cmapss

    short_units = 0
    for subset in ("FD002", "FD004"):
        frame = load_cmapss(subset, "test")  # type: ignore[arg-type]
        lengths = frame.groupby("unit").size()
        short_units += int((lengths < cmapss_config.window_length).sum())
    assert short_units > 0, "expected some truncated test trajectories"

    frame = load_cmapss("FD004", "test")
    frame["rul"] = 0
    _, _, units = make_last_windows(
        frame, cmapss_config.sensor_columns, window=cmapss_config.window_length
    )
    assert len(units) == frame["unit"].nunique(), "every test engine must yield one window"
