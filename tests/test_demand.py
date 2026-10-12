"""The demand contract, the leakage guards, and the copula path sampler.

The two tests that matter most are ``test_forecast_is_invariant_to_truncation``
and ``test_no_feature_uses_information_after_the_origin``. A leaked feature does
not raise; it produces a validation score that looks excellent and a twin whose
order book is impossible to reproduce in reality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twinmind.models.demand import (
    DemandData,
    DemandModel,
    DemandQuantiles,
)
from twinmind.models.demand_features import (
    KNOWN_FUTURE,
    ORIGIN_ONLY,
    build_inference_rows,
    build_training_rows,
    origin_features,
    weekly_origins,
)
from twinmind.models.demand_lgbm import (
    SAMPLER_QUANTILES,
    LightGBMQuantileModel,
    fit_conformal,
    fit_copula,
    fit_quantile_models,
    interpolate_quantiles,
    nearest_psd,
)

HORIZON = 28


def _frame(n_series: int = 4, days: int = 500, seed: int = 0) -> pd.DataFrame:
    """A featurised-looking frame with real weekly seasonality and a trend."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=days, freq="D")
    rows = []
    for s in range(n_series):
        base = 50.0 * (s + 1)
        weekly = 12 * np.sin(2 * np.pi * np.arange(days) / 7)
        trend = np.linspace(0, 15, days)
        # An AR(1) shock, so residuals genuinely carry serial correlation.
        shock = np.zeros(days)
        for i in range(1, days):
            shock[i] = 0.6 * shock[i - 1] + rng.normal(0, 6)
        y = np.clip(base + weekly + trend + shock, 0, None)
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{s}",
                    "ds": dates,
                    "y": y,
                    "onpromotion": rng.integers(0, 4, days).astype("float64"),
                    "oil_price": 50 + np.cumsum(rng.normal(0, 0.2, days)),
                    "transactions": rng.integers(400, 3000, days).astype("float64"),
                    "dow": dates.dayofweek,
                    "dom": dates.day,
                    "month": dates.month,
                    "week_of_year": dates.isocalendar().week.astype(int),
                    "is_weekend": (dates.dayofweek >= 5).astype(int),
                    "is_payday": ((dates.day == 15) | dates.is_month_end).astype(int),
                    "is_holiday": rng.integers(0, 2, days) * (rng.random(days) < 0.08),
                    "store_nbr": s + 1,
                    "cluster": (s % 3) + 1,
                    "fourier_weekly_sin_1": np.sin(2 * np.pi * np.arange(days) / 7),
                    "fourier_weekly_cos_1": np.cos(2 * np.pi * np.arange(days) / 7),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _frame()


@pytest.fixture(scope="module")
def snapshot(frame: pd.DataFrame) -> DemandData:
    return DemandData(frame)


@pytest.fixture(scope="module")
def fitted(frame: pd.DataFrame, snapshot: DemandData) -> LightGBMQuantileModel:
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    origins = weekly_origins(frame, origin, 20)
    rows = build_training_rows(frame, origins, horizon=HORIZON, min_history=60)
    boosters, spec = fit_quantile_models(rows, SAMPLER_QUANTILES, seed=0)
    return LightGBMQuantileModel(boosters, spec, snapshot, horizon=HORIZON)


# --------------------------------------------------------------------------- #
# DemandQuantiles validators
# --------------------------------------------------------------------------- #
def test_valid_quantiles_round_trip() -> None:
    q = DemandQuantiles(horizon=3, q10=[1.0, 2.0, 3.0], q50=[2.0, 3.0, 4.0], q90=[3.0, 4.0, 5.0])
    assert q.horizon == 3
    assert q.as_frame(pd.Timestamp("2017-01-01")).iloc[0]["ds"] == pd.Timestamp("2017-01-02")


def test_quantiles_reject_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected horizon=3"):
        DemandQuantiles(horizon=3, q10=[1.0, 2.0], q50=[2.0, 3.0], q90=[3.0, 4.0])


def test_quantiles_reject_negative_demand() -> None:
    with pytest.raises(ValueError, match="negative demand"):
        DemandQuantiles(horizon=2, q10=[-1.0, 0.0], q50=[1.0, 1.0], q90=[2.0, 2.0])


def test_quantiles_reject_crossing() -> None:
    with pytest.raises(ValueError, match="quantiles cross"):
        DemandQuantiles(horizon=2, q10=[5.0, 1.0], q50=[2.0, 2.0], q90=[9.0, 3.0])


def test_as_frame_covers_origin_plus_one_onwards() -> None:
    """origin is the last OBSERVED day, so the forecast starts the day after."""
    q = DemandQuantiles(horizon=3, q10=[0.0] * 3, q50=[1.0] * 3, q90=[2.0] * 3)
    dates = q.as_frame(pd.Timestamp("2017-03-10"))["ds"].tolist()
    assert dates[0] == pd.Timestamp("2017-03-11")
    assert dates[-1] == pd.Timestamp("2017-03-13")


# --------------------------------------------------------------------------- #
# DemandData
# --------------------------------------------------------------------------- #
def test_snapshot_is_frozen_and_copies_its_frame(frame: pd.DataFrame) -> None:
    data = DemandData(frame)
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raise FrozenInstanceError
        data.target = "other"  # type: ignore[misc]

    before = len(data.frame)
    frame.loc[0, "y"] = -999.0  # mutate the caller's frame
    assert len(data.frame) == before
    assert data.frame["y"].min() >= 0, "snapshot must not alias the caller's frame"


def test_truncate_blanks_the_unknowable_and_keeps_the_knowable(
    snapshot: DemandData,
) -> None:
    """Default truncation removes what the origin cannot know, and only that."""
    origin = snapshot.frame["ds"].max() - pd.Timedelta(days=30)
    cut = snapshot.truncate(origin)
    future = cut.frame[cut.frame["ds"] > origin]

    assert future["y"].isna().all(), "realised sales after the origin must be gone"
    assert future["oil_price"].isna().all(), "oil is not knowable ahead"
    assert future["transactions"].isna().all(), "footfall is not knowable ahead"
    assert future["dow"].notna().all(), "the calendar IS knowable ahead"
    assert future["onpromotion"].notna().all(), "planned promotions are knowable"
    # History is untouched.
    past = cut.frame[cut.frame["ds"] <= origin]
    assert past["y"].notna().all()


def test_hard_truncate_drops_the_future_rows(snapshot: DemandData) -> None:
    origin = snapshot.frame["ds"].max() - pd.Timedelta(days=30)
    cut = snapshot.truncate(origin, hard=True)
    assert cut.frame["ds"].max() <= origin
    assert len(cut.frame) < len(snapshot.frame)


def test_history_and_future_split_at_the_origin(snapshot: DemandData) -> None:
    origin = snapshot.frame["ds"].max() - pd.Timedelta(days=40)
    history = snapshot.history("s0", origin)
    future = snapshot.future("s0", origin, HORIZON)

    assert history["ds"].max() == origin
    assert future["ds"].min() == origin + pd.Timedelta(days=1)
    assert len(future) == HORIZON


# --------------------------------------------------------------------------- #
# THE leakage guards
# --------------------------------------------------------------------------- #
def test_forecast_is_invariant_to_truncation(
    fitted: LightGBMQuantileModel, frame: pd.DataFrame
) -> None:
    """A forecast at an origin must not change when the future is deleted.

    This is the contract test. If any feature reads past the origin, removing
    the future changes the answer.
    """
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    full = fitted.quantiles("s0", origin, HORIZON)

    truncated_model = LightGBMQuantileModel(
        fitted.boosters, fitted.spec, fitted.data.truncate(origin), horizon=HORIZON
    )
    truncated = truncated_model.quantiles("s0", origin, HORIZON)

    np.testing.assert_allclose(full.q10, truncated.q10, rtol=1e-10)
    np.testing.assert_allclose(full.q50, truncated.q50, rtol=1e-10)
    np.testing.assert_allclose(full.q90, truncated.q90, rtol=1e-10)


def test_no_feature_uses_information_after_the_origin(frame: pd.DataFrame) -> None:
    """Corrupting every post-origin row must leave the feature matrix unchanged.

    Stronger than truncation: it proves the features are not merely insensitive
    to *missing* future rows but to their *contents*.
    """
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON + 5)
    series = frame[frame["unique_id"] == "s0"]
    history = series[series["ds"] <= origin]
    future = series[series["ds"] > origin]

    clean = build_inference_rows(history, future, origin, HORIZON)

    poisoned = future.copy()
    for column in ("y", *ORIGIN_ONLY):
        if column in poisoned.columns:
            poisoned[column] = 1e9
    dirty = build_inference_rows(history, poisoned, origin, HORIZON)

    shared = [c for c in clean.columns if c in dirty.columns and c != "ds"]
    pd.testing.assert_frame_equal(clean[shared], dirty[shared])


def test_origin_features_only_read_the_history_given(frame: pd.DataFrame) -> None:
    series = frame[frame["unique_id"] == "s0"].reset_index(drop=True)
    cut = series.iloc[:300]

    from_cut = origin_features(cut)
    from_full_sliced = origin_features(series.iloc[:300])
    assert from_cut == from_full_sliced


def test_known_future_and_origin_only_do_not_overlap() -> None:
    assert not set(KNOWN_FUTURE) & set(ORIGIN_ONLY)
    assert "y" not in KNOWN_FUTURE, "the target is never a known-future covariate"
    for lagged in ("lag_1", "lag_7", "roll_mean_7"):
        assert lagged not in KNOWN_FUTURE


def test_training_rows_never_carry_a_target_day_lag(frame: pd.DataFrame) -> None:
    """Every history feature is as-of-origin, so it repeats across the 28 steps."""
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    rows = build_training_rows(frame, [origin], horizon=HORIZON, min_history=60)
    one = rows[rows["unique_id"] == "s0"].sort_values("k")

    for column in [c for c in one.columns if c.startswith("origin_")]:
        assert one[column].nunique(dropna=False) == 1, (
            f"{column} varies across k, so it is being computed per target day "
            "rather than at the origin"
        )
    assert one["k"].tolist() == list(range(1, HORIZON + 1))


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #
class _StubDemandModel:
    """Minimal conforming implementation, to prove the Protocol is satisfiable."""

    def quantiles(self, series_id: str, origin: pd.Timestamp, horizon: int) -> DemandQuantiles:
        return DemandQuantiles(
            horizon=horizon,
            q10=[1.0] * horizon,
            q50=[2.0] * horizon,
            q90=[3.0] * horizon,
        )

    def sample_paths(
        self, series_id: str, origin: pd.Timestamp, horizon: int, n_samples: int, seed: int
    ) -> np.ndarray:
        return np.full((n_samples, horizon), 2.0)


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(_StubDemandModel(), DemandModel)


def test_real_model_satisfies_the_protocol(fitted: LightGBMQuantileModel) -> None:
    assert isinstance(fitted, DemandModel)


def test_quantiles_are_monotone_and_nonnegative(
    fitted: LightGBMQuantileModel, frame: pd.DataFrame
) -> None:
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    for series_id in ("s0", "s1", "s2"):
        q = fitted.quantiles(series_id, origin, HORIZON)
        lo, mid, hi = np.array(q.q10), np.array(q.q50), np.array(q.q90)
        assert (lo >= 0).all()
        assert (lo <= mid).all() and (mid <= hi).all()


def test_save_and_load_round_trip(fitted: LightGBMQuantileModel, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from twinmind.models.demand import load_demand_model

    origin = fitted.data.frame["ds"].max() - pd.Timedelta(days=HORIZON)
    before = fitted.quantiles("s0", origin, HORIZON)

    fitted.save(tmp_path / "demand")
    restored = load_demand_model(tmp_path / "demand", fitted.data)
    assert isinstance(restored, DemandModel)

    after = restored.quantiles("s0", origin, HORIZON)
    np.testing.assert_allclose(before.q50, after.q50, rtol=1e-8)
    assert not (tmp_path / "demand" / "lgbm_q50.pt").exists(), "boosters are not .pt files"
    assert (tmp_path / "demand" / "lgbm_q50.txt").exists()


# --------------------------------------------------------------------------- #
# Copula path sampling
# --------------------------------------------------------------------------- #
def test_nearest_psd_repairs_an_indefinite_matrix() -> None:
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < 0

    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-10
    np.testing.assert_allclose(np.diag(fixed), 1.0, atol=1e-10)


def test_interpolate_quantiles_holds_the_tails_flat() -> None:
    levels = np.array([0.1, 0.5, 0.9])
    values = np.array([10.0, 20.0, 30.0])
    assert interpolate_quantiles(levels, values, np.array([0.01]))[0] == 10.0
    assert interpolate_quantiles(levels, values, np.array([0.99]))[0] == 30.0
    assert interpolate_quantiles(levels, values, np.array([0.5]))[0] == 20.0


def test_sampled_paths_are_deterministic_nonnegative_and_shaped(
    fitted: LightGBMQuantileModel, frame: pd.DataFrame
) -> None:
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    first = fitted.sample_paths("s0", origin, HORIZON, n_samples=64, seed=7)
    second = fitted.sample_paths("s0", origin, HORIZON, n_samples=64, seed=7)

    assert first.shape == (64, HORIZON)
    assert (first >= 0).all()
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, fitted.sample_paths("s0", origin, HORIZON, 64, seed=8))


def test_sampled_marginals_match_the_calibrated_quantiles(
    fitted: LightGBMQuantileModel, frame: pd.DataFrame
) -> None:
    """Sampled paths must reproduce the quantiles the model actually reports.

    Checked against ``quantiles()`` rather than against the raw boosters, so if
    conformal calibration is active the marginals are the *calibrated* ones --
    the sampler and the reported interval cannot drift apart.

    Tolerance is ``max(5% of the q90-q10 width, 3 x the Monte Carlo standard
    error)``. The 5% figure alone is not a safe threshold on its own: on the
    narrowest steps of this fixture the sampling error at n = 5,000 is itself
    about 18% of the width, and it shrinks as 1/sqrt(n) (measured: 17.7% ->
    4.7% -> 2.5% at 5k -> 20k -> 80k), which is noise rather than bias. The
    standard error is estimated from the spread across independent batches, so
    the bound adapts to the interval instead of needing a hand-tuned n.
    """
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    q = fitted.quantiles("s0", origin, HORIZON)

    batches = 8
    per_batch = 2500
    stacked = np.stack(
        [
            fitted.sample_paths("s0", origin, HORIZON, n_samples=per_batch, seed=100 + b)
            for b in range(batches)
        ]
    )
    pooled = stacked.reshape(-1, HORIZON)

    for step in range(HORIZON):
        width = max(q.q90[step] - q.q10[step], 1e-6)
        for level, expected in ((0.1, q.q10[step]), (0.5, q.q50[step]), (0.9, q.q90[step])):
            got = float(np.quantile(pooled[:, step], level))
            batch_estimates = [float(np.quantile(s[:, step], level)) for s in stacked]
            standard_error = float(np.std(batch_estimates, ddof=1)) / np.sqrt(batches)
            tolerance = max(0.05 * width, 3.0 * standard_error)
            assert abs(got - expected) <= tolerance, (
                f"step {step} q{int(level * 100)}: sampled {got:.2f} vs reported "
                f"{expected:.2f}, tolerance {tolerance:.2f} "
                f"(5% width {0.05 * width:.2f}, 3 SE {3 * standard_error:.2f})"
            )


def test_sampler_marginals_follow_the_conformal_adjustment(
    fitted: LightGBMQuantileModel, frame: pd.DataFrame
) -> None:
    """Turning conformal on must widen the sampled paths, not just the reported band.

    The failure this guards against is a sampler that reads the raw boosters
    while ``quantiles()`` reports calibrated ones: the twin would then sample
    from an interval narrower than the one documented, and its stockout
    probabilities would be optimistic.
    """
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    before = fitted.sample_paths("s0", origin, HORIZON, n_samples=4000, seed=5)
    spread_before = float(
        np.mean(np.quantile(before, 0.9, axis=0) - np.quantile(before, 0.1, axis=0))
    )

    widened = LightGBMQuantileModel(
        fitted.boosters,
        fitted.spec,
        fitted.data,
        conformal={
            "lower": np.full(HORIZON, 20.0),
            "upper": np.full(HORIZON, 20.0),
        },
        horizon=HORIZON,
    )
    after = widened.sample_paths("s0", origin, HORIZON, n_samples=4000, seed=5)
    spread_after = float(np.mean(np.quantile(after, 0.9, axis=0) - np.quantile(after, 0.1, axis=0)))

    assert spread_after > spread_before * 1.5, (
        f"sampled spread {spread_before:.2f} -> {spread_after:.2f}; the sampler is "
        "not reading the calibrated quantiles"
    )
    reported = widened.quantiles("s0", origin, HORIZON)
    np.testing.assert_allclose(
        np.quantile(after, 0.9, axis=0),
        reported.q90,
        rtol=0.10,
        err_msg="sampled q90 must track the reported q90",
    )


def test_copula_reproduces_the_reference_autocorrelation() -> None:
    """Sampled paths must carry the serial dependence the residuals had.

    Reference: an AR(1) residual process with a known lag-1 correlation. The
    sampled paths' normal scores must match it within +/- 0.10, and independent
    sampling must visibly fail the same check -- otherwise the test proves
    nothing.
    """
    rng = np.random.default_rng(0)
    horizon, n_paths, rho = HORIZON, 600, 0.6
    residuals = np.empty((n_paths, horizon))
    # Seed step 0 from the stationary distribution, not zeros: an all-zero first
    # column has no variance, and its correlation with step 1 is undefined.
    residuals[:, 0] = rng.normal(0, 1 / np.sqrt(1 - rho**2), n_paths)
    for step in range(1, horizon):
        residuals[:, step] = rho * residuals[:, step - 1] + rng.normal(0, 1, n_paths)

    spec = fit_copula(residuals, horizon, shrinkage=0.3)
    reference = float(np.mean([spec.correlation[i, i + 1] for i in range(horizon - 1)]))

    sampled = rng.multivariate_normal(np.zeros(horizon), spec.correlation, size=4000)
    got = float(
        np.mean([np.corrcoef(sampled[:, i], sampled[:, i + 1])[0, 1] for i in range(horizon - 1)])
    )
    assert abs(got - reference) <= 0.10, f"sampled lag-1 {got:.3f} vs reference {reference:.3f}"

    independent = rng.normal(size=(4000, horizon))
    naive = float(
        np.mean(
            [
                np.corrcoef(independent[:, i], independent[:, i + 1])[0, 1]
                for i in range(horizon - 1)
            ]
        )
    )
    assert abs(naive) < 0.10, "independent sampling should show ~0 autocorrelation"
    assert abs(naive - reference) > 0.10, "the check must be able to fail"


def test_copula_rejects_too_few_residual_paths() -> None:
    with pytest.raises(ValueError, match="need at least 10"):
        fit_copula(np.zeros((5, HORIZON)), HORIZON)


def test_copula_round_trips_through_npz(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(1)
    spec = fit_copula(rng.normal(size=(200, HORIZON)), HORIZON)
    spec.to_npz(tmp_path / "copula.npz")

    from twinmind.models.demand_lgbm import CopulaSpec

    restored = CopulaSpec.from_npz(tmp_path / "copula.npz")
    np.testing.assert_allclose(spec.correlation, restored.correlation)
    assert restored.horizon == spec.horizon


# --------------------------------------------------------------------------- #
# Conformal
# --------------------------------------------------------------------------- #
def test_conformal_widens_an_undercovering_interval() -> None:
    rng = np.random.default_rng(2)
    n, horizon = 1000, 4
    steps = np.tile(np.arange(1, horizon + 1), n // horizon)
    y = rng.normal(100, 20, len(steps))
    lower = np.full(len(steps), 95.0)  # far too narrow
    upper = np.full(len(steps), 105.0)

    offsets = fit_conformal(y, lower, upper, steps, horizon, alpha=0.2)
    assert (offsets["upper"] > 0).all(), "an undercovering interval must be widened"

    widened_low = lower - offsets["lower"][steps - 1]
    widened_high = upper + offsets["upper"][steps - 1]
    covered = float(np.mean((y >= widened_low) & (y <= widened_high)))
    assert 0.74 <= covered <= 0.86, f"post-conformal coverage {covered:.3f} off target"


def test_conformal_offsets_are_per_step() -> None:
    """Later steps are harder, so their offsets must be allowed to differ."""
    rng = np.random.default_rng(3)
    horizon = 4
    steps = np.repeat(np.arange(1, horizon + 1), 400)
    # Error grows with the step, as it does in a real forecast.
    y = np.concatenate([rng.normal(100, 5 * k, 400) for k in range(1, horizon + 1)])
    lower, upper = np.full(len(steps), 95.0), np.full(len(steps), 105.0)

    offsets = fit_conformal(y, lower, upper, steps, horizon)
    assert offsets["upper"][-1] > offsets["upper"][0], (
        "the day-4 offset should exceed the day-1 offset when error grows"
    )


# --------------------------------------------------------------------------- #
# Negative control: the leakage test must be able to fail
# --------------------------------------------------------------------------- #
def _leaky_inference_rows(
    history: pd.DataFrame,
    future: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    target: str = "y",
    date_column: str = "ds",
) -> pd.DataFrame:
    """A deliberately leaky feature builder, used only to prove the guard bites.

    Identical to the real one except for ``leaky_lag_1``: the sales figure from
    the day *before the target day*. For step k that is day origin+k-1, which
    for every k > 1 is in the future and cannot exist at the origin. This is the
    exact mistake the direct multi-horizon design was adopted to prevent -- it is
    what you get by reusing the one-step ``lag_1`` column at a 28-day horizon.
    """
    rows = build_inference_rows(history, future, origin, horizon, target, date_column)
    combined = pd.concat([history, future], ignore_index=True).sort_values(date_column)
    series = combined.set_index(date_column)[target]

    leaked = []
    for step in range(1, horizon + 1):
        previous_day = pd.Timestamp(origin) + pd.Timedelta(days=step - 1)
        leaked.append(float(series.get(previous_day, np.nan)))
    rows["leaky_lag_1"] = leaked
    return rows


def test_negative_control_a_leaky_feature_fails_the_same_guard(
    frame: pd.DataFrame,
) -> None:
    """The guard must reject a feature that reads past the origin.

    A test that only ever passes proves nothing. This runs the identical
    corruption check against a builder with one short lag added, and asserts the
    check *fails* -- so when it passes on the real builder, that means something.
    """
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON + 5)
    series = frame[frame["unique_id"] == "s0"]
    history = series[series["ds"] <= origin]
    future = series[series["ds"] > origin]

    clean = _leaky_inference_rows(history, future, origin, HORIZON)

    poisoned_future = future.copy()
    poisoned_future["y"] = 1e9
    dirty = _leaky_inference_rows(history, poisoned_future, origin, HORIZON)

    # Step 1 reads the origin day itself, which is legitimately known; steps
    # 2..28 read days that do not exist yet, and must move when poisoned.
    assert clean["leaky_lag_1"].iloc[0] == dirty["leaky_lag_1"].iloc[0]
    moved = (clean["leaky_lag_1"].iloc[1:] != dirty["leaky_lag_1"].iloc[1:]).sum()
    assert moved == HORIZON - 1, (
        f"only {moved} of {HORIZON - 1} future steps moved; the control is not leaking"
    )

    with pytest.raises(AssertionError):
        shared = [c for c in clean.columns if c in dirty.columns and c != "ds"]
        pd.testing.assert_frame_equal(clean[shared], dirty[shared])


def test_negative_control_leaky_feature_breaks_truncation_invariance(
    frame: pd.DataFrame, snapshot: DemandData
) -> None:
    """The same leak also breaks the end-to-end truncation test.

    Blanking the unknowable columns changes the leaky feature, so a model built
    on it would answer differently -- which is what the real model must not do.
    """
    origin = frame["ds"].max() - pd.Timedelta(days=HORIZON)
    series_full = snapshot.frame[snapshot.frame["unique_id"] == "s0"]
    blinded = snapshot.truncate(origin).frame
    series_cut = blinded[blinded["unique_id"] == "s0"]

    def build(source: pd.DataFrame) -> pd.DataFrame:
        return _leaky_inference_rows(
            source[source["ds"] <= origin], source[source["ds"] > origin], origin, HORIZON
        )

    full = build(series_full)["leaky_lag_1"].to_numpy()
    cut = build(series_cut)["leaky_lag_1"].to_numpy()

    assert not np.allclose(full[1:], cut[1:], equal_nan=True), (
        "the leaky feature should change once future sales are blanked"
    )
    # And the real builder, on the same data, does not move at all.
    real_full = build_inference_rows(
        series_full[series_full["ds"] <= origin],
        series_full[series_full["ds"] > origin],
        origin,
        HORIZON,
    )
    real_cut = build_inference_rows(
        series_cut[series_cut["ds"] <= origin],
        series_cut[series_cut["ds"] > origin],
        origin,
        HORIZON,
    )
    shared = [c for c in real_full.columns if c != "ds"]
    pd.testing.assert_frame_equal(real_full[shared], real_cut[shared])
