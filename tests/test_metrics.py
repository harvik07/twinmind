"""Evaluation metrics, and the C-MAPSS scoring protocol.

The protocol tests matter most. Averaging over every test window instead of
scoring one prediction per engine is the standard way published C-MAPSS numbers
get inflated, and it does not raise -- it just returns a better score.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from twinmind.eval.metrics import (
    NASA_EARLY_SCALE,
    NASA_LATE_SCALE,
    best_threshold_fbeta,
    brier_score,
    coverage,
    evaluate_cmapss_test,
    expected_calibration_error,
    format_mean_std,
    last_window_indices,
    mae,
    nasa_score,
    pinball_loss,
    pr_auc,
    recall_at_precision,
    reliability_curve,
    rmse,
    roc_auc,
    summarise,
    threshold_at_precision,
)


# --------------------------------------------------------------------------- #
# Regression basics
# --------------------------------------------------------------------------- #
def test_rmse_and_mae_hand_computed() -> None:
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    # errors +2, -2, +3 -> mean|e| = 7/3, rmse = sqrt(17/3)
    assert mae(y_true, y_pred) == pytest.approx(7 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(17 / 3))


def test_perfect_prediction_scores_zero() -> None:
    y = [1.0, 5.0, 9.0]
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert nasa_score(y, y) == 0.0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse([1.0, 2.0], [1.0])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        rmse([], [])


# --------------------------------------------------------------------------- #
# NASA score -- hand-computed at d = -10, 0, +10
# --------------------------------------------------------------------------- #
def test_nasa_score_at_zero_error() -> None:
    assert nasa_score([100.0], [100.0]) == pytest.approx(0.0)


def test_nasa_score_early_prediction_hand_computed() -> None:
    """d = -10: predicted 90 against a true 100, so exp(10/13) - 1."""
    expected = math.exp(10 / NASA_EARLY_SCALE) - 1
    assert nasa_score([100.0], [90.0]) == pytest.approx(expected)
    assert nasa_score([100.0], [90.0]) == pytest.approx(1.158106, abs=1e-6)


def test_nasa_score_late_prediction_hand_computed() -> None:
    """d = +10: predicted 110 against a true 100, so exp(10/10) - 1 = e - 1."""
    expected = math.exp(10 / NASA_LATE_SCALE) - 1
    assert nasa_score([100.0], [110.0]) == pytest.approx(expected)
    assert nasa_score([100.0], [110.0]) == pytest.approx(math.e - 1)
    assert nasa_score([100.0], [110.0]) == pytest.approx(1.718282, abs=1e-6)


def test_late_predictions_are_punished_harder_than_early_ones() -> None:
    """The whole point of the metric: an overestimate schedules maintenance too late."""
    late = nasa_score([100.0], [110.0])
    early = nasa_score([100.0], [90.0])
    assert late > early, "a late prediction must cost more than an early one of equal size"
    assert late / early == pytest.approx(1.4837, abs=1e-3)


@pytest.mark.parametrize("magnitude", [1.0, 5.0, 20.0, 50.0])
def test_asymmetry_holds_at_every_magnitude(magnitude: float) -> None:
    assert nasa_score([100.0], [100.0 + magnitude]) > nasa_score([100.0], [100.0 - magnitude])


def test_nasa_score_sums_over_engines() -> None:
    """It is a sum, not a mean -- more engines means a larger score."""
    one = nasa_score([100.0], [110.0])
    three = nasa_score([100.0] * 3, [110.0] * 3)
    assert three == pytest.approx(3 * one)


def test_nasa_score_is_dominated_by_its_worst_prediction() -> None:
    """Justifies always reporting it beside RMSE rather than instead of it."""
    mild = nasa_score([100.0] * 10, [105.0] * 10)
    with_outlier = nasa_score([100.0] * 10, [105.0] * 9 + [200.0])
    assert with_outlier > 100 * mild


# --------------------------------------------------------------------------- #
# The C-MAPSS test protocol
# --------------------------------------------------------------------------- #
def test_last_window_indices_picks_the_final_row_per_unit() -> None:
    units = [1, 1, 1, 2, 2, 3]
    np.testing.assert_array_equal(last_window_indices(units), [2, 4, 5])


def test_last_window_indices_preserves_first_appearance_order() -> None:
    units = [7, 7, 3, 3, 3, 5]
    np.testing.assert_array_equal(last_window_indices(units), [1, 4, 5])


def test_evaluate_cmapss_test_scores_one_prediction_per_engine() -> None:
    # Two engines, three windows each; only the last window of each is scored.
    units = [1, 1, 1, 2, 2, 2]
    y_true = [125.0, 60.0, 10.0, 125.0, 80.0, 20.0]
    y_pred = [125.0, 60.0, 15.0, 125.0, 80.0, 25.0]

    result = evaluate_cmapss_test(y_true, y_pred, units)

    assert result["n_units"] == 2
    assert result["n_windows_seen"] == 6
    # Only the final windows count: errors of +5 and +5.
    assert result["rmse"] == pytest.approx(5.0)
    assert result["mae"] == pytest.approx(5.0)
    assert result["nasa_score"] == pytest.approx(2 * (math.exp(0.5) - 1))


def test_protocol_is_not_the_average_over_all_windows() -> None:
    """The inflation this guards against, demonstrated numerically.

    Early windows sit at the cap where any model is nearly exact. Averaging over
    them hides the near-failure error the benchmark exists to measure.
    """
    units = [1] * 10
    y_true = [125.0] * 9 + [10.0]
    y_pred = [125.0] * 9 + [40.0]  # only the final window is wrong, badly

    protocol = evaluate_cmapss_test(y_true, y_pred, units)
    naive = rmse(y_true, y_pred)

    assert protocol["rmse"] == pytest.approx(30.0), "the protocol sees the real error"
    assert naive == pytest.approx(math.sqrt(900 / 10))
    assert naive < protocol["rmse"] / 3, (
        "averaging over all windows should look far better than it is; "
        f"naive {naive:.2f} vs protocol {protocol['rmse']:.2f}"
    )


def test_protocol_and_naive_agree_when_there_is_one_window_per_unit() -> None:
    units = [1, 2, 3]
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]

    result = evaluate_cmapss_test(y_true, y_pred, units)
    assert result["rmse"] == pytest.approx(rmse(y_true, y_pred))
    assert result["n_units"] == result["n_windows_seen"] == 3


def test_evaluate_rejects_mismatched_unit_ids() -> None:
    with pytest.raises(ValueError, match="does not match"):
        evaluate_cmapss_test([1.0, 2.0], [1.0, 2.0], [1])


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@pytest.fixture()
def imbalanced() -> tuple[list[int], list[float]]:
    """A 10% positive class with a usefully but imperfectly ranked score."""
    rng = np.random.default_rng(0)
    y = [1] * 20 + [0] * 180
    scores = list(rng.normal(0.7, 0.2, 20)) + list(rng.normal(0.3, 0.2, 180))
    return y, [float(np.clip(s, 0, 1)) for s in scores]


def test_pr_auc_and_roc_auc_beat_chance(imbalanced: tuple[list[int], list[float]]) -> None:
    y, scores = imbalanced
    assert pr_auc(y, scores) > 0.10, "must beat the 10% positive base rate"
    assert roc_auc(y, scores) > 0.5


def test_perfect_ranking_scores_one() -> None:
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert pr_auc(y, scores) == pytest.approx(1.0)
    assert roc_auc(y, scores) == pytest.approx(1.0)


def test_recall_at_precision_on_a_known_case() -> None:
    """Perfect separation: full recall is available at precision 1.0."""
    y = [0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.9, 0.95]
    assert recall_at_precision(y, scores, 0.80) == pytest.approx(1.0)


def test_recall_at_precision_returns_zero_when_unreachable() -> None:
    y = [0] * 99 + [1]
    scores = [0.5] * 100  # no threshold separates anything
    assert recall_at_precision(y, scores, 0.99) == 0.0


def test_recall_at_precision_rejects_a_bad_target() -> None:
    with pytest.raises(ValueError, match="target_precision"):
        recall_at_precision([0, 1], [0.1, 0.9], 0.0)


def test_threshold_at_precision_is_usable(imbalanced: tuple[list[int], list[float]]) -> None:
    y, scores = imbalanced
    threshold = threshold_at_precision(y, scores, 0.80)
    predicted = [1 if s >= threshold else 0 for s in scores]
    true_positive = sum(a and b for a, b in zip(y, predicted, strict=True))
    flagged = sum(predicted)
    if flagged:
        assert true_positive / flagged >= 0.79


def test_fbeta_threshold_favours_recall() -> None:
    """beta=2 should not pick a threshold so high it misses most positives."""
    y = [0] * 90 + [1] * 10
    rng = np.random.default_rng(1)
    scores = list(rng.uniform(0, 0.6, 90)) + list(rng.uniform(0.4, 1.0, 10))

    threshold, score = best_threshold_fbeta(y, scores, beta=2.0)
    assert 0.0 <= threshold <= 1.0
    assert score > 0.0
    caught = sum(1 for s, label in zip(scores, y, strict=True) if label == 1 and s >= threshold)
    assert caught >= 7, "an F2-optimal threshold should catch most positives"


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_brier_score_hand_computed() -> None:
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_perfectly_calibrated_probabilities_have_near_zero_ece() -> None:
    rng = np.random.default_rng(2)
    prob = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < prob).astype(int)
    assert expected_calibration_error(y, prob, n_bins=10) < 0.02


def test_systematically_overconfident_probabilities_have_large_ece() -> None:
    y = [0] * 100
    prob = [0.9] * 100  # claims 90% confident, is right 0% of the time
    assert expected_calibration_error(y, prob, n_bins=10) == pytest.approx(0.9)


def test_ece_counts_probability_one() -> None:
    """The last bin must be closed on the right or p=1 is silently dropped."""
    assert expected_calibration_error([0, 0], [1.0, 1.0], n_bins=10) == pytest.approx(1.0)


def test_ece_rejects_zero_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0, 1], [0.1, 0.9], n_bins=0)


def test_reliability_curve_shapes_agree() -> None:
    rng = np.random.default_rng(3)
    prob = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < prob).astype(int)

    predicted, observed, weights = reliability_curve(y, prob, n_bins=10)
    assert predicted.shape == observed.shape == weights.shape
    assert weights.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Forecasting helpers
# --------------------------------------------------------------------------- #
def test_pinball_loss_is_asymmetric_by_quantile() -> None:
    # At q=0.9 under-forecasting should hurt more than over-forecasting.
    under = pinball_loss([100.0], [90.0], 0.9)
    over = pinball_loss([100.0], [110.0], 0.9)
    assert under > over
    assert under == pytest.approx(0.9 * 10)
    assert over == pytest.approx(0.1 * 10)


def test_pinball_at_median_is_half_the_absolute_error() -> None:
    assert pinball_loss([100.0], [90.0], 0.5) == pytest.approx(5.0)


def test_pinball_rejects_an_out_of_range_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss([1.0], [1.0], 1.0)


def test_coverage_counts_inclusive_bounds() -> None:
    assert coverage([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]) == pytest.approx(2 / 3)
    assert coverage([1.0], [1.0], [1.0]) == 1.0


# --------------------------------------------------------------------------- #
# Multi-seed reporting
# --------------------------------------------------------------------------- #
def test_summarise_reports_mean_std_and_n() -> None:
    stats = summarise([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(math.sqrt(2 / 3))
    assert stats["n"] == 3


def test_summarise_of_one_seed_has_zero_std_not_nan() -> None:
    stats = summarise([5.0])
    assert stats["std"] == 0.0
    assert not math.isnan(stats["std"])


def test_format_mean_std_renders_for_a_table() -> None:
    assert format_mean_std([10.0, 12.0], places=2) == "11.00 +/- 1.00"


def test_summarise_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarise([])
