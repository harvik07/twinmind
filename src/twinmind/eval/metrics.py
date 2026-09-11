"""Evaluation metrics for every component.

The C-MAPSS section carries a protocol rule that matters more than any metric
definition here: see :func:`evaluate_cmapss_test`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Denominators of the NASA asymmetric score. Late predictions decay slower, so
#: they cost more -- see :func:`nasa_score`.
NASA_EARLY_SCALE = 13.0
NASA_LATE_SCALE = 10.0


def _as_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype="float64").ravel()
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true, pred = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    return true, pred


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    true, pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    true, pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """The standard C-MAPSS asymmetric penalty. Lower is better.

    With ``d = y_pred - y_true``::

        s = sum( exp(-d / 13) - 1   where d <  0    (early prediction)
                 exp( d / 10) - 1   where d >= 0    (late prediction) )

    The asymmetry is the point. A *late* prediction overestimates remaining life,
    so maintenance is scheduled after the engine has already failed; an *early*
    prediction merely wastes some remaining life. The smaller denominator on the
    late branch makes that error grow faster, so at equal magnitude a late
    prediction always costs more -- ``nasa_score(+10) > nasa_score(-10)``.

    Unbounded above and dominated by its worst few predictions, which is why it is
    always reported next to RMSE rather than instead of it.
    """
    true, pred = _pair(y_true, y_pred)
    d = pred - true
    penalties = np.where(
        d < 0, np.exp(-d / NASA_EARLY_SCALE) - 1.0, np.exp(d / NASA_LATE_SCALE) - 1.0
    )
    return float(np.sum(penalties))


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision).

    The right headline metric for a rare positive class: unlike ROC-AUC it does
    not flatter a model that is mostly correct about the abundant negatives.
    """
    from sklearn.metrics import average_precision_score

    true, score = _pair(y_true, y_score)
    return float(average_precision_score(true, score))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve."""
    from sklearn.metrics import roc_auc_score

    true, score = _pair(y_true, y_score)
    return float(roc_auc_score(true, score))


def recall_at_precision(
    y_true: ArrayLike, y_score: ArrayLike, target_precision: float = 0.80
) -> float:
    """Best recall achievable at or above ``target_precision``.

    Answers the operational question directly: if we accept at most
    ``1 - target_precision`` false alarms among our alerts, what fraction of real
    failures do we catch? Returns 0.0 when no threshold reaches that precision.
    """
    from sklearn.metrics import precision_recall_curve

    if not 0.0 < target_precision <= 1.0:
        raise ValueError(f"target_precision must be in (0, 1], got {target_precision}")
    true, score = _pair(y_true, y_score)
    precision, recall, _ = precision_recall_curve(true, score)
    feasible = precision >= target_precision
    if not feasible.any():
        return 0.0
    return float(np.max(recall[feasible]))


def threshold_at_precision(
    y_true: ArrayLike, y_score: ArrayLike, target_precision: float = 0.80
) -> float:
    """The score threshold achieving ``target_precision`` at the best recall."""
    from sklearn.metrics import precision_recall_curve

    true, score = _pair(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(true, score)
    # precision/recall have one more element than thresholds.
    feasible = np.where(precision[:-1] >= target_precision)[0]
    if feasible.size == 0:
        return 1.0
    best = feasible[np.argmax(recall[:-1][feasible])]
    return float(thresholds[best])


def best_threshold_fbeta(
    y_true: ArrayLike, y_score: ArrayLike, beta: float = 2.0
) -> tuple[float, float]:
    """Threshold maximising F-beta, and the score it achieves.

    ``beta=2`` weights recall four times as heavily as precision, which is the
    right trade for maintenance: a false alarm costs an unnecessary inspection, a
    miss costs an unplanned failure at 5x the price.
    """
    from sklearn.metrics import precision_recall_curve

    true, score = _pair(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(true, score)
    p, r = precision[:-1], recall[:-1]
    denominator = (beta**2 * p) + r
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denominator > 0, (1 + beta**2) * p * r / denominator, 0.0)
    if fbeta.size == 0:
        return 0.5, 0.0
    best = int(np.argmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error of predicted probabilities. Lower is better."""
    true, prob = _pair(y_true, y_prob)
    return float(np.mean((prob - true) ** 2))


def expected_calibration_error(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10) -> float:
    """Expected calibration error over ``n_bins`` equal-width probability bins.

    The weighted mean gap between confidence and accuracy. A model with ECE 0.20
    that claims 90% confidence is right about 70% of the time -- which matters
    here because the twin turns this probability into a failure hazard, so a
    miscalibrated head biases every downstream KPI.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    true, prob = _pair(y_true, y_prob)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Left-closed bins, with the last one closed on the right so p=1 counts.
        in_bin = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if not in_bin.any():
            continue
        weight = float(in_bin.mean())
        error += weight * abs(float(true[in_bin].mean()) - float(prob[in_bin].mean()))
    return float(error)


def reliability_curve(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(mean_predicted, observed_frequency, bin_weight)`` for a reliability diagram."""
    true, prob = _pair(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    predicted, observed, weights = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if not in_bin.any():
            continue
        predicted.append(float(prob[in_bin].mean()))
        observed.append(float(true[in_bin].mean()))
        weights.append(float(in_bin.mean()))
    return np.array(predicted), np.array(observed), np.array(weights)


# --------------------------------------------------------------------------- #
# The C-MAPSS test protocol
# --------------------------------------------------------------------------- #
def last_window_indices(unit_ids: ArrayLike) -> NDArray[np.int64]:
    """Index of the final row for each unit, preserving first-appearance order.

    Assumes rows are already ordered by cycle within a unit, which every loader
    and windowing function in this package guarantees.
    """
    units = np.asarray(unit_ids).ravel()
    if units.size == 0:
        raise ValueError("unit_ids is empty")
    order, last = [], {}
    for position, unit in enumerate(units):
        key = unit.item() if hasattr(unit, "item") else unit
        if key not in last:
            order.append(key)
        last[key] = position
    return np.array([last[key] for key in order], dtype="int64")


def evaluate_cmapss_test(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    unit_ids: ArrayLike,
) -> dict[str, float]:
    """Score a C-MAPSS test set under the benchmark protocol.

    **The protocol, and why it is not optional.** A C-MAPSS test set is scored
    with exactly *one* prediction per engine: the RUL at its final observed
    cycle, compared against ``RUL_FD00X.txt``. Windowing a test trajectory
    produces many windows per engine, and averaging the error over all of them
    is not the benchmark -- it is a different, easier task.

    It is easier for a specific reason. Most windows of a test engine sit early
    in its trajectory, where the capped target is pinned at 125 and any model
    predicting "healthy" is nearly exact. Averaging over them drowns the handful
    of near-failure windows that the metric is supposed to be about, so RMSE
    falls and the NASA score falls further. Numbers produced that way are not
    comparable with any published result, and the gap is large enough to look
    like a state-of-the-art claim.

    This function therefore reduces to the last window per unit before scoring,
    and reports how many rows it discarded so the reduction is visible in logs.

    Args:
        y_true: per-window ground-truth RUL.
        y_pred: per-window predicted RUL.
        unit_ids: engine id for each window, ordered by cycle within an engine.

    Returns:
        ``rmse``, ``mae``, ``nasa_score``, ``n_units`` and ``n_windows_seen``.
    """
    true, pred = _pair(y_true, y_pred)
    units = np.asarray(unit_ids).ravel()
    if units.shape != true.shape:
        raise ValueError(f"unit_ids {units.shape} does not match predictions {true.shape}")

    indices = last_window_indices(units)
    last_true, last_pred = true[indices], pred[indices]

    if indices.size < true.size:
        logger.info(
            "C-MAPSS protocol: scoring %d units from %d windows (one prediction per engine)",
            indices.size,
            true.size,
        )

    return {
        "rmse": rmse(last_true, last_pred),
        "mae": mae(last_true, last_pred),
        "nasa_score": nasa_score(last_true, last_pred),
        "n_units": float(indices.size),
        "n_windows_seen": float(true.size),
    }


# --------------------------------------------------------------------------- #
# Other components
# --------------------------------------------------------------------------- #
def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    """Quantile (pinball) loss for the demand model."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    true, pred = _pair(y_true, y_pred)
    delta = true - pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Empirical coverage of a prediction interval."""
    true = _as_1d(y_true, "y_true")
    lo, hi = _as_1d(lower, "lower"), _as_1d(upper, "upper")
    if not (true.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must have the same shape")
    return float(np.mean((true >= lo) & (true <= hi)))


def surrogate_fidelity(twin_kpis: ArrayLike, surrogate_kpis: ArrayLike) -> dict[str, float]:
    """Per-KPI agreement between the surrogate and the real twin."""
    raise NotImplementedError("surrogate_fidelity is not implemented yet")


def calls_per_query(call_log_path: str) -> dict[str, float]:
    """Summarise LLM calls and tokens per query from the JSONL call log."""
    raise NotImplementedError("calls_per_query is not implemented yet")


def summarise(values: list[float]) -> dict[str, float]:
    """``mean``, ``std`` and ``n`` for a list of per-seed results.

    Population std with ``ddof=0``, and 0.0 for a single seed rather than NaN, so
    a one-seed run still renders in a table.
    """
    if not values:
        raise ValueError("cannot summarise an empty list")
    array = np.asarray(values, dtype="float64")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "n": float(array.size),
    }


def format_mean_std(values: list[float], places: int = 3) -> str:
    """``"12.345 +/- 0.678"``, for result tables."""
    stats = summarise(values)
    return f"{stats['mean']:.{places}f} +/- {stats['std']:.{places}f}"
