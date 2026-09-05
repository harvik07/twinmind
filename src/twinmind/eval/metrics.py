"""Evaluation metrics for every component."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    raise NotImplementedError("rmse is not implemented yet")


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The asymmetric NASA scoring function (late predictions penalised harder)."""
    raise NotImplementedError("cmapss_score is not implemented yet")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss for the demand model."""
    raise NotImplementedError("pinball_loss is not implemented yet")


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage of a prediction interval."""
    raise NotImplementedError("coverage is not implemented yet")


def surrogate_fidelity(twin_kpis: np.ndarray, surrogate_kpis: np.ndarray) -> dict[str, float]:
    """Per-KPI agreement between the surrogate and the real twin."""
    raise NotImplementedError("surrogate_fidelity is not implemented yet")


def calls_per_query(call_log_path: str) -> dict[str, float]:
    """Summarise LLM calls and tokens per query from the JSONL call log."""
    raise NotImplementedError("calls_per_query is not implemented yet")
