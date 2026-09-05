"""Feature engineering and scaling shared by the models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_constant_sensors(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return the subset of ``columns`` with non-zero variance."""
    raise NotImplementedError("drop_constant_sensors is not implemented yet")


def fit_scaler(array: np.ndarray, kind: str = "minmax") -> dict[str, np.ndarray]:
    """Fit a scaler and return its parameters as a serialisable dict."""
    raise NotImplementedError("fit_scaler is not implemented yet")


def apply_scaler(array: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Apply a scaler fitted by ``fit_scaler``."""
    raise NotImplementedError("apply_scaler is not implemented yet")


def add_calendar_features(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Add day-of-week, month and holiday flags for the demand model."""
    raise NotImplementedError("add_calendar_features is not implemented yet")
