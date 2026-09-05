"""Sliding-window construction for the sequence models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    group_column: str,
    window_length: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(windows, last_index)`` arrays, windowed within each group."""
    raise NotImplementedError("make_windows is not implemented yet")


def compute_rul(frame: pd.DataFrame, group_column: str, cap: int) -> pd.Series:
    """Piecewise-linear remaining useful life, clipped at ``cap``."""
    raise NotImplementedError("compute_rul is not implemented yet")


def train_val_split_by_group(
    frame: pd.DataFrame, group_column: str, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unit id so no engine appears in both folds."""
    raise NotImplementedError("train_val_split_by_group is not implemented yet")
