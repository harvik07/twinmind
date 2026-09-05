"""Dataset loaders for C-MAPSS, AI4I 2020 and Favorita."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from twinmind.utils.config import AI4IConfig, CMAPSSConfig, FavoritaConfig


def load_cmapss(config: CMAPSSConfig, subset: str, split: str = "train") -> pd.DataFrame:
    """Load one C-MAPSS subset (FD001-FD004) as a tidy frame."""
    raise NotImplementedError("load_cmapss is not implemented yet")


def load_ai4i(config: AI4IConfig, path: Path | None = None) -> pd.DataFrame:
    """Load the UCI AI4I 2020 predictive-maintenance table."""
    raise NotImplementedError("load_ai4i is not implemented yet")


def load_favorita(config: FavoritaConfig, path: Path | None = None) -> pd.DataFrame:
    """Load Favorita store sales in Nixtla long format (unique_id, ds, y)."""
    raise NotImplementedError("load_favorita is not implemented yet")
