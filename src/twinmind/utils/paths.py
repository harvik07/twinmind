"""Filesystem locations, resolved from the package file rather than the cwd.

Every accessor creates its directory on first use so that scripts run from a
Kaggle notebook, a test subdirectory or the repo root all behave identically.
"""

from __future__ import annotations

from pathlib import Path

# paths.py lives at <root>/src/twinmind/utils/paths.py -> four parents up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

CONFIG_DIR: Path = PROJECT_ROOT / "configs"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Root of the (gitignored) data tree."""
    return _ensure(PROJECT_ROOT / "data")


def raw_dir() -> Path:
    """Untouched downloads: C-MAPSS, AI4I 2020, Favorita."""
    return _ensure(data_dir() / "raw")


def interim_dir() -> Path:
    """Partially cleaned intermediates."""
    return _ensure(data_dir() / "interim")


def processed_dir() -> Path:
    """Model-ready tensors/parquet."""
    return _ensure(data_dir() / "processed")


def models_dir() -> Path:
    """Trained checkpoints and scalers."""
    return _ensure(PROJECT_ROOT / "models")


def reports_dir() -> Path:
    """Metrics, logs and generated tables."""
    return _ensure(PROJECT_ROOT / "reports")


def figures_dir() -> Path:
    """Figures for the dissertation."""
    return _ensure(reports_dir() / "figures")


def rollouts_dir() -> Path:
    """Simulator rollouts used to train the Model C surrogate."""
    return _ensure(processed_dir() / "rollouts")
