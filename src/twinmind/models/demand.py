"""Model B: N-HiTS / PatchTST multi-quantile demand forecasting.

Requires the optional ``forecast`` extra (``pip install -e ".[forecast]"``);
neuralforecast is imported lazily so the rest of the package stays CPU-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from twinmind.utils.config import DemandModelConfig


def build_models(config: DemandModelConfig) -> list[Any]:
    """Instantiate the configured neuralforecast models with a multi-quantile loss."""
    raise NotImplementedError("build_models is not implemented yet")


def fit(config: DemandModelConfig, frame: pd.DataFrame) -> Any:
    """Fit a NeuralForecast object on a long-format frame."""
    raise NotImplementedError("fit is not implemented yet")


def predict_quantiles(model: Any, horizon: int) -> pd.DataFrame:
    """Forecast q10/q50/q90 for each series over ``horizon`` steps."""
    raise NotImplementedError("predict_quantiles is not implemented yet")


def save(model: Any, path: Path) -> None:
    """Persist a fitted forecaster."""
    raise NotImplementedError("save is not implemented yet")


def load(path: Path) -> Any:
    """Load a forecaster saved by :func:`save`."""
    raise NotImplementedError("load is not implemented yet")
