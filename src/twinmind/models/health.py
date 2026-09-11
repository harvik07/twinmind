"""Model A1 -- machine health from C-MAPSS sensor windows.

Architecture note
-----------------
A1 and A2 are **separate models**, not two heads on one trunk:

* **A1** (this module) is the C-MAPSS model. One CNN-BiLSTM trunk feeds *two*
  heads -- a regression head for remaining useful life and a binary head for
  "will fail within ``h`` cycles". These genuinely share a trunk because they
  share inputs and the second label is derived from the first.
* **A2** (:mod:`twinmind.models.ai4i`) is a small tabular classifier on AI4I
  2020. Different feature space, different domain, separate model. It exists for
  the class-imbalance and calibration story and as an independent check.

A trunk cannot be shared across the two: C-MAPSS is a 30-cycle window of turbofan
sensors, AI4I is a single row of milling-machine measurements. There is no common
input space to share.

The twin consumes A1 through :class:`HealthModel` and nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class RULPrediction(BaseModel):
    """One engine's health estimate. The twin's failure hazard is built from this.

    ``rul_mean`` and ``rul_std`` are in cycles and reported on the *capped* scale
    the model was trained on, so ``rul_mean`` saturates at the configured cap
    (125 by default) for a healthy engine. ``rul_std`` is the MC-dropout
    predictive standard deviation, not a fitted variance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rul_mean: float = Field(ge=0.0, description="Predicted remaining useful life, cycles.")
    rul_std: float = Field(ge=0.0, description="MC-dropout predictive std, cycles.")
    p_fail_within_h: float = Field(
        ge=0.0, le=1.0, description="Probability of failure within horizon_h cycles."
    )
    horizon_h: int = Field(gt=0, description="Failure horizon in cycles.")


@runtime_checkable
class HealthModel(Protocol):
    """The contract the digital twin depends on.

    Deliberately minimal: the twin asks for a health estimate given a window of
    sensor readings and gets back a validated :class:`RULPrediction`. Nothing
    about checkpoints, devices, normalisers or torch appears here, so the twin
    can be driven by the trained network, by a stub, or by a replay of recorded
    predictions without changing a line of simulator code.
    """

    def predict(self, window: NDArray[np.floating]) -> RULPrediction:
        """Predict from a single window of shape ``[T, C]``."""
        ...

    def predict_batch(self, windows: NDArray[np.floating]) -> list[RULPrediction]:
        """Predict from a batch of shape ``[B, T, C]``."""
        ...


class ConstantHealthModel:
    """A fixed-output :class:`HealthModel`, for tests and twin bring-up.

    Satisfies the Protocol exactly, so the simulator can be built and exercised
    before A1 is trained. It is not a baseline and must never appear in results.
    """

    def __init__(
        self,
        rul_mean: float = 125.0,
        rul_std: float = 0.0,
        p_fail_within_h: float = 0.0,
        horizon_h: int = 30,
    ) -> None:
        self._prediction = RULPrediction(
            rul_mean=rul_mean,
            rul_std=rul_std,
            p_fail_within_h=p_fail_within_h,
            horizon_h=horizon_h,
        )

    def predict(self, window: NDArray[np.floating]) -> RULPrediction:
        """Return the fixed prediction, after validating the window's rank."""
        array = np.asarray(window)
        if array.ndim != 2:
            raise ValueError(f"expected a window of shape [T, C], got {array.shape}")
        return self._prediction

    def predict_batch(self, windows: NDArray[np.floating]) -> list[RULPrediction]:
        """Return one fixed prediction per row of the batch."""
        array = np.asarray(windows)
        if array.ndim != 3:
            raise ValueError(f"expected a batch of shape [B, T, C], got {array.shape}")
        return [self._prediction] * array.shape[0]
