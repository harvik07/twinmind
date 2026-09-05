"""Model A: CNN-BiLSTM remaining-useful-life estimator with a failure head."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from twinmind.utils.config import HealthModelConfig


class CNNBiLSTMHealth(nn.Module):
    """Conv feature extractor -> BiLSTM -> RUL head + failure-probability head."""

    def __init__(self, config: HealthModelConfig, n_features: int) -> None:
        super().__init__()
        self.config = config
        self.n_features = n_features
        raise NotImplementedError("CNNBiLSTMHealth.__init__ is not implemented yet")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rul, failure_logit)`` for a batch of shape (B, T, F)."""
        raise NotImplementedError("CNNBiLSTMHealth.forward is not implemented yet")


def mc_dropout_predict(
    model: CNNBiLSTMHealth, x: torch.Tensor, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """MC-dropout predictive mean and standard deviation."""
    raise NotImplementedError("mc_dropout_predict is not implemented yet")


def failure_hazard(rul: np.ndarray, failure_probability: np.ndarray) -> np.ndarray:
    """Convert model outputs into the per-hour hazard the twin consumes."""
    raise NotImplementedError("failure_hazard is not implemented yet")
