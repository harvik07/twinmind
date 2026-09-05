"""Model C: heteroscedastic residual MLP emulating the digital twin."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from twinmind.twin.runner import KPIWithCI, Scenario
from twinmind.utils.config import SurrogateModelConfig


class HeteroscedasticMLP(nn.Module):
    """Predicts a mean and a log-variance per KPI from an 18-D scenario vector."""

    def __init__(self, config: SurrogateModelConfig) -> None:
        super().__init__()
        self.config = config
        raise NotImplementedError("HeteroscedasticMLP.__init__ is not implemented yet")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, log_variance)`` for a batch of shape (B, 18)."""
        raise NotImplementedError("HeteroscedasticMLP.forward is not implemented yet")


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood."""
    raise NotImplementedError("gaussian_nll is not implemented yet")


class Surrogate:
    """Inference wrapper: screens many scenarios far faster than the twin."""

    def __init__(self, model: HeteroscedasticMLP, scaler: dict[str, np.ndarray]) -> None:
        self.model = model
        self.scaler = scaler

    def predict(self, scenarios: list[Scenario]) -> list[KPIWithCI]:
        """Predict KPIs with uncertainty for a batch of scenarios."""
        raise NotImplementedError("Surrogate.predict is not implemented yet")
