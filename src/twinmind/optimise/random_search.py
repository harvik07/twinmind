"""Uniform random search over the scenario space."""

from __future__ import annotations

from collections.abc import Callable

from twinmind.twin.runner import Scenario, TwinResult
from twinmind.utils.config import TwinConfig


def sample_scenario(config: TwinConfig, seed: int) -> Scenario:
    """Draw one scenario uniformly from the configured bounds."""
    raise NotImplementedError("sample_scenario is not implemented yet")


def random_search(
    config: TwinConfig,
    evaluate: Callable[[Scenario], TwinResult],
    n_trials: int,
    seed: int = 42,
) -> list[TwinResult]:
    """Evaluate ``n_trials`` random scenarios."""
    raise NotImplementedError("random_search is not implemented yet")
