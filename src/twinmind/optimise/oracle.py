"""Expensive near-exhaustive search, used only to define the regret baseline."""

from __future__ import annotations

from collections.abc import Callable

from twinmind.twin.runner import Scenario, TwinResult
from twinmind.utils.config import TwinConfig


def build_oracle(
    config: TwinConfig,
    evaluate: Callable[[Scenario], TwinResult],
    budget: int = 20000,
    seed: int = 42,
) -> TwinResult:
    """Best scenario found under a very large evaluation budget."""
    raise NotImplementedError("build_oracle is not implemented yet")


def regret(result: TwinResult, oracle: TwinResult, objective: str = "profit") -> float:
    """Relative shortfall of ``result`` against the oracle."""
    raise NotImplementedError("regret is not implemented yet")
