"""Optuna TPE and CMA-ES baselines the agent is benchmarked against."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from twinmind.twin.runner import Scenario, TwinResult
from twinmind.utils.config import TwinConfig


def suggest_scenario(trial: Any, config: TwinConfig) -> Scenario:
    """Map an Optuna trial onto a Scenario using the configured bounds."""
    raise NotImplementedError("suggest_scenario is not implemented yet")


def optimise(
    config: TwinConfig,
    evaluate: Callable[[Scenario], TwinResult],
    n_trials: int,
    sampler: Literal["tpe", "cmaes"] = "tpe",
    objective: str = "profit",
    seed: int = 42,
) -> list[TwinResult]:
    """Run an Optuna study and return every trial's result."""
    raise NotImplementedError("optimise is not implemented yet")
