"""Deterministic tools the graph nodes call. None of these consult an LLM."""

from __future__ import annotations

from twinmind.twin.runner import KPIWithCI, Scenario, TwinResult


def evaluate_on_twin(scenario: Scenario, n_reps: int) -> TwinResult:
    """Run the real SimPy twin for one scenario."""
    raise NotImplementedError("evaluate_on_twin is not implemented yet")


def screen_with_surrogate(scenarios: list[Scenario], top_k: int, objective: str) -> list[Scenario]:
    """Rank scenarios with Model C and return the best ``top_k``."""
    raise NotImplementedError("screen_with_surrogate is not implemented yet")


def compare(baseline: KPIWithCI, candidate: KPIWithCI, objective: str) -> dict[str, float]:
    """Absolute and relative deltas plus a CI-overlap flag."""
    raise NotImplementedError("compare is not implemented yet")


def clip_to_bounds(scenario: Scenario) -> Scenario:
    """Project an LLM-proposed scenario back into the configured bounds."""
    raise NotImplementedError("clip_to_bounds is not implemented yet")
