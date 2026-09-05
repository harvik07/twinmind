"""Coarse grid search, used as a sanity baseline on a few dimensions."""

from __future__ import annotations

from collections.abc import Callable

from twinmind.twin.runner import Scenario, TwinResult
from twinmind.utils.config import TwinConfig


def build_grid(config: TwinConfig, dims: list[str], points_per_dim: int) -> list[Scenario]:
    """Cartesian product over ``dims``, holding every other dimension at its default."""
    raise NotImplementedError("build_grid is not implemented yet")


def grid_search(
    config: TwinConfig,
    evaluate: Callable[[Scenario], TwinResult],
    dims: list[str],
    points_per_dim: int = 5,
) -> list[TwinResult]:
    """Evaluate every point on the grid."""
    raise NotImplementedError("grid_search is not implemented yet")
