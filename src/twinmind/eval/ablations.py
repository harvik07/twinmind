"""Ablation harness: which parts of the system actually earn their place."""

from __future__ import annotations

from typing import Any, Literal

Ablation = Literal[
    "full",
    "no_surrogate",
    "no_critic",
    "no_memory",
    "random_explorer",
    "fixed_mtbf",
]


def run_ablation(name: Ablation, queries: list[str], **deps: Any) -> dict[str, Any]:
    """Run one ablation across a query set and return its aggregate metrics."""
    raise NotImplementedError("run_ablation is not implemented yet")


def compare_ablations(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the ablation comparison table."""
    raise NotImplementedError("compare_ablations is not implemented yet")
