"""The Prompt 9 agent-benchmark goals: natural language plus a gold spec.

Loaded from ``tests/data/agent_bench_goals.yaml``. Each gold spec is validated
through :class:`ObjectiveSpec` with its context id, so a benchmark arm runs on
exactly the spec the label says -- optimisation isolated from parsing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from twinmind.optim.contracts import CONTROLLABLE_DIMS, ObjectiveSpec
from twinmind.utils.paths import PROJECT_ROOT

BENCH_GOALS = PROJECT_ROOT / "tests" / "data" / "agent_bench_goals.yaml"
CATEGORIES = ("feasible", "infeasible", "no_room", "ambiguous_answerable")
#: Contexts reserved for the benchmark: unused by any dev, core or reserve goal.
BENCH_CONTEXTS = tuple(range(8, 20))


@dataclass(frozen=True)
class BenchGoal:
    """One benchmark goal."""

    id: str
    category: str
    context_id: int
    text: str
    gold: dict[str, Any]

    @property
    def spec(self) -> ObjectiveSpec:
        """The gold spec, validated, on this goal's context."""
        payload = dict(self.gold)
        fixed = dict(payload.get("fixed") or {})
        if payload.get("free_dims") is None:
            payload["free_dims"] = [d for d in CONTROLLABLE_DIMS if d not in fixed]
        payload["fixed"] = fixed
        payload["context_id"] = self.context_id
        payload["infeasible_by_design"] = self.category == "infeasible"
        return ObjectiveSpec.from_dict(payload)


def load_bench_goals(path: Path = BENCH_GOALS) -> list[BenchGoal]:
    """All benchmark goals, in file order."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    goals = [BenchGoal(**g) for g in raw["goals"]]
    for goal in goals:
        if goal.category not in CATEGORIES:
            raise ValueError(f"{goal.id}: unknown category {goal.category!r}")
        if goal.context_id not in BENCH_CONTEXTS:
            raise ValueError(f"{goal.id}: context {goal.context_id} is not a benchmark context")
    return goals


def bench_goals_hash(path: Path = BENCH_GOALS) -> str:
    """Hash of the goal file, for the manifest and the pre-registration."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
