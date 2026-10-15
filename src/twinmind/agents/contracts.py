"""The agent's output contract: one :class:`AgentResult` per goal.

The Prompt 9 dashboard reads these, so every field is plain JSON (dicts,
lists, numbers, strings) and the round trip ``AgentResult.from_dict(r.to_dict())
== r`` is exact. Nested structure is kept as dicts on purpose: a dashboard can
read a dict without importing this package.

STATUS PRECEDENCE, highest first:

``spec_invalid``        the planner refused, or no valid spec after repairs.
``needs_clarification`` the goal was ambiguous; ``question`` holds ONE question.
``infeasible``          no plan was point-feasible on the independent re-evaluation.
``degraded_template``   a plan exists but the LLM explanation failed the numeric
                        grounding check twice; a deterministic template was used.
``ok``                  everything else.

A plan that does not beat the current policy is still ``ok`` -- the critic
flags it and ``plan["recommended"]`` is False, so it is never presented as a
recommendation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

STATUSES: tuple[str, ...] = (
    "ok",
    "needs_clarification",
    "infeasible",
    "spec_invalid",
    "degraded_template",
)


def _to_jsonable(value: Any) -> Any:
    """Tuples become lists, recursively, so a round trip through JSON is exact."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class AgentResult:
    """Everything one goal produced, and what it cost."""

    goal_text: str
    context_id: int
    status: str
    spec: dict[str, Any] | None = None
    question: str | None = None
    refusal: str | None = None
    plan: dict[str, Any] | None = None
    baseline_comparison: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    closest_achievable: list[dict[str, Any]] = field(default_factory=list)
    explanation: str | None = None
    explanation_mode: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Normalise containers to their JSON shape, so equality survives the
        # round trip whatever the caller passed in (tuples vs lists).
        for name in (
            "spec",
            "plan",
            "baseline_comparison",
            "candidates",
            "flags",
            "closest_achievable",
            "trace",
        ):
            object.__setattr__(self, name, _to_jsonable(getattr(self, name)))
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} is not one of {STATUSES}")
        if self.status == "needs_clarification" and not self.question:
            raise ValueError("needs_clarification requires a question")

    @property
    def flag_codes(self) -> list[str]:
        """The critic's flag codes, in the order raised."""
        return [f["code"] for f in self.flags]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "goal_text": self.goal_text,
            "context_id": self.context_id,
            "status": self.status,
            "spec": _to_jsonable(self.spec),
            "question": self.question,
            "refusal": self.refusal,
            "plan": _to_jsonable(self.plan),
            "baseline_comparison": _to_jsonable(self.baseline_comparison),
            "candidates": _to_jsonable(self.candidates),
            "flags": _to_jsonable(self.flags),
            "closest_achievable": _to_jsonable(self.closest_achievable),
            "explanation": self.explanation,
            "explanation_mode": self.explanation_mode,
            "trace": _to_jsonable(self.trace),
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        """One line of JSON, for the run log."""
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentResult:
        """Inverse of :meth:`to_dict`; rejects unknown fields."""
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown AgentResult field(s): {sorted(unknown)}")
        return cls(**{k: _to_jsonable(v) for k, v in payload.items()})
