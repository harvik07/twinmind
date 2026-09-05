"""Individual LangGraph nodes.

Division of labour, enforced by construction: ``planner`` and ``explorer`` ask
the LLM for *parameters*, ``explainer`` asks it for *prose*, and ``evaluator``
and ``critic`` never call an LLM at all.
"""

from __future__ import annotations

from typing import Any

from twinmind.agents.graph import AgentState


def planner(state: AgentState, **deps: Any) -> AgentState:
    """Turn the query into an objective plus constraint emphasis (LLM: parameters only)."""
    raise NotImplementedError("planner is not implemented yet")


def explorer(state: AgentState, **deps: Any) -> AgentState:
    """Propose candidate scenarios and screen them with the surrogate."""
    raise NotImplementedError("explorer is not implemented yet")


def evaluator(state: AgentState, **deps: Any) -> AgentState:
    """Re-verify the top candidates on the real twin. Deterministic Python only."""
    raise NotImplementedError("evaluator is not implemented yet")


def critic(state: AgentState, **deps: Any) -> AgentState:
    """Check constraints and CI overlap, decide whether to iterate. No LLM."""
    raise NotImplementedError("critic is not implemented yet")


def explainer(state: AgentState, **deps: Any) -> AgentState:
    """Write the natural-language rationale over already-computed KPIs."""
    raise NotImplementedError("explainer is not implemented yet")
