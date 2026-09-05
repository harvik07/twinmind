"""LangGraph state machine: Planner -> Explorer -> Evaluator -> Critic -> Explainer."""

from __future__ import annotations

from typing import Any, TypedDict

from twinmind.twin.runner import Scenario, TwinResult
from twinmind.utils.config import AgentConfig


class AgentState(TypedDict, total=False):
    """State threaded through the graph."""

    query: str
    objective: str
    iteration: int
    proposals: list[Scenario]
    screened: list[Scenario]
    results: list[TwinResult]
    baseline: TwinResult
    best: TwinResult
    critique: dict[str, Any]
    explanation: str
    done: bool


def build_graph(config: AgentConfig, **dependencies: Any) -> Any:
    """Compile the LangGraph state machine."""
    raise NotImplementedError("build_graph is not implemented yet")


def run_agent(query: str, config: AgentConfig, **dependencies: Any) -> AgentState:
    """Run the agent end to end for a natural-language query."""
    raise NotImplementedError("run_agent is not implemented yet")
