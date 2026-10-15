"""The agent graph: planner -> explorer -> evaluator -> critic -> explainer.

    planner --(clarify / refuse / invalid)--> END
       |
    explorer -> evaluator -> critic --(swap)----------> evaluator
                               |  --(explore_again)--> explorer   (at most once)
                               v
                           explainer -> END

Only the planner and the explainer call the LLM, through one
:class:`CallBudget` per goal, so the cap holds across repairs and
regenerations. The explorer, evaluator and critic are deterministic Python.

:func:`run_agent` runs one goal and returns a frozen :class:`AgentResult`,
appending it to the JSONL run log. There is no learned memory: the roadmap's
memory feature was dropped for time, and the log is a record, not a store the
agent reads back.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, TypedDict

from twinmind.agents.contracts import AgentResult
from twinmind.agents.critic import critique
from twinmind.agents.explainer import explain
from twinmind.agents.llm_client import (
    CallBudget,
    CallLogger,
    GroqClient,
    LLMClient,
    ModelRule,
    RateLimiter,
    discover_model,
    extra_params_for,
    load_api_key,
    rank_models,
)
from twinmind.agents.planner import plan_goal
from twinmind.agents.tools import Toolkit, build_facts, closest_achievable
from twinmind.optim.contracts import ObjectiveSpec
from twinmind.optim.goals import DEV_CONTEXTS
from twinmind.utils.config import AgentConfig
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    """State threaded through the graph."""

    goal_text: str
    context_id: int
    outcome_kind: str
    spec: ObjectiveSpec
    question: str | None
    refusal: str | None
    planner_repairs: int
    margin_std: float
    retries_used: int
    exploration_status: str
    chosen: dict[str, Any]
    candidates: list[dict[str, Any]]
    evaluation: dict[str, Any]
    novelty: float
    flags: list[dict[str, Any]]
    recommended: bool
    closest: list[dict[str, Any]]
    explanation: str
    explanation_mode: str
    explanation_record: dict[str, Any]
    explorer_seconds: Annotated[float, operator.add]
    surrogate_evals: Annotated[int, operator.add]
    nodes_run: Annotated[list[str], operator.add]
    critic_actions: Annotated[list[str], operator.add]


@dataclass
class AgentDeps:
    """What a run needs: config, an LLM client, and the deterministic toolkit."""

    config: AgentConfig
    llm: LLMClient
    toolkit: Toolkit
    run_log: Path | None = None


def build_graph(deps: AgentDeps, budget: CallBudget) -> Any:
    """Compile the LangGraph state machine for one goal."""
    from langgraph.graph import END, START, StateGraph

    cfg = deps.config

    def planner(state: AgentState) -> dict[str, Any]:
        outcome = plan_goal(
            state["goal_text"],
            state["context_id"],
            deps.toolkit.context_summary(state["context_id"]),
            budget,
            cfg.budget.max_spec_repairs,
            cfg.planner.max_objective_terms,
            cfg.planner.max_constraints,
        )
        return {
            "nodes_run": ["planner"],
            "outcome_kind": outcome.kind,
            "spec": outcome.spec,
            "question": outcome.question,
            "refusal": outcome.reason,
            "planner_repairs": outcome.repairs,
            "margin_std": cfg.explorer.margin_std,
            "retries_used": 0,
            "candidates": [],
        }

    def explorer(state: AgentState) -> dict[str, Any]:
        spec = state["spec"]
        exploration = deps.toolkit.explore(
            spec, state["margin_std"], cfg.explorer.seed + state["retries_used"]
        )
        return {
            "nodes_run": ["explorer"],
            "chosen": exploration.chosen,
            "candidates": [*state.get("candidates", []), *exploration.candidates],
            "exploration_status": exploration.status,
            "explorer_seconds": exploration.wall_seconds,
            "surrogate_evals": exploration.surrogate_evals,
        }

    def evaluator(state: AgentState) -> dict[str, Any]:
        scenario = state["chosen"]["scenario"]
        return {
            "nodes_run": ["evaluator"],
            "evaluation": deps.toolkit.evaluate(state["spec"], scenario),
            "novelty": deps.toolkit.novelty(scenario),
        }

    def critic(state: AgentState) -> dict[str, Any]:
        verdict = critique(
            state["spec"],
            state["evaluation"],
            state["chosen"],
            state["candidates"],
            state["chosen"].get("status", state["exploration_status"]),
            state["novelty"],
            state["retries_used"],
            cfg.critic,
        )
        update: dict[str, Any] = {
            "nodes_run": ["critic"],
            "critic_actions": [verdict.action],
            "flags": verdict.flags,
            "recommended": verdict.recommended,
        }
        if verdict.action == "swap":
            update["chosen"] = verdict.swap_to
            update["retries_used"] = state["retries_used"] + 1
        elif verdict.action == "explore_again":
            update["margin_std"] = state["margin_std"] + cfg.critic.retry_margin_increment_std
            update["retries_used"] = state["retries_used"] + 1
        else:
            update["closest"] = (
                closest_achievable(state["spec"], state["evaluation"], state["candidates"])
                if not state["evaluation"]["feasible_point"]
                else []
            )
        return update

    def explainer(state: AgentState) -> dict[str, Any]:
        facts = build_facts(
            state["spec"],
            state["chosen"],
            state["evaluation"],
            state["flags"],
            state["closest"],
            state["recommended"],
        )
        result = explain(
            facts, budget, cfg.budget.max_explainer_regenerations, cfg.explainer.max_words
        )
        return {
            "nodes_run": ["explainer"],
            "explanation": result.text,
            "explanation_mode": result.mode,
            "explanation_record": {
                "attempts": result.attempts,
                "ungrounded": result.ungrounded,
                "fallback_reason": result.fallback_reason,
                "facts": facts,
            },
        }

    graph = StateGraph(AgentState)
    for name, node in (
        ("planner", planner),
        ("explorer", explorer),
        ("evaluator", evaluator),
        ("critic", critic),
        ("explainer", explainer),
    ):
        graph.add_node(name, node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda s: "explorer" if s["outcome_kind"] == "spec" else "end",
        {"explorer": "explorer", "end": END},
    )
    graph.add_edge("explorer", "evaluator")
    graph.add_edge("evaluator", "critic")
    graph.add_conditional_edges(
        "critic",
        lambda s: {"swap": "evaluator", "explore_again": "explorer"}.get(
            s["critic_actions"][-1], "explainer"
        ),
        {"evaluator": "evaluator", "explorer": "explorer", "explainer": "explainer"},
    )
    graph.add_edge("explainer", END)
    return graph.compile()


def _status(state: AgentState) -> str:
    kind = state.get("outcome_kind")
    if kind in ("refuse", "invalid"):
        return "spec_invalid"
    if kind == "clarify":
        return "needs_clarification"
    if not state["evaluation"]["feasible_point"]:
        return "infeasible"
    if state.get("explanation_mode") == "template":
        return "degraded_template"
    return "ok"


def run_agent(goal_text: str, context_id: int, deps: AgentDeps) -> AgentResult:
    """Run one goal end to end. Never raises for a bad goal; returns a status."""
    if context_id not in DEV_CONTEXTS:
        raise ValueError(
            f"context {context_id} is not a dev context {DEV_CONTEXTS}; the agent only "
            "runs on dev contexts, so no held-out result is produced here"
        )
    started = time.perf_counter()
    twin_before = int(getattr(deps.toolkit, "twin_runs", 0))
    budget = CallBudget(deps.llm, deps.config.budget.max_llm_calls_per_goal)
    graph = build_graph(deps, budget)
    state: AgentState = graph.invoke(
        {"goal_text": goal_text, "context_id": context_id},
        {"recursion_limit": 25},
    )
    status = _status(state)
    spec = state.get("spec")
    chosen = state.get("chosen")
    trace = {
        "nodes_run": state.get("nodes_run", []),
        "critic_actions": state.get("critic_actions", []),
        "retries_used": state.get("retries_used", 0),
        "planner_repairs": state.get("planner_repairs", 0),
        "llm_model": getattr(deps.llm, "model", "?"),
        "llm_calls": [r.to_dict() for r in budget.records],
        "llm_calls_total": budget.used,
        "llm_call_cap": budget.max_calls,
        "prompt_tokens": sum(r.prompt_tokens for r in budget.records),
        "completion_tokens": sum(r.completion_tokens for r in budget.records),
        "llm_latency_seconds": sum(r.latency_seconds for r in budget.records),
        "twin_runs": int(getattr(deps.toolkit, "twin_runs", 0)) - twin_before,
        "surrogate_evals": state.get("surrogate_evals", 0),
        "explorer_seconds": state.get("explorer_seconds", 0.0),
        "wall_seconds": time.perf_counter() - started,
        "explanation": {
            k: v for k, v in (state.get("explanation_record") or {}).items() if k != "facts"
        },
    }
    plan = None
    if chosen is not None:
        plan = {
            "scenario": chosen["scenario"],
            "recommended": bool(state.get("recommended", False)),
            "verification": {
                "mean": chosen["verification_mean"],
                "sem": chosen["verification_sem"],
                "stream": "search",
            },
            "origin": chosen.get("origin"),
            "novelty": state.get("novelty"),
        }
    result = AgentResult(
        goal_text=goal_text,
        context_id=context_id,
        status=status,
        spec=spec.to_dict() if spec is not None else None,
        question=state.get("question"),
        refusal=state.get("refusal"),
        plan=plan,
        baseline_comparison=state.get("evaluation"),
        candidates=state.get("candidates", []),
        flags=state.get("flags", []),
        closest_achievable=state.get("closest", []),
        explanation=state.get("explanation"),
        explanation_mode=state.get("explanation_mode"),
        trace=trace,
    )
    if deps.run_log is not None:
        deps.run_log.parent.mkdir(parents=True, exist_ok=True)
        with deps.run_log.open("a", encoding="utf-8") as fh:
            fh.write(result.to_json() + "\n")
    return result


# --------------------------------------------------------------------------- #
# Real dependencies
# --------------------------------------------------------------------------- #
def resolve_model(config: AgentConfig, raw_client: Any, probe: Any | None = None) -> str:
    """The configured model if Groq still lists it as eligible; else re-discover."""
    llm = config.llm
    rule = ModelRule(
        exclude_patterns=tuple(llm.selection.exclude_patterns),
        min_context_window=llm.selection.min_context_window,
        min_completion_tokens=llm.selection.min_completion_tokens,
    )
    ranked = rank_models(raw_client.models.list().data, rule)
    if llm.model in ranked:
        return llm.model
    logger.warning("configured model %s is no longer listed; re-discovering", llm.model)
    if probe is None:
        raise RuntimeError("model withdrawn and no probe supplied to re-discover")
    return discover_model(raw_client, rule, probe)


def groq_client(config: AgentConfig) -> GroqClient:
    """A GroqClient on the resolved model. The key is read, never printed."""
    from groq import Groq

    llm = config.llm
    raw = Groq(
        api_key=load_api_key(llm.api_key_env, PROJECT_ROOT / ".env"),
        timeout=llm.timeout_seconds,
        max_retries=0,
    )
    model = resolve_model(config, raw)
    return GroqClient(
        model,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        timeout_seconds=llm.timeout_seconds,
        max_transient_retries=llm.max_transient_retries,
        backoff_seconds=llm.backoff_seconds,
        extra_params=extra_params_for(model, llm.extra_params_by_prefix),
        client=raw,
        rate_limiter=RateLimiter(
            llm.rate_limit.requests_per_minute, llm.rate_limit.tokens_per_minute
        ),
        call_logger=CallLogger(PROJECT_ROOT / llm.call_log),
    )


@lru_cache(maxsize=1)
def _shared_state() -> dict[str, Any]:
    from twinmind.models.surrogate_data import load_rollouts
    from twinmind.models.surrogate_mlp import SurrogateEnsemble
    from twinmind.optim.novelty import NoveltyScorer
    from twinmind.twin.demand_proc import CachedDemandModel
    from twinmind.twin.runner import build_resources

    rollouts = PROJECT_ROOT / "data" / "processed" / "rollouts"
    model = CachedDemandModel.load(rollouts / "demand_grids.npz")
    train = load_rollouts(rollouts / "train")
    return {
        "resources": build_resources(demand_model=model, series_id=model.series_ids[0]),
        "surrogate": SurrogateEnsemble.load(PROJECT_ROOT / "models" / "surrogate"),
        "novelty": NoveltyScorer.from_scenarios(train.scenario_matrix()),
        "t1": load_rollouts(rollouts / "test_t1", allow_test=True),
    }


@lru_cache(maxsize=32)
def dev_context(context_id: int) -> Any:
    """The T1 context a dev ``context_id`` names -- the same mapping the goals use."""
    from twinmind.twin.context import sample_context

    if context_id not in DEV_CONTEXTS:
        raise ValueError(f"context {context_id} is not a dev context")
    state = _shared_state()
    t1, resources = state["t1"], state["resources"]
    blocks = sorted(set(t1.context_blocks.tolist()))
    block = blocks[context_id % len(blocks)]
    origin = t1.frame.loc[t1.context_blocks == block, "origin"].iloc[0]
    return sample_context(
        10_000 + int(block),
        cache=getattr(resources.lookup, "cache", None),
        series_id=resources.series_id,
        origin=origin,
    )


def twin_toolkit(config: AgentConfig) -> Any:
    """The real toolkit on one worker."""
    from twinmind.agents.tools import TwinToolkit

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass
    state = _shared_state()
    e = config.explorer
    return TwinToolkit(
        resources=state["resources"],
        surrogate=state["surrogate"],
        novelty_scorer=state["novelty"],
        contexts=dev_context,
        optimiser=e.optimiser,
        budget=e.budget,
        k=e.k,
        min_distance=e.min_distance,
        verify_reps=e.verify_reps,
        max_rounds=e.max_rounds,
        eval_reps=config.evaluator.reps,
        confidence=config.evaluator.confidence_level,
        goal_seed=e.seed,
    )
