"""The Prompt 9 benchmark machinery, offline (no twin, no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from typing import Any

import pytest

from twinmind.agents.bench import TimeUp, best_point, twin_only_timeboxed
from twinmind.agents.bench_goals import BENCH_CONTEXTS, load_bench_goals
from twinmind.optim.contracts import Constraint, ObjectiveSpec
from twinmind.optim.goals import load_goals
from twinmind.twin.runner import Scenario
from twinmind.utils.paths import PROJECT_ROOT


# --------------------------------------------------------------------------- #
# The goal set
# --------------------------------------------------------------------------- #
def test_twenty_goals_with_the_registered_mix() -> None:
    goals = load_bench_goals()
    assert len(goals) == 20 and len({g.id for g in goals}) == 20
    # 12 feasible were written; G11 was relabelled infeasible by the goal check
    # BEFORE pre-registration.
    assert Counter(g.category for g in goals) == {
        "feasible": 11,
        "infeasible": 4,
        "no_room": 3,
        "ambiguous_answerable": 2,
    }


def test_benchmark_contexts_are_untouched_by_any_other_goal() -> None:
    used = {g.spec.context_id for g in load_goals()}
    for goal in load_bench_goals():
        assert goal.context_id in BENCH_CONTEXTS
        assert goal.context_id not in used


def test_every_gold_spec_validates_and_weak_kpis_are_constrained() -> None:
    goals = load_bench_goals()
    specs = {g.id: g.spec for g in goals}
    weak = [
        g.id
        for g in goals
        if g.category == "feasible"
        and any(c.kpi in ("unplanned_failures", "downtime_hours") for c in specs[g.id].constraints)
    ]
    assert len(weak) >= 3, weak
    for goal in goals:
        assert specs[goal.id].infeasible_by_design == (goal.category == "infeasible")


def test_the_twin_check_confirmed_every_label_before_pre_registration() -> None:
    check = json.loads(
        (PROJECT_ROOT / "reports" / "results" / "agent_bench_goal_check.json").read_text(
            encoding="utf-8"
        )
    )
    goals = {g.id: g for g in load_bench_goals()}
    assert set(check) == set(goals)
    for goal_id, row in check.items():
        assert row["category"] == goals[goal_id].category
        assert row["label_confirmed"], goal_id


# --------------------------------------------------------------------------- #
# twin_only at equal wall-clock
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SlowEvaluator:
    """Each evaluation costs 1 s of fake clock; stockout falls with safety stock."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def evaluate(self, scenarios: Any) -> list[dict[str, float]]:
        out = []
        for s in scenarios:
            self.clock.now += 1.0
            out.append(
                {
                    "profit": 1000.0 * s.throughput_multiplier,
                    "stockout_probability": max(0.0, 0.5 - s.safety_stock / 4000.0),
                }
            )
        return out

    @property
    def cost(self) -> Any:
        return None


def _spec() -> ObjectiveSpec:
    return ObjectiveSpec(
        name="t",
        objective={"profit": 1.0},
        free_dims=("throughput_multiplier", "safety_stock"),
        constraints=(Constraint("stockout_probability", "<=", 0.2),),
    )


def test_twin_only_stops_at_its_wall_clock_budget_and_keeps_the_best_point() -> None:
    clock = Clock()

    def search(spec: Any, evaluator: Any, budget: int, seed: int) -> None:
        for i in range(1000):
            evaluator.evaluate(
                [Scenario(throughput_multiplier=0.8 + 0.001 * i, safety_stock=2000.0 * (i % 2))]
            )

    outcome = twin_only_timeboxed(
        _spec(), SlowEvaluator(clock), 30.0, 1, clock=clock, search=search
    )
    assert outcome.evaluations == 30, "one point per fake second, then TimeUp"
    assert outcome.feasible_on_search
    assert outcome.scenario is not None and outcome.scenario.safety_stock == 2000.0
    assert clock.now == pytest.approx(30.0)


def test_twin_only_restarts_when_the_search_converges_early() -> None:
    clock = Clock()
    seeds: list[int] = []

    def short_search(spec: Any, evaluator: Any, budget: int, seed: int) -> None:
        seeds.append(seed)
        for _ in range(4):
            evaluator.evaluate([Scenario()])

    outcome = twin_only_timeboxed(
        _spec(), SlowEvaluator(clock), 10.0, 3, clock=clock, search=short_search
    )
    assert outcome.restarts == 2 and seeds == [3000, 3001, 3002]
    assert outcome.evaluations == 10


def test_best_point_is_feasibility_first() -> None:
    spec = _spec()
    infeasible_rich = (Scenario(), {"profit": 9e9, "stockout_probability": 0.9})
    feasible_poor = (Scenario(), {"profit": 1.0, "stockout_probability": 0.1})
    assert best_point(spec, [infeasible_rich, feasible_poor]) == feasible_poor
    assert best_point(spec, []) is None
    assert issubclass(TimeUp, Exception)


# --------------------------------------------------------------------------- #
# Amendment A.1: CMA-ES on goals with one (or no) free dimension
# --------------------------------------------------------------------------- #
class FlatEvaluator:
    """A no-room objective: nothing the lever does changes any KPI."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def evaluate(self, scenarios: Any) -> list[dict[str, float]]:
        out = []
        for _ in scenarios:
            self.clock.now += 0.01
            out.append({"profit": 1000.0, "stockout_probability": 0.1})
        return out

    @property
    def cost(self) -> Any:
        return None


def _one_dim() -> ObjectiveSpec:
    return ObjectiveSpec(name="n", objective={"profit": 1.0}, free_dims=("pm_interval_M5",))


def test_negative_control_pycma_raises_in_one_dimension() -> None:
    """The library bug itself: a 1-D bounded run whose std crosses the cap raises.

    Seed 1007 with x0 0.5, sigma 0.25 and bounds [0, 1] -- exactly what
    cmaes_search builds for seed 1006 -- crosses pycma's automatic maxstd of
    1/3 at its first tell, on a flat objective. The OLD cmaes_search called
    tell unguarded and so raised this straight out of the twin_only arm.
    """
    import cma

    es = cma.CMAEvolutionStrategy([0.5], 0.25, {"seed": 1007, "bounds": [0.0, 1.0], "verbose": -9})
    points = es.ask()
    with pytest.raises(ValueError, match="dimension needed"):
        es.tell(points, [1.0] * len(points))


def test_cmaes_search_survives_the_one_dimensional_std_limit() -> None:
    """Same seed through cmaes_search (seed 1006 -> pycma seed 1007): no raise,
    and the batch evaluated before the failed tell is kept."""
    from twinmind.optim.optimisers import cmaes_search

    clock = Clock()
    found = cmaes_search(_one_dim(), FlatEvaluator(clock), 10**6, seed=1006)
    assert found, "the evaluated batch survives"


def test_twin_only_on_a_one_dimensional_flat_goal_uses_its_whole_budget() -> None:
    """The N01-N03 shape: one free dimension, flat objective, many restarts."""
    clock = Clock()
    outcome = twin_only_timeboxed(_one_dim(), FlatEvaluator(clock), 30.0, 1, clock=clock)
    assert outcome.evaluations > 100 and outcome.restarts > 10
    assert outcome.scenario is not None and outcome.feasible_on_search
    assert clock.now >= 30.0


def test_twin_only_with_zero_free_dimensions_evaluates_the_only_plan() -> None:
    """ObjectiveSpec refuses such a goal, so build one past it, as a guard test."""
    from twinmind.optim.contracts import SpecError

    with pytest.raises(SpecError):
        ObjectiveSpec(
            name="z",
            objective={"profit": 1.0},
            free_dims=("pm_interval_M5",),
            fixed={"pm_interval_M5": 200.0},
        )
    spec = _one_dim()
    object.__setattr__(spec, "free_dims", ())
    clock = Clock()
    outcome = twin_only_timeboxed(spec, FlatEvaluator(clock), 30.0, 1, clock=clock)
    assert outcome.evaluations == 1 and outcome.restarts == 0
    assert outcome.scenario == spec.base_scenario()
    assert "no free dimensions" in outcome.note


# --------------------------------------------------------------------------- #
# The agent arm on a gold spec: no LLM, and the no-critic arm comes for free
# --------------------------------------------------------------------------- #
def test_run_on_spec_makes_no_llm_call_and_records_the_pre_critic_choice() -> None:
    loader = importlib.util.spec_from_file_location(
        "test_agent_fakes", PROJECT_ROOT / "tests" / "test_agent.py"
    )
    assert loader and loader.loader
    ta: Any = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = ta  # dataclasses look their module up here
    loader.loader.exec_module(ta)

    from twinmind.agents.graph import AgentDeps, run_on_spec
    from twinmind.agents.llm_client import FakeLLM

    fragile = {**ta.BASE, "profit": 290_000.0, "stockout_probability": 0.095}
    explored = ta._exploration(
        ta._candidate(1.1, 290_000.0, robust=False), [ta._candidate(1.2, 270_000.0, robust=True)]
    )
    toolkit = ta.FakeToolkit(
        [explored], {1.1: fragile, 1.2: ta.GOOD_PLAN}, noise={"stockout_probability": 0.05}
    )
    llm = FakeLLM([])
    spec = ObjectiveSpec(
        name="gold",
        objective={"profit": 1.0},
        constraints=(Constraint("stockout_probability", "<=", 0.1),),
        context_id=8,
    )
    result = run_on_spec(spec, AgentDeps(config=ta._config(), llm=llm, toolkit=toolkit), seed=2)
    assert llm.calls == [] and result.trace["llm_calls_total"] == 0
    assert result.trace["nodes_run"][0] == "explorer"
    history = result.trace["history"]
    assert len(history) == 2, "the first-round plan, then the critic's swap"
    assert history[0]["scenario"]["throughput_multiplier"] == pytest.approx(1.1)
    assert result.plan is not None
    assert result.plan["scenario"]["throughput_multiplier"] == pytest.approx(1.2)
    assert result.plan["recommendation_reason"]


# --------------------------------------------------------------------------- #
# The runner's bookkeeping
# --------------------------------------------------------------------------- #
def _runner() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_agent_bench.py"
    spec = importlib.util.spec_from_file_location("run_agent_bench_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cells_are_atomic_and_only_successes_count(tmp_path: Any) -> None:
    runner = _runner()
    path = tmp_path / runner.cell_name("G01", "agent", 1)
    assert path.name == "G01__agent__seed1.json"
    runner.write_atomic({"ok": False}, path)
    assert not runner.cell_done(path)
    runner.write_atomic({"ok": True}, path)
    assert runner.cell_done(path)
    assert not list(tmp_path.glob("*.tmp.json"))


def test_appending_an_amendment_keeps_the_manifest_hash_but_edits_do_not(tmp_path: Any) -> None:
    runner = _runner()
    original = "# Pre-registration\n\n## Hypotheses\n\nB1 ...\n"
    path = tmp_path / "prereg.md"
    path.write_text(original, encoding="utf-8")
    before = runner.preregistration_base_hash(path)
    path.write_text(original + "\n## Amendment A.1 — fix\n\nWhat changed.\n", encoding="utf-8")
    assert runner.preregistration_base_hash(path) == before
    path.write_text(original.replace("B1", "B9") + "\n## Amendment A.1\n", encoding="utf-8")
    assert runner.preregistration_base_hash(path) != before


def test_the_manifest_is_invalidated_by_the_preregistration_and_goals() -> None:
    runner = _runner()
    assert {"preregistration_hash", "goals_hash", "agent_config_hash", "seeds"} <= set(
        runner.INVALIDATING
    )
    assert "created" not in runner.INVALIDATING
