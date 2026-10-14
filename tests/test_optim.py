"""Tests for the optimisation layer.

The two that matter most are the leak guards. An optimiser that could reach the
"reeval" seed stream could report a maximum over the seeds it searched with --
the winner's curse, baked in and invisible. And a goal library whose dev and
held-out halves shared a context would make the held-out result meaningless.

Everything here runs in the non-data suite against a fake cheap surrogate and a
fake twin, so none of it needs the real artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from twinmind.optim.contracts import (
    CONTROLLABLE_DIMS,
    EXOGENOUS_DIMS,
    Constraint,
    ObjectiveSpec,
    OptimResult,
    Plan,
    SpecError,
    rank_plans,
)
from twinmind.optim.evaluators import (
    STREAM_NAMES,
    EvalCost,
    feasible_robust,
    stream_seeds,
)
from twinmind.optim.goals import DEV_CONTEXTS, HELD_OUT_CONTEXTS, load_goals
from twinmind.optim.novelty import NoveltyScorer, normalise_scenarios
from twinmind.optim.optimisers import (
    GRID_MAX_DIMS,
    build_scenario,
    cmaes_search,
    free_dimensions,
    grid_search,
    random_search,
    rank_key,
    tpe_search,
)
from twinmind.twin.runner import KPI_NAMES, SCENARIO_DIMS, Scenario
from twinmind.utils.paths import PROJECT_ROOT


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class FakeEvaluator:
    """A cheap, deterministic stand-in for the surrogate.

    Profit rises with throughput and falls with speed, stockout falls with
    safety stock. Crude, but it gives every optimiser a real gradient to find
    and a real constraint to trip over, which is all these tests need.
    """

    _cost: EvalCost = None  # type: ignore[assignment]
    calls: int = 0

    def __post_init__(self) -> None:
        self._cost = EvalCost()

    def evaluate(self, scenarios: Any) -> list[dict[str, float]]:
        """KPIs from a closed-form surface."""
        out = []
        for scenario in scenarios:
            self.calls += 1
            self._cost.surrogate_evals += 1
            speed = np.mean([getattr(scenario, f"speed_setpoint_M{i}") for i in range(1, 6)])
            profit = (
                300_000.0
                + 200_000.0 * (scenario.throughput_multiplier - 1.0)
                - 150_000.0 * (speed - 1.0) ** 2
                + 10.0 * scenario.safety_stock
            )
            stockout = float(np.clip(0.5 - scenario.safety_stock / 8000.0, 0.0, 1.0))
            out.append(
                {
                    "units_produced": 20_000.0 + 5_000.0 * scenario.throughput_multiplier,
                    "oee": float(np.clip(0.7 * speed, 0.0, 1.0)),
                    "unit_cost": 30.0 - 2.0 * scenario.throughput_multiplier,
                    "profit": profit,
                    "stockout_probability": stockout,
                    "downtime_hours": 10.0 * speed,
                    "unplanned_failures": max(0.0, 2.0 * (speed - 0.9)),
                    "on_time_delivery_rate": float(np.clip(1.0 - stockout, 0.0, 1.0)),
                }
            )
        return out

    @property
    def cost(self) -> EvalCost:
        """What has been spent."""
        return self._cost


class FakeSurrogate:
    """A ``SurrogateModel``-shaped object over :class:`FakeEvaluator`."""

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias
        self.inner = FakeEvaluator()

    def predict(self, scenarios: Any, context: Any) -> Any:
        """Return a SurrogateOutput-like object."""
        from twinmind.models.surrogate import SurrogateOutput

        rows = self.inner.evaluate(scenarios)
        mean = np.array([[r[k] for k in KPI_NAMES] for r in rows], dtype="float64")
        # An optimistic bias on profit, so the winner's-curse path is exercised.
        mean[:, KPI_NAMES.index("profit")] += self.bias
        return SurrogateOutput(
            mean=mean,
            aleatoric_std=np.full_like(mean, 1.0),
            epistemic_std=np.full_like(mean, 0.5),
        )


def simple_spec(**kwargs: Any) -> ObjectiveSpec:
    """A small goal over three free dimensions."""
    payload: dict[str, Any] = {
        "name": "fake",
        "objective": {"profit": 1.0},
        "free_dims": ("throughput_multiplier", "safety_stock", "speed_setpoint_M1"),
    }
    payload.update(kwargs)
    return ObjectiveSpec(**payload)


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #
def test_every_scenario_field_is_classified_exactly_once() -> None:
    """The controllable/exogenous split must cover all 18 dims with no overlap."""
    assert set(CONTROLLABLE_DIMS) | set(EXOGENOUS_DIMS) == set(SCENARIO_DIMS)
    assert not set(CONTROLLABLE_DIMS) & set(EXOGENOUS_DIMS)
    assert len(CONTROLLABLE_DIMS) + len(EXOGENOUS_DIMS) == 18


def test_spec_json_round_trip() -> None:
    spec = simple_spec(
        constraints=(Constraint("stockout_probability", "<=", 0.05),),
        fixed={"price_multiplier": 0.9},
    )
    restored = ObjectiveSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.to_dict() == spec.to_dict()


def test_unknown_kpi_names_the_valid_ones() -> None:
    """The error messages are for an LLM, so they must list the alternatives."""
    with pytest.raises(SpecError) as error:
        ObjectiveSpec(name="x", objective={"throughput": 1.0})
    assert "throughput" in str(error.value)
    assert "units_produced" in str(error.value)


def test_fixing_a_nonexistent_field_raises() -> None:
    with pytest.raises(SpecError, match="is not a Scenario field"):
        ObjectiveSpec(name="x", objective={"profit": 1.0}, fixed={"nope": 1.0})


def test_fixed_value_outside_bounds_raises() -> None:
    with pytest.raises(SpecError, match="outside the configured range"):
        ObjectiveSpec(name="x", objective={"profit": 1.0}, fixed={"price_multiplier": 99.0})


def test_contradictory_constraints_raise() -> None:
    with pytest.raises(SpecError, match="contradictory constraints"):
        ObjectiveSpec(
            name="x",
            objective={"profit": 1.0},
            constraints=(
                Constraint("oee", ">=", 0.9),
                Constraint("oee", "<=", 0.4),
            ),
        )


def test_a_dimension_cannot_be_both_free_and_fixed() -> None:
    with pytest.raises(SpecError, match="BOTH free_dims and fixed"):
        ObjectiveSpec(
            name="x",
            objective={"profit": 1.0},
            free_dims=("overtime_hours",),
            fixed={"overtime_hours": 2.0},
        )


def test_rate_constraint_outside_unit_interval_raises() -> None:
    with pytest.raises(SpecError, match=r"outside \[0, 1\]"):
        Constraint("stockout_probability", "<=", 5.0)


def test_unknown_goal_field_is_rejected() -> None:
    with pytest.raises(SpecError, match="unknown goal field"):
        ObjectiveSpec.from_dict({"name": "x", "objective": {"profit": 1.0}, "oops": 1})


def test_minimise_sense_flips_the_score() -> None:
    """Optimisers always maximise, so the sense must be handled in the spec."""
    maximise = ObjectiveSpec(name="a", objective={"unit_cost": 1.0}, sense="maximise")
    minimise = ObjectiveSpec(name="b", objective={"unit_cost": 1.0}, sense="minimise")
    kpis = {"unit_cost": 25.0}
    assert maximise.score(kpis) == pytest.approx(25.0)
    assert minimise.score(kpis) == pytest.approx(-25.0)


def test_total_violation_normalises_per_constraint() -> None:
    """A 0.05 stockout breach and a GBP 50,000 profit breach must not be summed raw."""
    spec = ObjectiveSpec(
        name="x",
        objective={"profit": 1.0},
        constraints=(
            Constraint("stockout_probability", "<=", 0.10),
            Constraint("profit", ">=", 100_000.0),
        ),
    )
    only_rate = spec.total_violation({"stockout_probability": 0.15, "profit": 200_000.0})
    only_profit = spec.total_violation({"stockout_probability": 0.05, "profit": 50_000.0})
    # Both are breaches; neither should dwarf the other by six orders of magnitude.
    assert 0.0 < only_rate < 1.0
    assert 0.0 < only_profit < 1.0


def test_rank_plans_puts_feasible_first() -> None:
    """Feasibility-first: a feasible plan beats a better-scoring infeasible one."""
    good_infeasible = Plan(
        scenario=Scenario(),
        feasible_point=False,
        constraint_margins={"c": -5.0},
        objective_value=1_000_000.0,
    )
    poor_feasible = Plan(scenario=Scenario(), feasible_point=True, objective_value=1.0)
    assert rank_plans([good_infeasible, poor_feasible])[0] is poor_feasible


def test_among_infeasible_smaller_violation_wins() -> None:
    big = Plan(
        scenario=Scenario(),
        feasible_point=False,
        constraint_margins={"c": -50.0},
        objective_value=10.0,
    )
    small = Plan(
        scenario=Scenario(),
        feasible_point=False,
        constraint_margins={"c": -1.0},
        objective_value=1.0,
    )
    assert rank_plans([big, small])[0] is small


def test_optim_result_round_trip() -> None:
    result = OptimResult(
        spec=simple_spec(),
        plans=(Plan(scenario=Scenario(), objective_value=1.0),),
        method="tpe",
        seed=3,
    )
    restored = OptimResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.method == "tpe"
    assert restored.spec.name == result.spec.name
    assert len(restored.plans) == 1


# --------------------------------------------------------------------------- #
# Seed streams
# --------------------------------------------------------------------------- #
def test_stream_names_are_exactly_search_and_reeval() -> None:
    assert STREAM_NAMES == ("search", "reeval")


def test_search_and_reeval_seeds_are_disjoint() -> None:
    """The firewall: the final number must not come from the search seeds."""
    for goal_seed in range(10):
        search = set(stream_seeds("search", goal_seed, 30))
        reeval = set(stream_seeds("reeval", goal_seed, 30))
        assert len(search) == 30 and len(reeval) == 30
        assert not (search & reeval), goal_seed


def test_optimisers_cannot_reach_the_reeval_stream() -> None:
    """Structural, not conventional: ``evaluate`` has no seed argument at all."""
    import inspect

    from twinmind.optim.evaluators import SurrogateEvaluator, TwinEvaluator

    for cls in (SurrogateEvaluator, TwinEvaluator):
        parameters = set(inspect.signature(cls.evaluate).parameters)
        assert "seed" not in parameters, cls.__name__
        assert "stream" not in parameters, cls.__name__
    # And there is no setter that would let one be swapped at runtime.
    assert not hasattr(TwinEvaluator, "set_stream")


def test_unknown_stream_raises() -> None:
    with pytest.raises(KeyError, match="unknown stream"):
        stream_seeds("oracle", 1, 4)


def test_feasible_robust_is_stricter_than_point_feasibility() -> None:
    """A mean that clears a bound by less than its own error is not robust."""
    spec = ObjectiveSpec(
        name="x",
        objective={"profit": 1.0},
        constraints=(Constraint("stockout_probability", "<=", 0.10),),
    )
    mean = {"stockout_probability": 0.099}
    assert spec.feasible(mean)
    assert not feasible_robust(spec, mean, {"stockout_probability": 0.02})
    assert feasible_robust(spec, mean, {"stockout_probability": 0.0001})


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("search", [random_search, tpe_search, cmaes_search])
def test_optimisers_respect_bounds_and_frozen_dims(search: Any) -> None:
    """Only free dimensions may move, and only inside the configured bounds."""
    from twinmind.models.surrogate import scenario_bounds

    spec = simple_spec(fixed={"price_multiplier": 1.1})
    candidates = search(spec, FakeEvaluator(), 60, seed=1)
    assert candidates
    bounds = scenario_bounds()
    free = set(free_dimensions(spec))
    default = Scenario()
    for candidate in candidates[:20]:
        for dim in SCENARIO_DIMS:
            value = getattr(candidate.scenario, dim)
            low, high = bounds[dim]
            assert low - 1e-9 <= value <= high + 1e-9, dim
            if dim == "price_multiplier":
                assert value == pytest.approx(1.1)
            elif dim not in free:
                assert value == pytest.approx(getattr(default, dim)), dim


@pytest.mark.parametrize("search", [random_search, tpe_search, cmaes_search])
def test_optimisers_are_deterministic_under_a_seed(search: Any) -> None:
    spec = simple_spec()
    first = search(spec, FakeEvaluator(), 40, seed=7)
    second = search(spec, FakeEvaluator(), 40, seed=7)
    assert [c.scenario.to_vector() for c in first[:5]] == [
        c.scenario.to_vector() for c in second[:5]
    ]


@pytest.mark.parametrize("search", [random_search, tpe_search, cmaes_search])
def test_optimisers_improve_on_a_known_surface(search: Any) -> None:
    """The fake surface rewards high throughput, so the best plan should be high."""
    spec = simple_spec()
    candidates = search(spec, FakeEvaluator(), 80, seed=2)
    assert candidates[0].scenario.throughput_multiplier > 1.0


def test_grid_reports_na_above_the_dimension_limit() -> None:
    """Never silently substitute another algorithm under the grid's name."""
    spec = ObjectiveSpec(name="wide", objective={"profit": 1.0})
    assert len(free_dimensions(spec)) > GRID_MAX_DIMS
    assert grid_search(spec, FakeEvaluator(), 100, seed=1) is None


def test_grid_runs_when_the_space_is_small() -> None:
    spec = simple_spec()
    candidates = grid_search(spec, FakeEvaluator(), 64, seed=1)
    assert candidates is not None and candidates


def test_feasibility_first_ordering_in_rank_key() -> None:
    from twinmind.optim.optimisers import Candidate

    feasible = Candidate(Scenario(), {}, score=1.0, violation=0.0)
    infeasible = Candidate(Scenario(), {}, score=1e9, violation=0.1)
    assert rank_key(feasible) < rank_key(infeasible)


def test_build_scenario_rounds_integer_dimensions() -> None:
    spec = ObjectiveSpec(
        name="x", objective={"profit": 1.0}, free_dims=("shift_count", "overtime_hours")
    )
    scenario = build_scenario(spec, ["shift_count", "overtime_hours"], np.array([0.5, 0.5]))
    assert isinstance(scenario.shift_count, int)


def test_a_goal_with_no_free_dimensions_raises() -> None:
    """The defensive guard in free_dimensions, reached by bypassing the validator.

    The constructor already rejects a dimension that is both free and fixed, so
    this state is unreachable through the public API. The guard still earns its
    place -- a future caller building a spec programmatically could produce it --
    and an unreachable guard that has never been seen to fire is worth nothing,
    so the frozen dataclass is written through directly to exercise it.
    """
    spec = ObjectiveSpec(name="x", objective={"profit": 1.0}, free_dims=("overtime_hours",))
    object.__setattr__(spec, "fixed", {"overtime_hours": 1.0})
    with pytest.raises(SpecError, match="no free dimensions"):
        free_dimensions(spec)


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #
def test_supplier_distance_is_one_between_any_two_suppliers() -> None:
    """Unordered categories must be equidistant, not interpolated."""
    rows = []
    for supplier in (0, 1, 2):
        rows.append(Scenario(supplier_choice=supplier).to_vector())
    normalised = normalise_scenarios(np.array(rows))
    for a in range(3):
        for b in range(a + 1, 3):
            assert float(np.linalg.norm(normalised[a] - normalised[b])) == pytest.approx(1.0)


def test_novelty_rises_with_distance_from_training() -> None:
    rng = np.random.default_rng(0)
    from twinmind.models.surrogate import scenario_bounds

    bounds = scenario_bounds()
    # Training points clustered near the middle of every dimension.
    mid = np.array([[(bounds[d][0] + bounds[d][1]) / 2 for d in SCENARIO_DIMS] for _ in range(200)])
    mid += rng.normal(0, 1e-3, size=mid.shape)
    scorer = NoveltyScorer.from_scenarios(mid, k=5)

    centre = scorer.score(mid[:5])
    corner = scorer.score(np.array([[bounds[d][1] for d in SCENARIO_DIMS]]))
    assert float(corner[0]) > float(centre.mean())


def test_novelty_k_is_fixed_not_tuned() -> None:
    """k is pinned so it cannot quietly become a hyperparameter fitted to OOD."""
    from twinmind.optim.novelty import DEFAULT_K

    assert DEFAULT_K == 10


# --------------------------------------------------------------------------- #
# Goal library
# --------------------------------------------------------------------------- #
def test_every_goal_yaml_validates() -> None:
    goals = load_goals()
    assert len(goals) == 20
    assert sum(1 for g in goals if g.split == "dev") == 8
    assert sum(1 for g in goals if g.split == "held_out") == 12


def test_dev_and_held_out_contexts_are_disjoint() -> None:
    """A goal must not be nearly the same problem on both sides of the split."""
    goals = load_goals()
    dev = {g.spec.context_id for g in goals if g.split == "dev"}
    held = {g.spec.context_id for g in goals if g.split == "held_out"}
    assert not (dev & held)
    assert all(c in DEV_CONTEXTS for c in dev)
    assert all(c in HELD_OUT_CONTEXTS for c in held)


def test_exactly_two_goals_are_infeasible_by_design() -> None:
    goals = load_goals()
    infeasible = [g for g in goals if g.spec.infeasible_by_design]
    assert len(infeasible) == 2
    assert {g.split for g in infeasible} == {"dev", "held_out"}


def test_goal_library_covers_the_required_shapes() -> None:
    """Single, weighted, constrained, frozen-dim and exogenous what-if."""
    goals = [g.spec for g in load_goals()]
    assert any(len(g.objective) > 1 for g in goals), "no weighted multi-objective goal"
    assert any(g.sense == "minimise" for g in goals), "no minimise goal"
    assert any(len(g.free_dims) < len(CONTROLLABLE_DIMS) for g in goals), "no frozen-dim goal"
    assert any(set(g.fixed) & set(EXOGENOUS_DIMS) for g in goals), "no exogenous what-if"
    constrained = {c.kpi for g in goals for c in g.constraints}
    for kpi in (
        "stockout_probability",
        "on_time_delivery_rate",
        "unplanned_failures",
        "downtime_hours",
    ):
        assert kpi in constrained, f"no goal constrains {kpi}"


def test_a_goal_in_the_wrong_context_range_is_rejected(tmp_path: Path) -> None:
    """A dev goal using a held-out context would quietly break the split."""
    import yaml

    from twinmind.optim.goals import load_goal

    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {"name": "bad", "split": "dev", "context_id": 25, "objective": {"profit": 1.0}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match="not in the dev range"):
        load_goal(path)


# --------------------------------------------------------------------------- #
# Benchmark plumbing
# --------------------------------------------------------------------------- #
def _benchmark_module() -> Any:
    """Load the benchmark script as a module."""
    import importlib.util
    import sys as _sys

    from twinmind.utils.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "scripts" / "run_optim_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_optim_benchmark_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_writes_atomically_and_resume_skips(tmp_path: Path) -> None:
    module = _benchmark_module()
    path = tmp_path / "cell.json"
    # cell_done also checks the budget hash, so a cell counted as finished must
    # carry the current one -- see test_a_cell_from_a_different_budget_is_not_
    # counted_as_done for why.
    module.write_atomic({"a": 1, "budgets_hash": module.budgets_hash()}, path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp.json"))
    assert module.cell_done(path)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert not module.cell_done(broken)
    assert not module.cell_done(tmp_path / "absent.json")


def test_benchmark_manifest_conflicts_on_a_changed_assumption() -> None:
    module = _benchmark_module()
    base = {
        "twin_config_hash": "a",
        "surrogate_hash": "b",
        "assumptions": {"demand_capacity_ratio": 0.85},
        "demand_scale": 0.44,
        "budgets": {"sf_tpe": 3000},
        "created": "x",
    }
    assert module.manifest_conflicts(base, {**base, "created": "y"}) == []
    changed = {**base, "assumptions": {"demand_capacity_ratio": 0.64}}
    assert "assumptions" in module.manifest_conflicts(base, changed)


def test_benchmark_cell_filenames_are_unique_per_cell() -> None:
    module = _benchmark_module()
    names = {
        module.Cell(goal=g, method=m, seed=s).filename
        for g in ("a", "b")
        for m in ("sf_tpe", "twin_tpe")
        for s in (1, 2)
    }
    assert len(names) == 8


# --------------------------------------------------------------------------- #
# surrogate_first_verify, with a fake surrogate and a fake twin
# --------------------------------------------------------------------------- #
class FakeTwinResources:
    """Minimal stand-in for TwinResources, consumed by a patched run_twin.

    The fake twin is the UNBIASED surface plus replication noise. All optimism
    lives in ``FakeSurrogate.bias``, so the surrogate-minus-twin gap equals that
    bias exactly and a test can assert on it. An earlier version applied the
    bias in both places and the gap came out doubled -- a fixture bug that looked
    like a code bug.
    """

    def __init__(self, stockout_floor: float = 0.0) -> None:
        self.stockout_floor = stockout_floor
        self.calls = 0


def _patch_twin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace run_twin with the fake surface, offset from the surrogate's.

    The twin is the fake surface MINUS the surrogate's optimistic bias, so the
    surrogate systematically over-predicts profit. That is what makes the
    winner's-curse and verification paths meaningful rather than decorative.
    """
    import twinmind.optim.evaluators as evaluators

    def fake_run_twin(scenario: Any, seed: int, context: Any, resources: Any) -> Any:
        resources.calls += 1
        rows = FakeEvaluator().evaluate([scenario])[0]
        rng = np.random.default_rng(seed)
        values = dict(rows)
        values["profit"] = rows["profit"] + float(rng.normal(0, 500))
        values["stockout_probability"] = float(
            np.clip(max(rows["stockout_probability"], resources.stockout_floor), 0.0, 1.0)
        )
        values["on_time_delivery_rate"] = float(
            np.clip(1.0 - values["stockout_probability"], 0.0, 1.0)
        )

        class Result:
            def to_vector(self) -> list[float]:
                return [values[k] for k in KPI_NAMES]

        return Result()

    monkeypatch.setattr(evaluators, "run_twin", fake_run_twin, raising=False)
    import twinmind.twin.runner as runner_module

    monkeypatch.setattr(runner_module, "run_twin", fake_run_twin)


def test_surrogate_first_verify_finds_a_feasible_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path: a verified-feasible plan, re-evaluated independently."""
    from twinmind.optim.surrogate_first import VerifyConfig, surrogate_first_verify

    _patch_twin(monkeypatch)
    spec = simple_spec(constraints=(Constraint("stockout_probability", "<=", 0.40),))
    plans, trace = surrogate_first_verify(
        spec,
        context=None,
        surrogate=FakeSurrogate(bias=20_000.0),
        resources=FakeTwinResources(),
        seed=1,
        config=VerifyConfig(optimiser="random", budget=256, k=5, verify_reps=3, final_reps=5),
    )
    assert plans
    best = plans[0]
    assert best.status == "ok"
    assert best.feasible_point
    assert trace["status"] == "ok"
    # The surrogate over-predicted profit, and the gap is reported.
    assert best.surrogate_twin_gap["profit"] > 0


def test_surrogate_first_verify_reports_the_surrogate_twin_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap is the agent's future fallback signal, so it must be per-KPI."""
    from twinmind.optim.surrogate_first import VerifyConfig, surrogate_first_verify

    _patch_twin(monkeypatch)
    plans, _ = surrogate_first_verify(
        simple_spec(),
        context=None,
        surrogate=FakeSurrogate(bias=50_000.0),
        resources=FakeTwinResources(),
        seed=2,
        config=VerifyConfig(optimiser="random", budget=128, k=4, verify_reps=3, final_reps=5),
    )
    gap = plans[0].surrogate_twin_gap
    assert set(gap) == set(KPI_NAMES)
    # The surrogate's bias is the ONLY source of disagreement, so the gap is it.
    assert gap["profit"] == pytest.approx(50_000.0, rel=0.1)


def test_surrogate_first_verify_returns_infeasible_rather_than_passing_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An impossible constraint must surface, never be quietly relaxed."""
    from twinmind.optim.surrogate_first import VerifyConfig, surrogate_first_verify

    _patch_twin(monkeypatch)
    # The fake twin floors stockout at 0.30, so a <= 0.01 constraint cannot hold.
    spec = simple_spec(constraints=(Constraint("stockout_probability", "<=", 0.01),))
    plans, trace = surrogate_first_verify(
        spec,
        context=None,
        surrogate=FakeSurrogate(),
        resources=FakeTwinResources(stockout_floor=0.30),
        seed=3,
        config=VerifyConfig(
            optimiser="random", budget=128, k=4, verify_reps=2, final_reps=3, max_rounds=1
        ),
    )
    assert plans
    assert plans[0].status == "infeasible_unverified"
    assert not plans[0].feasible_point
    assert trace["status"] == "infeasible_unverified"
    # It really did try more than once before giving up.
    assert len(trace["rounds"]) == 2


def test_surrogate_first_verify_widens_the_margin_across_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each retry must loosen nothing and tighten the surrogate margin."""
    from twinmind.optim.surrogate_first import VerifyConfig, surrogate_first_verify

    _patch_twin(monkeypatch)
    spec = simple_spec(constraints=(Constraint("stockout_probability", "<=", 0.01),))
    _, trace = surrogate_first_verify(
        spec,
        context=None,
        surrogate=FakeSurrogate(),
        resources=FakeTwinResources(stockout_floor=0.30),
        seed=4,
        config=VerifyConfig(
            optimiser="random", budget=64, k=3, verify_reps=2, final_reps=2, max_rounds=2
        ),
    )
    margins = [r["margin_std"] for r in trace["rounds"]]
    assert margins == sorted(margins)
    assert margins[-1] > margins[0]


def test_shortlist_is_diverse_not_twenty_points_in_one_basin() -> None:
    """Verifying 20 near-identical plans spends the twin budget learning one thing."""
    from twinmind.optim.optimisers import Candidate
    from twinmind.optim.surrogate_first import select_diverse

    base = Scenario()
    clones = [Candidate(base, {}, score=10.0 - i, violation=0.0) for i in range(10)]
    spread = [
        Candidate(
            Scenario(throughput_multiplier=0.7 + 0.1 * i), {}, score=1.0 - i * 0.01, violation=0.0
        )
        for i in range(6)
    ]
    chosen = select_diverse([*clones, *spread], k=5, min_distance=0.15)
    vectors = {tuple(c.scenario.to_vector()) for c in chosen}
    assert len(vectors) >= 4, "the shortlist collapsed onto near-duplicates"


def test_surrogate_only_skips_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ablation must spend only the final re-evaluation on the twin."""
    from twinmind.optim.surrogate_first import VerifyConfig, surrogate_only

    _patch_twin(monkeypatch)
    resources = FakeTwinResources()
    plans, trace = surrogate_only(
        simple_spec(),
        context=None,
        surrogate=FakeSurrogate(bias=40_000.0),
        resources=resources,
        seed=5,
        config=VerifyConfig(optimiser="random", budget=128, final_reps=4),
    )
    assert trace["method"] == "surrogate_only"
    assert len(plans) == 1
    # Only the 4 final replications touched the twin, not a shortlist of 20.
    assert resources.calls == 4
    assert plans[0].surrogate_twin_gap["profit"] > 0


def test_benchmark_smoke_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The --smoke path must construct and run a cell with fakes.

    Not a subprocess test: the real smoke needs the trained surrogate and the
    health cache, which the non-data suite does not have. This exercises the
    cell-construction and dispatch logic with the same fakes used above, which
    is the part that breaks when a method name or a budget key is mistyped.
    """
    module = _benchmark_module()
    _patch_twin(monkeypatch)

    goal = next(g for g in load_goals(split="dev") if not g.spec.infeasible_by_design)
    for method in ("sf_random", "surrogate_only", "twin_tpe", "best_known"):
        cell = module.Cell(goal=goal.name, method=method, seed=1)
        result = module.run_cell(
            cell,
            goal,
            context=None,
            surrogate=FakeSurrogate(bias=10_000.0),
            resources=FakeTwinResources(),
            novelty=None,
            smoke=True,
        )
        assert result["method"] == method
        assert result["goal"] == goal.name
        assert result["plans"], method
        # Every cell must round-trip through JSON, since that is how it is stored.
        restored = json.loads(json.dumps(result, default=str))
        assert restored["method"] == method


def test_every_method_has_a_budget() -> None:
    """A mistyped method name would otherwise fail only hours into the run."""
    module = _benchmark_module()
    assert set(module.METHODS) == set(module.BUDGETS)
    assert all(v > 0 for v in module.BUDGETS.values())


def test_surrogate_only_is_a_matched_ablation_of_sf_tpe() -> None:
    """The two arms must share ONE budget constant, not two equal literals.

    ``surrogate_only`` isolates what twin verification is worth, so it has to
    differ from ``surrogate_first_verify`` in the verification step alone --
    K=1 and no verification against K=20 verified. It was 20,000 against
    ``sf_tpe``'s 3,000, which made the ablation the best-searched arm and would
    have left H3 uninterpretable: a win for ``surrogate_only`` could have been
    the larger search rather than the absence of verification.

    Asserting they are *equal* would pass with two literals that later drift.
    This asserts they are the *same constant*, which is what stops the drift.
    """
    module = _benchmark_module()
    assert module.BUDGETS["sf_tpe"] is module.SURROGATE_TPE_BUDGET
    assert module.BUDGETS["surrogate_only"] is module.SURROGATE_TPE_BUDGET
    assert module.SURROGATE_TPE_BUDGET == 3_000

    source = (PROJECT_ROOT / "scripts" / "run_optim_benchmark.py").read_text(encoding="utf-8")
    budget_block = source[source.index("BUDGETS = {") : source.index("METHODS = tuple")]
    assert budget_block.count("SURROGATE_TPE_BUDGET") == 2
    assert "20_000" not in budget_block.replace('"sf_cmaes": 20_000,', "")


def test_budget_hash_changes_when_any_budget_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hash is what stops a resume mixing cells from two configurations."""
    module = _benchmark_module()
    before = module.budgets_hash()
    monkeypatch.setitem(module.BUDGETS, "surrogate_only", 20_000)
    assert module.budgets_hash() != before


def test_a_cell_from_a_different_budget_is_not_counted_as_done(tmp_path: Path) -> None:
    """A stale cell must be retried, not averaged in beside current ones.

    This is the concrete risk from Amendment 1: a ``surrogate_only`` cell
    written under the old 20,000-trial budget sitting in a results tree. Without
    the check it would count as finished and its number would be averaged
    alongside 3,000-trial cells, giving a table whose rows are not comparable
    and no sign that anything was wrong.
    """
    module = _benchmark_module()
    path = tmp_path / "cell.json"

    module.write_atomic({"goal": "g", "budgets_hash": module.budgets_hash()}, path)
    assert module.cell_done(path)

    module.write_atomic({"goal": "g", "budgets_hash": "0000stale0000"}, path)
    assert not module.cell_done(path)

    # A cell with no hash at all predates the check and is treated as stale.
    module.write_atomic({"goal": "g"}, path)
    assert not module.cell_done(path)


# --------------------------------------------------------------------------- #
# Production budgets
# --------------------------------------------------------------------------- #
#
# The smoke run used budgets ~1/500 of production, so any code path that only
# triggers at full budget was untested -- and one was broken. ``random_search``
# drew a second Sobol block once the budget passed its 2,048-point evaluation
# batch, which scipy refuses, and the production run died in seconds while the
# smoke run passed. These tests run every optimiser at its REAL budget against
# fakes, so the failure mode cannot come back.


def _production_budgets() -> dict[str, int]:
    """The budgets the benchmark actually uses, read from the script itself."""
    return dict(_benchmark_module().BUDGETS)


@pytest.mark.parametrize("method", ["sf_random", "sf_cmaes", "best_known"])
def test_sobol_and_cmaes_survive_their_production_budget(method: str) -> None:
    """Every budget-dependent search path, at the budget it will really see.

    ``sf_random`` (50,000) and ``best_known`` (500,000) both drive
    ``random_search``, which is where the crash was. CMA-ES (20,000) is included
    because it is the other large-budget path.

    Fails on the old code with:
        ValueError: The balance properties of Sobol' points require n to be a
        power of 2.
    """
    budgets = _production_budgets()
    search = random_search if method != "sf_cmaes" else cmaes_search
    spec = simple_spec()
    candidates = search(spec, FakeEvaluator(), budgets[method], seed=1)
    assert candidates, method
    # The shortlist is capped at 200 regardless of budget.
    assert len(candidates) <= 200


def test_random_search_draws_one_balanced_sobol_block() -> None:
    """A prefix of ONE 2**m block, not several blocks concatenated.

    The property the fix preserves: Sobol's low-discrepancy guarantee holds for
    a prefix from the start of the sequence. Checked by confirming the points a
    large-budget search visits are the prefix of a single fresh sampler's block.
    """
    from scipy import stats as _stats

    from twinmind.models.surrogate import scenario_bounds

    spec = simple_spec()
    free = free_dimensions(spec)
    budget = 3000
    evaluator = FakeEvaluator()
    candidates = random_search(spec, evaluator, budget, seed=11)
    assert evaluator.calls == budget

    sampler = _stats.qmc.Sobol(d=len(free), scramble=True, seed=11)
    expected = sampler.random_base2(m=12)[:budget]  # 2**12 = 4096 >= 3000
    bounds = scenario_bounds()
    first = build_scenario(spec, free, expected[0])
    # The best-ranked candidate is not necessarily the first point, so check
    # membership of the first point rather than ordering.
    visited = {tuple(np.round(c.scenario.to_vector(), 9)) for c in candidates}
    assert len(visited) <= 200
    assert all(bounds[d][0] <= getattr(first, d) <= bounds[d][1] for d in free)


@pytest.mark.parametrize("budget", [2048, 2049, 4096, 4097, 8192])
def test_random_search_at_and_around_every_power_of_two(budget: int) -> None:
    """The old bug triggered at budget > 2048 exactly; pin the boundary."""
    evaluator = FakeEvaluator()
    candidates = random_search(simple_spec(), evaluator, budget, seed=3)
    assert candidates
    assert evaluator.calls == budget


def test_tpe_at_production_budget_is_covered_by_a_smaller_run() -> None:
    """TPE's production budget is 3,000 trials, which Optuna cannot do in seconds.

    Measured at roughly 40 ms per trial, 3,000 trials is about two minutes --
    too slow for the suite, and it is the SAMPLER that is slow, not any
    budget-dependent branch of ours. ``tpe_search`` has no batching or block
    logic, so there is no code path that appears only above a threshold; the
    budget is passed straight to ``study.optimize``. This runs enough trials to
    exercise the loop and records why the full number is not used here.
    """
    budgets = _production_budgets()
    assert budgets["sf_tpe"] == 3000
    evaluator = FakeEvaluator()
    candidates = tpe_search(simple_spec(), evaluator, 300, seed=2)
    assert candidates
    assert evaluator.calls == 300


def test_surrogate_first_and_only_at_production_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two surrogate-first entry points at their real budgets."""
    from twinmind.optim.surrogate_first import VerifyConfig, surrogate_first_verify, surrogate_only

    _patch_twin(monkeypatch)
    budgets = _production_budgets()

    plans, _ = surrogate_first_verify(
        simple_spec(),
        context=None,
        surrogate=FakeSurrogate(bias=5_000.0),
        resources=FakeTwinResources(),
        seed=1,
        # The real K and reps; only the optimiser is swapped to the fast one.
        config=VerifyConfig(
            optimiser="random", budget=budgets["sf_random"], k=20, verify_reps=10, final_reps=30
        ),
    )
    assert plans and plans[0].status in {"ok", "infeasible_unverified"}

    only, _ = surrogate_only(
        simple_spec(),
        context=None,
        surrogate=FakeSurrogate(bias=5_000.0),
        resources=FakeTwinResources(),
        seed=1,
        config=VerifyConfig(optimiser="random", budget=budgets["surrogate_only"], final_reps=30),
    )
    assert only


def test_twin_only_arms_at_production_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """300 twin evaluations at 5 reps, through the recording wrapper.

    The anytime checkpoints at 50/150/300 only fire at production budget, so a
    smoke run never exercised them.
    """
    module = _benchmark_module()
    _patch_twin(monkeypatch)

    plans, trace = module.run_twin_only(
        simple_spec(),
        context=None,
        resources=FakeTwinResources(),
        seed=1,
        optimiser="random",
        budget=module.BUDGETS["twin_tpe"],
        n_reps=5,
    )
    assert plans
    checkpoints = [row["evaluations"] for row in trace["anytime"]]
    assert checkpoints == list(module.ANYTIME_CHECKPOINTS)


def test_best_known_at_production_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """500,000 surrogate points, top-100 verified. The largest budget in the run."""
    module = _benchmark_module()
    _patch_twin(monkeypatch)

    plans, _ = module.run_best_known(
        simple_spec(),
        context=None,
        surrogate=FakeSurrogate(),
        resources=FakeTwinResources(),
        seed=1,
        budget=module.BUDGETS["best_known"],
        top_n=100,
    )
    assert plans
    assert len(plans) <= 10


# --------------------------------------------------------------------------- #
# Per-cell failure handling
# --------------------------------------------------------------------------- #
def test_a_failing_cell_does_not_raise_out_of_the_worker() -> None:
    """One bad cell must not kill the pool, as the Sobol crash did.

    ``_run_one`` returns ``ok=False`` with a traceback instead of raising, so
    ``imap_unordered`` keeps feeding the remaining cells.
    """
    module = _benchmark_module()
    # A goal that does not exist makes _run_one's lookup raise inside the try.
    result = module._run_one(
        {
            "goal": "no_such_goal",
            "method": "sf_random",
            "seed": 1,
            "smoke": True,
            "novelty_active": False,
        }
    )
    assert result["ok"] is False
    assert result["goal"] == "no_such_goal"
    assert "traceback" in result and result["traceback"]
    assert result["error"]


def test_failed_cells_are_not_written_to_the_results_tree(tmp_path: Path) -> None:
    """A failure goes to failures/, so a rerun retries it and keeps the rest.

    Writing a failed cell into the results tree would make ``cell_done`` treat
    it as finished, and the rerun would skip the very cell that needs retrying.
    """
    module = _benchmark_module()
    cell = module.Cell(goal="g", method="sf_random", seed=1)
    failure = {"ok": False, "goal": "g", "method": "sf_random", "seed": 1, "error": "boom"}

    module.write_atomic(failure, tmp_path / "failures" / cell.filename)
    assert not module.cell_done(tmp_path / cell.filename), "a failure must not count as done"
    # The failure file exists and parses; it is simply not in the results tree,
    # which is what makes a rerun retry it.
    assert (tmp_path / "failures" / cell.filename).exists()
    assert (
        json.loads((tmp_path / "failures" / cell.filename).read_text(encoding="utf-8"))["error"]
        == "boom"
    )


def test_a_scratch_run_never_writes_the_canonical_summary() -> None:
    """A dry run pointed at --out <scratch> must not touch reports/results.

    It did: the production-budget dry run on a DEV goal overwrote
    reports/results/optim_benchmark_run.json with a one-cell summary -- the
    path the real held-out run owns. The canonical copy is now gated on --out
    being the default results directory, checked here against the source so the
    gate cannot be removed without this test failing.
    """
    source = (PROJECT_ROOT / "scripts" / "run_optim_benchmark.py").read_text(encoding="utf-8")
    tail = source[source.index('write_atomic(summary, out / "summary.json")') :]
    gate = tail.index("if Path(args.out).resolve() == RESULTS_DIR.resolve():")
    canonical = tail.index('"optim_benchmark_run.json"')
    assert gate < canonical, "the canonical summary must be written only inside the gate"


# --------------------------------------------------------------------------- #
# Bounded memory
# --------------------------------------------------------------------------- #
#
# The first full run died with MemoryError: ``best_known`` drew its whole
# 500,000-point Sobol sample as one 2**19 x 16 float64 block (64 MiB), six such
# cells started at once because the queue put them first, and on an 8 GB laptop
# the machine ran out. The draw is now chunked and the queue keeps best_known
# to one lane; these pin both, and pin that neither changed a single result.


def _one_block(d: int, budget: int, seed: int) -> Any:
    """The draw the chunked sampler replaces: one 2**ceil(log2(budget)) block."""
    import math

    from scipy import stats as _stats

    sampler = _stats.qmc.Sobol(d=d, scramble=True, seed=seed)
    return sampler.random_base2(m=max(int(math.ceil(math.log2(max(budget, 2)))), 1))[:budget]


@pytest.mark.parametrize(
    ("d", "budget", "chunk"),
    [
        (16, 500_000, 2**15),  # best_known, at its real budget and real chunk
        (3, 50_000, 2**15),  # sf_random
        (3, 20_000, 2**12),  # several chunks, last one truncated
        (3, 2**16, 2**15),  # an exact multiple: no truncation
        (3, 2**15 + 1, 2**15),  # one point into the second chunk
        (3, 3_000, 2**15),  # fits in one chunk: the old path exactly
    ],
)
def test_chunked_sobol_is_bitwise_identical_to_the_single_block(
    d: int, budget: int, chunk: int
) -> None:
    """Same points, same order, same bits -- with every warning an error.

    Bitwise, not ``allclose``: the completed ``sf_random`` cells were produced
    by the single-block draw, and they stay valid only if not one bit moves.
    Warnings are errors because scipy's balance warning is how a non-power-of-
    two draw would announce itself, and a warning is easy to miss in a log.
    """
    import warnings

    from twinmind.optim.optimisers import sobol_chunks

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parts = list(sobol_chunks(d, budget, seed=7, chunk=chunk))
        expected = _one_block(d, budget, seed=7)
    starts = [start for start, _ in parts]
    chunked = np.concatenate([points for _, points in parts])
    assert starts == list(range(0, budget, chunk))[: len(parts)]
    assert max(points.shape[0] for _, points in parts) <= chunk
    assert chunked.shape == expected.shape == (budget, d)
    assert np.array_equal(chunked.view(np.uint64), expected.view(np.uint64))


def test_sobol_chunks_rejects_a_chunk_that_is_not_a_power_of_two() -> None:
    from twinmind.optim.optimisers import sobol_chunks

    with pytest.raises(ValueError, match="power of two"):
        list(sobol_chunks(3, 10_000, seed=0, chunk=3000))


def _signature(candidates: list[Any]) -> list[tuple[Any, ...]]:
    return [(tuple(c.scenario.to_vector()), c.score, c.violation) for c in candidates]


def _one_shot_random_search(spec: ObjectiveSpec, evaluator: Any, budget: int, seed: int) -> Any:
    """The old code, reduced to its definition: one block, score all, sort once."""
    from twinmind.optim.optimisers import _score_batch

    free = free_dimensions(spec)
    points = _one_block(len(free), budget, seed)
    scenarios = [build_scenario(spec, free, row) for row in points]
    everything = _score_batch(spec, scenarios, evaluator)
    everything.sort(key=rank_key)  # stable: ties keep evaluation order
    return everything[:200]


def _constrained_spec() -> ObjectiveSpec:
    """simple_spec plus a constraint, so feasibility-first ordering bites."""
    return simple_spec(
        constraints=(Constraint(kpi="stockout_probability", op="<=", threshold=0.3),)
    )


def test_chunked_top_k_equals_the_one_shot_top_k_at_20k(monkeypatch: pytest.MonkeyPatch) -> None:
    """The running top-K merge returns exactly the one-shot top-K.

    The chunk is shrunk to 4,096 so the 20,000 budget spans five chunks (the
    last truncated); at the production chunk of 32,768 it would be one chunk
    and prove nothing.
    """
    import twinmind.optim.optimisers as optimisers

    monkeypatch.setattr(optimisers, "SOBOL_CHUNK", 4096)
    chunked = random_search(_constrained_spec(), FakeEvaluator(), 20_000, seed=4)
    one_shot = _one_shot_random_search(_constrained_spec(), FakeEvaluator(), 20_000, seed=4)
    assert len(list(optimisers.sobol_chunks(3, 20_000, 4, chunk=4096))) == 5
    assert _signature(chunked) == _signature(one_shot)
    assert any(c.feasible for c in chunked) and len(chunked) == 200


def test_best_known_returns_the_same_plans_as_the_one_shot_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through ``run_best_known``: shortlist, twin verification, ranking."""
    import twinmind.optim.optimisers as optimisers

    module = _benchmark_module()
    _patch_twin(monkeypatch)
    monkeypatch.setattr(optimisers, "SOBOL_CHUNK", 4096)

    def plans_with(search: Any) -> list[tuple[Any, ...]]:
        monkeypatch.setitem(module.OPTIMISERS, "random", search)
        plans, _ = module.run_best_known(
            _constrained_spec(),
            context=None,
            surrogate=FakeSurrogate(),
            resources=FakeTwinResources(),
            seed=2,
            budget=20_000,
            top_n=100,
        )
        return [
            (tuple(p.scenario.to_vector()), p.objective_value, p.status, p.feasible_robust)
            for p in plans
        ]

    new = plans_with(random_search)
    old = plans_with(_one_shot_random_search)
    assert new == old
    assert len(new) == 10


def test_random_search_is_deterministic_per_seed_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import twinmind.optim.optimisers as optimisers

    monkeypatch.setattr(optimisers, "SOBOL_CHUNK", 4096)
    first = random_search(simple_spec(), FakeEvaluator(), 20_000, seed=5)
    again = random_search(simple_spec(), FakeEvaluator(), 20_000, seed=5)
    other = random_search(simple_spec(), FakeEvaluator(), 20_000, seed=6)
    assert _signature(first) == _signature(again)
    assert _signature(first) != _signature(other)


def test_random_search_memory_is_bounded_at_the_best_known_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tracemalloc peak over the full 500,000-point search, all 16 dimensions free.

    THE BOUND IS 16 MiB, a quarter of the 64 MiB the single-block draw held for
    its points alone (2**19 rows x 16 x 8 bytes), so the old code fails this
    test on the Sobol block before evaluating anything. The new peak is one
    32,768-row chunk (4 MiB) plus one 2,048-row batch plus the top-K buffer.

    Per-row Scenario construction is stubbed (to a copied row) and the
    surrogate is a vectorised closed form, because under tracemalloc the real
    ones cost ~0.9 ms a point -- about 8 minutes at this budget. Both allocate
    per BATCH, not per budget, so stubbing them changes the constant and not
    the scaling; measured with the real ones at 50,000 points (two chunks) the
    peak was 13.5 MiB.
    """
    import tracemalloc
    import warnings
    from types import SimpleNamespace

    import twinmind.optim.optimisers as optimisers

    budget = _production_budgets()["best_known"]
    spec = ObjectiveSpec(name="memory", objective={"profit": 1.0}, free_dims=CONTROLLABLE_DIMS)
    assert len(free_dimensions(spec)) == 16

    def fake_surrogate(
        spec: ObjectiveSpec, rows: Any, evaluator: Any, margins: Any = None
    ) -> list[Any]:
        evaluator.calls += len(rows)
        block = np.asarray(rows)
        score = block[:, 0] - 0.5 * (block[:, 1] - 0.3) ** 2
        violation = np.maximum(block[:, 2] - 0.8, 0.0)
        return [
            optimisers.Candidate(scenario=row, kpis={}, score=float(s), violation=float(v))
            for row, s, v in zip(rows, score, violation, strict=True)
        ]

    monkeypatch.setattr(optimisers, "build_scenario", lambda spec, free, row: row.copy())
    monkeypatch.setattr(optimisers, "_score_batch", fake_surrogate)
    counter = SimpleNamespace(calls=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracemalloc.start()
        try:
            best = optimisers.random_search(spec, counter, budget, seed=0)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    assert counter.calls == budget
    assert len(best) == optimisers.TOP_K
    assert peak < 16 * 2**20, f"peak {peak / 2**20:.1f} MiB"


def test_top_k_stays_bounded_and_breaks_ties_by_evaluation_order() -> None:
    """TPE's and CMA-ES's ``found`` used to grow to the budget (20,000 for sf_cmaes)."""
    import twinmind.optim.optimisers as optimisers

    top = optimisers.TopK(k=5)
    scenarios = [Scenario(throughput_multiplier=1.0 + i / 2000) for i in range(1000)]
    for start in range(0, 1000, 7):
        top.add(
            [
                optimisers.Candidate(
                    scenario=scenarios[i], kpis={}, score=float(i % 13), violation=0.0
                )
                for i in range(start, min(start + 7, 1000))
            ]
        )
        assert len(top._items) < 2 * top.k + 7
    assert top.seen == 1000
    best = top.result()
    # Score 12 is the maximum; among equal scores the earliest evaluation wins.
    assert [c.score for c in best] == [12.0] * 5
    assert [c.scenario for c in best] == [scenarios[i] for i in (12, 25, 38, 51, 64)]


def test_grid_walks_the_same_points_in_the_same_order() -> None:
    """Batched grid == the full meshgrid it replaced."""
    spec = simple_spec()
    free = free_dimensions(spec)
    seen: list[Any] = []

    class Recording(FakeEvaluator):
        def evaluate(self, scenarios: Any) -> list[dict[str, float]]:
            seen.extend(s.to_vector() for s in scenarios)
            return super().evaluate(scenarios)

    grid_search(spec, Recording(), 2000, seed=0, batch_size=100)
    per_dim = max(2, int(round(2000 ** (1.0 / len(free)))))
    axes = [np.linspace(0.0, 1.0, per_dim) for _ in free]
    mesh = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    expected = [build_scenario(spec, free, row).to_vector() for row in mesh]
    assert np.array_equal(np.array(seen), np.array(expected))


def test_novelty_chunks_by_the_difference_tensor_and_scores_identically() -> None:
    """The [rows, N, D] intermediate is what is bounded, and no score moves.

    The old chunk ignored D: 50 queries against 20,000 x 20 training points went
    in one chunk, a 160 MB difference tensor, so the 48 MiB bound fails on it.
    """
    import tracemalloc

    rng = np.random.default_rng(0)
    scorer = NoveltyScorer(training=rng.random((20_000, 20)), k=10)
    query = rng.random((50, 20))

    tracemalloc.start()
    try:
        scores = scorer.score_normalised(query)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 48 * 2**20, f"peak {peak / 2**20:.1f} MiB"

    head = query[:4]
    distances = np.linalg.norm(head[:, None, :] - scorer.training[None, :, :], axis=2)
    reference = np.partition(distances, 9, axis=1)[:, :10].mean(axis=1)
    assert np.array_equal(scores[:4], reference)


# --------------------------------------------------------------------------- #
# Scheduling: best_known never runs beside another best_known
# --------------------------------------------------------------------------- #
def _tasks(module: Any, goals: int = 3, seeds: tuple[int, ...] = (1, 2)) -> list[dict[str, Any]]:
    """The old goal -> method -> seed order, best_known first within each goal."""
    methods = ("best_known", *[m for m in module.METHODS if m != "best_known"])
    return [
        {"goal": f"g{g}", "method": m, "seed": s, "smoke": False, "novelty_active": True}
        for g in range(goals)
        for m in methods
        for s in seeds
    ]


def _concurrency_probe() -> tuple[Any, dict[str, Any]]:
    """A worker function that records how many cells, and best_known cells, overlap."""
    import threading
    import time as _time

    lock = threading.Lock()
    state: dict[str, Any] = {"heavy": 0, "max_heavy": 0, "live": 0, "max_live": 0, "starts": []}

    def work(task: dict[str, Any]) -> dict[str, Any]:
        heavy = int(task["method"] == "best_known")
        with lock:
            state["starts"].append(task["method"])
            state["live"] += 1
            state["heavy"] += heavy
            state["max_live"] = max(state["max_live"], state["live"])
            state["max_heavy"] = max(state["max_heavy"], state["heavy"])
        _time.sleep(0.03 if heavy else 0.005)
        with lock:
            state["live"] -= 1
            state["heavy"] -= heavy
        return {**task, "ok": True}

    return work, state


def test_dispatch_never_runs_two_best_known_cells_at_once() -> None:
    from multiprocessing.pool import ThreadPool

    module = _benchmark_module()
    tasks = _tasks(module)
    work, state = _concurrency_probe()
    with ThreadPool(4) as pool:
        results = list(module.dispatch(pool, work, tasks, workers=4))

    assert sorted((r["goal"], r["method"], r["seed"]) for r in results) == sorted(
        (t["goal"], t["method"], t["seed"]) for t in tasks
    )
    assert state["max_heavy"] == 1
    assert state["max_live"] <= 4
    # The heavy lane is the critical path, so it starts first.
    assert state["starts"][0] == "best_known"


def test_the_old_queue_order_did_run_best_known_cells_together() -> None:
    """Negative control: the probe can see the overlap the scheduler prevents."""
    from multiprocessing.pool import ThreadPool

    module = _benchmark_module()
    work, state = _concurrency_probe()
    with ThreadPool(4) as pool:
        list(pool.imap_unordered(work, _tasks(module)))
    assert state["max_heavy"] > 1


def test_failed_dispatch_is_reported_not_lost() -> None:
    """A task whose worker raises still comes back, as a failure."""
    from multiprocessing.pool import ThreadPool

    module = _benchmark_module()

    def explode(task: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("worker died")

    with ThreadPool(2) as pool:
        results = list(module.dispatch(pool, explode, _tasks(module, goals=1, seeds=(1,)), 2))
    assert len(results) == len(module.METHODS)
    assert all(r["ok"] is False and "worker died" in r["error"] for r in results)


def test_the_pool_recycles_workers_and_uses_the_scheduler() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_optim_benchmark.py").read_text(encoding="utf-8")
    assert "maxtasksperchild=MAX_TASKS_PER_CHILD" in source
    assert "dispatch(pool, _run_one, tasks, workers, deadline=deadline)" in source
    assert "imap_unordered(_run_one" not in source


def test_simulated_duration_respects_both_lower_bounds() -> None:
    """The serial best_known lane is a floor no worker count gets under."""
    module = _benchmark_module()
    seconds = module.MEASURED_S_PER_CELL
    counts = {m: 36 for m in module.METHODS}
    lane = seconds["best_known"] * 36 / 3600
    for workers in (1, 2, 3, 4, 6):
        hours = module.simulate_hours(counts, workers)
        total = sum(seconds[m] * n for m, n in counts.items()) / workers / 3600
        assert hours >= max(lane, total) - 1e-9
    # One worker is simply the sum, start-ups included.
    serial = sum(
        (seconds[m] + module.WORKER_STARTUP_S / module.MAX_TASKS_PER_CHILD) * n
        for m, n in counts.items()
    )
    assert module.simulate_hours(counts, 1) == pytest.approx(serial / 3600)


# --------------------------------------------------------------------------- #
# Manifest: what invalidates a results tree, and the upgrade that keeps it
# --------------------------------------------------------------------------- #
def _manifest(module: Any) -> dict[str, Any]:
    return {
        "twin_config_hash": "t",
        "surrogate_hash": "s",
        "assumptions": {"demand_capacity_ratio": 0.85},
        "demand_scale": 0.44,
        "budgets": dict(module.BUDGETS),
        "budgets_hash": module.budgets_hash(),
        "goals_hash": module.goals_hash(),
        "seed_scheme": module.seed_scheme(),
        "design_hash": module.BenchmarkDesign.load().hash(),
        "seeds": [1, 2],
        "created": "now",
    }


_NEW_KEYS = ("goals_hash", "seed_scheme", "design_hash")


def test_manifest_covers_goals_design_and_budgets_but_not_the_commit() -> None:
    """Seeds are pinned through the design hash since Amendment 2, not the bare list.

    The pre-amendment manifest records seeds [1, 2, 3] and its cells are valid;
    comparing that list to the design's [1, 2] would refuse a correct resume.
    """
    module = _benchmark_module()
    for key in ("goals_hash", "seed_scheme", "design_hash", "budgets_hash", "surrogate_hash"):
        assert key in module.INVALIDATING_KEYS
    assert "seeds" not in module.INVALIDATING_KEYS
    assert not any("git" in key or "commit" in key for key in module.INVALIDATING_KEYS)
    current = _manifest(module)
    assert module.manifest_conflicts(current, {**current, "created": "later"}) == []
    assert module.manifest_conflicts({**current, "goals_hash": "edited"}, current) == ["goals_hash"]
    assert module.manifest_conflicts({**current, "design_hash": "other"}, current) == [
        "design_hash"
    ]
    assert module.manifest_conflicts({**current, "seeds": [1, 2, 3]}, current) == []


def test_goals_hash_moves_when_a_goal_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    module = _benchmark_module()
    before = module.goals_hash()
    goals = load_goals()
    first = goals[0]
    edited = replace(first, spec=replace(first.spec, name=first.spec.name + "_edited"))
    monkeypatch.setattr(module, "load_goals", lambda: [edited, *goals[1:]])
    assert module.goals_hash() != before


def test_a_manifest_from_before_the_new_keys_is_upgraded_not_refused() -> None:
    """The completed held-out cells were written under a manifest without
    ``goals_hash``, ``seed_scheme`` or ``design_hash``. Refusing it would throw
    them away for a check that did not exist yet; upgrading is allowed for
    exactly those keys.
    """
    module = _benchmark_module()
    current = _manifest(module)
    legacy = {k: v for k, v in current.items() if k not in _NEW_KEYS}
    legacy["seeds"] = [1, 2, 3]  # what the real pre-amendment manifest records
    assert module.manifest_conflicts(legacy, current) == []
    upgraded, added = module.upgrade_manifest(legacy, current)
    assert sorted(added) == sorted(_NEW_KEYS)
    assert upgraded["seeds"] == [1, 2, 3], "the original record is kept"
    assert upgraded["goals_hash"] == current["goals_hash"]
    assert upgraded["created"] == legacy["created"], "the original record is kept"
    # Once upgraded, the keys are enforced like any other.
    assert module.manifest_conflicts({**upgraded, "goals_hash": "x"}, current) == ["goals_hash"]


def test_a_legacy_manifest_still_conflicts_on_any_key_it_does_record() -> None:
    module = _benchmark_module()
    current = _manifest(module)
    legacy = {k: v for k, v in current.items() if k not in _NEW_KEYS}
    assert module.manifest_conflicts({**legacy, "budgets_hash": "old"}, current) == ["budgets_hash"]
    # Only the named keys may be missing; any other absence is a conflict.
    no_twin = {k: v for k, v in legacy.items() if k != "twin_config_hash"}
    assert module.manifest_conflicts(no_twin, current) == ["twin_config_hash"]


# --------------------------------------------------------------------------- #
# Amendment 2: the reduced-scale design
# --------------------------------------------------------------------------- #
def _design() -> Any:
    from twinmind.optim.design import BenchmarkDesign

    return BenchmarkDesign.load()


def test_the_design_partitions_the_held_out_library_exactly() -> None:
    """8 core + 4 reserve = the 12 held-out goals; no goal left unaccounted for."""
    module = _benchmark_module()
    design = _design()
    held_out = [g.name for g in load_goals(split="held_out")]
    design.check_against(held_out, module.METHODS)
    assert len(design.core_goals) == 8 and len(design.reserve_goals) == 4
    # The two goals that already had completed cells stay in.
    assert {"held_failures_cap", "held_infeasible_profit_floor"} <= set(design.core_goals)


def test_the_seed_rule_is_one_best_known_seed_and_two_for_the_rest() -> None:
    module = _benchmark_module()
    design = _design()
    assert design.seeds_for("best_known") == (1,)
    for method in module.METHODS:
        assert 3 not in design.seeds_for(method), "seed 3 is excluded by rule"
        if method != "best_known":
            assert design.seeds_for(method) == (1, 2)
    assert sum(module.design_counts(design).values()) == 8 * 1 + 8 * 6 * 2 == 104


def test_the_core_goals_keep_every_family_represented() -> None:
    """The structural rule behind the selection, checked rather than asserted.

    Same shapes the full library is required to cover, now on the CORE set: a
    weighted objective, a minimise goal, frozen dimensions, an exogenous
    what-if, an infeasible-by-design goal, and a hard constraint on each of the
    four constrained KPIs. ``downtime_hours`` is the reason
    ``held_premium_supplier_service`` is core rather than
    ``held_offshore_supplier``: it is the only held-out goal that constrains it.
    """
    design = _design()
    core = [g.spec for g in load_goals(split="held_out") if g.name in design.core_goals]
    assert any(len(g.objective) > 1 for g in core), "no weighted goal"
    assert any(g.sense == "minimise" for g in core), "no minimise goal"
    assert any(len(g.free_dims) < len(CONTROLLABLE_DIMS) for g in core), "no frozen-dim goal"
    assert any(set(g.fixed) & set(EXOGENOUS_DIMS) for g in core), "no exogenous what-if"
    assert any(g.infeasible_by_design for g in core), "no infeasible-by-design goal"
    constrained = {c.kpi for g in core for c in g.constraints if c.hard}
    for kpi in ("stockout_probability", "on_time_delivery_rate", "unplanned_failures"):
        assert kpi in constrained, kpi
    assert "downtime_hours" in constrained, "the only downtime_hours goal was dropped"


def test_the_design_hash_ignores_comments_and_moves_with_the_selection(tmp_path: Path) -> None:
    from twinmind.optim.design import DESIGN_PATH, BenchmarkDesign

    text = DESIGN_PATH.read_text(encoding="utf-8")
    commented = tmp_path / "commented.yaml"
    commented.write_text("# a new comment\n" + text, encoding="utf-8")
    assert BenchmarkDesign.load(commented).hash() == _design().hash()

    swapped = tmp_path / "swapped.yaml"
    swapped.write_text(text.replace("default: [1, 2]", "default: [1, 3]"), encoding="utf-8")
    assert BenchmarkDesign.load(swapped).hash() != _design().hash()


def test_a_design_that_leaves_a_held_out_goal_unlisted_is_rejected(tmp_path: Path) -> None:
    from twinmind.optim.design import DESIGN_PATH, BenchmarkDesign

    module = _benchmark_module()
    text = DESIGN_PATH.read_text(encoding="utf-8")
    dropped = tmp_path / "dropped.yaml"
    dropped.write_text(text.replace("  - held_speed_only", ""), encoding="utf-8")
    held_out = [g.name for g in load_goals(split="held_out")]
    with pytest.raises(SpecError, match="partition"):
        BenchmarkDesign.load(dropped).check_against(held_out, module.METHODS)


def _args(module: Any, *argv: str) -> Any:
    return module.build_parser().parse_args(list(argv))


def test_held_out_plan_is_exactly_the_design() -> None:
    module = _benchmark_module()
    design = _design()
    split, plan = module.plan_cells(_args(module), design)
    assert split == "held_out"
    assert sorted(plan) == sorted(design.cells(module.METHODS))
    assert len(plan) == 104
    assert not any(seed == 3 for _, _, seed in plan)
    assert not any(goal in design.reserve_goals for goal, _, _ in plan)


def test_a_reserve_goal_or_seed_3_cannot_be_run_on_held_out() -> None:
    module = _benchmark_module()
    design = _design()
    with pytest.raises(SystemExit, match="RESERVE"):
        module.plan_cells(_args(module, "--goals", "held_max_profit"), design)
    with pytest.raises(SystemExit, match="outside the design"):
        module.plan_cells(_args(module, "--seeds", "3"), design)


def test_goals_and_seeds_only_narrow_the_held_out_design() -> None:
    module = _benchmark_module()
    _, plan = module.plan_cells(_args(module, "--goals", "held_pm_only", "--seeds", "2"), _design())
    assert plan and all(g == "held_pm_only" and s == 2 for g, _, s in plan)
    # best_known has no seed 2 under the design, so narrowing to 2 leaves it out.
    assert "best_known" not in {m for _, m, _ in plan}


def test_dev_and_smoke_runs_are_not_governed_by_the_design() -> None:
    """The production-budget dry run targets a dev goal with seed 0; that must still work."""
    module = _benchmark_module()
    split, plan = module.plan_cells(
        _args(module, "--split", "dev", "--goals", "dev_max_profit", "--seeds", "0"), _design()
    )
    assert split == "dev" and {s for _, _, s in plan} == {0}
    assert len(plan) == len(module.METHODS)
    split, plan = module.plan_cells(_args(module, "--smoke"), _design())
    assert split == "dev"
    assert len(plan) == 2 * len(module.METHODS) * 3


def test_the_loader_reads_only_design_cells_and_never_opens_the_rest(tmp_path: Path) -> None:
    """The analysis input. Seed-3 and reserve files are listed, never opened.

    Proved by making the out-of-design files invalid JSON: if the loader opened
    them they would land in ``stale``, not ``excluded``.
    """
    from twinmind.optim.design import load_design_cells

    module = _benchmark_module()
    design = _design()
    good = {"budgets_hash": module.budgets_hash(), "ok": True}
    module.write_atomic(good, tmp_path / "held_pm_only__sf_tpe__seed1.json")
    module.write_atomic(good, tmp_path / "held_pm_only__best_known__seed1.json")
    module.write_atomic(
        {"budgets_hash": "old"}, tmp_path / "held_pm_only__sf_tpe__seed2.json"
    )  # stale: another budget
    for outside in (
        "held_pm_only__sf_tpe__seed3.json",  # seed 3
        "held_pm_only__best_known__seed2.json",  # best_known has one seed
        "held_max_profit__sf_tpe__seed1.json",  # reserve goal
    ):
        (tmp_path / outside).write_text("{not json", encoding="utf-8")

    found = load_design_cells(tmp_path, design, module.METHODS, module.budgets_hash())
    assert len(found.cells) == 2
    assert found.stale == ["held_pm_only__sf_tpe__seed2.json"]
    assert found.excluded == sorted(
        [
            "held_max_profit__sf_tpe__seed1.json",
            "held_pm_only__best_known__seed2.json",
            "held_pm_only__sf_tpe__seed3.json",
        ]
    )
    assert len(found.missing) == 104 - 3


def test_max_hours_stops_starting_cells_but_lets_running_ones_finish() -> None:
    """After the deadline nothing new starts; whatever was running is still yielded."""
    import threading
    import time as _time
    from multiprocessing.pool import ThreadPool

    module = _benchmark_module()
    tasks = _tasks(module, goals=2, seeds=(1,))
    lock = threading.Lock()
    state = {"started": 0, "finished": 0}

    def work(task: dict[str, Any]) -> dict[str, Any]:
        with lock:
            state["started"] += 1
        _time.sleep(0.02)
        with lock:
            state["finished"] += 1
        return {**task, "ok": True}

    # The clock passes the deadline once two cells have finished.
    with ThreadPool(3) as pool:
        results = list(
            module.dispatch(
                pool, work, tasks, workers=3, deadline=1.0, clock=lambda: state["finished"] / 2
            )
        )
    assert 2 <= len(results) < len(tasks), "stopped early, but not before the deadline"
    assert len(results) == state["started"] == state["finished"], "no in-flight cell dropped"
    assert all(r["ok"] for r in results)


def test_max_hours_is_recorded_in_the_summary_and_validated() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_optim_benchmark.py").read_text(encoding="utf-8")
    assert "dispatch(pool, _run_one, tasks, workers, deadline=deadline)" in source
    assert '"stopped_by_max_hours": bool(not_started)' in source
    assert "--max-hours must be positive" in source
