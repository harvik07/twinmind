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
    module.write_atomic({"a": 1}, path)
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
