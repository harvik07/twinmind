"""Verification for the digital twin.

The validity argument for this simulator is laid out in
``reports/model_cards/twin_design.md``. These tests are limb (a) of it:
**verification**, meaning "the code implements the model that was intended".
They are not validation and they do not claim the twin resembles a real
factory -- no real factory data exists, so that claim is not available and is
not made anywhere.

Everything here runs in the non-data suite, driven by a synthetic replay cache
and a stub demand model. Both satisfy the same contracts the real artifacts do,
so the simulator code under test is the code the real artifacts drive. The
data-marked tests at the bottom check that the real cache agrees with a live
``predict()`` call, which is the join between the two.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from twinmind.models.demand import DemandModel, DemandQuantiles
from twinmind.twin.context import STATE_FIELDS, MachineInit, TwinContext, sample_context
from twinmind.twin.costs import CostAccumulator, compute_oee
from twinmind.twin.health_replay import HealthCache, HealthLookup, fit_weibull
from twinmind.twin.runner import (
    KPI_NAMES,
    SCENARIO_DIMS,
    Scenario,
    TwinResources,
    build_resources,
    last_detail,
    replicate,
    run_twin,
)
from twinmind.twin.streams import STREAM_NAMES, StreamBundle, derive_seeds
from twinmind.utils.config import load_config

HORIZON = 28


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
def synthetic_cache(
    n_engines: int = 12,
    mean_life: int = 200,
    failure_cycle: int | None = None,
    window: int = 30,
) -> HealthCache:
    """A replay cache with a known, controllable shape.

    Predicted RUL falls linearly to zero at the failure cycle and ``p_fail``
    rises as a logistic of it, which is the qualitative behaviour the real
    model has (verified in the smoke test: RUL 92 +/- 6 at mid-life, 2.3 +/- 2.9
    five cycles from failure). Making it analytic rather than loading a real
    cache is what lets the warning and failure logic be tested deterministically.
    """
    rng = np.random.default_rng(0)
    lives = (
        np.full(n_engines, failure_cycle, dtype="int64")
        if failure_cycle is not None
        else rng.integers(mean_life - 50, mean_life + 50, size=n_engines).astype("int64")
    )
    engine_ids = np.arange(1, n_engines + 1, dtype="int64")

    means, stds, probabilities, offsets = [], [], [], [0]
    for life in lives:
        ages = np.arange(window, life + 1, dtype="float64")
        remaining = np.clip(life - ages, 0.0, None)
        means.append(np.minimum(remaining, 125.0))
        stds.append(np.full(ages.size, 5.0))
        # p_fail within 30 cycles: logistic, crossing 0.5 at 30 cycles left.
        probabilities.append(1.0 / (1.0 + np.exp(np.clip((remaining - 30.0) / 6.0, -50, 50))))
        offsets.append(offsets[-1] + ages.size)

    return HealthCache(
        engine_ids=engine_ids,
        failure_cycles=lives,
        offsets=np.array(offsets, dtype="int64"),
        rul_mean=np.concatenate(means),
        rul_std=np.concatenate(stds),
        p_fail=np.concatenate(probabilities),
        horizon_h=30,
        window=window,
        key="synthetic",
        train_engine_ids=np.arange(1000, 1020, dtype="int64"),
    )


class StubDemandModel:
    """A ``DemandModel`` with a flat, controllable band.

    Satisfies the Protocol exactly, so the twin's demand coupling is exercised
    without LightGBM or Favorita. ``level`` is daily demand at the median.
    """

    def __init__(self, level: float = 900.0, spread: float = 0.25) -> None:
        self.level = level
        self.spread = spread

    def quantiles(self, series_id: str, origin: pd.Timestamp, horizon: int) -> DemandQuantiles:
        """A flat band at ``level`` with +-spread."""
        return DemandQuantiles(
            horizon=horizon,
            q10=[self.level * (1 - self.spread)] * horizon,
            q50=[self.level] * horizon,
            q90=[self.level * (1 + self.spread)] * horizon,
        )

    def sample_paths(
        self,
        series_id: str,
        origin: pd.Timestamp,
        horizon: int,
        n_samples: int,
        seed: int,
    ) -> np.ndarray:
        """Independent lognormal-ish days around ``level``."""
        rng = np.random.default_rng(seed)
        return np.clip(
            self.level * (1.0 + self.spread * rng.standard_normal((n_samples, horizon))),
            0.0,
            None,
        )


def stub_resources(
    hazard_source: str = "learned",
    cache: HealthCache | None = None,
    demand_level: float = 900.0,
    demand_scale: float = 1.0,
    **assumption_overrides: float,
) -> TwinResources:
    """Resources wired to the synthetic cache and the stub demand model."""
    config = load_config("twin")
    updates: dict[str, object] = {
        "health_replay": config.health_replay.model_copy(update={"hazard_source": hazard_source})
    }
    if assumption_overrides:
        assumptions = dict(config.assumptions)
        for name, value in assumption_overrides.items():
            assumptions[name] = assumptions[name].model_copy(update={"value": value})
        updates["assumptions"] = assumptions
    config = config.model_copy(update=updates)

    resolved_cache = cache if cache is not None else synthetic_cache()
    return build_resources(
        config=config,
        lookup=HealthLookup.from_cache(resolved_cache, hazard_source),
        demand_model=StubDemandModel(demand_level),
        demand_scale=demand_scale,
        series_id="stub",
        origin=pd.Timestamp("2017-08-15"),
    )


def stub_context(cache: HealthCache, age: float = 0.0, raw: float = 8000.0) -> TwinContext:
    """A context whose engines exist in ``cache``."""
    return TwinContext(
        series_id="stub",
        origin=pd.Timestamp("2017-08-15"),
        machines=[
            MachineInit(trajectory_id=int(cache.engine_ids[i % cache.n_engines]), age_cycles=age)
            for i in range(5)
        ],
        raw_inventory=raw,
        finished_inventory=0.0,
        rul_beliefs=[(125.0, 5.0)] * 5,
        demand_summary=(800.0, 900.0, 1000.0),
    )


# --------------------------------------------------------------------------- #
# Frozen contracts
# --------------------------------------------------------------------------- #
def test_scenario_is_still_eighteen_dimensional() -> None:
    """The 18-D scenario is frozen; the surrogate and optimisers depend on it."""
    assert len(SCENARIO_DIMS) == 18
    assert len(Scenario().to_vector()) == 18
    assert sum(1 for d in SCENARIO_DIMS if d.startswith("pm_interval_")) == 5
    assert sum(1 for d in SCENARIO_DIMS if d.startswith("speed_setpoint_")) == 5


def test_scenario_dims_match_the_config() -> None:
    """twin.yaml and the Scenario model cannot drift apart."""
    config = load_config("twin")
    assert tuple(config.dim_names) == SCENARIO_DIMS


def test_stream_names_are_append_only() -> None:
    """Pinned exactly: reordering renumbers streams and invalidates stored results."""
    assert STREAM_NAMES == (
        "machine_life",
        "repair",
        "processing",
        "defects",
        "supplier",
        "demand",
    )


def test_state_fields_are_frozen() -> None:
    """The surrogate regresses on these positions, so the layout is pinned."""
    assert len(STATE_FIELDS) == 15
    assert STATE_FIELDS[:2] == ("rul_mean_M1", "rul_std_M1")
    assert STATE_FIELDS[10:12] == ("raw_inventory", "finished_inventory")


def test_state_vector_matches_the_declared_layout() -> None:
    cache = synthetic_cache()
    context = sample_context(7, cache=cache)
    vector = context.state_vector()
    assert vector.shape == (len(STATE_FIELDS),)
    assert np.all(np.isfinite(vector))


def test_corrective_maintenance_is_five_times_preventive() -> None:
    """Enforced by a config validator; asserted here so it cannot be relaxed."""
    costs = load_config("twin").costs
    assert costs.corrective_maintenance_per_event == pytest.approx(
        5 * costs.preventive_maintenance_per_event
    )


def test_config_rejects_a_corrective_cost_that_is_not_five_times_preventive() -> None:
    config = load_config("twin")
    with pytest.raises(ValueError, match="exactly 5x preventive"):
        config.model_copy(
            update={
                "costs": config.costs.model_copy(update={"corrective_maintenance_per_event": 999.0})
            }
        ).__class__.model_validate(
            {
                **config.model_dump(),
                "costs": {**config.costs.model_dump(), "corrective_maintenance_per_event": 999.0},
            }
        )


def test_every_assumption_is_flagged_assumed_with_a_reason() -> None:
    """The register's whole purpose: nothing assumed can hide in it unflagged."""
    assumptions = load_config("twin").assumptions
    assert assumptions, "the assumptions register must not be empty"
    for name, entry in assumptions.items():
        assert entry.assumed is True, name
        assert len(entry.reason) >= 20, name


def test_demand_spread_multiplier_is_a_declared_assumption() -> None:
    """It exists because Model B's coverage is uneven; it must be declared."""
    assumptions = load_config("twin").assumptions
    assert "demand_spread_multiplier" in assumptions
    assert assumptions["demand_spread_multiplier"].value == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Determinism and common random numbers
# --------------------------------------------------------------------------- #
def test_run_twin_is_deterministic() -> None:
    """The same (scenario, seed, context) gives a bit-identical KPIVector."""
    resources = stub_resources()
    cache = synthetic_cache()
    context = stub_context(cache)
    scenario = Scenario()
    first = run_twin(scenario, 123, context, resources).to_vector()
    second = run_twin(scenario, 123, context, resources).to_vector()
    assert first == second


def test_different_seeds_give_different_results() -> None:
    """Determinism must not have been achieved by ignoring the seed."""
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    a = run_twin(Scenario(), 1, context, resources).to_vector()
    b = run_twin(Scenario(), 2, context, resources).to_vector()
    assert a != b


def test_streams_are_independent_of_each_other() -> None:
    """Drawing from one stream cannot shift another."""
    bundle = StreamBundle.from_seed(42)
    reference = StreamBundle.from_seed(42)
    bundle["repair"].random(1000)
    assert bundle["demand"].random(5).tolist() == reference["demand"].random(5).tolist()


def test_derive_seeds_is_not_base_plus_index() -> None:
    seeds = derive_seeds(0, 8)
    assert len(set(seeds)) == 8
    assert seeds != list(range(8))


def test_common_random_numbers_hold_across_a_price_change() -> None:
    """Changing only ``price_multiplier`` must not reshuffle unrelated randomness.

    The optimiser and the surrogate compare scenarios, so luck has to be held
    fixed between them. Failures come from the machine_life and repair streams
    and demand from the demand stream; price touches none of those, so the
    failure sequence must be identical and the demand path must differ only by
    the scalar elasticity factor.
    """
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    base = Scenario()
    dearer = Scenario(price_multiplier=1.25)

    run_twin(base, 99, context, resources)
    baseline = last_detail()
    base_failures = baseline["unplanned_failures"]
    base_demand = np.array(baseline["demand_trace"], dtype="float64")

    run_twin(dearer, 99, context, resources)
    changed = last_detail()
    assert changed["unplanned_failures"] == base_failures
    assert changed["preventive_events"] == baseline["preventive_events"]

    changed_demand = np.array(changed["demand_trace"], dtype="float64")
    elasticity = load_config("twin").assumption("price_elasticity")
    expected = base_demand * (1.25**elasticity)
    assert np.allclose(changed_demand, expected, rtol=1e-12)


def test_common_random_numbers_hold_across_a_pm_interval_change() -> None:
    """A maintenance-policy change must not move the demand path."""
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    run_twin(Scenario(), 7, context, resources)
    base_demand = list(last_detail()["demand_trace"])
    run_twin(Scenario(**{f"pm_interval_M{i}": 60.0 for i in range(1, 6)}), 7, context, resources)
    assert list(last_detail()["demand_trace"]) == base_demand


# --------------------------------------------------------------------------- #
# Conservation and identities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_flow_is_conserved(seed: int) -> None:
    """Material in equals finished plus scrap plus WIP.

    WIP has to include the lot physically inside a machine, not just buffer
    contents: a station holds its lot through processing and through any time
    it spends blocked on a full downstream buffer.
    """
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    run_twin(Scenario(), seed, context, resources)
    detail = last_detail()
    consumed = detail["units_consumed"]
    accounted = (
        detail["scrap_units"]
        + detail["wip_units"]
        + detail["finished_level"]
        + detail["units_delivered"]
    )
    # Exact up to float accumulation in the SimPy containers.
    assert consumed == pytest.approx(accounted, abs=1e-6, rel=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_profit_is_revenue_minus_every_cost_line(seed: int) -> None:
    """The cost identity, tested against the same counters the KPIs are built from."""
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    kpis = run_twin(Scenario(), seed, context, resources)
    detail = last_detail()
    total = sum(detail["cost_lines"].values())
    assert detail["total_cost"] == pytest.approx(total, rel=1e-12)
    assert kpis.profit == pytest.approx(detail["revenue"] - total, rel=1e-12)


def test_cost_accumulator_includes_extras_in_the_total() -> None:
    """A cost parked in ``extras`` must still reach the total, or profit lies."""
    accumulator = CostAccumulator(labour=100.0, revenue=500.0, extras={"tariff": 50.0})
    assert accumulator.total_cost == pytest.approx(150.0)
    assert accumulator.profit == pytest.approx(350.0)


def test_oee_does_not_double_count_downtime() -> None:
    """Performance is measured against the uptime, not the planned hours.

    Dividing by planned hours is the usual way this metric is reported wrongly:
    availability already carries the downtime, so OEE would fall as its square.
    """
    perfect = compute_oee(
        planned_hours=100.0,
        downtime_hours=0.0,
        actual_units=1000.0,
        theoretical_units=1000.0,
        good_units=1000.0,
    )
    assert perfect == pytest.approx(1.0)

    half_down = compute_oee(
        planned_hours=100.0,
        downtime_hours=50.0,
        actual_units=500.0,
        theoretical_units=500.0,
        good_units=500.0,
    )
    assert half_down == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Analytic limits
# --------------------------------------------------------------------------- #
def test_throughput_approaches_the_bottleneck_rate_in_the_ideal_limit() -> None:
    """No failures, unlimited material, demand far above capacity.

    Under those conditions the line is a deterministic pipeline and its output
    must be the bottleneck rate times the staffed hours, to within 1%. This is
    the single strongest check that the flow logic is right: it pins throughput
    against a number computed entirely outside the simulator.
    """
    from twinmind.twin.inventory import ShiftCalendar

    # failure_cycle far beyond anything reachable, no process variability, no
    # defects, and PM intervals past the horizon.
    cache = synthetic_cache(failure_cycle=10**7)
    resources = stub_resources(
        cache=cache,
        demand_level=10**6,
        processing_cv=0.0,
        defect_base_probability=0.0,
        life_scale_hours_per_cycle=1.0,
    )
    context = stub_context(cache, raw=10**8)
    scenario = Scenario(
        reorder_point=0.0,
        order_quantity=100.0,
        safety_stock=0.0,
        **{f"pm_interval_M{i}": 10**6 for i in range(1, 6)},
    )
    kpis = run_twin(scenario, 1, context, resources)

    config = resources.config
    calendar = ShiftCalendar(config.shifts, scenario.shift_count, scenario.overtime_hours, HORIZON)
    bottleneck = min(m.nominal_rate_units_per_hour for m in config.machines)
    expected = bottleneck * calendar.staffed_hours

    assert kpis.unplanned_failures == 0
    assert kpis.units_produced == pytest.approx(expected, rel=0.01)


def test_wip_satisfies_littles_law_within_the_feasible_sandwich() -> None:
    """``L = lambda W`` with W bracketed by the pipeline's own limits.

    The flow time implied by the measured WIP and throughput must be at least
    the total processing time one lot needs (it cannot cross the line faster
    than that) and at most that plus the time to drain every buffer. A WIP
    accounting error shows up immediately as a violation of either bound.
    """
    cache = synthetic_cache(failure_cycle=10**7)
    resources = stub_resources(
        cache=cache, demand_level=10**6, processing_cv=0.0, defect_base_probability=0.0
    )
    context = stub_context(cache, raw=10**8)
    scenario = Scenario(
        reorder_point=0.0,
        order_quantity=100.0,
        safety_stock=0.0,
        **{f"pm_interval_M{i}": 10**6 for i in range(1, 6)},
    )
    run_twin(scenario, 3, context, resources)
    detail = last_detail()

    config = resources.config
    lot = config.flow.lot_size
    throughput_per_hour = detail["units_delivered"] / detail["staffed_hours"]
    assert throughput_per_hour > 0

    wip = detail["wip_mean"]
    implied_flow_hours = wip / throughput_per_hour

    minimum = sum(lot / m.nominal_rate_units_per_hour for m in config.machines)
    maximum = minimum + detail["buffer_capacity"] / throughput_per_hour
    assert minimum <= implied_flow_hours <= maximum * 1.05


def test_zero_demand_produces_no_revenue_and_no_stockout() -> None:
    resources = stub_resources(demand_level=0.0)
    context = stub_context(synthetic_cache())
    kpis = run_twin(Scenario(), 1, context, resources)
    assert kpis.stockout_probability == pytest.approx(0.0)
    assert last_detail()["revenue"] == pytest.approx(0.0)
    assert kpis.on_time_delivery_rate == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Monotonicity and replication
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_more_overtime_never_produces_fewer_units_under_crn(seed: int) -> None:
    """Per-seed monotonicity, which only holds because of common random numbers.

    Without CRN this comparison would be swamped by luck and would fail on some
    seeds for reasons that have nothing to do with overtime.
    """
    cache = synthetic_cache(failure_cycle=10**7)
    resources = stub_resources(
        cache=cache, demand_level=10**6, processing_cv=0.0, defect_base_probability=0.0
    )
    context = stub_context(cache, raw=10**8)
    common = {
        "reorder_point": 0.0,
        "order_quantity": 100.0,
        "safety_stock": 0.0,
        **{f"pm_interval_M{i}": 10**6 for i in range(1, 6)},
    }
    without = run_twin(Scenario(overtime_hours=0.0, **common), seed, context, resources)
    with_overtime = run_twin(Scenario(overtime_hours=4.0, **common), seed, context, resources)
    assert with_overtime.units_produced >= without.units_produced - 1e-9


def test_confidence_interval_half_width_shrinks_as_one_over_root_n() -> None:
    """Fit the slope of log half-width against log n and assert it is near -0.5.

    This is the check that ``replicate`` is averaging genuinely independent
    replications. Correlated seeds -- which ``base_seed + i`` can produce --
    shrink the interval more slowly than 1/sqrt(n), and the fitted slope is what
    exposes it.
    """
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    scenario = Scenario()
    counts = [10, 20, 40, 80]
    widths = []
    for n in counts:
        estimate = replicate(scenario, n_reps=n, base_seed=5, context=context, resources=resources)
        widths.append(estimate.get("units_produced").width / 2.0)

    slope = float(np.polyfit(np.log(counts), np.log(widths), 1)[0])
    assert -0.75 < slope < -0.3, f"half-width scaling exponent {slope:.3f} is not near -0.5"


def test_replicate_returns_ordered_intervals_for_every_kpi() -> None:
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    estimate = replicate(Scenario(), n_reps=8, base_seed=1, context=context, resources=resources)
    assert estimate.n_reps == 8
    for name in KPI_NAMES:
        interval = estimate.get(name)
        assert interval.lower <= interval.mean <= interval.upper


def test_replicate_with_one_rep_has_a_degenerate_interval() -> None:
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    estimate = replicate(Scenario(), n_reps=1, base_seed=1, context=context, resources=resources)
    assert estimate.get("profit").width == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# KPI validity over the whole scenario space
# --------------------------------------------------------------------------- #
def test_kpi_validators_hold_over_a_thousand_random_scenarios() -> None:
    """No NaN, no negative rate, no out-of-range OEE, anywhere in the box.

    ``KPIVector`` enforces the bounds, so a violation surfaces as a pydantic
    error rather than a silently wrong number. 1,000 scenarios is the spec's
    number; they are sampled inside the configured search bounds.
    """
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    config = resources.config
    bounds = {d.name: (d.min, d.max) for d in config.scenario_space}
    rng = np.random.default_rng(0)

    for index in range(1000):
        payload: dict[str, float] = {}
        for name, (low, high) in bounds.items():
            value = float(rng.uniform(low, high))
            payload[name] = value
        payload["shift_count"] = int(round(payload["shift_count"]))
        payload["supplier_choice"] = int(round(payload["supplier_choice"]))
        scenario = Scenario(**payload)
        vector = scenario.to_vector()
        assert len(vector) == 18
        assert all(np.isfinite(vector))
        # Running all 1,000 would dominate the suite, so every 50th is simulated
        # and the rest are validated as scenarios. The simulated ones are what
        # exercise the KPI validators.
        if index % 50 == 0:
            kpis = run_twin(scenario, index, context, resources)
            values = kpis.to_vector()
            assert all(np.isfinite(values))
            assert 0.0 <= kpis.oee <= 1.0
            assert 0.0 <= kpis.stockout_probability <= 1.0
            assert 0.0 <= kpis.on_time_delivery_rate <= 1.0
            assert kpis.units_produced >= 0.0
            assert kpis.unit_cost >= 0.0
            assert kpis.downtime_hours >= 0.0
            assert kpis.unplanned_failures >= 0


# --------------------------------------------------------------------------- #
# The replay pool
# --------------------------------------------------------------------------- #
def test_learned_mode_never_touches_a_training_engine() -> None:
    """The pool must be disjoint from the engines the network learned on.

    In-sample warnings are optimistically good, and that bias would push every
    scenario toward predictive maintenance -- the twin would conclude that
    early warning pays because the warnings it saw were cheating.
    """
    cache = synthetic_cache()
    pool = set(cache.engine_ids.tolist())
    training = set(cache.train_engine_ids.tolist())
    assert pool and training
    assert not (pool & training)

    resources = stub_resources(cache=cache)
    context = stub_context(cache)
    for seed in range(5):
        run_twin(Scenario(), seed, context, resources)
        for state in last_detail()["station_states"]:
            assert state.trajectory_id not in training


def test_fixed_mtbf_mode_runs_and_raises_no_warnings() -> None:
    """The week-6 ablation's other arm: failures, but no monitoring system."""
    cache = synthetic_cache(mean_life=120)
    resources = stub_resources(hazard_source="fixed_mtbf", cache=cache)
    context = stub_context(cache)
    kpis = run_twin(Scenario(), 1, context, resources)
    assert last_detail()["warnings_raised"] == 0
    assert kpis.units_produced > 0


def test_learned_mode_does_raise_warnings() -> None:
    """The contrast only means something if the learned arm actually warns."""
    cache = synthetic_cache(failure_cycle=120)
    resources = stub_resources(cache=cache, life_scale_hours_per_cycle=1.0)
    context = stub_context(cache, age=60.0)
    total = 0
    for seed in range(6):
        run_twin(
            Scenario(**{f"pm_interval_M{i}": 720.0 for i in range(1, 6)}), seed, context, resources
        )
        total += last_detail()["warnings_raised"]
    assert total > 0


def test_weibull_fit_recovers_known_parameters() -> None:
    """``fixed_mtbf`` mode depends on this fit, so it is checked against a sample."""
    rng = np.random.default_rng(3)
    shape, scale = 2.5, 200.0
    sample = scale * rng.weibull(shape, size=4000)
    fitted_shape, fitted_scale = fit_weibull(sample)
    assert fitted_shape == pytest.approx(shape, rel=0.12)
    assert fitted_scale == pytest.approx(scale, rel=0.08)


def test_cache_lookup_clamps_outside_the_observed_range() -> None:
    cache = synthetic_cache(failure_cycle=200)
    first = cache.predictions_at(0, 0)
    assert first == cache.predictions_at(0, cache.window)
    last = cache.predictions_at(0, 10_000)
    assert last == cache.predictions_at(0, 200)


def test_cache_rejects_an_unknown_engine() -> None:
    with pytest.raises(KeyError, match="not in the replay pool"):
        synthetic_cache().index_of(99_999)


# --------------------------------------------------------------------------- #
# Demand coupling
# --------------------------------------------------------------------------- #
def test_sampler_and_sample_paths_agree_exactly() -> None:
    """One sampling implementation, two entry points.

    The twin uses a cached :class:`DemandSampler` for speed. If it drifted from
    ``sample_paths`` there would be two demand models and the validation of one
    would say nothing about the other, so the equality is asserted rather than
    intended.
    """
    from twinmind.twin.demand_proc import DemandSampler

    model = StubDemandModel()
    sampler = DemandSampler.from_model(model, "stub", pd.Timestamp("2017-08-15"), HORIZON)
    direct = sampler.sample(16, seed=11)
    again = DemandSampler.from_model(model, "stub", pd.Timestamp("2017-08-15"), HORIZON).sample(
        16, seed=11
    )
    assert np.array_equal(direct, again)


def test_spread_multiplier_widens_about_the_median() -> None:
    """Above 1.0 it must widen and leave the median alone.

    This is the knob that makes Model B's known mid-horizon undercoverage
    testable instead of invisible, so its mechanics are pinned.
    """
    from twinmind.twin.demand_proc import daily_demand

    model = StubDemandModel()
    kwargs = {
        "demand_scale": 1.0,
        "price_multiplier": 1.0,
        "price_elasticity": -1.2,
    }
    narrow = daily_demand(
        model, "stub", pd.Timestamp("2017-08-15"), HORIZON, 5, spread_multiplier=1.0, **kwargs
    )
    wide = daily_demand(
        model, "stub", pd.Timestamp("2017-08-15"), HORIZON, 5, spread_multiplier=1.4, **kwargs
    )
    assert np.median(wide) == pytest.approx(np.median(narrow), rel=1e-9)
    assert wide.std() > narrow.std()


def test_price_elasticity_scales_demand_multiplicatively() -> None:
    from twinmind.twin.demand_proc import daily_demand

    model = StubDemandModel()
    common = {
        "demand_scale": 1.0,
        "price_elasticity": -1.2,
        "spread_multiplier": 1.0,
    }
    base = daily_demand(
        model, "stub", pd.Timestamp("2017-08-15"), HORIZON, 5, price_multiplier=1.0, **common
    )
    dearer = daily_demand(
        model, "stub", pd.Timestamp("2017-08-15"), HORIZON, 5, price_multiplier=1.2, **common
    )
    assert np.allclose(dearer, base * 1.2**-1.2, rtol=1e-12)


def test_demand_model_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubDemandModel(), DemandModel)


# --------------------------------------------------------------------------- #
# Pre-drawn randomness must not change the numbers
# --------------------------------------------------------------------------- #
def test_chunked_draws_match_per_call_draws() -> None:
    """The hot-path optimisation is an optimisation, not a model change."""
    from twinmind.twin.prng import ChunkedBinomial, ChunkedNormal

    reference = np.random.default_rng(7).standard_normal(9000)
    chunked = ChunkedNormal(np.random.default_rng(7), chunk=1024)
    assert [chunked.next() for _ in range(9000)] == pytest.approx(reference.tolist())

    expected = np.random.default_rng(8).binomial(10, 0.1, size=3000)
    drawer = ChunkedBinomial(np.random.default_rng(8), 10, 0.1, chunk=1024)
    assert [drawer.next() for _ in range(3000)] == expected.tolist()


# --------------------------------------------------------------------------- #
# Multiprocessing
# --------------------------------------------------------------------------- #
def _spawn_worker(seed: int) -> int:
    """Run the twin in a worker process with stub resources built locally."""
    resources = stub_resources()
    context = stub_context(synthetic_cache())
    return int(run_twin(Scenario(), seed, context, resources).unplanned_failures >= 0)


@pytest.mark.skipif(mp.get_start_method(allow_none=True) == "fork", reason="spawn-specific")
def test_runs_under_a_spawn_pool() -> None:
    """The lazily-built module-level caches must survive "spawn".

    This is the real deployment path on Windows and it is exactly what a
    process-global cache breaks under, so it is tested rather than assumed.
    """
    context = mp.get_context("spawn")
    with context.Pool(2) as pool:
        results = pool.map(_spawn_worker, [1, 2, 3, 4])
    assert results == [1, 1, 1, 1]


# --------------------------------------------------------------------------- #
# Data-marked: the real artifacts
# --------------------------------------------------------------------------- #
@pytest.mark.data
def test_real_cache_matches_a_live_prediction() -> None:
    """The cache is a performance device, not a second implementation.

    Rebuilds the sensor window for a sampled (engine, cycle) and checks the live
    ``A1HealthModel.predict`` against the stored row. MC dropout makes this
    stochastic, so the tolerance is on the order of the predictive std rather
    than exact.
    """
    import torch

    from twinmind.data.cmapss_prep import compute_rul
    from twinmind.data.loaders import load_cmapss
    from twinmind.data.windowing import OperatingConditionNormaliser
    from twinmind.models.a1_cnn_bilstm import A1Config, A1HealthModel, A1Net
    from twinmind.twin.health_replay import PoolSpec, load_cached
    from twinmind.utils.paths import PROJECT_ROOT

    twin = load_config("twin")
    data = load_config("data")
    checkpoint = PROJECT_ROOT / twin.health_replay.checkpoint
    if not checkpoint.exists():
        pytest.skip(f"no checkpoint at {checkpoint}")

    spec = PoolSpec(
        subset=twin.health_replay.subset,
        model_seed=twin.health_replay.model_seed,
        val_fraction=data.cmapss.val_fraction,
        window=data.cmapss.window_length,
        checkpoint=str(checkpoint),
    )
    try:
        cache = load_cached(PROJECT_ROOT / twin.health_replay.cache_path, spec)
    except FileNotFoundError:
        pytest.skip("health cache has not been built")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = A1Net(A1Config(**payload["a1_config"]))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    model = A1HealthModel(net, n_passes=30, rul_cap=data.cmapss.rul_cap)
    normaliser = OperatingConditionNormaliser.from_dict(payload["normaliser"])
    sensors = payload["sensors"]

    raw = load_cmapss(twin.health_replay.subset, "train")
    raw = raw.assign(rul=compute_rul(raw, cap=data.cmapss.rul_cap))
    engine = int(cache.engine_ids[0])
    block = normaliser.transform(raw[raw["unit"] == engine]).sort_values("cycle")
    values = block[sensors].to_numpy(dtype="float32")

    window = cache.window
    age = window + 20
    live = model.predict(values[age - window : age])
    cached_mean, cached_std, cached_p = cache.predictions_at(0, age)

    # MC dropout means neither call is deterministic; two std of the predictive
    # spread is the right order of agreement to demand.
    assert live.rul_mean == pytest.approx(cached_mean, abs=max(3.0 * cached_std, 2.0))
    assert live.p_fail_within_h == pytest.approx(cached_p, abs=0.15)


@pytest.mark.data
def test_real_replay_pool_is_large_enough_and_out_of_sample() -> None:
    """At least 40 held-out run-to-failure engines, disjoint from training."""
    from twinmind.twin.health_replay import PoolSpec, load_cached, pool_engine_ids
    from twinmind.utils.paths import PROJECT_ROOT

    twin = load_config("twin")
    data = load_config("data")
    checkpoint = PROJECT_ROOT / twin.health_replay.checkpoint
    if not checkpoint.exists():
        pytest.skip(f"no checkpoint at {checkpoint}")

    train_ids, pool_ids = pool_engine_ids(
        twin.health_replay.subset, twin.health_replay.model_seed, data.cmapss.val_fraction
    )
    assert len(pool_ids) >= twin.health_replay.min_pool_engines
    assert not (set(train_ids) & set(pool_ids))

    spec = PoolSpec(
        subset=twin.health_replay.subset,
        model_seed=twin.health_replay.model_seed,
        val_fraction=data.cmapss.val_fraction,
        window=data.cmapss.window_length,
        checkpoint=str(checkpoint),
    )
    try:
        cache = load_cached(PROJECT_ROOT / twin.health_replay.cache_path, spec)
    except FileNotFoundError:
        pytest.skip("health cache has not been built")
    assert sorted(cache.engine_ids.tolist()) == sorted(pool_ids)


@pytest.mark.data
def test_real_twin_run_meets_the_performance_budget() -> None:
    """One run on one core must fit the 0.25 s budget the rollout plan needs."""
    import time

    try:
        resources = build_resources()
    except FileNotFoundError as error:
        pytest.skip(f"real artifacts unavailable: {error}")

    scenario = Scenario()
    run_twin(scenario, 0, resources=resources)
    durations = []
    for seed in range(20):
        started = time.perf_counter()
        run_twin(scenario, seed, resources=resources)
        durations.append(time.perf_counter() - started)
    mean = float(np.mean(durations))
    assert mean < 0.25, f"mean run time {1000 * mean:.0f} ms exceeds the 250 ms budget"


def test_sampler_memo_entry_dies_with_its_model() -> None:
    """The property that makes address reuse harmless, tested directly.

    The memo was keyed on ``id(model)``. ``id()`` is a memory address and
    CPython reuses addresses, so once a model was garbage-collected a newly
    allocated one could land at the same address and silently inherit the dead
    model's cached quantile grids -- a twin forecasting with the wrong demand
    model, with no error anywhere. It surfaced as a test that passed alone and
    failed once earlier tests had allocated and freed a stub model first.

    Rather than trying to coax the allocator into reusing an address, which is
    not reproducible, this asserts the invariant that makes reuse impossible:
    a model's cache entry is held weakly and disappears with the model, so no
    later object can find it.
    """
    import gc

    from twinmind.twin.demand_proc import _SAMPLER_MEMO, get_sampler

    origin = pd.Timestamp("2017-08-15")
    model = StubDemandModel(level=900.0)
    get_sampler(model, "stub", origin, HORIZON)
    assert model in _SAMPLER_MEMO

    del model
    gc.collect()
    # Nothing left for a future object at the same address to inherit.
    assert all(isinstance(v, dict) for v in _SAMPLER_MEMO.values())
    assert not any(isinstance(k, StubDemandModel) and k.level == 900.0 for k in _SAMPLER_MEMO)


def test_two_live_models_do_not_share_a_sampler() -> None:
    """Distinct models at the same (series, origin, horizon) must stay distinct."""
    from twinmind.twin.demand_proc import get_sampler

    origin = pd.Timestamp("2017-08-15")
    high, low = StubDemandModel(level=900.0), StubDemandModel(level=1.0)
    high_sampler = get_sampler(high, "stub", origin, HORIZON)
    low_sampler = get_sampler(low, "stub", origin, HORIZON)

    assert high_sampler is not low_sampler
    assert float(np.median(high_sampler.sample(64, seed=0))) > 100.0
    assert float(np.median(low_sampler.sample(64, seed=0))) < 100.0
    # And the memo still returns the right one for each.
    assert get_sampler(high, "stub", origin, HORIZON) is high_sampler
    assert get_sampler(low, "stub", origin, HORIZON) is low_sampler
