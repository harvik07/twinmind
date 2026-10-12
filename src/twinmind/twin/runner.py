"""Data contracts and entry point for the digital twin.

These Pydantic models are the interface every later component depends on: the
surrogate (Model C) regresses onto ``KPIVector``, the optimisers search over
``Scenario``, and the agent's Evaluator/Critic nodes compare ``KPIWithCI``
objects. The simulation itself is not implemented yet.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import simpy
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats as _stats

if TYPE_CHECKING:  # pragma: no cover
    from twinmind.twin.context import TwinContext
    from twinmind.utils.config import TwinConfig

# Canonical order of the 18 scenario dimensions. Must match configs/twin.yaml.
SCENARIO_DIMS: tuple[str, ...] = (
    "throughput_multiplier",
    "shift_count",
    "overtime_hours",
    "pm_interval_M1",
    "pm_interval_M2",
    "pm_interval_M3",
    "pm_interval_M4",
    "pm_interval_M5",
    "speed_setpoint_M1",
    "speed_setpoint_M2",
    "speed_setpoint_M3",
    "speed_setpoint_M4",
    "speed_setpoint_M5",
    "supplier_choice",
    "reorder_point",
    "order_quantity",
    "safety_stock",
    "price_multiplier",
)

# Canonical order of the 8 KPIs the surrogate predicts.
KPI_NAMES: tuple[str, ...] = (
    "units_produced",
    "oee",
    "unit_cost",
    "profit",
    "stockout_probability",
    "downtime_hours",
    "unplanned_failures",
    "on_time_delivery_rate",
)


class Scenario(BaseModel):
    """One point in the 18-dimensional decision space.

    Bounds here are the hard structural limits; the *searchable* bounds live in
    ``configs/twin.yaml`` so they can be narrowed without a code change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    throughput_multiplier: float = Field(default=1.0, gt=0.0)
    shift_count: int = Field(default=2, ge=1, le=3)
    overtime_hours: float = Field(default=0.0, ge=0.0)

    pm_interval_M1: float = Field(default=168.0, gt=0.0)
    pm_interval_M2: float = Field(default=168.0, gt=0.0)
    pm_interval_M3: float = Field(default=120.0, gt=0.0)
    pm_interval_M4: float = Field(default=168.0, gt=0.0)
    pm_interval_M5: float = Field(default=240.0, gt=0.0)

    speed_setpoint_M1: float = Field(default=1.0, gt=0.0)
    speed_setpoint_M2: float = Field(default=1.0, gt=0.0)
    speed_setpoint_M3: float = Field(default=1.0, gt=0.0)
    speed_setpoint_M4: float = Field(default=1.0, gt=0.0)
    speed_setpoint_M5: float = Field(default=1.0, gt=0.0)

    supplier_choice: int = Field(default=0, ge=0)
    # Defaults chosen so the default scenario is a FEASIBLE plan. See
    # configs/twin.yaml for why 1200/2500/800 was not: with (s, Q) reviewed on
    # inventory position, an order quantity above the reorder point means two
    # orders can never be in flight, capping supply far below demand.
    reorder_point: float = Field(default=7000.0, ge=0.0)
    order_quantity: float = Field(default=6000.0, gt=0.0)
    safety_stock: float = Field(default=1500.0, ge=0.0)
    price_multiplier: float = Field(default=1.0, gt=0.0)

    def to_vector(self) -> list[float]:
        """Return the scenario in canonical dimension order."""
        return [float(getattr(self, name)) for name in SCENARIO_DIMS]

    @classmethod
    def from_vector(cls, vector: list[float] | tuple[float, ...]) -> Scenario:
        """Build a Scenario from a canonically ordered vector."""
        if len(vector) != len(SCENARIO_DIMS):
            raise ValueError(f"expected {len(SCENARIO_DIMS)} values, got {len(vector)}")
        payload: dict[str, Any] = dict(zip(SCENARIO_DIMS, vector, strict=True))
        payload["shift_count"] = int(round(payload["shift_count"]))
        payload["supplier_choice"] = int(round(payload["supplier_choice"]))
        return cls(**payload)


class KPIVector(BaseModel):
    """Deterministic KPIs from a single simulation replication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    units_produced: float = Field(ge=0.0)
    oee: float = Field(ge=0.0, le=1.0)
    unit_cost: float = Field(ge=0.0)
    profit: float
    stockout_probability: float = Field(ge=0.0, le=1.0)
    downtime_hours: float = Field(ge=0.0)
    unplanned_failures: int = Field(ge=0)
    on_time_delivery_rate: float = Field(ge=0.0, le=1.0)

    def to_vector(self) -> list[float]:
        """Return the KPIs in canonical order (the surrogate's target layout)."""
        return [float(getattr(self, name)) for name in KPI_NAMES]

    @classmethod
    def from_vector(cls, vector: list[float] | tuple[float, ...]) -> KPIVector:
        """Build a KPIVector from a canonically ordered vector."""
        if len(vector) != len(KPI_NAMES):
            raise ValueError(f"expected {len(KPI_NAMES)} values, got {len(vector)}")
        payload: dict[str, Any] = dict(zip(KPI_NAMES, vector, strict=True))
        payload["unplanned_failures"] = int(round(payload["unplanned_failures"]))
        return cls(**payload)


class CIEstimate(BaseModel):
    """A mean with a two-sided confidence interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def _check_ordering(self) -> CIEstimate:
        if not self.lower <= self.mean <= self.upper:
            raise ValueError(f"CI must satisfy lower <= mean <= upper, got {self}")
        return self

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.upper - self.lower


class KPIWithCI(BaseModel):
    """KPIs aggregated across replications, each with a confidence interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_reps: int = Field(gt=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    units_produced: CIEstimate
    oee: CIEstimate
    unit_cost: CIEstimate
    profit: CIEstimate
    stockout_probability: CIEstimate
    downtime_hours: CIEstimate
    unplanned_failures: CIEstimate
    on_time_delivery_rate: CIEstimate

    def means(self) -> dict[str, float]:
        """Point estimates keyed by KPI name."""
        return {name: getattr(self, name).mean for name in KPI_NAMES}

    def get(self, kpi: str) -> CIEstimate:
        """Look up one KPI's estimate by name."""
        if kpi not in KPI_NAMES:
            raise KeyError(f"unknown KPI {kpi!r}; expected one of {KPI_NAMES}")
        estimate: CIEstimate = getattr(self, kpi)
        return estimate


class ConstraintReport(BaseModel):
    """Deterministic pass/fail record for one constraint. Never written by an LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kpi: str
    direction: Literal["min", "max"]
    threshold: float
    observed: float
    satisfied: bool


class TwinResult(BaseModel):
    """Everything one twin evaluation returns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: Scenario
    kpis: KPIWithCI
    constraints: list[ConstraintReport] = Field(default_factory=list)
    wall_clock_seconds: float = Field(default=0.0, ge=0.0)

    @property
    def feasible(self) -> bool:
        """True when every constraint is satisfied."""
        return all(c.satisfied for c in self.constraints)


class TwinRunner:
    """Runs the SimPy digital twin for a scenario. Not implemented yet."""

    def __init__(self, config: TwinConfig, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed if seed is not None else config.seed

    def run_once(self, scenario: Scenario, seed: int) -> KPIVector:
        """Run a single replication and return its KPIs."""
        raise NotImplementedError("TwinRunner.run_once is not implemented yet")

    def run(self, scenario: Scenario, n_reps: int | None = None) -> TwinResult:
        """Run ``n_reps`` replications and return aggregated KPIs with CIs."""
        raise NotImplementedError("TwinRunner.run is not implemented yet")

    def check_constraints(self, kpis: KPIWithCI) -> list[ConstraintReport]:
        """Evaluate configured constraints against aggregated KPIs."""
        raise NotImplementedError("TwinRunner.check_constraints is not implemented yet")


# =========================================================================== #
# The simulation
# =========================================================================== #
#
# ``run_twin`` is a pure function of ``(scenario, seed, context)``. Everything
# expensive and immutable -- the config, the health cache, the demand sampler --
# lives in a :class:`TwinResources` built lazily, once per process, and never
# mutated. That is what makes the hot path free of file I/O and what makes the
# twin safe under multiprocessing "spawn": a worker imports this module, finds
# ``_RESOURCES`` empty, and builds its own read-only copy.
#
# There is deliberately no module-level mutable state beyond that cache.

HORIZON_DAYS: int = 28

_RESOURCES: dict[str, TwinResources] = {}


@_dataclass(frozen=True)
class TwinResources:
    """Immutable, process-local things the simulation reads but never changes."""

    config: TwinConfig
    #: The hazard façade (``HealthLookup``, or a test's synthetic one) and the
    #: demand model are injectable seams, so they are annotated ``Any`` rather
    #: than pinned to a concrete class. That is what lets the non-data tests
    #: drive the simulator with stubs through the same code path.
    lookup: Any
    demand_model: Any
    demand_scale: float
    series_id: str
    origin: pd.Timestamp

    @property
    def assumptions(self) -> dict[str, float]:
        """Every declared assumption, resolved to its value."""
        return {name: entry.value for name, entry in self.config.assumptions.items()}


def build_resources(
    config: TwinConfig | None = None,
    lookup: Any | None = None,
    demand_model: Any | None = None,
    demand_scale: float | None = None,
    series_id: str | None = None,
    origin: pd.Timestamp | None = None,
) -> TwinResources:
    """Assemble the resources, loading the real artifacts for anything omitted.

    Every argument is injectable so the non-data test suite can drive the twin
    with stub health and demand models. The stubs satisfy the same Protocols, so
    the simulator code they exercise is the same code the real artifacts drive.
    """
    from twinmind.utils.config import load_config

    resolved_config = config if config is not None else cast("TwinConfig", load_config("twin"))

    if lookup is None:
        from twinmind.twin.health_replay import HealthLookup, PoolSpec, load_cached
        from twinmind.utils.paths import PROJECT_ROOT

        data_config = cast("Any", load_config("data"))
        replay = resolved_config.health_replay
        spec = PoolSpec(
            subset=replay.subset,
            model_seed=replay.model_seed,
            val_fraction=data_config.cmapss.val_fraction,
            window=data_config.cmapss.window_length,
            checkpoint=str(PROJECT_ROOT / replay.checkpoint),
        )
        cache = load_cached(PROJECT_ROOT / replay.cache_path, spec)
        if cache.n_engines < replay.min_pool_engines:
            raise ValueError(
                f"replay pool has {cache.n_engines} engines, below the configured "
                f"minimum of {replay.min_pool_engines}"
            )
        lookup = HealthLookup.from_cache(cache, replay.hazard_source)

    resolved_series = series_id if series_id is not None else "1_GROCERY I"
    resolved_origin = pd.Timestamp("2017-08-15") if origin is None else pd.Timestamp(origin)

    if demand_model is None:
        from twinmind.data.features import build_demand_features
        from twinmind.data.loaders import load_favorita
        from twinmind.models.demand import DemandData, load_demand_model
        from twinmind.utils.paths import PROJECT_ROOT

        data_config = cast("Any", load_config("data"))
        frame = build_demand_features(load_favorita(), data_config.favorita, top_n=200)
        demand_model = load_demand_model(PROJECT_ROOT / "models" / "demand", DemandData(frame))
        resolved_series = (
            resolved_series
            if resolved_series in DemandData(frame).series_ids
            else (DemandData(frame).series_ids[0])
        )

    if demand_scale is None:
        from twinmind.twin.demand_proc import calibrate_demand_scale

        # Capacity means ACHIEVABLE capacity, which is the bottleneck rate over
        # the hours the line is actually staffed. Multiplying the rate by a
        # nominal 16-hour day instead overstates it by 40%: the line works five
        # days in seven, so 28 calendar days are only 20 staffed ones. Demand
        # arrives every day including weekends, so calibrating against the
        # overstated figure makes the baseline structurally unable to serve its
        # own order book and every scenario looks like a stockout.
        from twinmind.twin.inventory import ShiftCalendar

        nominal = min(m.nominal_rate_units_per_hour for m in resolved_config.machines)
        baseline_calendar = ShiftCalendar(resolved_config.shifts, 2, 0.0, HORIZON_DAYS)
        capacity_per_day = nominal * baseline_calendar.staffed_hours / HORIZON_DAYS
        demand_scale = calibrate_demand_scale(
            cast("Any", demand_model),
            resolved_series,
            resolved_origin,
            HORIZON_DAYS,
            nominal_capacity_per_day=capacity_per_day,
            target_ratio=resolved_config.assumption("demand_capacity_ratio"),
        )

    return TwinResources(
        config=resolved_config,
        lookup=lookup,
        demand_model=demand_model,
        demand_scale=float(demand_scale),
        series_id=resolved_series,
        origin=resolved_origin,
    )


def get_resources() -> TwinResources:
    """The process-local resources, built on first use."""
    cached = _RESOURCES.get("default")
    if cached is None:
        cached = build_resources()
        _RESOURCES["default"] = cached
    return cached


@_dataclass
class _Ledger:
    """Counters the day-level process accumulates."""

    units_demanded: float = 0.0
    units_delivered: float = 0.0
    shortfall_units: float = 0.0
    stockout_days: int = 0
    finished_unit_days: float = 0.0
    days_elapsed: int = 0
    finished_trace: list[float] = _field(default_factory=list)
    demand_trace: list[float] = _field(default_factory=list)
    delivered_trace: list[float] = _field(default_factory=list)
    wip_samples: list[float] = _field(default_factory=list)


def _demand_process(
    env: simpy.Environment,
    finished: simpy.Container,
    demand_path: NDArray[np.float64],
    ledger: _Ledger,
) -> Generator[simpy.Event, Any, Any]:
    """One order arrival per day, fulfilled from finished stock; shortfall is lost.

    Demand lands at the *end* of each day, after that day's production. Landing
    it at the start would make day 1 a guaranteed stockout for any context that
    begins with empty finished stock, which is an artefact of the ordering
    rather than anything about the line.
    """
    for day in range(demand_path.size):
        yield env.timeout(24.0)
        wanted = float(demand_path[day])
        available = float(finished.level)
        shipped = min(wanted, available)
        if shipped > 0:
            yield finished.get(shipped)
        unmet = wanted - shipped

        ledger.units_demanded += wanted
        ledger.units_delivered += shipped
        ledger.shortfall_units += unmet
        if unmet > 1e-9:
            ledger.stockout_days += 1
        ledger.finished_unit_days += float(finished.level)
        ledger.days_elapsed = day + 1
        ledger.finished_trace.append(float(finished.level))
        ledger.demand_trace.append(wanted)
        ledger.delivered_trace.append(shipped)


def _wip_monitor(
    env: simpy.Environment, buffers: list[simpy.Container], ledger: _Ledger
) -> Generator[simpy.Event, Any, Any]:
    """Sample work in progress once a day, for the time-average Little's law needs."""
    while True:
        ledger.wip_samples.append(float(sum(b.level for b in buffers)))
        yield env.timeout(24.0)


def run_twin(
    scenario: Scenario,
    seed: int,
    context: TwinContext | None = None,
    resources: TwinResources | None = None,
) -> KPIVector:
    """One replication. A pure function of ``(scenario, seed, context)``.

    Deterministic by construction rather than by care: every random number
    comes from :class:`~twinmind.twin.streams.StreamBundle`, which derives its
    generators from ``seed`` alone. Nothing on this path reads the clock, the
    filesystem or a global counter.
    """
    from twinmind.twin.context import default_context
    from twinmind.twin.demand_proc import daily_demand
    from twinmind.twin.inventory import (
        InventoryPolicy,
        ShiftCalendar,
        Warehouse,
        WarehouseState,
    )
    from twinmind.twin.machines import Station, StationState
    from twinmind.twin.streams import StreamBundle

    resolved = resources if resources is not None else get_resources()
    config = resolved.config
    assumption = config.assumption
    ctx = (
        context
        if context is not None
        else default_context(resolved.series_id, resolved.origin, len(config.machines))
    )
    streams = StreamBundle.from_seed(seed)

    calendar = ShiftCalendar(
        config.shifts, scenario.shift_count, scenario.overtime_hours, HORIZON_DAYS
    )
    env = simpy.Environment()

    # The demand path is drawn first and from the demand stream alone, so adding
    # or removing any other source of randomness cannot shift it. That is what
    # the common-random-numbers test in the model card checks.
    demand_path = daily_demand(
        cast(Any, resolved.demand_model),
        resolved.series_id,
        resolved.origin,
        HORIZON_DAYS,
        seed=int(streams["demand"].integers(0, 2**31 - 1)),
        demand_scale=resolved.demand_scale,
        price_multiplier=scenario.price_multiplier,
        price_elasticity=assumption("price_elasticity"),
        spread_multiplier=assumption("demand_spread_multiplier"),
    )

    # Buffers: raw yard, four inter-station buffers with the configured finite
    # capacities, then the despatch store. The yard and the store are
    # effectively unbounded; it is the four middle ones that produce blocking.
    unbounded = 1e9
    raw = simpy.Container(env, capacity=unbounded, init=float(ctx.raw_inventory))
    finished = simpy.Container(env, capacity=unbounded, init=float(ctx.finished_inventory))
    buffers: list[simpy.Container] = [raw]
    for machine in config.machines[:-1]:
        buffers.append(simpy.Container(env, capacity=float(machine.buffer_capacity), init=0.0))
    buffers.append(finished)

    lookup = resolved.lookup
    hazard_source = config.health_replay.hazard_source
    states: list[StationState] = []
    for index, machine in enumerate(config.machines):
        initial = ctx.machines[index] if index < len(ctx.machines) else ctx.machines[-1]
        state = StationState(age_cycles=float(initial.age_cycles))
        state.trajectory_id = int(initial.trajectory_id)
        if hazard_source == "fixed_mtbf":
            state.failure_cycle = float(lookup.draw_lifetime(streams["machine_life"]))
        else:
            slot = _resolve_slot(lookup, int(initial.trajectory_id), streams["machine_life"])
            state.trajectory_slot = slot
            state.failure_cycle = float(lookup.cache.failure_cycles[slot])
        states.append(state)

        env.process(
            Station(
                env,
                machine,
                inbound=buffers[index],
                outbound=buffers[index + 1],
                state=state,
                speed_setpoint=float(getattr(scenario, f"speed_setpoint_{machine.id}")),
                pm_interval_hours=float(getattr(scenario, f"pm_interval_{machine.id}")),
                lot_size=config.flow.lot_size,
                throughput_multiplier=scenario.throughput_multiplier,
                processing_cv=assumption("processing_cv"),
                wear_exponent=assumption("wear_exponent"),
                life_scale_hours_per_cycle=assumption("life_scale_hours_per_cycle"),
                defect_base=assumption("defect_base_probability"),
                defect_speed_exponent=assumption("defect_speed_exponent"),
                repair_multiplier=assumption("unplanned_repair_multiplier"),
                warning_threshold=config.health_replay.warning_threshold,
                hazard_source=hazard_source,
                health_lookup=lookup,
                shift_calendar=calendar,
                rng_processing=streams["processing"],
                rng_repair=streams["repair"],
                rng_defects=streams["defects"],
                rng_life=streams["machine_life"],
            ).run()
        )

    supplier = config.suppliers[min(scenario.supplier_choice, len(config.suppliers) - 1)]
    warehouse_state = WarehouseState()
    env.process(
        Warehouse(
            env,
            InventoryPolicy(scenario.reorder_point, scenario.order_quantity, scenario.safety_stock),
            supplier,
            initial_stock=float(ctx.raw_inventory),
            container=raw,
            state=warehouse_state,
            rng=streams["supplier"],
            lead_time_multiplier=assumption("lead_time_multiplier"),
            short_ship_fraction=assumption("supplier_short_ship_fraction"),
        ).run()
    )

    ledger = _Ledger()
    env.process(_demand_process(env, finished, demand_path, ledger))
    # Daily WIP samples: 28 extra events, which is nothing against ~12,000 lot
    # events, and without a time-average of WIP there is no way to check
    # Little's law at all.
    env.process(_wip_monitor(env, buffers[1:-1], ledger))
    env.run(until=HORIZON_DAYS * 24.0)

    return _accumulate(
        config=config,
        scenario=scenario,
        calendar=calendar,
        states=states,
        warehouse=warehouse_state,
        ledger=ledger,
        buffers=buffers,
        supplier=supplier,
        context=ctx,
    )


def _resolve_slot(lookup: Any, trajectory_id: int, rng: np.random.Generator) -> int:
    """Pool slot for a context's engine id, drawing a fresh one if it is absent.

    A context may name an engine that is not in this pool -- the default context
    uses id 0 as a placeholder, and a context sampled against one cache may be
    replayed against another. Falling back to a draw keeps that usable instead
    of raising, and the draw comes from the machine_life stream so it stays
    inside the common-random-numbers guarantee.
    """
    cache = getattr(lookup, "cache", None)
    if cache is not None:
        try:
            return cache.index_of(trajectory_id)
        except KeyError:
            pass
    slot, _, _ = lookup.draw_engine(rng)
    return int(slot)


def _accumulate(
    config: TwinConfig,
    scenario: Scenario,
    calendar: Any,
    states: list[Any],
    warehouse: Any,
    ledger: _Ledger,
    buffers: list[simpy.Container],
    supplier: Any,
    context: TwinContext,
) -> KPIVector:
    """Turn the simulation's counters into costs and the canonical KPI vector.

    Separated from ``run_twin`` so the cost identity and flow conservation can
    be tested against the same counters the KPIs are built from, rather than
    against a reconstruction.
    """
    from twinmind.twin.costs import CostAccumulator, build_kpi_vector

    costs = config.costs
    assumption = config.assumption
    shifts = config.shifts

    units_produced = float(states[-1].units_out)
    good_units = units_produced
    scrap_units = float(sum(s.scrap_units for s in states))
    downtime_hours = float(sum(s.downtime_hours for s in states))
    unplanned = int(sum(s.unplanned_failures for s in states))
    preventive = int(sum(s.preventive_events for s in states))

    # Labour: capacity-hours, not individual workers. Regular hours are paid at
    # the base rate and overtime at the multiplier, per staffed day.
    staffed_days = int(calendar.starts.size)
    regular_hours_per_day = scenario.shift_count * shifts.hours_per_shift
    operators = config.labour.operators_per_shift * scenario.shift_count
    labour = (
        staffed_days
        * operators
        * costs.labour_per_operator_hour
        * (regular_hours_per_day + scenario.overtime_hours * costs.overtime_multiplier)
    )

    # Energy follows run time and the square of speed, per machine.
    energy_exponent = assumption("energy_speed_exponent")
    energy = 0.0
    for machine, state in zip(config.machines, states, strict=True):
        speed = float(getattr(scenario, f"speed_setpoint_{machine.id}"))
        energy += state.processing_hours * costs.energy_per_machine_hour * speed**energy_exponent

    holding = (
        warehouse.holding_unit_days + ledger.finished_unit_days
    ) * costs.holding_cost_per_unit_per_day

    accumulator = CostAccumulator(
        maintenance_preventive=preventive * costs.preventive_maintenance_per_event,
        maintenance_corrective=unplanned * costs.corrective_maintenance_per_event,
        downtime=downtime_hours * costs.downtime_per_hour,
        labour=labour,
        holding=holding,
        stockout=ledger.shortfall_units * costs.stockout_penalty_per_unit,
        late_delivery=0.0,
        raw_material=warehouse.material_cost,
        energy=energy,
        overhead=HORIZON_DAYS * costs.fixed_overhead_per_day,
        revenue=ledger.units_delivered
        * costs.base_selling_price_per_unit
        * scenario.price_multiplier,
    )

    # OEE's performance term is measured against what the bottleneck could have
    # produced in the *uptime*, which is why staffed hours and downtime are
    # passed separately rather than netted off here.
    bottleneck_rate = min(
        machine.nominal_rate_units_per_hour
        * float(getattr(scenario, f"speed_setpoint_{machine.id}"))
        * scenario.throughput_multiplier
        for machine in config.machines
    )
    uptime = max(calendar.staffed_hours - downtime_hours / len(states), 0.0)
    theoretical_units = bottleneck_rate * uptime

    stockout_probability = (
        ledger.stockout_days / ledger.days_elapsed if ledger.days_elapsed else 0.0
    )

    kpis = build_kpi_vector(
        accumulator,
        costs,
        units_produced=units_produced,
        good_units=good_units,
        planned_hours=calendar.staffed_hours,
        downtime_hours=downtime_hours,
        theoretical_units=theoretical_units,
        units_demanded=ledger.units_demanded,
        units_delivered=ledger.units_delivered,
        stockout_probability=stockout_probability,
        unplanned_failures=unplanned,
    )
    _LAST_DETAIL["detail"] = {
        "cost_lines": accumulator.cost_lines(),
        "revenue": accumulator.revenue,
        "total_cost": accumulator.total_cost,
        "scrap_units": scrap_units,
        "wip_units": float(sum(b.level for b in buffers[1:-1]))
        + float(sum(s.in_process for s in states)),
        "wip_buffers": float(sum(b.level for b in buffers[1:-1])),
        "wip_in_machines": float(sum(s.in_process for s in states)),
        "wip_mean": float(np.mean(ledger.wip_samples)) if ledger.wip_samples else 0.0,
        "units_delivered": float(ledger.units_delivered),
        "units_demanded": float(ledger.units_demanded),
        "buffer_capacity": float(sum(b.capacity for b in buffers[1:-1])),
        "finished_level": float(buffers[-1].level),
        "raw_level": float(buffers[0].level),
        "raw_in": float(context.raw_inventory) + float(warehouse.units_received),
        "units_consumed": float(states[0].units_in),
        "preventive_events": preventive,
        "unplanned_failures": unplanned,
        "warnings_raised": int(sum(s.warnings_raised for s in states)),
        "warned_before_failure": int(sum(s.warned_before_failure for s in states)),
        "unwarned_failures": int(sum(s.unwarned_failures for s in states)),
        "warning_lead_hours": [h for s in states for h in s.warning_lead_hours],
        "staffed_hours": calendar.staffed_hours,
        "demand_trace": list(ledger.demand_trace),
        "delivered_trace": list(ledger.delivered_trace),
        "finished_trace": list(ledger.finished_trace),
        "raw_trace": list(warehouse.raw_level_trace),
        "station_states": states,
        "supplier_id": supplier.id,
    }
    return kpis


#: Side-channel for the verification tests and the figure scripts. ``run_twin``
#: returns a plain ``KPIVector`` -- the surrogate's target and nothing more --
#: but the conservation and cost-identity tests need the internal counters, and
#: threading a second return value through would complicate the hot path's
#: signature for every caller. Overwritten on every run, so it is only ever read
#: immediately after a single-threaded call; the tests do exactly that.
_LAST_DETAIL: dict[str, dict[str, Any]] = {}


def last_detail() -> dict[str, Any]:
    """Internal counters from the most recent ``run_twin`` call in this process."""
    detail = _LAST_DETAIL.get("detail")
    if detail is None:
        raise RuntimeError("no run_twin call has completed in this process yet")
    return detail


def replicate(
    scenario: Scenario,
    n_reps: int = 30,
    base_seed: int = 0,
    context: TwinContext | None = None,
    resources: TwinResources | None = None,
    confidence_level: float = 0.95,
) -> KPIWithCI:
    """Mean and a 95% t-interval per KPI over ``n_reps`` replications.

    Seeds come from ``SeedSequence.spawn``, never ``base_seed + i``. The
    t-distribution rather than the normal because 30 replications is small
    enough for the difference to matter -- t(29) is about 4% wider than z, and
    reporting the narrower interval would overstate the precision of every
    number in the results table.
    """
    if n_reps < 1:
        raise ValueError(f"n_reps must be >= 1, got {n_reps}")
    from twinmind.twin.streams import derive_seeds

    seeds = derive_seeds(base_seed, n_reps)
    rows = np.array(
        [run_twin(scenario, seed, context, resources).to_vector() for seed in seeds],
        dtype="float64",
    )
    means = rows.mean(axis=0)
    if n_reps > 1:
        sem = rows.std(axis=0, ddof=1) / math.sqrt(n_reps)
        critical = float(_stats.t.ppf(0.5 + confidence_level / 2.0, df=n_reps - 1))
    else:
        sem = np.zeros_like(means)
        critical = 0.0
    half = critical * sem

    estimates: dict[str, Any] = {"n_reps": n_reps, "confidence_level": confidence_level}
    for index, name in enumerate(KPI_NAMES):
        low, high = means[index] - half[index], means[index] + half[index]
        # KPIVector clamps several KPIs to [0, 1]; CIEstimate does not, and a
        # bound-crossing interval on a rate would be nonsense to report.
        if name in {"oee", "stockout_probability", "on_time_delivery_rate"}:
            low, high = max(low, 0.0), min(high, 1.0)
        if name in {"units_produced", "unit_cost", "downtime_hours", "unplanned_failures"}:
            low = max(low, 0.0)
        estimates[name] = CIEstimate(
            mean=float(means[index]),
            lower=float(min(low, means[index])),
            upper=float(max(high, means[index])),
        )
    return KPIWithCI(**estimates)
