"""Data contracts and entry point for the digital twin.

These Pydantic models are the interface every later component depends on: the
surrogate (Model C) regresses onto ``KPIVector``, the optimisers search over
``Scenario``, and the agent's Evaluator/Critic nodes compare ``KPIWithCI``
objects. The simulation itself is not implemented yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover
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
    reorder_point: float = Field(default=1200.0, ge=0.0)
    order_quantity: float = Field(default=2500.0, gt=0.0)
    safety_stock: float = Field(default=800.0, ge=0.0)
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
