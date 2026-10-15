"""Frozen contracts for the optimisation layer.

THE AUDIENCE FOR THE ERROR MESSAGES IS AN LLM.

In week 6 the agent proposes an ``ObjectiveSpec`` as JSON. It will get things
wrong: a KPI that does not exist, a bound outside the Scenario's range, a
constraint pair that cannot both hold, a fixed field that is not a field. Every
one of those raises :class:`SpecError` with a message naming the offending value
**and the valid alternatives**, because "invalid spec" tells a model nothing and
"unknown KPI 'throughput'; valid KPIs are ..." tells it exactly what to emit
next. That is the whole reason the validation is this verbose.

THE DECISION / EXOGENOUS SPLIT (A1).

Not every Scenario field is something a plant manager can choose. The optimiser
may only move **controllables**; a goal pins **exogenous** variables to describe
a what-if. Mixing them would let the optimiser "solve" a demand slump by
quietly turning demand back up.

======================  ==========  ===================================================
field                   class       why
======================  ==========  ===================================================
throughput_multiplier   CONTROL     the production target the planner sets
shift_count             CONTROL     how many shifts are staffed
overtime_hours          CONTROL     paid overtime appended to a staffed day
pm_interval_M1..M5      CONTROL     the maintenance policy, one interval per machine
speed_setpoint_M1..M5   CONTROL     how hard each machine is driven
reorder_point           CONTROL     the (s, Q) trigger
order_quantity          CONTROL     the (s, Q) quantity
safety_stock            CONTROL     the buffer the planner holds
supplier_choice         EXOGENOUS   a sourcing decision, but used here as the what-if
                                    lever ("what if we had to use the offshore
                                    supplier?"); a goal may free it explicitly
price_multiplier        EXOGENOUS   the selling price is a commercial decision taken
                                    outside the plant, and it moves demand through the
                                    elasticity assumption, so optimising it would be
                                    optimising the market rather than the line
======================  ==========  ===================================================

``supplier_choice`` is the one genuine judgement call. It *is* a decision
somebody makes, but it is the natural handle for the supply-disruption what-ifs
the goal library needs, so it is exogenous by default and a goal can put it in
``free_dims`` to optimise over it. That is stated rather than hidden because a
reader could reasonably classify it the other way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from twinmind.models.surrogate import KPI_ORDER
from twinmind.twin.runner import SCENARIO_DIMS, Scenario

#: Fields the optimiser may move.
CONTROLLABLE_DIMS: tuple[str, ...] = (
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
    "reorder_point",
    "order_quantity",
    "safety_stock",
)

#: Fields a goal pins to describe a what-if. See the module docstring.
EXOGENOUS_DIMS: tuple[str, ...] = ("supplier_choice", "price_multiplier")

#: KPIs where a larger value is better. Used to orient constraint margins and to
#: sanity-check that a goal is not asking to minimise something it should
#: maximise.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"units_produced", "oee", "profit", "on_time_delivery_rate"}
)


class SpecError(ValueError):
    """A malformed goal specification, with a message aimed at a model.

    Deliberately its own type: the agent layer needs to distinguish "you wrote
    the spec wrong, here is how to fix it" from an internal failure, and catching
    bare ``ValueError`` would swallow both.
    """


def _assert_known_kpi(name: str, where: str) -> None:
    """Raise a message that names the valid KPIs."""
    if name not in KPI_ORDER:
        raise SpecError(f"{where}: unknown KPI {name!r}. Valid KPIs are: {', '.join(KPI_ORDER)}.")


def _assert_known_dim(name: str, where: str) -> None:
    """Raise a message that names the valid Scenario fields and their class."""
    if name not in SCENARIO_DIMS:
        raise SpecError(
            f"{where}: {name!r} is not a Scenario field. Valid fields are: "
            f"{', '.join(SCENARIO_DIMS)}."
        )


@dataclass(frozen=True)
class Constraint:
    """One KPI threshold. ``hard`` constraints must hold; soft ones are advisory."""

    kpi: str
    op: Literal["<=", ">="]
    threshold: float
    hard: bool = True

    def __post_init__(self) -> None:
        _assert_known_kpi(self.kpi, "constraint")
        if self.op not in ("<=", ">="):
            raise SpecError(
                f"constraint on {self.kpi!r}: op must be '<=' or '>=', got {self.op!r}."
            )
        if self.kpi in {"oee", "stockout_probability", "on_time_delivery_rate"} and not (
            0.0 <= self.threshold <= 1.0
        ):
            raise SpecError(
                f"constraint on {self.kpi!r}: threshold {self.threshold} is outside [0, 1], "
                f"but {self.kpi} is a rate defined on [0, 1]."
            )

    def satisfied(self, value: float) -> bool:
        """Does an observed KPI value satisfy this constraint?"""
        return value <= self.threshold if self.op == "<=" else value >= self.threshold

    def margin(self, value: float) -> float:
        """Signed slack: positive is satisfied, negative is the violation size."""
        return self.threshold - value if self.op == "<=" else value - self.threshold

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"kpi": self.kpi, "op": self.op, "threshold": self.threshold, "hard": self.hard}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Constraint:
        """Parse, raising :class:`SpecError` on anything unexpected."""
        missing = {"kpi", "op", "threshold"} - set(payload)
        if missing:
            raise SpecError(
                f"constraint is missing {sorted(missing)}. A constraint needs "
                "'kpi', 'op' ('<=' or '>='), 'threshold', and optionally 'hard'."
            )
        return cls(
            kpi=str(payload["kpi"]),
            op=payload["op"],
            threshold=float(payload["threshold"]),
            hard=bool(payload.get("hard", True)),
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    """What to optimise, under what constraints, with what held fixed."""

    name: str
    objective: dict[str, float]
    sense: Literal["maximise", "minimise"] = "maximise"
    constraints: tuple[Constraint, ...] = ()
    free_dims: tuple[str, ...] = ()
    fixed: dict[str, float] = field(default_factory=dict)
    context_id: int = 0
    infeasible_by_design: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise SpecError("a goal needs a non-empty 'name'.")
        if not self.objective:
            raise SpecError(
                "'objective' must map at least one KPI to a weight, e.g. {'profit': 1.0} "
                f"or a weighted sum like {{'profit': 1.0, 'oee': 50000.0}}. Valid KPIs: "
                f"{', '.join(KPI_ORDER)}."
            )
        for kpi in self.objective:
            _assert_known_kpi(kpi, f"goal {self.name!r} objective")
        if self.sense not in ("maximise", "minimise"):
            raise SpecError(
                f"goal {self.name!r}: 'sense' must be 'maximise' or 'minimise', got {self.sense!r}."
            )

        free = self.free_dims or CONTROLLABLE_DIMS
        for dim in free:
            _assert_known_dim(dim, f"goal {self.name!r} free_dims")
        for dim in self.fixed:
            _assert_known_dim(dim, f"goal {self.name!r} fixed")

        overlap = set(free) & set(self.fixed)
        if overlap:
            raise SpecError(
                f"goal {self.name!r}: {sorted(overlap)} appear in BOTH free_dims and fixed. "
                "A dimension is either searched or pinned, not both."
            )
        object.__setattr__(self, "free_dims", tuple(free))
        self._check_bounds()
        self._check_contradictions()

    def _check_bounds(self) -> None:
        """Every fixed value must be inside the Scenario's own bounds."""
        from twinmind.models.surrogate import scenario_bounds

        bounds = scenario_bounds()
        for dim, value in self.fixed.items():
            low, high = bounds[dim]
            if not low - 1e-9 <= float(value) <= high + 1e-9:
                raise SpecError(
                    f"goal {self.name!r}: fixed {dim}={value} is outside the configured "
                    f"range [{low}, {high}]. Pick a value inside it."
                )

    def _check_contradictions(self) -> None:
        """Catch constraint pairs that cannot both hold."""
        by_kpi: dict[str, list[Constraint]] = {}
        for constraint in self.constraints:
            by_kpi.setdefault(constraint.kpi, []).append(constraint)
        for kpi, group in by_kpi.items():
            lower = [c.threshold for c in group if c.op == ">="]
            upper = [c.threshold for c in group if c.op == "<="]
            if lower and upper and max(lower) > min(upper):
                raise SpecError(
                    f"goal {self.name!r}: contradictory constraints on {kpi!r} -- "
                    f">= {max(lower)} and <= {min(upper)} cannot both hold. "
                    "Widen one of them."
                )

    # ------------------------------------------------------------- scoring --
    def score(self, kpis: dict[str, float]) -> float:
        """The scalar objective, always oriented so **larger is better**.

        Minimise goals are negated here rather than in each optimiser, so every
        optimiser can maximise unconditionally and none of them has to remember
        the sense.
        """
        total = sum(weight * float(kpis[kpi]) for kpi, weight in self.objective.items())
        return -total if self.sense == "minimise" else total

    def violations(self, kpis: dict[str, float]) -> dict[str, float]:
        """Per-constraint violation size; 0.0 where satisfied. Hard ones only."""
        out: dict[str, float] = {}
        for constraint in self.constraints:
            if not constraint.hard:
                continue
            margin = constraint.margin(float(kpis[constraint.kpi]))
            out[f"{constraint.kpi}{constraint.op}{constraint.threshold}"] = max(-margin, 0.0)
        return out

    def total_violation(self, kpis: dict[str, float]) -> float:
        """Total hard-constraint violation, normalised per constraint.

        Each violation is divided by ``max(|threshold|, 1)`` so a stockout
        probability breach of 0.05 and a profit breach of £50,000 are not summed
        as if they were the same quantity.
        """
        total = 0.0
        for constraint in self.constraints:
            if not constraint.hard:
                continue
            margin = constraint.margin(float(kpis[constraint.kpi]))
            if margin < 0:
                total += -margin / max(abs(constraint.threshold), 1.0)
        return total

    def feasible(self, kpis: dict[str, float]) -> bool:
        """Do all hard constraints hold?"""
        return all(c.satisfied(float(kpis[c.kpi])) for c in self.constraints if c.hard)

    def base_scenario(self) -> Scenario:
        """A default Scenario with this goal's fixed values applied.

        ``shift_count`` and ``supplier_choice`` are rounded to int because a
        goal's ``fixed`` map is JSON, where every number arrives as a float, and
        Scenario types them as integers.
        """
        payload: dict[str, Any] = dict(self.fixed)
        for dim in ("shift_count", "supplier_choice"):
            if dim in payload:
                payload[dim] = int(round(float(payload[dim])))
        return Scenario(**payload)

    # --------------------------------------------------------------- JSON --
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "name": self.name,
            "objective": dict(self.objective),
            "sense": self.sense,
            "constraints": [c.to_dict() for c in self.constraints],
            "free_dims": list(self.free_dims),
            "fixed": dict(self.fixed),
            "context_id": self.context_id,
            "infeasible_by_design": self.infeasible_by_design,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObjectiveSpec:
        """Parse a spec, raising :class:`SpecError` with actionable messages."""
        if not isinstance(payload, dict):
            raise SpecError(f"a goal must be a JSON object, got {type(payload).__name__}.")
        unknown = set(payload) - {
            "name",
            "objective",
            "sense",
            "constraints",
            "free_dims",
            "fixed",
            "context_id",
            "infeasible_by_design",
            "notes",
        }
        if unknown:
            raise SpecError(
                f"unknown goal field(s) {sorted(unknown)}. Valid fields are: name, objective, "
                "sense, constraints, free_dims, fixed, context_id, infeasible_by_design, notes."
            )
        return cls(
            name=str(payload.get("name", "")),
            objective={str(k): float(v) for k, v in dict(payload.get("objective", {})).items()},
            sense=payload.get("sense", "maximise"),
            constraints=tuple(
                Constraint.from_dict(c) for c in payload.get("constraints", []) or []
            ),
            free_dims=tuple(payload.get("free_dims", []) or []),
            fixed={str(k): float(v) for k, v in dict(payload.get("fixed", {})).items()},
            context_id=int(payload.get("context_id", 0)),
            infeasible_by_design=bool(payload.get("infeasible_by_design", False)),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class Plan:
    """One candidate, with what the surrogate said and what the twin confirmed."""

    scenario: Scenario
    surrogate_pred: dict[str, float] = field(default_factory=dict)
    twin_verified: dict[str, dict[str, float]] = field(default_factory=dict)
    feasible_point: bool = False
    feasible_robust: bool = False
    constraint_margins: dict[str, float] = field(default_factory=dict)
    surrogate_twin_gap: dict[str, float] = field(default_factory=dict)
    novelty: float = float("nan")
    cost: dict[str, float] = field(default_factory=dict)
    status: Literal["ok", "infeasible_unverified", "no_candidate"] = "ok"
    objective_value: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "scenario": {d: getattr(self.scenario, d) for d in SCENARIO_DIMS},
            "surrogate_pred": dict(self.surrogate_pred),
            "twin_verified": {k: dict(v) for k, v in self.twin_verified.items()},
            "feasible_point": self.feasible_point,
            "feasible_robust": self.feasible_robust,
            "constraint_margins": dict(self.constraint_margins),
            "surrogate_twin_gap": dict(self.surrogate_twin_gap),
            "novelty": self.novelty,
            "cost": dict(self.cost),
            "status": self.status,
            "objective_value": self.objective_value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Plan:
        """Parse a plan."""
        return cls(
            scenario=Scenario(**payload["scenario"]),
            surrogate_pred=dict(payload.get("surrogate_pred", {})),
            twin_verified={k: dict(v) for k, v in payload.get("twin_verified", {}).items()},
            feasible_point=bool(payload.get("feasible_point", False)),
            feasible_robust=bool(payload.get("feasible_robust", False)),
            constraint_margins=dict(payload.get("constraint_margins", {})),
            surrogate_twin_gap=dict(payload.get("surrogate_twin_gap", {})),
            novelty=float(payload.get("novelty", float("nan"))),
            cost=dict(payload.get("cost", {})),
            status=payload.get("status", "ok"),
            objective_value=float(payload.get("objective_value", float("nan"))),
        )


@dataclass(frozen=True)
class OptimResult:
    """Everything one optimisation run returns."""

    spec: ObjectiveSpec
    plans: tuple[Plan, ...]
    method: str
    seed: int
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Plan | None:
        """The top-ranked plan, or None when nothing was found."""
        return self.plans[0] if self.plans else None

    @property
    def status(self) -> str:
        """The best plan's status, or ``no_candidate``."""
        return self.plans[0].status if self.plans else "no_candidate"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "spec": self.spec.to_dict(),
            "plans": [p.to_dict() for p in self.plans],
            "method": self.method,
            "seed": self.seed,
            "trace": self.trace,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OptimResult:
        """Parse a result."""
        return cls(
            spec=ObjectiveSpec.from_dict(payload["spec"]),
            plans=tuple(Plan.from_dict(p) for p in payload.get("plans", [])),
            method=str(payload.get("method", "")),
            seed=int(payload.get("seed", 0)),
            trace=dict(payload.get("trace", {})),
        )


def rank_plans(plans: list[Plan]) -> list[Plan]:
    """Feasibility-first ordering, with no penalty weight to choose.

    Feasible beats infeasible, always. Among feasible, the higher objective
    wins. Among infeasible, the smaller total violation wins, and the objective
    breaks ties.

    A weighted penalty would need a weight, and the right weight depends on
    units nobody has calibrated -- how many pounds of profit is one point of
    stockout probability worth? Lexicographic ordering sidesteps that question
    entirely, which is why it is used instead.
    """
    return sorted(
        plans,
        key=lambda p: (
            not p.feasible_point,
            sum(v for v in p.constraint_margins.values() if v < 0) * -1.0
            if not p.feasible_point
            else 0.0,
            -p.objective_value,
        ),
    )
