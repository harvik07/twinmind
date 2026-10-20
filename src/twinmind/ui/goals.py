"""Goal cards: the Plan page's form, turned into a validated goal.

A card plus its limits builds an ``ObjectiveSpec`` directly, so a card goal
needs no language model to be understood. Every card also names the saved
example shown in demo mode, and says when that example is only a related goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from twinmind.optim.contracts import CONTROLLABLE_DIMS, ObjectiveSpec, SpecError
from twinmind.ui.glossary import KPIS, PERIOD

#: The demand month a card goal plans for unless the user picks another.
DEFAULT_MONTH = 8


class GoalFormError(ValueError):
    """The form describes a goal that can't be planned; the message is plain."""


@dataclass(frozen=True)
class LimitField:
    """One "Any limits?" input, in business units."""

    id: str
    label: str
    kpi: str
    op: str  # "<=" | ">="
    unit: str  # "count" | "percent" | "gbp" | "gbp_unit"
    min: float
    max: float
    step: float
    default: float
    help: str

    def threshold(self, value: float) -> float:
        """The constraint threshold in the KPI's own unit (percent -> fraction)."""
        return value / 100.0 if self.unit == "percent" else float(value)


LIMITS: dict[str, LimitField] = {
    f.id: f
    for f in (
        LimitField(
            "max_breakdowns",
            f"Most breakdowns I accept in {PERIOD}",
            "unplanned_failures",
            "<=",
            "count",
            0.0,
            10.0,
            0.5,
            1.0,
            f"Unexpected machine failures over the {PERIOD}, averaged over the simulated "
            "months. Breakdown numbers are rough guides.",
        ),
        LimitField(
            "min_on_time",
            "Lowest on-time delivery I accept (%)",
            "on_time_delivery_rate",
            ">=",
            "percent",
            50.0,
            100.0,
            1.0,
            80.0,
            KPIS["on_time_delivery_rate"].meaning,
        ),
        LimitField(
            "max_short_days",
            "Highest share of days short of stock I accept (%)",
            "stockout_probability",
            "<=",
            "percent",
            0.0,
            100.0,
            1.0,
            30.0,
            KPIS["stockout_probability"].meaning,
        ),
        LimitField(
            "min_profit",
            f"Lowest profit I accept (£ over {PERIOD})",
            "profit",
            ">=",
            "gbp",
            0.0,
            600_000.0,
            10_000.0,
            200_000.0,
            KPIS["profit"].meaning,
        ),
        LimitField(
            "max_unit_cost",
            "Highest cost per unit I accept (£)",
            "unit_cost",
            "<=",
            "gbp_unit",
            10.0,
            100.0,
            0.5,
            30.0,
            KPIS["unit_cost"].meaning,
        ),
    )
}

OVERTIME_HELP = "No keeps overtime at zero hours a day; every other setting may still change."


@dataclass(frozen=True)
class GoalCard:
    id: str
    icon: str
    title: str
    example: str
    kpi: str
    sense: str
    limits: tuple[tuple[str, bool], ...]  # (limit id, on by default)
    overtime_choice: bool
    demo_id: str  # saved example shown in demo mode

    @property
    def limit_fields(self) -> list[tuple[LimitField, bool]]:
        return [(LIMITS[i], on) for i, on in self.limits]


CARDS: dict[str, GoalCard] = {
    c.id: c
    for c in (
        GoalCard(
            "profit",
            "💷",
            "Make more profit",
            "“Earn more without missing deliveries”",
            "profit",
            "maximise",
            (("max_breakdowns", False), ("min_on_time", False), ("max_short_days", False)),
            True,
            "g01",
        ),
        GoalCard(
            "breakdowns",
            "🔧",
            "Reduce breakdowns",
            "“Fewer machine breakdowns a month”",
            "unplanned_failures",
            "minimise",
            (("min_profit", True), ("min_on_time", False)),
            False,
            "g10",
        ),
        GoalCard(
            "on_time",
            "🚚",
            "Deliver orders on time",
            "“Ship more orders on the day they're due”",
            "on_time_delivery_rate",
            "maximise",
            (("min_profit", True), ("max_unit_cost", False)),
            False,
            "g07",
        ),
        GoalCard(
            "unit_cost",
            "🏷️",
            "Lower cost per unit",
            "“Make each unit cheaper”",
            "unit_cost",
            "minimise",
            (("min_on_time", True), ("max_short_days", False)),
            False,
            "g03",
        ),
        GoalCard(
            "stock",
            "📦",
            "Keep stock from running out",
            "“Fewer days short of stock”",
            "stockout_probability",
            "minimise",
            (("min_profit", True),),
            True,
            "aa1",
        ),
    )
}
OWN_WORDS = "own_words"
OWN_WORDS_EXAMPLES = (
    "Make more profit but keep breakdowns to one a month.",
    "Cut the cost per unit while delivering at least 85% of orders on time.",
    "Produce as many units as possible with no overtime.",
)


def build_spec(
    card_id: str,
    limits: dict[str, float | None],
    allow_overtime: bool = True,
    month: int = DEFAULT_MONTH,
) -> ObjectiveSpec:
    """The goal a card and its limits describe. ``None`` = limit not used.

    Raises :class:`GoalFormError` with a plain message when the limits can't
    all hold or a value is out of range; never a technical exception.
    """
    card = CARDS[card_id]
    constraints = []
    for limit_id, value in limits.items():
        if value is None:
            continue
        field = LIMITS[limit_id]
        if limit_id not in dict(card.limits):
            raise GoalFormError(f"“{field.label}” isn't a limit for this goal.")
        if not field.min <= float(value) <= field.max:
            raise GoalFormError(f"“{field.label}” must be between {field.min:g} and {field.max:g}.")
        constraints.append(
            {"kpi": field.kpi, "op": field.op, "threshold": field.threshold(float(value))}
        )
    free = [] if allow_overtime else [d for d in CONTROLLABLE_DIMS if d != "overtime_hours"]
    payload: dict[str, Any] = {
        "name": goal_text_from(card.kpi, card.sense, constraints, allow_overtime),
        "objective": {card.kpi: 1.0},
        "sense": card.sense,
        "constraints": constraints,
        "free_dims": free,
        "context_id": int(month),
    }
    try:
        return ObjectiveSpec.from_dict(payload)
    except (SpecError, ValueError) as error:
        raise GoalFormError(
            "These limits contradict each other. Change one of them and try again."
        ) from error


# --------------------------------------------------------------------------- #
# Plain goal text
# --------------------------------------------------------------------------- #
_GOAL_PHRASE = {
    ("profit", "maximise"): "Make more profit",
    ("unplanned_failures", "minimise"): "Reduce breakdowns",
    ("on_time_delivery_rate", "maximise"): "Deliver more orders on time",
    ("unit_cost", "minimise"): "Lower the cost per unit",
    ("stockout_probability", "minimise"): "Have fewer days short of stock",
    ("units_produced", "maximise"): "Make more units",
    ("oee", "maximise"): "Raise equipment efficiency (OEE)",
    ("downtime_hours", "minimise"): "Reduce downtime",
}


def limit_phrase(kpi: str, op: str, threshold: float) -> str:
    """One limit in plain words: "at most 1 breakdown in 4 weeks"."""
    most = "at most" if op == "<=" else "at least"
    if kpi == "unplanned_failures":
        noun = "breakdown" if threshold == 1 else "breakdowns"
        return f"{most} {threshold:g} {noun} in {PERIOD}"
    if kpi == "downtime_hours":
        return f"{most} {threshold:g} hours of downtime"
    if kpi == "on_time_delivery_rate":
        return f"on-time delivery of {most} {100 * threshold:g}%"
    if kpi == "stockout_probability":
        return f"short of stock on {most} {100 * threshold:g}% of days"
    if kpi == "profit":
        return f"profit of {most} £{threshold:,.0f}"
    if kpi == "unit_cost":
        return f"cost per unit of {most} £{threshold:,.2f}"
    if kpi == "units_produced":
        return f"{most} {threshold:,.0f} units made"
    return f"{KPIS[kpi].name.lower()} {most} {threshold:g}"


def goal_text_from(
    objective: str | dict[str, float],
    sense: str,
    constraints: list[dict[str, Any]],
    allow_overtime: bool = True,
) -> str:
    """A goal in one plain sentence, from its parts."""
    terms = [objective] if isinstance(objective, str) else list(objective)
    phrases = [_GOAL_PHRASE.get((k, sense), f"Improve {KPIS[k].name.lower()}") for k in terms]
    text = " and ".join([phrases[0], *(p[0].lower() + p[1:] for p in phrases[1:])])
    limits = [limit_phrase(c["kpi"], c["op"], float(c["threshold"])) for c in constraints]
    if limits:
        text += ", with " + " and ".join(limits)
    if not allow_overtime:
        text += ", without overtime"
    return text + "."


def goal_text(spec: dict[str, Any]) -> str:
    """Plain sentence for a stored goal (an AgentResult's ``spec``)."""
    free = spec.get("free_dims") or []
    no_overtime = bool(free) and "overtime_hours" not in free and len(free) == 15
    return goal_text_from(
        spec.get("objective", {}),
        spec.get("sense", "maximise"),
        list(spec.get("constraints", [])),
        allow_overtime=not no_overtime,
    )


def is_related_only(card_id: str, saved_spec: dict[str, Any]) -> bool:
    """True when the saved example's goal improves something other than the card's."""
    card = CARDS[card_id]
    return set(saved_spec.get("objective", {})) != {card.kpi} or (
        saved_spec.get("sense", "maximise") != card.sense
    )
