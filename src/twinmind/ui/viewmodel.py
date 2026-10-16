"""View models: everything the dashboard shows, built from an AgentResult.

Pure Python -- no Streamlit import -- so every rule about what the UI may say
is unit-tested here once, and every page inherits it:

a. A surrogate number is an **estimate** and is never given an interval. A twin
   number is **verified on twin (N reps)** and always carries its 95% CI.
b. The UI never says "optimal", "guaranteed" or "proven" (see
   :func:`assert_no_overclaim`).
c. The surrogate ensemble's epistemic std is never shown as uncertainty: it
   failed out-of-distribution detection (AUROC 0.590). :func:`estimate_table`
   takes predicted MEANS only. Coverage is shown as the novelty score's band.
d. ``downtime_hours`` and ``unplanned_failures`` always carry the indicative
   badge (:data:`WEAK_KPI_BADGE`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from twinmind.agents.contracts import AgentResult
from twinmind.agents.explanation_v2 import FORBIDDEN_WORDS, KPI_FORMAT
from twinmind.optim.contracts import HIGHER_IS_BETTER
from twinmind.twin.runner import KPI_NAMES

ESTIMATE_LABEL = "estimate"
VERIFIED_LABEL = "verified on twin ({n} reps)"

#: The two KPIs the surrogate predicts worst, and the badge they always carry.
WEAK_KPIS = ("downtime_hours", "unplanned_failures")
WEAK_KPI_BADGE = (
    "indicative: surrogate R² 0.91 / 0.73, and sensitive to the health-model assumption "
    "(feasibility flipped in 7 of 40 benchmark cells under a fixed-failure-rate hazard)"
)

#: Novelty bands, from the NoveltyScorer's distribution over the 1,000 T1
#: scenarios (never the OOD set): inside up to its 95th percentile -- the
#: critic's threshold -- edge up to the largest T1 score, outside beyond.
NOVELTY_T1_P95 = 0.987382
NOVELTY_T1_MAX = 1.097809

KPI_DISPLAY: dict[str, str] = {
    "units_produced": "Units produced",
    "oee": "OEE",
    "unit_cost": "Unit cost",
    "profit": "Profit",
    "stockout_probability": "Stockout probability",
    "downtime_hours": "Downtime",
    "unplanned_failures": "Unplanned failures",
    "on_time_delivery_rate": "On-time delivery",
}

#: Plain-English critic flags. Each says what the check found, not more.
FLAG_TEXT: dict[str, str] = {
    "weak_kpi_constraint": "A constraint uses a KPI the surrogate predicts poorly "
    "(downtime or unplanned failures), so the search was steered by its weakest estimates.",
    "not_robustly_feasible": "The plan meets the constraints on average, but its 95% "
    "interval crosses a limit: it may not meet them in a bad month.",
    "novel_plan": "The plan lies beyond most of what the surrogate was trained on, so "
    "the surrogate's estimates for it are less trustworthy.",
    "no_improvement": "The plan does not beat the current policy by more than the noise, "
    "so it is not recommended.",
    "infeasible_unverified": "No shortlisted plan met every constraint on the twin.",
}

STATUS_TEXT: dict[str, tuple[str, str]] = {
    # status -> (label, tone)
    "ok": ("Plan found", "good"),
    "infeasible": ("No feasible plan found", "warning"),
    "needs_clarification": ("Needs clarification", "info"),
    "spec_invalid": ("Could not use this goal", "error"),
    "degraded_template": ("Plan found; explanation from the fallback template", "warning"),
}


class OverclaimError(ValueError):
    """UI text used a word the dashboard must never use."""


def assert_no_overclaim(text: str) -> str:
    """Return ``text`` unchanged, or raise if it says optimal/guaranteed/proven."""
    found = [w for w in FORBIDDEN_WORDS if re.search(rf"\b{w}\b", text.lower())]
    if found:
        raise OverclaimError(f"forbidden word(s) {found} in UI text: {text[:120]!r}")
    return text


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def format_value(kpi: str, value: float | None) -> str:
    """A KPI value in its unit; fractions as percentages. Plain decimals only."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    unit, decimals = KPI_FORMAT[kpi]
    if unit == "fraction 0-1":
        return f"{100.0 * value:.1f}%"
    sign = "−" if value < 0 else ""
    if unit == "GBP":
        return f"{sign}£{abs(value):,.0f}"
    if unit == "GBP per unit":
        return f"{sign}£{abs(value):,.2f}"
    return f"{value:,.{decimals}f}" + {"hours": " h", "units": "", "failures": ""}.get(unit, "")


def format_delta(kpi: str, value: float) -> str:
    """A signed change; fractions in percentage points."""
    unit, decimals = KPI_FORMAT[kpi]
    sign = "+" if value > 0 else ("−" if value < 0 else "±")
    magnitude = abs(value)
    if unit == "fraction 0-1":
        return f"{sign}{100.0 * magnitude:.1f} pp"
    if unit == "GBP":
        return f"{sign}£{magnitude:,.0f}"
    if unit == "GBP per unit":
        return f"{sign}£{magnitude:,.2f}"
    return f"{sign}{magnitude:,.{decimals}f}" + (" h" if unit == "hours" else "")


def format_ci(kpi: str, low: float, high: float) -> str:
    """A 95% interval in the KPI's unit."""
    return f"95% CI {format_value(kpi, low)} to {format_value(kpi, high)}"


# --------------------------------------------------------------------------- #
# View objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KPIRow:
    """One KPI line of a table."""

    kpi: str
    name: str
    source_label: str  # ESTIMATE_LABEL or VERIFIED_LABEL.format(n=...)
    value: str  # the plan's value (or the estimate)
    value_ci: str | None  # None for an estimate, always set when verified
    baseline: str | None
    baseline_ci: str | None
    change: str | None
    change_ci: str | None
    better_when: str
    improved: bool | None
    badge: str | None


@dataclass(frozen=True)
class RecommendationCard:
    """The headline of a result."""

    goal_text: str
    status: str
    status_label: str
    tone: str
    recommended: bool
    headline: str
    reason: str | None
    explanation: str | None
    explanation_source: str | None
    question: str | None
    refusal: str | None
    objective: str | None
    constraints: list[str] = field(default_factory=list)
    plan_changes: list[str] = field(default_factory=list)
    verified_label: str | None = None


@dataclass(frozen=True)
class NoveltyView:
    """Where a plan sits relative to the surrogate's training coverage."""

    score: float | None
    band: str  # "inside" | "edge" | "outside" | "unknown"
    text: str


@dataclass(frozen=True)
class FlagView:
    code: str
    text: str


@dataclass(frozen=True)
class OptionView:
    constraint: str
    closest: str
    option: str
    note: str = "An option for you to decide on, not a decision."


@dataclass(frozen=True)
class RunStatusView:
    label: str
    tone: str
    detail: str


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _objective_text(spec: dict[str, Any]) -> str:
    terms = spec.get("objective", {})
    sense = "Maximise" if spec.get("sense", "maximise") == "maximise" else "Minimise"
    if len(terms) == 1:
        return f"{sense} {KPI_DISPLAY.get(next(iter(terms)), next(iter(terms)))}"
    parts = [f"{w:g} × {KPI_DISPLAY.get(k, k)}" for k, w in terms.items()]
    return f"{sense} " + " + ".join(parts)


def _constraint_text(c: dict[str, Any]) -> str:
    kpi = c["kpi"]
    op = "at most" if c["op"] == "<=" else "at least"
    return f"{KPI_DISPLAY.get(kpi, kpi)} {op} {format_value(kpi, float(c['threshold']))}"


def plan_changes(result: AgentResult) -> list[str]:
    """Levers the plan moves away from the current policy, as text."""
    from twinmind.optim.contracts import ObjectiveSpec

    if result.plan is None or result.spec is None:
        return []
    current = ObjectiveSpec.from_dict(result.spec).base_scenario().model_dump()
    out = []
    for dim, value in result.plan["scenario"].items():
        before = float(current[dim])
        after = float(value)
        if abs(before - after) > 1e-6 * max(1.0, abs(before)):
            out.append(f"{dim}: {before:,.2f} → {after:,.2f}")
    return out


def recommendation_card(result: AgentResult) -> RecommendationCard:
    """The headline card for a result of any status."""
    label, tone = STATUS_TEXT[result.status]
    recommended = bool(result.plan and result.plan.get("recommended"))
    reason = result.plan.get("recommendation_reason") if result.plan else None
    if result.status == "needs_clarification":
        headline = "The goal needs one more detail before anything is run."
    elif result.status == "spec_invalid":
        headline = "This goal could not be turned into a valid plan request."
    elif result.status == "infeasible":
        headline = "No plan met every constraint on the twin."
    elif recommended:
        headline = "Recommended plan, verified on the twin."
    else:
        headline = "A plan was found but is not recommended."
    n_reps = (result.baseline_comparison or {}).get("n_reps")
    if result.explanation_mode == "llm":
        source = "Written by the language model from the verified facts, then checked."
    elif result.explanation_mode == "template":
        source = "Generated from a fixed template (the model's text did not pass the checks)."
    else:
        source = None
    card = RecommendationCard(
        goal_text=result.goal_text,
        status=result.status,
        status_label=label,
        tone=tone,
        recommended=recommended,
        headline=headline,
        reason=reason,
        explanation=result.explanation,
        explanation_source=source,
        question=result.question,
        refusal=result.refusal,
        objective=_objective_text(result.spec) if result.spec else None,
        constraints=[_constraint_text(c) for c in (result.spec or {}).get("constraints", [])],
        plan_changes=plan_changes(result),
        verified_label=VERIFIED_LABEL.format(n=n_reps) if n_reps else None,
    )
    for text in (card.headline, card.reason or "", card.explanation or "", card.status_label):
        assert_no_overclaim(text)
    return card


def kpi_table(result: AgentResult) -> list[KPIRow]:
    """Plan vs current policy, verified on the twin, with CIs. Empty if no plan."""
    evaluation = result.baseline_comparison
    if not evaluation:
        return []
    label = VERIFIED_LABEL.format(n=evaluation["n_reps"])
    rows = []
    for kpi in KPI_NAMES:
        row = evaluation["kpis"][kpi]
        delta = float(row["diff"]["mean"])
        better_when = "higher" if kpi in HIGHER_IS_BETTER else "lower"
        improved = None if delta == 0 else (delta > 0) == (better_when == "higher")
        rows.append(
            KPIRow(
                kpi=kpi,
                name=KPI_DISPLAY[kpi],
                source_label=label,
                value=format_value(kpi, row["plan"]["mean"]),
                value_ci=format_ci(kpi, row["plan"]["ci_low"], row["plan"]["ci_high"]),
                baseline=format_value(kpi, row["baseline"]["mean"]),
                baseline_ci=format_ci(kpi, row["baseline"]["ci_low"], row["baseline"]["ci_high"]),
                change=format_delta(kpi, delta),
                change_ci=f"95% CI {format_delta(kpi, row['diff']['ci_low'])} to "
                f"{format_delta(kpi, row['diff']['ci_high'])}",
                better_when=better_when,
                improved=improved,
                badge=WEAK_KPI_BADGE if kpi in WEAK_KPIS else None,
            )
        )
    return rows


def estimate_table(means: dict[str, float]) -> list[KPIRow]:
    """Surrogate estimates: means ONLY, labelled "estimate", never an interval.

    Takes the predicted means and nothing else on purpose: the ensemble's
    epistemic std failed OOD detection (AUROC 0.590) and is not shown.
    """
    rows = []
    for kpi in KPI_NAMES:
        rows.append(
            KPIRow(
                kpi=kpi,
                name=KPI_DISPLAY[kpi],
                source_label=ESTIMATE_LABEL,
                value=format_value(kpi, float(means[kpi])),
                value_ci=None,
                baseline=None,
                baseline_ci=None,
                change=None,
                change_ci=None,
                better_when="higher" if kpi in HIGHER_IS_BETTER else "lower",
                improved=None,
                badge=WEAK_KPI_BADGE if kpi in WEAK_KPIS else None,
            )
        )
    return rows


def twin_table(
    means: dict[str, float],
    lows: dict[str, float],
    highs: dict[str, float],
    n_reps: int,
) -> list[KPIRow]:
    """A what-if plan run on the twin: means WITH 95% CIs, labelled with the
    number of replications. At least 2: one run has no interval, and a twin
    number is never shown without one."""
    if n_reps < 2:
        raise ValueError("a twin-verified number needs at least 2 replications for its CI")
    label = VERIFIED_LABEL.format(n=n_reps)
    return [
        KPIRow(
            kpi=kpi,
            name=KPI_DISPLAY[kpi],
            source_label=label,
            value=format_value(kpi, float(means[kpi])),
            value_ci=format_ci(kpi, float(lows[kpi]), float(highs[kpi])),
            baseline=None,
            baseline_ci=None,
            change=None,
            change_ci=None,
            better_when="higher" if kpi in HIGHER_IS_BETTER else "lower",
            improved=None,
            badge=WEAK_KPI_BADGE if kpi in WEAK_KPIS else None,
        )
        for kpi in KPI_NAMES
    ]


def flags_plain(result: AgentResult) -> list[FlagView]:
    """The critic's flags in plain English, in the order raised."""
    return [
        FlagView(code=f["code"], text=assert_no_overclaim(FLAG_TEXT.get(f["code"], f["message"])))
        for f in result.flags
    ]


def closest_options(result: AgentResult) -> list[OptionView]:
    """Closest-achievable values for an infeasible goal, as options."""
    out = []
    for c in result.closest_achievable:
        kpi = c["kpi"]
        out.append(
            OptionView(
                constraint=_constraint_text(
                    {"kpi": kpi, "op": c["op"], "threshold": c["threshold"]}
                ),
                closest="Closest any verified plan came: "
                + format_value(kpi, c["closest_achievable"]),
                option=c["option"],
            )
        )
    return out


def novelty_view(score: float | None) -> NoveltyView:
    """The novelty band -- shown instead of the surrogate's epistemic std."""
    if score is None or not math.isfinite(float(score)):
        return NoveltyView(score=None, band="unknown", text="Coverage not assessed.")
    if score <= NOVELTY_T1_P95:
        band, text = "inside", "Inside training coverage: similar plans were in the training data."
    elif score <= NOVELTY_T1_MAX:
        band, text = (
            "edge",
            "Edge of training coverage: the surrogate has seen few plans like this.",
        )
    else:
        band, text = (
            "outside",
            "Outside training coverage: treat the surrogate's estimate with caution; "
            "only the twin-verified numbers carry weight.",
        )
    return NoveltyView(score=float(score), band=band, text=text)


def run_status(result: AgentResult) -> RunStatusView:
    """What happened, in one line, plus the cost of finding out."""
    label, tone = STATUS_TEXT[result.status]
    trace = result.trace or {}
    detail = (
        f"{trace.get('llm_calls_total', 0)} language-model call(s), "
        f"{trace.get('twin_runs', 0)} twin runs, "
        f"{float(trace.get('wall_seconds', 0.0)):.0f} s"
    )
    if trace.get("critic_actions") and trace["critic_actions"][0] != "accept":
        detail += "; the critic retried once"
    return RunStatusView(label=label, tone=tone, detail=detail)
