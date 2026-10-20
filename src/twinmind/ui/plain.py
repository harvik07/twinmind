"""Plain-language view of a result, for a factory manager.

Everything here is computed from the AgentResult's twin numbers and plan by
fixed rules, with no Streamlit import:

- the status chip and the one-sentence headline;
- "What to change": every moved setting in plain words with units;
- "What you get": four KPI cards and a "More numbers" table;
- "Things to watch": side effects and critic flags, with a fixed severity;
- "Why this plan": a summary of numbers and setting changes only, in a fixed
  template that makes no causal claim;
- "How much to trust this": a deterministic three-level rule (docs/ux_spec.md
  A3 f), with its reasons and no probabilities.

The exact technical labels stay in ``viewmodel`` and are shown under "Show
technical details".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from twinmind.agents.contracts import AgentResult
from twinmind.optim.contracts import CONTROLLABLE_DIMS, ObjectiveSpec
from twinmind.twin.runner import KPI_NAMES
from twinmind.ui import viewmodel as vm
from twinmind.ui.glossary import KPIS, MACHINES, PERIOD, ROUGH_KPI_NOTE, SETTINGS
from twinmind.ui.goals import goal_text, limit_phrase

CHECKED_LABEL = "Checked with the detailed simulation ({n} runs)"
ESTIMATE_LABEL = "Quick estimate (instant, rough: not checked)"
SAVED_LABEL = "Saved example result, not a live run"
HEADLINE_KPIS = ("profit", "on_time_delivery_rate", "unplanned_failures", "unit_cost")
WEAK_GOAL_REASON = (
    "Breakdown and downtime numbers are less reliable in this model, so treat this as a "
    "rough guide."
)


# --------------------------------------------------------------------------- #
# Numbers in plain units
# --------------------------------------------------------------------------- #
def value_text(kpi: str, value: float | None) -> str:
    """A KPI value with its unit (fractions as %, money with £)."""
    return vm.format_value(kpi, value)


def change_text(kpi: str, delta: float) -> str:
    """A signed change with an ASCII sign (st.metric reads it for the arrow)."""
    unit, decimals = vm.KPI_FORMAT[kpi]
    sign = "+" if delta > 0 else ("-" if delta < 0 else "±")
    size = abs(delta)
    if unit == "fraction 0-1":
        return f"{sign}{100 * size:.1f} points"
    if unit == "GBP":
        return f"{sign}£{size:,.0f}"
    if unit == "GBP per unit":
        return f"{sign}£{size:,.2f}"
    if unit == "hours":
        return f"{sign}{size:,.1f} h"
    return f"{sign}{size:,.{max(decimals, 0)}f}"


def _better(kpi: str, delta: float) -> bool:
    return (delta > 0) == KPIS[kpi].higher_is_better


@dataclass(frozen=True)
class Verdict:
    word: str  # "better" | "worse" | "no clear change"
    clear: bool


def verdict(kpi: str, delta: float, low: float, high: float) -> Verdict:
    """Better/worse only when the change's 95% interval excludes zero."""
    if low <= 0 <= high or delta == 0:
        return Verdict("no clear change", False)
    return Verdict("better" if _better(kpi, delta) else "worse", True)


# --------------------------------------------------------------------------- #
# Status chip and headline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chip:
    icon: str
    text: str
    color: str  # st.badge colour; the text always carries the meaning too


CHIPS = {
    "recommended": Chip("✅", "Recommended plan", "green"),
    "no_better": Chip("➖", "No better plan found: your current setup is already close", "orange"),
    "not_found": Chip(
        "🔍",
        "We couldn't find a plan that meets these limits. Closest options below.",
        "orange",
    ),
    "question": Chip("❓", "We need one more detail", "blue"),
    "unusable": Chip("⛔", "We couldn't use this goal", "red"),
}


def outcome(result: AgentResult) -> str:
    """Which of the five outcomes a result is."""
    if result.status == "needs_clarification":
        return "question"
    if result.status == "spec_invalid":
        return "unusable"
    if result.status == "infeasible":
        return "not_found"
    if result.plan and result.plan.get("recommended"):
        return "recommended"
    return "no_better"


def chip(result: AgentResult) -> Chip:
    return CHIPS[outcome(result)]


def _objective_kpis(result: AgentResult) -> list[str]:
    return list((result.spec or {}).get("objective", {}))


def _constraints(result: AgentResult) -> list[dict[str, Any]]:
    return list((result.spec or {}).get("constraints", []))


def _row(result: AgentResult, kpi: str) -> dict[str, Any]:
    return (result.baseline_comparison or {})["kpis"][kpi]


def _moves(result: AgentResult) -> list[str]:
    parts = []
    for kpi in _objective_kpis(result):
        row = _row(result, kpi)
        parts.append(
            f"{KPIS[kpi].name.lower() if kpi != 'oee' else KPIS[kpi].name} goes from "
            f"{value_text(kpi, row['baseline']['mean'])} to "
            f"{value_text(kpi, row['plan']['mean'])} "
            f"({change_text(kpi, row['diff']['mean'])})"
        )
    return parts


def _robust(result: AgentResult) -> bool:
    return bool((result.baseline_comparison or {}).get("feasible_robust"))


def headline(result: AgentResult, unreachable: bool = False) -> str:
    """One plain sentence: what to do and what it gives."""
    kind = outcome(result)
    if kind == "question":
        return f"We need one more detail: {result.question}"
    if kind == "unusable":
        return error_message("ai_unreachable" if unreachable else "not_understood")
    if kind == "not_found":
        options = closest_lines(result)
        first = f" The closest any checked plan came: {options[0]}" if options else ""
        return f"None of the plans we checked met all your limits.{first}"
    changes = setting_changes(result)
    n = len(changes)
    moves = " and ".join(_moves(result))
    if kind == "recommended":
        limits = _constraints(result)
        tail = ""
        if limits:
            margin = "with a safety margin" if _robust(result) else "on average"
            tail = f", with your limits met {margin}"
        return (
            f"Change {n} setting{'s' if n != 1 else ''}: {moves[0].upper() + moves[1:]} "
            f"over {PERIOD}{tail}."
        )
    objective = (result.baseline_comparison or {}).get("objective", {}).get("diff", {})
    unclear = objective.get("ci_low", -1) <= 0 <= objective.get("ci_high", 1)
    size = (
        "too small to tell apart from normal month-to-month variation"
        if unclear
        else "not an improvement"
    )
    return f"Keep your current setup. With the best plan found, {moves}: {size}."


def closest_lines(result: AgentResult) -> list[str]:
    """Closest-achievable values, as plain options."""
    out = []
    for c in result.closest_achievable:
        kpi = c["kpi"]
        asked = limit_phrase(kpi, c["op"], float(c["threshold"]))
        out.append(
            f"{KPIS[kpi].name} {value_text(kpi, c['closest_achievable'])} (you asked for {asked})."
        )
    return out


# --------------------------------------------------------------------------- #
# What to change
# --------------------------------------------------------------------------- #
def _hours_per_day(scenario: dict[str, Any]) -> float:
    return 8.0 * float(scenario.get("shift_count", 2)) + float(scenario.get("overtime_hours", 0))


def _days(hours: float, per_day: float) -> str:
    days = hours / per_day if per_day > 0 else float("nan")
    if not math.isfinite(days):
        return "?"
    return f"{days:.1f}".rstrip("0").rstrip(".") if days < 10 else f"{days:.0f}"


@dataclass(frozen=True)
class SettingChange:
    code: str
    text: str
    size: float  # |change| as a share of the setting's range, for sorting


def setting_changes(result: AgentResult) -> list[SettingChange]:
    """Every controllable setting the plan moves, biggest change first."""
    if result.plan is None or result.spec is None:
        return []
    from twinmind.ui.services import slider_specs

    ranges = {s.name: (s.min, s.max) for s in slider_specs()}
    current = ObjectiveSpec.from_dict(result.spec).base_scenario().model_dump()
    plan = result.plan["scenario"]
    out = []
    for code in CONTROLLABLE_DIMS:
        before, after = float(current[code]), float(plan[code])
        if abs(before - after) <= 1e-6 * max(1.0, abs(before)):
            continue
        low, high = ranges[code]
        out.append(
            SettingChange(
                code,
                _change_line(code, before, after, current, plan),
                abs(after - before) / (high - low),
            )
        )
    return sorted(out, key=lambda c: -c.size)


def _change_line(
    code: str, before: float, after: float, current: dict[str, Any], plan: dict[str, Any]
) -> str:
    term = SETTINGS[code]
    if code.startswith("pm_interval_"):
        m = code.rsplit("_", 1)[1]
        how = "less often" if after > before else "more often"
        return (
            f"Service the {MACHINES[m]} machine ({m}) every {after:,.0f} running hours "
            f"instead of {before:,.0f}: {how} (about every "
            f"{_days(after, _hours_per_day(plan))} working days instead of "
            f"{_days(before, _hours_per_day(current))})"
        )
    if term.display == "percent":
        return f"{term.name}: {100 * after:.0f}% of normal instead of {100 * before:.0f}%"
    if code == "shift_count":
        return f"Shifts per day: {after:.0f} instead of {before:.0f}"
    if code == "overtime_hours":
        was = "none" if before == 0 else f"{before:.1f} hours"
        now = "none" if after == 0 else f"{after:.1f} hours a day"
        return f"Overtime: {now} instead of {was}"
    return f"{term.name}: {after:,.0f} units instead of {before:,.0f}"


# --------------------------------------------------------------------------- #
# What you get
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KPICard:
    """One headline number: the new value large, the current one under it, and
    the change as an arrow, a coloured line and a WORD (never colour alone)."""

    kpi: str
    label: str
    after: str
    before: str | None  # None when there is nothing to compare with
    change: str  # "+£84,109"
    word: str  # "better" | "worse" | "no clear change" | "about the same"
    help: str
    rough: bool
    rough_estimate: bool = False  # a quick estimate: the word is marked "(rough)"

    @property
    def color(self) -> str:
        return {"better": "green", "worse": "red"}.get(self.word, "gray")

    @property
    def arrow(self) -> str:
        if self.word in ("no clear change", "about the same") or not self.change:
            return "➖"
        return "⬆" if self.change.startswith("+") else "⬇"

    @property
    def line(self) -> str:
        suffix = " (rough)" if self.rough_estimate else ""
        return f"{self.arrow} {self.change} · {self.word}{suffix}".strip()


def card_kpis(result: AgentResult) -> list[str]:
    """The goal's own KPIs first, then the usual headline four, four in all."""
    order = _objective_kpis(result) + [c["kpi"] for c in _constraints(result)]
    order += list(HEADLINE_KPIS)
    out: list[str] = []
    for kpi in order:
        if kpi not in out:
            out.append(kpi)
    return out[:4]


def kpi_cards(result: AgentResult) -> list[KPICard]:
    if not result.baseline_comparison:
        return []
    cards = []
    for kpi in card_kpis(result):
        row = _row(result, kpi)
        diff = row["diff"]
        v = verdict(kpi, diff["mean"], diff["ci_low"], diff["ci_high"])
        term = KPIS[kpi]
        likely = (
            f" Checked value likely between {value_text(kpi, row['plan']['ci_low'])} and "
            f"{value_text(kpi, row['plan']['ci_high'])}."
        )
        cards.append(
            KPICard(
                kpi=kpi,
                label=term.name,
                after=value_text(kpi, row["plan"]["mean"]),
                before=value_text(kpi, row["baseline"]["mean"]),
                change=change_text(kpi, diff["mean"]),
                word=v.word,
                help=term.help + likely,
                rough=term.rough,
            )
        )
    return cards


def estimate_cards(now: dict[str, float], new: dict[str, float]) -> list[KPICard]:
    """Quick-estimate cards for Try changes: no range, and the word says "(rough)"."""
    out = []
    for kpi in HEADLINE_KPIS:
        term = KPIS[kpi]
        delta = new[kpi] - now[kpi]
        if abs(delta) < 1e-12:
            word = "about the same"
        else:
            word = "better" if _better(kpi, delta) else "worse"
        out.append(
            KPICard(
                kpi=kpi,
                label=term.name,
                after=value_text(kpi, new[kpi]),
                before=value_text(kpi, now[kpi]),
                change=change_text(kpi, delta),
                word=word,
                help=term.help + " Quick estimate: no range is given.",
                rough=term.rough,
                rough_estimate=True,
            )
        )
    return out


def more_numbers(result: AgentResult) -> list[dict[str, str]]:
    """The other KPIs, one table, one label (checked with the detailed simulation)."""
    if not result.baseline_comparison:
        return []
    shown = set(card_kpis(result))
    rows = []
    for kpi in KPI_NAMES:
        if kpi in shown:
            continue
        row = _row(result, kpi)
        diff = row["diff"]
        term = KPIS[kpi]
        rows.append(
            {
                "Number": term.name,
                "Current setup": value_text(kpi, row["baseline"]["mean"]),
                "With the plan": value_text(kpi, row["plan"]["mean"]),
                "Change": change_text(kpi, diff["mean"]),
                "Verdict": verdict(kpi, diff["mean"], diff["ci_low"], diff["ci_high"]).word,
                "Note": "⚠ rough guide" if term.rough else "",
            }
        )
    return rows


def checked_label(result: AgentResult) -> str:
    n = (result.baseline_comparison or {}).get("n_reps", 0)
    return CHECKED_LABEL.format(n=n)


# --------------------------------------------------------------------------- #
# Things to watch
# --------------------------------------------------------------------------- #
SEVERITY = {
    "high": ("🔴", "High"),
    "medium": ("🟠", "Medium"),
    "low": ("🟡", "Low"),
    "info": ("ℹ️", "Info"),
}
FLAG_WATCH: dict[str, tuple[str, str]] = {
    "not_robustly_feasible": (
        "medium",
        "Your limits are met on average, but not with a safety margin: in a bad month they "
        "may not hold.",
    ),
    "weak_kpi_constraint": (
        "low",
        "Your goal uses breakdown or downtime numbers. " + WEAK_GOAL_REASON,
    ),
    "novel_plan": (
        "low",
        "These settings are unlike most the model has seen, so its quick estimates were less "
        "reliable. The numbers shown come from the detailed simulation.",
    ),
    "no_improvement": ("info", "The best plan found isn't clearly better than your current setup."),
    "infeasible_unverified": (
        "info",
        "None of the shortlisted plans met every limit in the detailed simulation.",
    ),
}


@dataclass(frozen=True)
class WatchItem:
    icon: str
    severity: str
    text: str


def things_to_watch(result: AgentResult) -> list[WatchItem]:
    """Side effects first, then the critic's flags, each with a fixed severity."""
    items: list[tuple[str, str]] = []
    for effect in vm.side_effects(result):
        kpi = effect.kpi
        row = _row(result, kpi)
        level = "high" if kpi in ("profit", "on_time_delivery_rate") else "medium"
        before, after = (
            value_text(kpi, row["baseline"]["mean"]),
            value_text(kpi, row["plan"]["mean"]),
        )
        items.append(
            (
                level,
                f"{KPIS[kpi].name} gets worse: {before} → {after}. It isn't part of your goal, "
                "so check you can live with it.",
            )
        )
    codes = [f["code"] for f in result.flags]
    for code in codes:
        if code in FLAG_WATCH:
            items.append(FLAG_WATCH[code])
    if result.baseline_comparison and "weak_kpi_constraint" not in codes:
        items.append(("low", ROUGH_KPI_NOTE))
    order = list(SEVERITY)
    items.sort(key=lambda i: order.index(i[0]))
    return [WatchItem(SEVERITY[s][0], SEVERITY[s][1], t) for s, t in items]


# --------------------------------------------------------------------------- #
# Why this plan: numbers and setting changes only, no causal claims
# --------------------------------------------------------------------------- #
def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def plain_summary(result: AgentResult) -> str:
    """A fixed-template summary. It states numbers; it never says why."""
    kind = outcome(result)
    if kind in ("question", "unusable") or not result.baseline_comparison:
        return ""
    if kind == "not_found":
        n = len(result.candidates)
        closest = " ".join(closest_lines(result))
        return f"We checked {n} candidate plans. None met all your limits. {closest}".strip()
    sentences = []
    for kpi in _objective_kpis(result):
        row = _row(result, kpi)
        d = row["diff"]
        v = verdict(kpi, d["mean"], d["ci_low"], d["ci_high"])
        clarity = "a clear difference" if v.clear else "not a clear difference"
        sentences.append(
            f"{KPIS[kpi].name}: {value_text(kpi, row['plan']['mean'])} with the "
            f"{'plan' if kind == 'recommended' else 'best plan found'}, against "
            f"{value_text(kpi, row['baseline']['mean'])} now "
            f"({change_text(kpi, d['mean'])}, {clarity})."
        )
    for c in _constraints(result):
        kpi = c["kpi"]
        row = _row(result, kpi)
        sentences.append(
            f"Your limit, {limit_phrase(kpi, c['op'], float(c['threshold']))}: "
            f"{value_text(kpi, row['plan']['mean'])} with the plan."
        )
    changes = setting_changes(result)
    if kind == "recommended" and changes:
        top = ", ".join(_lower_first(SETTINGS[c.code].name) for c in changes[:3])
        sentences.append(
            f"{len(changes)} setting{'s' if len(changes) != 1 else ''} change; the biggest "
            f"changes are to {top}."
        )
    if kind == "no_better":
        sentences.append("The recommendation is to keep the current setup.")
    return " ".join(sentences)


# --------------------------------------------------------------------------- #
# How much to trust this
# --------------------------------------------------------------------------- #
TRUST_LEVELS = {
    "checked": ("●●●", "Checked", "green"),
    "partly": ("●●○", "Partly checked", "orange"),
    "rough": ("●○○", "Rough guide", "gray"),
}


@dataclass(frozen=True)
class TrustCheck:
    passed: bool | None  # None = not applicable (counts as passed)
    text: str

    @property
    def icon(self) -> str:
        return {True: "✓", False: "✗", None: "–"}[self.passed]


@dataclass(frozen=True)
class Trust:
    level: str  # key of TRUST_LEVELS
    conclusion: str
    checks: list[TrustCheck] = field(default_factory=list)

    @property
    def dots(self) -> str:
        return TRUST_LEVELS[self.level][0]

    @property
    def label(self) -> str:
        return TRUST_LEVELS[self.level][1]

    @property
    def color(self) -> str:
        return TRUST_LEVELS[self.level][2]


def trust(result: AgentResult) -> Trust | None:
    """The deterministic rule of docs/ux_spec.md A3 (f). None when no numbers.

    T1 checked with the detailed simulation; T2 limits met with a safety margin
    (only for a recommended plan with limits); T3 familiar territory (novelty
    band "inside"); T4 the goal avoids breakdown and downtime numbers.
    T1 fails -> Rough guide; else 0 / 1 / 2+ failures among T2-T4 ->
    Checked / Partly checked / Rough guide.
    """
    evaluation = result.baseline_comparison
    if not evaluation:
        return None
    kind = outcome(result)
    n_reps = int(evaluation.get("n_reps", 0))
    n_cand = len(result.candidates)

    t1 = n_reps >= 2
    if kind == "recommended":
        conclusion = "This describes how solid the plan's numbers are."
        t1_text = (
            f"The plan and your current setup were each checked with the detailed simulation "
            f"({n_reps} runs)."
        )
    elif kind == "no_better":
        conclusion = f"We searched and checked {n_cand} candidates; none beat your current setup."
        t1_text = f"The best candidate was checked with the detailed simulation ({n_reps} runs)."
    else:
        conclusion = f"We searched and checked {n_cand} candidates; none met all your limits."
        t1_text = f"The closest candidate was checked with the detailed simulation ({n_reps} runs)."
    if not t1:
        t1_text = "These numbers are a quick estimate only, not checked."

    if kind != "recommended":
        t2 = TrustCheck(None, "No plan is recommended, so no limits are claimed to be met.")
    elif not _constraints(result):
        t2 = TrustCheck(None, "You set no limits, so there were none to meet.")
    elif _robust(result):
        t2 = TrustCheck(True, "Your limits are met with a safety margin.")
    else:
        t2 = TrustCheck(False, "Your limits are met on average, but not with a safety margin.")

    band = vm.novelty_view((result.plan or {}).get("novelty")).band
    subject = "The plan's settings" if kind == "recommended" else "The best candidate's settings"
    t3 = TrustCheck(
        band == "inside",
        {
            "inside": f"{subject} are close to ones the model has seen before.",
            "edge": f"{subject} are less familiar to the model.",
            "outside": f"{subject} are unfamiliar to the model.",
        }.get(band, f"{subject} could not be compared with what the model has seen."),
    )

    weak = bool(vm.goal_kpis(result) & set(vm.WEAK_KPIS))
    t4 = TrustCheck(
        not weak,
        WEAK_GOAL_REASON if weak else "Your goal doesn't depend on breakdown or downtime numbers.",
    )

    checks = [TrustCheck(t1, t1_text), t2, t3, t4]
    if not t1:
        level = "rough"
    else:
        failed = sum(c.passed is False for c in (t2, t3, t4))
        level = ("checked", "partly", "rough", "rough")[failed]
    return Trust(level, conclusion, checks)


# --------------------------------------------------------------------------- #
# Messages (docs/ux_spec.md A8)
# --------------------------------------------------------------------------- #
MESSAGES = {
    "ai_unreachable": "We couldn't reach the AI service, so no new plan was found. Check your "
    "internet connection and try again. Meanwhile, here is a saved example result for a "
    "similar goal.",
    "ai_busy": "The AI service is busy right now. Wait a minute and press Find a plan again.",
    "not_understood": "We couldn't turn that into a goal we can plan for. Try one of the goal "
    "cards above, or say what you want more or less of, for example: “Make more profit with "
    "at most one breakdown a month”.",
    "cancelled": "You stopped this run, so there is no result. Press Find a plan to start again.",
    "timed_out": "This run hit the time limit and was stopped, so there is no result. Try a "
    "goal with fewer limits, or try again when the computer is less busy.",
    "stopped": "This run stopped before finishing, so there is no result. Press Find a plan to "
    "start again.",
    "failed": "Something went wrong while working out the plan, and no result was produced. Try "
    "again; if it happens again, see Technical details.",
    "busy": "A plan is already being worked out. It will appear on the Plan page when it's ready.",
    "demo": "Demo mode: this is a saved example for a similar goal. Your limits are not used.",
    "no_quick_estimate": "Quick estimates and checks need the full installation. Start from a "
    "saved example's plan to see its checked numbers.",
    "empty_history": "Nothing here yet. Results you find on the Plan page appear here.",
}


def error_message(kind: str) -> str:
    return MESSAGES[kind]


def goal_sentence(result: AgentResult) -> str:
    """The goal as the manager would say it."""
    if result.spec:
        return goal_text(result.spec)
    return result.goal_text
