"""Plain-language glossary: the single source of truth for every KPI and
setting name, unit and one-line meaning the dashboard shows.

Pages, the plain view models, the explanation captions and the tests all read
from here, so a name changes in one place. Nothing here imports Streamlit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from twinmind.optim.contracts import CONTROLLABLE_DIMS, HIGHER_IS_BETTER
from twinmind.twin.runner import KPI_NAMES

PERIOD = "4 weeks"
MACHINES = {
    "M1": "cutting",
    "M2": "forming",
    "M3": "welding",
    "M4": "finishing",
    "M5": "packing",
}


@dataclass(frozen=True)
class KPITerm:
    code: str
    name: str
    unit: str
    meaning: str
    higher_is_better: bool
    rough: bool = False  # breakdowns and downtime: the model is less accurate

    @property
    def better(self) -> str:
        return "Higher" if self.higher_is_better else "Lower"

    @property
    def help(self) -> str:
        """One line for the ⓘ next to a number."""
        text = f"{self.meaning} Unit: {self.unit}. {self.better} is better."
        if self.rough:
            text += " " + ROUGH_KPI_NOTE
        return text


@dataclass(frozen=True)
class SettingTerm:
    code: str
    section: str
    name: str
    unit: str
    meaning: str
    display: str  # "percent" | "hours" | "units" | "count"

    @property
    def help(self) -> str:
        return f"{self.meaning} Unit: {self.unit}."


#: The plain weak-KPI warning (Prompt 10's badge, in plain words).
ROUGH_KPI_NOTE = "Breakdown and downtime numbers are rough guides: the model is less accurate here."

KPIS: dict[str, KPITerm] = {
    t.code: t
    for t in (
        KPITerm(
            "profit",
            "Profit",
            f"£ over {PERIOD}",
            "Money left after all costs.",
            True,
        ),
        KPITerm(
            "units_produced",
            "Units made",
            f"units over {PERIOD}",
            "Finished units off the end of the line.",
            True,
        ),
        KPITerm(
            "on_time_delivery_rate",
            "On-time delivery",
            "% of units ordered",
            "Share of ordered units delivered on the day they were due.",
            True,
        ),
        KPITerm(
            "unplanned_failures",
            "Breakdowns",
            f"count over {PERIOD}",
            "Unexpected machine failures (not planned services).",
            False,
            rough=True,
        ),
        KPITerm(
            "downtime_hours",
            "Downtime",
            f"machine-hours over {PERIOD}",
            "Hours machines stood still, planned services included.",
            False,
            rough=True,
        ),
        KPITerm(
            "unit_cost",
            "Cost per unit",
            "£ per unit",
            "All costs divided by units made.",
            False,
        ),
        KPITerm(
            "stockout_probability",
            "Days short of stock",
            "% of days",
            "Share of days with some customer demand you could not supply.",
            False,
        ),
        KPITerm(
            "oee",
            "Equipment efficiency (OEE)",
            "%",
            "How much of the staffed time machines spent making good units at full "
            "speed; 100% is perfect.",
            True,
        ),
    )
}


def _settings() -> dict[str, SettingTerm]:
    out = [
        SettingTerm(
            "shift_count",
            "Shifts and overtime",
            "Shifts per day",
            "8-hour shifts (1 to 3)",
            "How many 8-hour shifts run each working day.",
            "count",
        ),
        SettingTerm(
            "overtime_hours",
            "Shifts and overtime",
            "Overtime per day",
            "hours (0 to 4)",
            "Paid extra hours added to each working day.",
            "hours",
        ),
        SettingTerm(
            "throughput_multiplier",
            "Machine speed",
            "Overall line pace",
            "% of normal (70 to 130)",
            "Target production rate for the whole line.",
            "percent",
        ),
    ]
    for m, machine in MACHINES.items():
        out.append(
            SettingTerm(
                f"speed_setpoint_{m}",
                "Machine speed",
                f"Speed of the {machine} machine ({m})",
                "% of normal (80 to 120)",
                "How fast this machine runs. Faster can wear it sooner.",
                "percent",
            )
        )
    for m, machine in MACHINES.items():
        out.append(
            SettingTerm(
                f"pm_interval_{m}",
                "Maintenance",
                f"Service interval, {machine} machine ({m})",
                "running hours (24 to 720)",
                "Running hours between planned services. Shorter means serviced more often.",
                "hours",
            )
        )
    out += [
        SettingTerm(
            "reorder_point",
            "Stock and ordering",
            "Reorder when stock falls to",
            "units (0 to 9,000)",
            "When stock on hand plus stock on order drops to this, order more.",
            "units",
        ),
        SettingTerm(
            "order_quantity",
            "Stock and ordering",
            "Order size",
            "units (100 to 10,000)",
            "How many units each order brings in.",
            "units",
        ),
        SettingTerm(
            "safety_stock",
            "Stock and ordering",
            "Safety stock",
            "units (0 to 4,000)",
            "Extra stock kept as a buffer against surprises.",
            "units",
        ),
    ]
    return {s.code: s for s in out}


SETTINGS: dict[str, SettingTerm] = _settings()
SECTIONS = ("Shifts and overtime", "Maintenance", "Machine speed", "Stock and ordering")

#: Technical term -> the plain words the default view uses instead.
PLAIN_WORDS: dict[str, str] = {
    "surrogate prediction": "quick estimate",
    "verified on twin (N reps)": "checked with the detailed simulation (N runs)",
    "replayed benchmark result": "saved example result, not a live run",
    "novelty band": "familiar / less familiar / unfamiliar territory",
    "robustly feasible": "meets your limits in every simulated month",
    "infeasible": "we couldn't find a plan that meets these limits",
    "constraint": "limit",
    "objective": "what you want to improve",
    "current policy": "current setup",
    "confidence interval": "likely between X and Y",
    "context": "example month of orders",
}

#: Words the default view never shows (outside the technical expanders).
BANNED_JARGON: tuple[str, ...] = (
    "surrogate",
    "twin",
    "epistemic",
    "novelty",
    "replication",
    "replications",
    "hazard",
    "seed",
    "context",
    "spec",
    "lever",
    "levers",
    "sobol",
    "ood",
    "benchmark",
    "critic",
    "objective",
    "constraint",
    "constraints",
    "feasible",
    "infeasible",
    "verified",
    "confidence interval",
)
#: Case-sensitive: "CI" as a word (not "circuit" or "decision").
_CI = re.compile(r"\bCI\b")
_OEE_ALLOWED = "Equipment efficiency (OEE)"

#: Expanders whose contents are technical on purpose and exempt from the ban.
TECHNICAL_EXPANDERS: tuple[str, ...] = (
    "Show technical details",
    "Show the exact test",
    "For reviewers",
    "Read the full section",
    "AI-written explanation (technical wording)",
)


def jargon_in(text: str) -> list[str]:
    """Banned words found in ``text`` (empty when the text is plain)."""
    lowered = text.lower()
    found = [w for w in BANNED_JARGON if re.search(rf"\b{re.escape(w)}\b", lowered)]
    if _CI.search(text):
        found.append("CI")
    if re.search(r"\boee\b", lowered) and _OEE_ALLOWED.lower() not in lowered:
        found.append("OEE without its explanation")
    return found


def kpi_name(code: str) -> str:
    return KPIS[code].name


def setting_name(code: str) -> str:
    return SETTINGS[code].name


def check_complete() -> None:
    """Every KPI and every controllable setting has an entry (used by tests)."""
    missing = [k for k in KPI_NAMES if k not in KPIS] + [
        d for d in CONTROLLABLE_DIMS if d not in SETTINGS
    ]
    if missing:
        raise KeyError(f"glossary is missing {missing}")
    wrong = [k for k, t in KPIS.items() if t.higher_is_better != (k in HIGHER_IS_BETTER)]
    if wrong:
        raise ValueError(f"glossary direction disagrees with the optimiser for {wrong}")


#: Technical terms used on the "How reliable is this?" page's detailed record
#: (the examiner's view). The page links to this list in one line.
TECH_TERMS: dict[str, str] = {
    "Digital twin (twin)": "The detailed simulation of the five-machine line, run many times "
    "with different random events.",
    "Surrogate": "The learned model that gives the instant quick estimate of the twin's numbers.",
    "Replications (reps)": "Repeated simulation runs of the same plan; 30 reps = 30 simulated "
    "months.",
    "95% CI": "A 95% confidence interval: the range the average would likely fall in if the "
    "simulation were repeated.",
    "Pre-registered": "The test, its threshold and its PASS/FAIL rule were written down before "
    "the results were seen.",
    "Twin-only": "The comparison planner: a plain search on the twin, with no AI, surrogate or "
    "critic.",
    "Normalised gap": "How far a plan's objective falls short of the best plan any method "
    "found, as a fraction; 0 = as good as the best found.",
    "Non-inferior": "Not worse than the comparison by more than an agreed margin (here 0.10 of "
    "normalised gap).",
    "Robustly feasible": "Meets every limit with its 95% interval inside the limit (a safety "
    "margin), not just on average.",
    "Critic": "The rule-based reviewer step that checks a plan and can swap it or search again.",
    "No-room goal": "A goal whose allowed changes cannot improve on the current setup.",
    "Spec-match": "The planner turned the plain-English goal into the intended precise goal.",
    "Explanation checker v1 / v2": "Automatic checks of an AI-written explanation's numbers, "
    "directions and stated reason. v1 was used in the test; v2 is the improved version, not "
    "yet retested.",
    "Mutation test": "A check of the checker: correct explanations are deliberately corrupted "
    "to see which corruptions it catches.",
    "Novelty": "How far a plan's settings are from those the surrogate was trained on.",
    "Epistemic std": "The surrogate ensemble's disagreement; not shown in the app because it "
    "did not flag unfamiliar plans (AUROC 0.590).",
    "OEE": "Overall equipment effectiveness: availability × speed × quality, 100% is perfect.",
    "Seed": "The starting number for the random events, so a run can be repeated exactly.",
}
