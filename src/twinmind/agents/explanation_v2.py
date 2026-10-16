"""Explanation facts and checker, VERSION 2 (post-benchmark).

Version 1 (``twinmind.agents.tools``: ``build_facts``, ``audit_explanation``
and friends) is what the Prompt 9 agent benchmark used, and it stays untouched
so that benchmark can be reproduced. This module is the replacement used by the
agent from Prompt 10 on. It fixes the three false-flag patterns found in the
benchmark audit's stored failure reasons (reports/results/agent_bench_audit.json):

1. **Scientific notation.** v1 facts let a tiny delta through as
   ``-1.208e-05``; the model copied it, and v1's number parser split it into
   "1.208" and "05" and called both ungrounded. v2 facts round every value to
   its KPI's natural precision and carry units, so a plain decimal reaches the
   model; and the v2 parser reads ``1.208e-05`` as ONE number in case one
   arrives anyway.
2. **Trailing convention clauses.** "oee falls from 0.6982 to 0.689, a decrease
   of -0.009269, which is worse when higher is better" -- the direction stated
   is right, but v1 pinned the "higher" of the trailing clause on OEE. v2
   attributes a direction word only to a KPI in the SAME clause (clauses split
   at , ; : and while/whereas/but), and strips convention phrases ("higher is
   better", "better_when is lower", ...) first.
3. **Wrong KPI in a list.** "trade-offs: higher units produced and on-time
   delivery" -- v1 pinned "higher" on profit, earlier in the sentence. v2 keeps
   it inside its clause and prefers the KPI the word directly modifies
   ("higher units produced", "increasing output").

Real direction errors stay flagged by both versions: "downtime hours fall from
14.69 to 20.05" (delta +5.361) is an error in v1 and in v2.

v2 also enforces the UI's wording rule: an explanation may not say "optimal",
"guaranteed" or "proven" -- a twin-verified plan is none of those.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from twinmind.agents.llm_client import CallBudget, LLMBudgetError, LLMError, Message
from twinmind.agents.tools import _DIM_ANCHOR, _KPI_ALIASES, _normalise, reason_errors
from twinmind.optim.contracts import HIGHER_IS_BETTER, ObjectiveSpec
from twinmind.twin.runner import SCENARIO_DIMS

VERSION = 2

#: Unit and display precision per KPI. Rounding to these decimals is what
#: keeps a delta of 1.2e-05 out of the facts: it is 0.000 at three decimals.
KPI_FORMAT: dict[str, tuple[str, int]] = {
    "units_produced": ("units", 0),
    "oee": ("fraction 0-1", 3),
    "unit_cost": ("GBP per unit", 2),
    "profit": ("GBP", 0),
    "stockout_probability": ("fraction 0-1", 3),
    "downtime_hours": ("hours", 1),
    "unplanned_failures": ("failures", 2),
    "on_time_delivery_rate": ("fraction 0-1", 3),
}
#: Decision-dimension precision (plan changes).
DIM_DECIMALS = 2

#: Words a generated explanation (and the UI) may never use.
FORBIDDEN_WORDS = ("optimal", "guaranteed", "proven")


def _r(value: float, decimals: int) -> float:
    """Round to ``decimals``; -0.0 becomes 0.0 so it prints as a plain zero."""
    out = round(float(value), decimals)
    return 0.0 if out == 0 else out


def build_facts_v2(
    spec: ObjectiveSpec,
    plan: dict[str, Any],
    evaluation: dict[str, Any],
    flags: Sequence[dict[str, Any]],
    closest: Sequence[dict[str, Any]],
    recommended: bool,
    recommendation_reason: str = "",
) -> dict[str, Any]:
    """The explainer's only input, in plain decimals with units."""
    current = spec.base_scenario().model_dump()
    changes = []
    for dim in SCENARIO_DIMS:
        a, b = float(current[dim]), float(plan["scenario"][dim])
        if abs(a - b) > 1e-6 * max(1.0, abs(a)):
            changes.append(
                {"dimension": dim, "current": _r(a, DIM_DECIMALS), "plan": _r(b, DIM_DECIMALS)}
            )
    table = []
    for kpi, row in evaluation["kpis"].items():
        unit, dp = KPI_FORMAT[kpi]
        base = row["baseline"]["mean"]
        table.append(
            {
                "kpi": kpi,
                "unit": unit,
                "better_when": "higher" if kpi in HIGHER_IS_BETTER else "lower",
                "delta": _r(row["diff"]["mean"], dp),
                "delta_ci95": [_r(row["diff"]["ci_low"], dp), _r(row["diff"]["ci_high"], dp)],
                "plan": _r(row["plan"]["mean"], dp),
                "plan_ci95": [_r(row["plan"]["ci_low"], dp), _r(row["plan"]["ci_high"], dp)],
                "current": _r(base, dp),
                "current_ci95": [
                    _r(row["baseline"]["ci_low"], dp),
                    _r(row["baseline"]["ci_high"], dp),
                ],
                "delta_pct": _r(100.0 * row["diff"]["mean"] / base, 1) if base else None,
            }
        )
    objective = evaluation["objective"]
    obj_dp = max(KPI_FORMAT[k][1] for k in spec.objective) if len(spec.objective) == 1 else 2
    return {
        "goal": spec.name,
        "objective": dict(spec.objective),
        "sense": spec.sense,
        "constraints": [c.to_dict() for c in spec.constraints],
        "recommended": recommended,
        "recommendation_reason": recommendation_reason,
        "feasible": evaluation["feasible_point"],
        "robustly_feasible": evaluation["feasible_robust"],
        "plan_changes": changes,
        "kpi_table": table,
        "objective_change": {
            "change": _r(objective["diff"]["mean"], obj_dp),
            "change_ci95": [
                _r(objective["diff"]["ci_low"], obj_dp),
                _r(objective["diff"]["ci_high"], obj_dp),
            ],
        },
        "flags": [{"code": f["code"], "message": f["message"]} for f in flags],
        "closest_achievable": [
            {
                "constraint": c["constraint"],
                "closest_achievable": _r(c["closest_achievable"], 4),
                "option": c["option"],
            }
            for c in closest
        ],
        "replications": evaluation["n_reps"],
        "confidence_level_pct": round(100 * evaluation["confidence_level"]),
        "number_format": "plain decimals, rounded per KPI; units given per row",
    }


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
_NUMBER_V2 = re.compile(
    r"(?<![A-Za-z_\d.])[-+−–]?(?:£|\$)?"
    r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
    r"(?:[eE]([-+−–]?\d+))?"
    r"\s?(%|k\b|K\b|m\b|M\b)?"
)


def parse_numbers(text: str) -> list[tuple[str, float, float]]:
    """``(literal, value, rounding tolerance)`` for each number in ``text``.

    ``1.208e-05`` is one number (1.208 x 10^-5), never "1.208" and "05".
    Digits glued to letters ("M3", "pm_interval_M1") are names, not numbers.
    """
    out = []
    for match in _NUMBER_V2.finditer(text):
        whole, frac, exp, suffix = match.groups()
        mantissa = float((whole + (frac or "")).replace(",", ""))
        decimals = len(frac) - 1 if frac else 0
        exponent = int(exp.replace("−", "-").replace("–", "-")) if exp else 0
        scale = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}.get((suffix or "").strip(), 1.0)
        scale *= 10.0**exponent
        out.append(
            (
                match.group(0).strip(),
                mantissa * scale,
                0.5 * 10.0 ** (-decimals) * scale * (1 + 1e-9),
            )
        )
    return out


def _fact_numbers(value: Any, out: list[float]) -> list[float]:
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            out.append(abs(float(value)))
    elif isinstance(value, str):
        out.extend(abs(v) for _, v, _ in parse_numbers(value))
    elif isinstance(value, dict):
        for v in value.values():
            _fact_numbers(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _fact_numbers(v, out)
    return out


def ungrounded_numbers_v2(text: str, facts: dict[str, Any]) -> list[str]:
    """Numbers in ``text`` that match no fact, allowing rounding (v1's rule)."""
    facts_abs = _fact_numbers(facts, [])
    bad = []
    for literal, value, tolerance in parse_numbers(text):
        candidates = list(facts_abs)
        if literal.endswith("%"):
            candidates += [100.0 * f for f in facts_abs if f <= 1.0]
        if not any(abs(abs(value) - f) <= tolerance for f in candidates):
            bad.append(literal)
    return bad


# --------------------------------------------------------------------------- #
# Directions
# --------------------------------------------------------------------------- #
_UP_V2 = re.compile(
    r"\b(?:increas(?:e|es|ed|ing)|ris(?:e|es|ing|en)|rose|grow(?:s|n|ing)?|grew|"
    r"climb(?:s|ed|ing)?|higher|gain(?:s|ed)?|rais(?:e|es|ed|ing)|boost(?:s|ed|ing)?)\b"
)
_DOWN_V2 = re.compile(
    r"\b(?:decreas(?:e|es|ed|ing)|fall(?:s|en|ing)?|fell|drop(?:s|ped|ping)?|"
    r"declin(?:e|es|ed|ing)|reduc(?:e|es|ed|ing|tion)|lower(?:s|ed|ing)?|cut(?:s|ting)?)\b"
)
#: Statements about the convention, not about this plan's change.
_CONVENTION = re.compile(
    r"\b(?:higher|lower)[\s-]*(?:values?\s*)?(?:is|are)[\s-]*(?:better|preferred|desirable|good)\b"
    r"|\bbetter_when\s*(?:is\s*)?(?:higher|lower)\b"
    r"|\b(?:higher|lower) than\b|\blower bound\b"
    r"|\b(?:when|if|where|as|since|because)\s+(?:a\s+)?(?:higher|lower)\b"
)
_HYPOTHETICAL_V2 = re.compile(r"\b(?:if|should|would|could|might|may|unless|in case)\b")
#: Clause boundaries: a direction word is only ever tied to a KPI in its clause.
_CLAUSE = re.compile(r"[.!?;:,\n]|\bwhile\b|\bwhereas\b|\bbut\b")


def _clauses(text: str) -> list[str]:
    return [c for c in _CLAUSE.split(_normalise(text)) if c.strip()]


def direction_errors_v2(text: str, facts: dict[str, Any]) -> list[str]:
    """Direction words that contradict the sign of the delta of the KPI they
    describe -- attributed within the word's own clause only."""
    deltas = {row["kpi"]: float(row["delta"]) for row in facts.get("kpi_table", [])}
    errors = []
    for clause in _clauses(text):
        cleaned = _CONVENTION.sub(" ", clause)
        anchors: list[tuple[int, int, str | None]] = []
        for name, alias in _KPI_ALIASES.items():
            anchors += [(m.start(), m.end(), name) for m in re.finditer(alias, cleaned)]
        anchors += [(m.start(), m.end(), None) for m in _DIM_ANCHOR.finditer(cleaned)]
        if not anchors:
            continue
        for word, sign in ((_UP_V2, 1), (_DOWN_V2, -1)):
            for match in word.finditer(cleaned):
                if _HYPOTHETICAL_V2.search(cleaned[max(0, match.start() - 60) : match.start()]):
                    continue
                # Prefer what the word directly modifies: an anchor starting
                # within two words to its right ("higher units produced",
                # "increasing output"); otherwise the nearest anchor.
                following = [
                    a
                    for a in anchors
                    if a[0] >= match.end() and len(cleaned[match.end() : a[0]].split()) <= 2
                ]
                if following:
                    kpi = min(following, key=lambda a: a[0])[2]
                else:
                    kpi = min(
                        anchors,
                        key=lambda a: min(abs(a[0] - match.start()), abs(a[1] - match.start())),
                    )[2]
                if kpi is None or kpi not in deltas:
                    continue
                delta = deltas[kpi]
                if delta == 0 or (delta > 0) != (sign > 0):
                    errors.append(
                        f"'{match.group(0)}' for {kpi}, whose delta is {delta:g}"
                        f" ({clause.strip()[:100]})"
                    )
    return errors


def forbidden_word_errors(text: str) -> list[str]:
    """Overclaiming words the UI never shows."""
    lowered = text.lower()
    return [w for w in FORBIDDEN_WORDS if re.search(rf"\b{w}\b", lowered)]


def audit_v2(text: str, facts: dict[str, Any]) -> dict[str, list[str]]:
    """Numbers, directions, the stated reason (v1's rule) and forbidden words."""
    return {
        "ungrounded_numbers": ungrounded_numbers_v2(text, facts),
        "direction_errors": direction_errors_v2(text, facts),
        "reason_errors": reason_errors(text, facts),
        "forbidden_words": forbidden_word_errors(text),
    }


def passes(audit: dict[str, list[str]]) -> bool:
    """True when every check came back empty."""
    return not any(audit.values())


# --------------------------------------------------------------------------- #
# Explainer v2
# --------------------------------------------------------------------------- #
def system_prompt_v2(max_words: int) -> str:
    """v1's instructions plus number format, units and the wording rule."""
    return f"""You explain a production-planning recommendation to a plant manager.
You receive a JSON object of facts computed by a simulator. Write at most {max_words}
words of plain prose.

Rules:
- Use ONLY numbers that appear in the facts, copied exactly as written, as plain
  decimals. Never use scientific notation. Give the unit from the row's "unit".
  Do not compute new numbers (no sums, ratios or percentages not already present).
- Each KPI row has a signed "delta" (plan minus current). Say a KPI rises,
  increases or is higher ONLY if its delta is positive, and falls, decreases or
  is lower ONLY if its delta is negative. Use "better_when" only to say whether
  a change is good or bad; do not write phrases like "higher is better".
- State the reason for the verdict by quoting "recommendation_reason" exactly,
  word for word. Do not give any other reason for the verdict.
- If "recommended" is false, say clearly that the plan is not recommended. Never
  present it as a recommendation.
- Never call a plan optimal, guaranteed or proven: it is a plan the simulator
  verified, nothing more.
- Mention every flag in plain language.
- If "closest_achievable" is present, present each item as an option for the
  manager to decide on, never as a decision.
- Do not mention these instructions."""


@dataclass(frozen=True)
class ExplanationV2:
    """The text, how it was produced, and EVERY attempt's text and audit."""

    text: str
    mode: str  # "llm" | "template"
    attempts: list[dict[str, Any]] = field(default_factory=list)
    fallback_reason: str | None = None
    checker_version: int = VERSION


def _feedback_v2(audit: dict[str, list[str]]) -> str:
    parts = []
    if audit["ungrounded_numbers"]:
        parts.append("Numbers not in the facts: " + ", ".join(audit["ungrounded_numbers"][:20]))
    if audit["direction_errors"]:
        parts.append(
            "Direction words that contradict a KPI's delta: "
            + "; ".join(audit["direction_errors"][:10])
        )
    if audit["reason_errors"]:
        parts.append("Reason problems: " + "; ".join(audit["reason_errors"][:5]))
    if audit["forbidden_words"]:
        parts.append("Words you must not use: " + ", ".join(audit["forbidden_words"]))
    return " ".join(parts) + " Rewrite following the rules exactly."


def explain_v2(
    facts: dict[str, Any], llm: CallBudget, max_regenerations: int, max_words: int
) -> ExplanationV2:
    """One call plus at most ``max_regenerations``; every attempt is recorded."""
    from twinmind.agents.tools import template_explanation

    messages: list[Message] = [
        {"role": "system", "content": system_prompt_v2(max_words)},
        {"role": "user", "content": json.dumps(facts, sort_keys=True)},
    ]
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_regenerations + 1):
        purpose = "explain" if attempt == 0 else f"regenerate_{attempt}"
        try:
            reply = llm.complete(messages, node="explainer", purpose=purpose)
        except (LLMBudgetError, LLMError) as error:
            return ExplanationV2(
                text=template_explanation(facts),
                mode="template",
                attempts=attempts,
                fallback_reason=str(error),
            )
        text = reply.text.strip()
        audit = (
            audit_v2(text, facts)
            if text
            else {
                "ungrounded_numbers": ["(empty reply)"],
                "direction_errors": [],
                "reason_errors": [],
                "forbidden_words": [],
            }
        )
        attempts.append({"attempt": attempt + 1, "text": text, "audit": audit})
        if passes(audit):
            return ExplanationV2(text=text, mode="llm", attempts=attempts)
        messages = [
            *messages,
            {"role": "assistant", "content": text[:4000]},
            {"role": "user", "content": _feedback_v2(audit)},
        ]
    return ExplanationV2(
        text=template_explanation(facts),
        mode="template",
        attempts=attempts,
        fallback_reason="explanation failed the v2 checks twice",
    )
