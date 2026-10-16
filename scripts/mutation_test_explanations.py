"""Post-benchmark MUTATION test of explanation checkers v1 and v2. No LLM call.

    python scripts/mutation_test_explanations.py

NOT a test result and NOT a change to B5. Each of the 40 final explanation
texts stored in reports/results/agent_bench_audit.json (all of which pass both
checkers) is corrupted mechanically, one edit per mutant, and both checkers
audit every mutant against the facts the explainer saw (the v1 facts, as in the
benchmark -- the stored texts quote v1's 4-significant-figure numbers, so v2
must see the same facts for a like-for-like comparison).

Corruption types (sites are found by THIS script's own patterns, not by either
checker's, so the test is not circular):

- ``direction_flip``: one direction word that states a change ("rises",
  "a decrease of", "higher units") becomes its antonym. Convention phrases
  ("higher is better", "lower than", "better_when is lower") and hypotheticals
  ("if failures rise") are not sites -- flipping them is not an error.
- ``number_change``: one stated number is scaled by 1.25 (or 0.6) at the same
  precision, and only if the new value matches no number in the facts.
- ``kpi_swap``: one KPI name becomes the next KPI in a fixed cycle. Reported
  overall and for the swaps whose new KPI has a delta of the opposite sign.
- ``negation``: "X rises" becomes "X does not rise", or "is recommended"
  becomes "is not recommended" (and back).

A mutant counts as caught when the checker's audit is not empty.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterator
from typing import Any

from twinmind.agents.critic import recommendation_reason
from twinmind.agents.explanation_v2 import audit_v2, passes
from twinmind.agents.tools import audit_explanation, audit_passes, build_facts
from twinmind.eval.registry import results_dir
from twinmind.optim.contracts import ObjectiveSpec
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("mutation_test_explanations")
CELLS = PROJECT_ROOT / "data" / "processed" / "agent_bench" / "full"
OUT = results_dir() / "explanation_mutation_v2.json"

# --------------------------------------------------------------------------- #
# The generator's own vocabulary
# --------------------------------------------------------------------------- #
#: Direction word -> antonym. Antonyms are chosen inside BOTH checkers' word
#: lists, so a miss is an attribution failure, not a vocabulary gap.
ANTONYM = {
    "increase": "decrease", "increases": "decreases", "increased": "decreased",
    "increasing": "decreasing", "decrease": "increase", "decreases": "increases",
    "decreased": "increased", "decreasing": "increasing",
    "rise": "fall", "rises": "falls", "rising": "falling", "risen": "fallen", "rose": "fell",
    "fall": "rise", "falls": "rises", "falling": "rising", "fallen": "risen", "fell": "rose",
    "higher": "lower", "lower": "higher",
    "drop": "rise", "drops": "rises", "dropped": "rose", "dropping": "rising",
    "climb": "drop", "climbs": "drops", "climbed": "dropped",
    "gain": "drop", "gains": "drops", "gained": "dropped",
    "grow": "fall", "grows": "falls", "grew": "fell", "growing": "falling",
    "decline": "rise", "declines": "rises", "declined": "rose", "declining": "rising",
    "reduce": "increase", "reduces": "increases", "reduced": "increased",
    "reducing": "increasing", "reduction": "increase",
    "cut": "increase", "cuts": "increases",
}  # fmt: skip
_DIRECTION = re.compile(r"\b(" + "|".join(sorted(ANTONYM, key=len, reverse=True)) + r")\b", re.I)
#: A direction word that states a convention or a comparison, not a change.
_NOT_A_CLAIM_AFTER = re.compile(
    r"^[\s-]*(?:values?\s+)?(?:is|are|was|were|than|bound|=)\b|^[\s-]*(?:better|worse)\b", re.I
)
_NOT_A_CLAIM_BEFORE = re.compile(
    r"(?:better_when|better when|when|if|where|as|since|because)\s+(?:is\s+)?(?:a\s+)?$", re.I
)
_HYPOTHETICAL = re.compile(r"\b(?:if|should|would|could|might|may|unless|in case)\b", re.I)

#: Verb (3rd person) -> its "does not" form, for claim negation.
NEGATE = {
    "increases": "does not increase", "decreases": "does not decrease",
    "rises": "does not rise", "falls": "does not fall", "drops": "does not drop",
    "climbs": "does not climb", "grows": "does not grow", "declines": "does not decline",
}  # fmt: skip
_NEG_SITE = re.compile(r"\b(" + "|".join(NEGATE) + r")\b", re.I)
_VERDICT = re.compile(r"\bis (not )?recommended\b", re.I)

#: KPI display names, in the fixed swap cycle.
KPI_CYCLE = [
    ("units_produced", "units produced", r"\bunits produced\b"),
    ("oee", "OEE", r"\boee\b"),
    ("unit_cost", "unit cost", r"\bunit[- ]cost\b"),
    ("profit", "profit", r"\bprofit\b"),
    ("stockout_probability", "stockout probability", r"\bstock-?out probability\b"),
    ("downtime_hours", "downtime hours", r"\bdowntime(?: hours)?\b"),
    ("unplanned_failures", "unplanned failures", r"\bunplanned failures\b"),
    ("on_time_delivery_rate", "on-time delivery rate", r"\bon-time delivery(?: rate)?\b"),
]


def _case_like(new: str, old: str) -> str:
    if old[:1].isupper() and not new[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


#: Decision-variable names: a sentence naming one describes a plan change,
#: whose direction neither checker verifies.
_PLAN_CHANGE = re.compile(
    r"safety[ _]stock|reorder[ _]point|order[ _]quantity|overtime|shift|throughput|"
    r"preventive|pm[ _]interval|speed|setpoint|supplier|price",
    re.I,
)


def _sentence_around(text: str, m: re.Match[str]) -> str:
    end = min([i for i in (text.find(c, m.end()) for c in ".!?\n") if i >= 0] or [len(text)])
    return _sentence_before(text, m.start()) + text[m.start() : end]


def _sentence_before(text: str, pos: int) -> str:
    start = max(text.rfind(c, 0, pos) for c in ".!?\n") + 1
    return text[start:pos]


# --------------------------------------------------------------------------- #
# Mutants: (type, detail, mutated text, extra)
# --------------------------------------------------------------------------- #
Mutant = tuple[str, str, str, dict[str, Any]]


def direction_flips(text: str, facts: dict[str, Any]) -> Iterator[Mutant]:
    for m in _DIRECTION.finditer(text):
        after, before = text[m.end() : m.end() + 20], text[max(0, m.start() - 40) : m.start()]
        if _NOT_A_CLAIM_AFTER.search(after) or _NOT_A_CLAIM_BEFORE.search(before):
            continue
        if _HYPOTHETICAL.search(_sentence_before(text, m.start())[-60:]):
            continue
        word = m.group(0)
        new = _case_like(ANTONYM[word.lower()], word)
        yield (
            "direction_flip",
            f"{word} -> {new}",
            text[: m.start()] + new + text[m.end() :],
            {"at": m.start(), "plan_change": bool(_PLAN_CHANGE.search(_sentence_around(text, m)))},
        )


def _fact_values(value: Any, out: list[float]) -> list[float]:
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out.append(abs(float(value)))
    elif isinstance(value, dict):
        for v in value.values():
            _fact_values(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _fact_values(v, out)
    return out


def number_changes(text: str, facts: dict[str, Any]) -> Iterator[Mutant]:
    values = _fact_values(facts, [])
    pattern = re.compile(r"(?<![A-Za-z_\d.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\deE])")
    for m in pattern.finditer(text):
        whole, frac = m.group(1), m.group(2) or ""
        decimals = len(frac) - 1 if frac else 0
        value = float((whole + frac).replace(",", ""))
        if value == 0:
            continue
        for factor in (1.25, 0.6):
            new_value = round(value * factor, decimals)
            tol = 0.5 * 10.0**-decimals
            if abs(new_value - value) < 2 * tol:
                continue
            # Must not land on any fact, also read as a percentage of a fraction.
            if any(abs(new_value - f) <= tol or abs(new_value - 100 * f) <= tol for f in values):
                continue
            literal = f"{new_value:,.{decimals}f}" if "," in whole else f"{new_value:.{decimals}f}"
            yield (
                "number_change",
                f"{m.group(0)} -> {literal}",
                text[: m.start()] + literal + text[m.end() :],
                {"at": m.start()},
            )
            break


def kpi_swaps(text: str, facts: dict[str, Any]) -> Iterator[Mutant]:
    deltas = {row["kpi"]: float(row["delta"]) for row in facts["kpi_table"]}
    for i, (kpi, _, pattern) in enumerate(KPI_CYCLE):
        new_kpi, new_name, _ = KPI_CYCLE[(i + 1) % len(KPI_CYCLE)]
        for m in re.finditer(pattern, text, re.I):
            new = _case_like(new_name, m.group(0))
            sign_change = (deltas.get(kpi, 0) > 0) != (deltas.get(new_kpi, 0) > 0)
            yield (
                "kpi_swap",
                f"{m.group(0)} -> {new}",
                text[: m.start()] + new + text[m.end() :],
                {"sign_change": sign_change, "at": m.start()},
            )


def negations(text: str, facts: dict[str, Any]) -> Iterator[Mutant]:
    for m in _NEG_SITE.finditer(text):
        new = NEGATE[m.group(0).lower()]
        yield (
            "negation",
            f"{m.group(0)} -> {new}",
            text[: m.start()] + new + text[m.end() :],
            {"at": m.start()},
        )
    for m in _VERDICT.finditer(text):
        new = "is recommended" if m.group(1) else "is not recommended"
        yield (
            "negation",
            f"{m.group(0)} -> {new}",
            text[: m.start()] + new + text[m.end() :],
            {"at": m.start()},
        )


GENERATORS: list[Callable[[str, dict[str, Any]], Iterator[Mutant]]] = [
    direction_flips,
    number_changes,
    kpi_swaps,
    negations,
]


def _v1_facts(key: str) -> dict[str, Any]:
    result = json.loads((CELLS / f"{key}.json").read_text(encoding="utf-8"))["result"]
    spec = ObjectiveSpec.from_dict(result["spec"])
    evaluation = result["baseline_comparison"]
    return build_facts(
        spec,
        result["plan"],
        evaluation,
        result["flags"],
        result["closest_achievable"],
        result["plan"]["recommended"],
        recommendation_reason(spec, evaluation, {f["code"] for f in result["flags"]}),
    )


def _rate(rows: list[dict[str, Any]], version: str) -> dict[str, Any]:
    caught = sum(r[f"{version}_caught"] for r in rows)
    return {
        "caught": caught,
        "n": len(rows),
        "fraction": round(caught / len(rows), 3) if rows else None,
    }


def main() -> int:
    audit = json.loads((results_dir() / "agent_bench_audit.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    excluded: dict[str, Any] = {}
    for key, entry in sorted(audit["explainer"].items()):
        facts, text = _v1_facts(key), entry["text"]
        before = {"v1": audit_explanation(text, facts), "v2": audit_v2(text, facts)}
        if not (audit_passes(before["v1"]) and passes(before["v2"])):
            # A text that already fails cannot show whether a mutant is caught.
            excluded[key] = {v: {k: e for k, e in a.items() if e} for v, a in before.items()}
            continue
        for generate in GENERATORS:
            for kind, detail, mutated, extra in generate(text, facts):
                v1, v2 = audit_explanation(mutated, facts), audit_v2(mutated, facts)
                rows.append(
                    {
                        "cell": key,
                        "type": kind,
                        "edit": detail,
                        "v1_caught": not audit_passes(v1),
                        "v2_caught": not passes(v2),
                        **extra,
                    }
                )
    types = ["direction_flip", "number_change", "kpi_swap", "negation"]
    by_type = {t: [r for r in rows if r["type"] == t] for t in types}
    by_type["kpi_swap (opposite-sign KPI)"] = [r for r in by_type["kpi_swap"] if r["sign_change"]]
    flips = by_type["direction_flip"]
    by_type["direction_flip (KPI sentence)"] = [r for r in flips if not r["plan_change"]]
    by_type["direction_flip (plan-change sentence)"] = [r for r in flips if r["plan_change"]]
    summary: dict[str, Any] = {
        "label": "post-benchmark mutation test, not a test result; B5 is unchanged",
        "facts": "v1 facts (what the stored texts were written from), for both checkers",
        "texts": len(audit["explainer"]),
        "excluded_texts_failing_before_mutation": excluded,
        "per_type": {
            t: {"v1": _rate(rs, "v1"), "v2": _rate(rs, "v2"), "cells": len({r["cell"] for r in rs})}
            for t, rs in by_type.items()
        },
        "v1_catches_v2_misses": [r for r in rows if r["v1_caught"] and not r["v2_caught"]],
        "v2_catches_v1_misses": len([r for r in rows if r["v2_caught"] and not r["v1_caught"]]),
        "rows": rows,
    }
    head = json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1)
    body = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows)
    OUT.write_text(
        head[:-2] + ',\n "rows": [\n' + body + "\n ]\n}\n", encoding="utf-8", newline="\n"
    )
    for t, stats in summary["per_type"].items():
        logger.info(
            "%-30s v1 %d/%d  v2 %d/%d",
            t,
            stats["v1"]["caught"],
            stats["v1"]["n"],
            stats["v2"]["caught"],
            stats["v2"]["n"],
        )
    logger.info("v1 catches, v2 misses: %d", len(summary["v1_catches_v2_misses"]))
    logger.info("excluded (fail before mutation): %s", excluded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
