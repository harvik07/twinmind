"""Regression tests for the two explanation flaws found in the Prompt 8 live demo.

Both use the REAL demo outputs committed in reports/results/agent_demo_{a,c}.json:
the facts are rebuilt from each AgentResult exactly as the explainer saw them,
and the explanation is the text the model actually wrote.

* demo (a): "Profit increases from 272000.0 to 137100.0" -- profit halved.
* demo (c): "recommended because it maximises profit" -- it was recommended for
  restoring the unplanned-failures cap, at a profit cost.

Each test shows the OLD check (numbers only) passing the flawed text, the NEW
audit failing it, and a minimally corrected text passing the new audit.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from twinmind.agents.critic import recommendation_reason
from twinmind.agents.tools import (
    audit_explanation,
    audit_passes,
    build_facts,
    direction_errors,
    reason_errors,
    template_explanation,
    ungrounded_numbers,
)
from twinmind.optim.contracts import ObjectiveSpec
from twinmind.utils.paths import PROJECT_ROOT


def _demo(name: str) -> tuple[dict[str, Any], str]:
    """(facts as the new explainer would see them, the text the model wrote)."""
    path = PROJECT_ROOT / "reports" / "results" / f"agent_demo_{name}.json"
    result = json.loads(path.read_text(encoding="utf-8"))["result"]
    spec = ObjectiveSpec.from_dict(result["spec"])
    evaluation = result["baseline_comparison"]
    codes = {f["code"] for f in result["flags"]}
    facts = build_facts(
        spec,
        result["plan"],
        evaluation,
        result["flags"],
        result["closest_achievable"],
        result["plan"]["recommended"],
        recommendation_reason(spec, evaluation, codes),
    )
    return facts, result["explanation"]


def test_demo_a_profit_said_to_increase_while_it_halved() -> None:
    facts, text = _demo("a")
    profit = next(r for r in facts["kpi_table"] if r["kpi"] == "profit")
    assert profit["delta"] < 0, "profit fell in demo (a)"
    assert "Profit increases from 272000.0 to 137100.0" in text

    # The old check passed it: every number is in the facts.
    assert ungrounded_numbers(text, facts) == []
    # The new one does not.
    errors = direction_errors(text, facts)
    assert errors and all("profit" in e for e in errors), errors

    corrected = text.replace("Profit increases from", "Profit falls from") + (
        f" Reason: {facts['recommendation_reason']}."
    )
    assert audit_passes(audit_explanation(corrected, facts)), audit_explanation(corrected, facts)


def test_demo_c_wrong_reason_for_the_recommendation() -> None:
    facts, text = _demo("c")
    reason = facts["recommendation_reason"]
    assert reason.startswith("restores the unplanned_failures <= 1 constraint"), reason
    assert "because it maximises profit" in text

    assert ungrounded_numbers(text, facts) == []
    errors = reason_errors(text, facts)
    assert any("is not stated" in e for e in errors)
    assert any("claims ['improves']" in e for e in errors), errors
    assert direction_errors(text, facts) == [], "its direction words were all right"

    corrected = text.replace(
        "because it maximises profit while meeting the hard constraint that unplanned "
        "failures stay at or below 1.0",
        f"because it {reason}",
    )
    assert corrected != text
    assert audit_passes(audit_explanation(corrected, facts)), audit_explanation(corrected, facts)


@pytest.mark.parametrize("name", ["a", "b", "c"])
def test_the_template_passes_the_full_audit_on_every_demo(name: str) -> None:
    facts, _ = _demo(name)
    template = template_explanation(facts)
    assert audit_passes(audit_explanation(template, facts)), audit_explanation(template, facts)


FACTS: dict[str, Any] = {
    "kpi_table": [
        {"kpi": "profit", "delta": -1000.0, "better_when": "higher"},
        {"kpi": "downtime_hours", "delta": 2.5, "better_when": "lower"},
        {"kpi": "unplanned_failures", "delta": -1.0, "better_when": "lower"},
    ],
    "recommendation_reason": "improves the objective over the current policy",
}


@pytest.mark.parametrize(
    "text",
    [
        "Profit falls and downtime rises. Verdict: improves the objective over the current policy.",
        "Failures drop, while downtime increases. improves the objective over the current policy",
        "Safety stock rises a lot. improves the objective over the current policy",
        "Watch out in case failures rise later. improves the objective over the current policy",
        "Downtime stays lower than the cap. improves the objective over the current policy",
    ],
)
def test_direction_wording_that_agrees_or_is_not_a_claim_passes(text: str) -> None:
    assert direction_errors(text, FACTS) == []


@pytest.mark.parametrize(
    ("text", "kpi"),
    [
        ("Profit rises under this plan.", "profit"),
        ("Downtime falls to a new low.", "downtime_hours"),
        ("There are fewer stoppages because unplanned failures increase.", "unplanned_failures"),
        ("It increases profit.", "profit"),
    ],
)
def test_direction_wording_that_contradicts_the_delta_is_caught(text: str, kpi: str) -> None:
    errors = direction_errors(text, FACTS)
    assert errors and kpi in errors[0]


def test_a_different_stated_reason_is_caught_even_with_the_right_one_elsewhere_absent() -> None:
    facts = {**FACTS, "recommendation_reason": "no improvement over current policy"}
    bad = "The plan is shown because it maximises profit."
    assert reason_errors(bad, facts)
    good = "It is not recommended: no improvement over current policy."
    assert reason_errors(good, facts) == []
