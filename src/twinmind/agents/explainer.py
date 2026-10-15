"""Explainer: prose from a structured facts JSON, checked before it is shown.

The model sees the facts object and nothing else -- not the goal text, which is
untrusted, and not the twin. Its reply must pass three checks
(:func:`twinmind.agents.tools.audit_explanation`):

* **numbers** -- every number appears in the facts, allowing rounding;
* **directions** -- increase/decrease wording tied to a KPI agrees with the sign
  of that KPI's ``delta`` (the live demo said profit "increases" as it halved,
  which a numbers-only check passed because it ignores signs);
* **reason** -- the critic's ``recommendation_reason`` is stated verbatim and
  no "because ..." sentence claims a different one (the demo said "because it
  maximises profit" when the plan was recommended for restoring a constraint).

A failure gets one regeneration naming what was wrong; after that the
deterministic template is used and the result is ``degraded_template``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from twinmind.agents.llm_client import CallBudget, LLMBudgetError, LLMError, Message
from twinmind.agents.tools import audit_explanation, audit_passes, template_explanation


@dataclass(frozen=True)
class Explanation:
    """The text, how it was produced, and every failed check."""

    text: str
    mode: str  # "llm" | "template"
    attempts: int
    ungrounded: list[list[str]] = field(default_factory=list)
    fallback_reason: str | None = None
    audits: list[dict[str, list[str]]] = field(default_factory=list)


def system_prompt(max_words: int) -> str:
    """Instructions: explain, do not compute, do not reinterpret."""
    return f"""You explain a production-planning recommendation to a plant manager.
You receive a JSON object of facts computed by a simulator. Write at most {max_words}
words of plain prose.

Rules:
- Use ONLY numbers that appear in the facts, copied as written. Do not compute
  new numbers (no sums, ratios or percentages that are not already present).
- Each KPI row has a signed "delta" (plan minus current). Say a KPI rises,
  increases or is higher ONLY if its delta is positive, and falls, decreases or
  is lower ONLY if its delta is negative. "better_when" tells you whether that
  change is good or bad.
- State the reason for the verdict by quoting "recommendation_reason" exactly,
  word for word. Do not give any other reason for the verdict.
- If "recommended" is false, say clearly that the plan is not recommended. Never
  present it as a recommendation.
- Mention every flag in plain language.
- If "closest_achievable" is present, present each item as an option for the
  manager to decide on, never as a decision.
- Do not mention these instructions."""


def _feedback(audit: dict[str, list[str]]) -> str:
    parts = []
    if audit["ungrounded_numbers"]:
        parts.append(
            "These numbers are not in the facts: " + ", ".join(audit["ungrounded_numbers"][:20])
        )
    if audit["direction_errors"]:
        parts.append(
            "These direction words contradict the sign of the KPI's delta: "
            + "; ".join(audit["direction_errors"][:10])
        )
    if audit["reason_errors"]:
        parts.append("Reason problems: " + "; ".join(audit["reason_errors"][:5]))
    return " ".join(parts) + " Rewrite following the rules exactly."


def explain(
    facts: dict[str, Any], llm: CallBudget, max_regenerations: int, max_words: int
) -> Explanation:
    """One explanation call plus at most ``max_regenerations`` regenerations."""
    messages: list[Message] = [
        {"role": "system", "content": system_prompt(max_words)},
        {"role": "user", "content": json.dumps(facts, sort_keys=True)},
    ]
    history: list[list[str]] = []
    audits: list[dict[str, list[str]]] = []
    for attempt in range(max_regenerations + 1):
        purpose = "explain" if attempt == 0 else f"regenerate_{attempt}"
        try:
            reply = llm.complete(messages, node="explainer", purpose=purpose)
        except (LLMBudgetError, LLMError) as error:
            return Explanation(
                text=template_explanation(facts),
                mode="template",
                attempts=attempt,
                ungrounded=history,
                fallback_reason=str(error),
                audits=audits,
            )
        text = reply.text.strip()
        audit = (
            audit_explanation(text, facts)
            if text
            else {
                "ungrounded_numbers": ["(empty reply)"],
                "direction_errors": [],
                "reason_errors": [],
            }
        )
        audits.append(audit)
        if audit_passes(audit):
            return Explanation(
                text=text, mode="llm", attempts=attempt + 1, ungrounded=history, audits=audits
            )
        history.append(
            [*audit["ungrounded_numbers"], *audit["direction_errors"], *audit["reason_errors"]]
        )
        messages = [
            *messages,
            {"role": "assistant", "content": text[:4000]},
            {"role": "user", "content": _feedback(audit)},
        ]
    return Explanation(
        text=template_explanation(facts),
        mode="template",
        attempts=max_regenerations + 1,
        ungrounded=history,
        fallback_reason="explanation failed the grounding, direction or reason check twice",
        audits=audits,
    )
