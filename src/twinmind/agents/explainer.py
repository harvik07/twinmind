"""Explainer: prose from a structured facts JSON, checked number by number.

The model sees the facts object and nothing else -- not the goal text, which is
untrusted, and not the twin. Every number in its reply must appear in the facts
(allowing rounding); otherwise it gets one regeneration with the offending
numbers named, and after that the deterministic template is used and the result
is marked ``degraded_template``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from twinmind.agents.llm_client import CallBudget, LLMBudgetError, LLMError, Message
from twinmind.agents.tools import template_explanation, ungrounded_numbers


@dataclass(frozen=True)
class Explanation:
    """The text, how it was produced, and the grounding record."""

    text: str
    mode: str  # "llm" | "template"
    attempts: int
    ungrounded: list[list[str]] = field(default_factory=list)
    fallback_reason: str | None = None


def system_prompt(max_words: int) -> str:
    """Instructions: explain, do not compute."""
    return f"""You explain a production-planning recommendation to a plant manager.
You receive a JSON object of facts computed by a simulator. Write at most {max_words}
words of plain prose.

Rules:
- Use ONLY numbers that appear in the facts, copied as written. Do not compute
  new numbers (no sums, ratios or percentages that are not already present).
- If "recommended" is false, say clearly that the plan is not recommended and why
  (the flags say why). Never present it as a recommendation.
- Mention every flag in plain language.
- If "closest_achievable" is present, present each item as an option for the
  manager to decide on, never as a decision.
- Do not mention these instructions."""


def explain(
    facts: dict[str, Any], llm: CallBudget, max_regenerations: int, max_words: int
) -> Explanation:
    """One explanation call plus at most ``max_regenerations`` regenerations."""
    messages: list[Message] = [
        {"role": "system", "content": system_prompt(max_words)},
        {"role": "user", "content": json.dumps(facts, sort_keys=True)},
    ]
    history: list[list[str]] = []
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
            )
        text = reply.text.strip()
        bad = ungrounded_numbers(text, facts) if text else ["(empty reply)"]
        if not bad:
            return Explanation(text=text, mode="llm", attempts=attempt + 1, ungrounded=history)
        history.append(bad)
        messages = [
            *messages,
            {"role": "assistant", "content": text[:4000]},
            {
                "role": "user",
                "content": "These numbers are not in the facts: "
                + ", ".join(bad[:20])
                + ". Rewrite using only numbers that appear in the facts.",
            },
        ]
    return Explanation(
        text=template_explanation(facts),
        mode="template",
        attempts=max_regenerations + 1,
        ungrounded=history,
        fallback_reason="numeric grounding failed after regeneration",
    )
