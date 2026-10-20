"""How reliable is this? What was tested, what worked, what did not.

The five pre-registered tests are five identical cards, passed and failed
alike. The exact wording, numbers and caveats sit one click away.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from twinmind.ui import services as sv

analysis: dict[str, Any] = json.loads(
    (sv.RESULTS_DIR / "agent_bench_analysis.json").read_text(encoding="utf-8")
)
rows = {r.id: r for r in sv.benchmark_rows(analysis)}
b2, b3, b5 = analysis["B2"], analysis["B3"], analysis["B5"]["explainer"]
passed = sum(r.verdict == "PASS" for r in rows.values())

st.title("How reliable is this?")
st.markdown("### What was tested, what worked, what did not")
st.markdown(
    "We tested the planner on 20 goals, twice each, against a simple search with no AI and "
    "no reviewer step, and decided in advance what would count as a pass. "
    f"**{passed} of 5 tests passed and {5 - passed} failed.** It found plans at least as good "
    "as the simple search, its reviewer step caught bad recommendations, and it said so when "
    "it found no plan that met the limits. It fell short on two things: its recommended "
    "plans met the limits with a safety margin slightly less often than the simple search's, "
    "and some of its written explanations failed the automatic check at the time."
)

PLAIN = {
    "B1": (
        "Does it find plans at least as good as a simple search?",
        "Its plans were better on average than the simple search's.",
    ),
    "B2": (
        "Do its recommended plans meet your limits with a safety margin at least as often?",
        f"{b2['agent']['robust']} of {b2['agent']['recommended']} of its recommended plans did, "
        f"against {b2['twin_only']['robust']} of {b2['twin_only']['recommended']} for the "
        "simple search.",
    ),
    "B3": (
        "Does the reviewer step stop bad recommendations?",
        f"{b3['agent']['count']} bad recommendation with the reviewer step, "
        f"{b3['agent_nocritic']['count']} without it.",
    ),
    "B4": (
        "Does it say so when it finds no plan that meets the limits, or there's nothing to gain?",
        "Yes for all 4 goals whose limits were set out of reach on purpose, and all 3 with "
        "nothing to gain.",
    ),
    "B5": (
        "Does it understand goals and explain results correctly?",
        f"It understood all {analysis['B5']['planner_spec_match']['n']} goals, but "
        f"{b5['n'] - b5['llm_text_pass_within_two_attempts']} of {b5['n']} written "
        "explanations failed the automatic check. The check has been improved since, but the "
        "improved version has not been retested.",
    ),
}

st.markdown("### The five tests")
for test_id in ("B1", "B2", "B3", "B4", "B5"):
    row = rows[test_id]
    question, answer = PLAIN[test_id]
    ok = row.verdict == "PASS"
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.markdown(f"**Test {test_id[1]}. {question}**")
        right.badge("Passed" if ok else "Failed", icon="✅" if ok else "❌",
                    color="green" if ok else "red")  # fmt: skip
        st.markdown(answer)
        with st.expander("Show the exact test"):
            st.markdown(
                f"**{row.id}** · {row.claim}  \n**Verdict:** {row.verdict}  \n"
                f"**Deciding number:** {row.deciding}"
            )
            if test_id == "B5":
                st.markdown(sv.B5_NOTE)
                st.markdown("\n".join(f"- {c}" for c in sv.CHECKER_CAVEATS))

st.markdown("### Known limits")
st.markdown(
    "- It plans for a computer model of a production line, not your actual factory. The model "
    "has not been compared against a real line.\n"
    "- Breakdown and downtime numbers are rough guides: the model is less accurate here.\n"
    "- The automatic check of the written explanations misses some mistakes: a wrong "
    "“not”, wrong directions for setting changes, and numbers attached to the wrong measure "
    "when both moved the same way. The checked numbers on the Plan page are the record.\n"
    "- The test was small: 20 goals, twice each.\n"
    "- When a goal is vague, it does not always ask what you meant."
)
with st.expander("Read the full section"):
    st.markdown("\n".join(f"- {c}" for c in sv.CHECKER_CAVEATS))
    for title, body in sv.underperformed_sections():
        st.markdown(f"#### {title}")
        st.markdown(body)

with st.expander("For reviewers"):
    st.markdown("**Agent benchmark, B1–B5 as registered**")
    st.markdown(
        "| | Pre-registered claim | Verdict | Deciding number |\n|---|---|---|---|\n"
        + "\n".join(
            f"| **{r.id}** | {r.claim} | **{r.verdict}** | {r.deciding} |" for r in rows.values()
        )
    )
    st.markdown(sv.B5_NOTE)
    mutation = sv.RESULTS_DIR / "explanation_mutation_v2.json"
    if mutation.exists():
        data = json.loads(mutation.read_text(encoding="utf-8"))
        st.caption(data["label"])
        st.dataframe(
            [
                {
                    "Corruption": name.replace("_", " "),
                    "v1 caught": f"{v['v1']['caught']}/{v['v1']['n']}",
                    "v2 caught": f"{v['v2']['caught']}/{v['v2']['n']}",
                }
                for name, v in data["per_type"].items()
            ],
            hide_index=True,
            width="stretch",
        )
    cards = {c.title: c for c in sv.model_cards()}
    chosen = cards[st.selectbox("Model card", list(cards), index=len(cards) - 2, key="rv_card")]
    for kind, body in sv.split_markdown(
        chosen.path.read_text(encoding="utf-8"), chosen.path.parent
    ):
        if kind == "img":
            st.image(body)
        else:
            st.markdown(body)
    figures = sorted(sv.FIGURES_DIR.glob("*.png"))
    if figures:
        figure = st.selectbox(
            "Figure", figures, format_func=lambda p: p.stem.replace("_", " "), key="rv_fig"
        )
        st.image(str(figure))
