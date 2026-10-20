"""How reliable is this? A short plain summary first, then the full record.

The summary (the only part under the jargon ban) says what was tested, what
worked and what did not. Below it is everything the v1.0 Evidence page showed,
visible and in the same order: the B1-B5 table with PASS and FAIL given equal
weight, what each failure means and does not mean, the checker-v1/v2 note and
caveats, the post-benchmark check, the Limitations, the model cards and the
figures. That part keeps its technical terms (it is also the examiner's view)
and links to a glossary of them.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from twinmind.ui import services as sv
from twinmind.ui.glossary import TECH_TERMS

analysis: dict[str, Any] = json.loads(
    (sv.RESULTS_DIR / "agent_bench_analysis.json").read_text(encoding="utf-8")
)
rows = sv.benchmark_rows(analysis)
by_id = {r.id: r for r in rows}
b2, b3, b5 = analysis["B2"], analysis["B3"], analysis["B5"]["explainer"]
passed = sum(r.verdict == "PASS" for r in rows)
failed_n = b5["n"] - b5["llm_text_pass_within_two_attempts"]

#: Each test as a plain question and a plain result (used in the summary and as
#: a visible column of the table).
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
        f"{failed_n} of {b5['n']} written explanations failed the automatic check. The check "
        "has been improved since, but the improved version has not been retested.",
    ),
}


#: Neutral labels in equal visual weight: a calm green tick and a grey mark.
MET = ":green[✓] **Met**"
NOT_MET = ":gray[○] **Not met**"


def result_label(verdict: str) -> str:
    return MET if verdict == "PASS" else NOT_MET


met = [r for r in rows if r.verdict == "PASS"]
not_met = [r for r in rows if r.verdict != "PASS"]

st.title("How reliable is this?")

# --------------------------------------------------------------------------- #
# Plain summary (the jargon ban applies here): what works first, then care
# --------------------------------------------------------------------------- #
with st.container(border=True, key="reliability_summary"):
    st.markdown("### What was tested, what worked, what did not")
    st.markdown(
        "We tested the planner on 20 goals, twice each, against a simple search with no AI "
        "and no reviewer step, and decided in advance what each test had to show. "
        f"**{len(met)} of 5 tests were met and {len(not_met)} were not.**"
    )
    st.markdown("#### What works well")
    st.markdown(
        "\n".join(f"- {MET} · Test {r.id[1]}. {PLAIN[r.id][0]} {PLAIN[r.id][1]}" for r in met)
    )
    st.markdown("#### Where it needs care")
    st.markdown(
        "\n".join(
            f"- {NOT_MET} · Test {r.id[1]}. {PLAIN[r.id][0]} {PLAIN[r.id][1]}" for r in not_met
        )
        + "\n- The automatic check of written explanations misses some mistakes: a wrong "
        "“not”, wrong directions for setting changes, and numbers attached to the wrong "
        "measure when both moved the same way. The checked numbers are the record."
    )

# --------------------------------------------------------------------------- #
# The full record (technical terms allowed; glossary one click away)
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 1])
left.caption(
    "Below is the full record as registered, in its technical wording. The terms are "
    "explained in the glossary →"
)
with right.popover("Glossary of technical terms", width="stretch"):
    st.markdown("\n".join(f"- **{term}**: {meaning}" for term, meaning in TECH_TERMS.items()))

st.subheader("Agent benchmark: B1–B5")
st.markdown(
    "| | Pre-registered claim | Result | Registered verdict | Plain result | Deciding number |\n"
    "|---|---|---|---|---|---|\n"
    + "\n".join(
        f"| **{r.id}** | {r.claim} | {result_label(r.verdict)} | {r.verdict} | "
        f"{PLAIN[r.id][1]} | {r.deciding} |"
        for r in rows
    )
)
st.markdown(f"**{len(met)} of {len(rows)} met (PASS); {len(not_met)} not met (FAIL).**")

st.markdown("**What the two not-met results mean**")
st.markdown(
    f"- **B2 · not met (FAIL).** *What it means:* recommended plans were robustly feasible in "
    f"{b2['agent']['robust']} of {b2['agent']['recommended']} cells against "
    f"{b2['twin_only']['robust']} of {b2['twin_only']['recommended']} for twin-only, and the "
    'registered bar was "at least as often". *What it does not mean:* that the agent’s '
    "plans usually break limits; the shortfall is one cell, and twin-only had no critic.\n"
    f"- **B5 · not met (FAIL).** *What it means:* {failed_n} of {b5['n']} explanations did "
    "not pass checker v1 within the one permitted regeneration (a fixed template was shown "
    "instead), so the 100% bar was missed. *What it does not mean:* that the plans or their "
    "numbers were wrong; every number comes from the twin, not the language model."
)
st.warning(sv.B5_NOTE, icon="📌")
st.markdown(
    "**Reading the explanation-checker evidence**\n\n"
    + "\n".join(f"- {caveat}" for caveat in sv.CHECKER_CAVEATS)
)
st.caption(
    "20 goals × 2 seeds. The reference is the best plan any arm found, so gaps are lower "
    "bounds. Full detail: model card F."
)

mutation = sv.RESULTS_DIR / "explanation_mutation_v2.json"
if mutation.exists():
    st.subheader("Post-benchmark check of the explanation checker (not a test result)")
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
    st.caption(
        "Neither checker catches negation, plan-change directions or same-sign KPI swaps; "
        "the twin's KPI table is the record, the explanation a reading of it."
    )

# --------------------------------------------------------------------------- #
# Limitations
# --------------------------------------------------------------------------- #
st.subheader("Limitations: where results underperformed the spec")
for title, body in sv.underperformed_sections():
    st.markdown(f"#### {title}")
    st.markdown(body)

# --------------------------------------------------------------------------- #
# Model cards and figures
# --------------------------------------------------------------------------- #
st.subheader("Model cards")
cards = {c.title: c for c in sv.model_cards()}
chosen = cards[st.selectbox("Card", list(cards), index=len(cards) - 2, key="ev_card")]
with st.container(border=True):
    for kind, body in sv.split_markdown(
        chosen.path.read_text(encoding="utf-8"), chosen.path.parent
    ):
        if kind == "img":
            st.image(body)
        else:
            st.markdown(body)

st.subheader("Figures")
figures = sorted(sv.FIGURES_DIR.glob("*.png"))
if figures:
    figure = st.selectbox(
        "Figure", figures, format_func=lambda p: p.stem.replace("_", " "), key="ev_fig"
    )
    st.image(str(figure))
