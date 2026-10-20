"""Evidence: the pre-registered results B1-B5, PASS and FAIL alike, the model
cards, the figures, and what underperformed."""

from __future__ import annotations

import json

import streamlit as st

from twinmind.ui import services as sv

st.title("Evidence")
st.caption(
    "What was tested, how, and what failed. Every hypothesis was registered before its "
    "benchmark ran; all five are shown the same way."
)

# --------------------------------------------------------------------------- #
# B1-B5
# --------------------------------------------------------------------------- #
st.subheader("Agent benchmark: B1–B5")
analysis = json.loads((sv.RESULTS_DIR / "agent_bench_analysis.json").read_text(encoding="utf-8"))
rows = sv.benchmark_rows(analysis)
st.markdown(
    "| | Pre-registered claim | Verdict | Deciding number |\n|---|---|---|---|\n"
    + "\n".join(f"| **{r.id}** | {r.claim} | **{r.verdict}** | {r.deciding} |" for r in rows)
)
passed = sum(r.verdict == "PASS" for r in rows)
st.markdown(f"**{passed} of {len(rows)} passed; {len(rows) - passed} failed.**")
st.warning(sv.B5_NOTE, icon="📌")
st.caption(
    "20 goals × 2 seeds. The reference is the best plan any arm found, so gaps are lower "
    "bounds. Full detail: model card F."
)

mutation = sv.RESULTS_DIR / "explanation_mutation_v2.json"
if mutation.exists():
    with st.expander("Post-benchmark check of the explanation checker (not a test result)"):
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
st.markdown(
    "**Reading the explanation-checker evidence**\n\n"
    + "\n".join(f"- {caveat}" for caveat in sv.CHECKER_CAVEATS)
)
for title, body in sv.underperformed_sections():
    with st.expander(title):
        st.markdown(body)

# --------------------------------------------------------------------------- #
# Model cards and figures
# --------------------------------------------------------------------------- #
st.subheader("Model cards")
cards = sv.model_cards()
names = {c.title: c for c in cards}
chosen = names[st.selectbox("Card", list(names), index=len(names) - 2, key="ev_card")]
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
