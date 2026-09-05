"""TwinMind dashboard.

Placeholder shell: navigation and status only. Each page is wired up as the
corresponding component lands.
"""

from __future__ import annotations

import streamlit as st

from twinmind.utils.config import CONFIG_MODELS, load_config

PAGES = [
    "Overview",
    "Machine Health (Model A)",
    "Demand (Model B)",
    "Digital Twin",
    "Surrogate (Model C)",
    "Agent",
    "Benchmark",
]


def render_sidebar() -> str:
    """Draw the navigation sidebar and return the selected page."""
    st.sidebar.title("TwinMind")
    st.sidebar.caption("Agentic digital twin of a manufacturing line")
    page = st.sidebar.radio("Navigation", PAGES, index=0)
    st.sidebar.divider()
    st.sidebar.subheader("Configs")
    for name in CONFIG_MODELS:
        try:
            load_config(name)
            st.sidebar.write(f"OK  {name}.yaml")
        except Exception as exc:  # surfacing any config error in the UI is the point
            st.sidebar.error(f"{name}.yaml: {exc}")
    return str(page)


def main() -> None:
    """Render the app."""
    st.set_page_config(page_title="TwinMind", page_icon="*", layout="wide")
    page = render_sidebar()
    st.title("TwinMind")
    st.subheader(page)
    st.info(
        "Scaffold only. Models, the SimPy twin and the agent layer are not "
        "implemented yet -- this page confirms the package imports and the "
        "configs validate."
    )
    st.markdown(
        """
        **Architecture**

        1. **Model A** - CNN-BiLSTM on C-MAPSS -> RUL, plus a failure head on AI4I 2020.
        2. **Model B** - N-HiTS / PatchTST on Favorita, multi-quantile (q10/q50/q90).
        3. **Digital twin** - SimPy line whose hazard comes from A and order book from B.
        4. **Model C** - heteroscedastic residual MLP surrogate of the twin.
        5. **Agent** - LangGraph Planner/Explorer/Evaluator/Critic/Explainer on Groq.
        """
    )


if __name__ == "__main__":
    main()
