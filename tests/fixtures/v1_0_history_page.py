# ruff: noqa: I001, E402
# Fixture: the v1.0-dashboard page, verbatim except this import, which loads
# the vendored v1.0 components (tests/fixtures/v1_0_components.py).
"""History: every dashboard event, replays and live runs labelled apart, plus
the agent's own run log (every live run, from the CLI too)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).parent))
import v1_0_components as ui

from typing import Any

import streamlit as st

from twinmind.ui import services as sv

st.title("History")

KIND_ICON = {"live": "🟢", "replayed": "🔁", "aborted": "⛔"}

items = sv.read_history(sv.history_path())
st.subheader("This dashboard")
if not items:
    st.caption("Nothing yet. Runs and replays from the Ask page appear here.")
else:
    st.dataframe(
        [
            {
                "When": i.when,
                "Kind": f"{KIND_ICON[i.kind]} {i.label}",
                "Goal": i.goal_text,
                "Status": i.status,
                "Detail": i.detail,
            }
            for i in items
        ],
        hide_index=True,
        width="stretch",
    )
    with_result = [i for i in items if i.result is not None]
    if with_result:
        options = {f"{i.when} · {i.label} · {i.goal_text[:60]}": i for i in with_result}
        picked = options[st.selectbox("Open", list(options), key="hist_pick")]
        assert picked.result is not None
        ui.result(picked.result, replayed=picked.kind == "replayed")

st.subheader("Agent run log")
cfg: Any = ui.agent_config()
from twinmind.utils.paths import PROJECT_ROOT  # noqa: E402

runs = sv.read_run_log(PROJECT_ROOT / cfg.run_log)
if not runs:
    st.caption("No live runs logged on this machine.")
else:
    st.caption(f"{len(runs)} live run(s) logged in {cfg.run_log}, newest first.")
    st.dataframe(
        [
            {
                "Kind": f"{KIND_ICON['live']} {sv.HISTORY_LABELS['live']}",
                "Goal": r.goal_text,
                "Status": r.status,
                "LLM calls": (r.trace or {}).get("llm_calls_total", 0),
                "Twin runs": (r.trace or {}).get("twin_runs", 0),
            }
            for r in runs
        ],
        hide_index=True,
        width="stretch",
    )
