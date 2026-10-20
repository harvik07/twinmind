"""Past results: every plan and saved example, newest first; click to reopen.

Live runs and saved examples are labelled apart in the "Type" column.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from twinmind.ui import components as ui
from twinmind.ui import plain as pl
from twinmind.ui import services as sv

st.title("Past results")

runner = ui.job_runner()
status = runner.status()
items = sv.read_history(sv.history_path(), running_id=status.id if status.active else "")

OUTCOME = {
    "running": "⏳ Being worked out",
    "stopped": "⏹ Stopped before finishing",
    "cancelled": "⏹ Cancelled",
    "timed_out": "⏹ Stopped at the time limit",
    "failed": "⚠️ Failed, no result",
}


def outcome(item: sv.HistoryItem) -> str:
    if item.result is not None:
        chip = pl.chip(item.result)
        return f"{chip.icon} {chip.text}"
    return OUTCOME.get(item.status, "⚠️ No result")


def type_label(item: sv.HistoryItem) -> str:
    return f"{'📁' if item.kind == 'replayed' else '🟢'} {item.label}"


if not items:
    st.info(pl.error_message("empty_history"), icon="🕘")
else:
    table: list[dict[str, Any]] = [
        {
            "When": i.when.replace(" UTC", ""),
            "Your goal": pl.goal_sentence(i.result) if i.result else i.goal_text,
            "Outcome": outcome(i),
            "Type": type_label(i),
        }
        for i in items
    ]
    st.caption("Click a row to open it.")
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="past_table",
    )
    selected = list(getattr(event, "selection", {}).get("rows", []))
    if selected:
        item = items[selected[0]]
        st.divider()
        if item.result is not None:
            ui.plan_result(item.result, saved=item.kind == "replayed")
        else:
            key = item.status if item.status in pl.MESSAGES else "stopped"
            st.warning(pl.error_message(key if key != "failed" else "failed"), icon="⏹")
            if item.detail:
                with st.expander(ui.TECH):
                    st.code(item.detail)

with st.expander(ui.TECH):
    from twinmind.utils.paths import PROJECT_ROOT

    cfg: Any = ui.agent_config()
    runs = sv.read_run_log(PROJECT_ROOT / cfg.run_log)
    st.caption(f"Agent run log ({cfg.run_log}): {len(runs)} live run(s), CLI runs included.")
    if runs:
        st.dataframe(
            [
                {
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
