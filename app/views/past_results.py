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


def outcome(item: sv.HistoryItem) -> str:
    """The result's status chip; a run without a result is "Stopped", with why."""
    if item.result is not None:
        chip = pl.chip(item.result)
        return f"{chip.icon} {chip.text}"
    if item.status == "running":
        return "⏳ Being worked out"
    return f"⏹ Stopped: {pl.stop_reason(item.status, item.detail)}"


def saved_entry(item: sv.HistoryItem) -> sv.DemoEntry | None:
    """The saved example a replayed item showed (for its provenance line)."""
    if item.result is None:
        return None
    for entry in ui.demo_cache():
        if entry.result.to_dict() == item.result.to_dict():
            return entry
    return None


def type_label(item: sv.HistoryItem) -> str:
    return f"{'📁' if item.kind == 'replayed' else '🟢'} {item.label}"


if not items:
    st.info(pl.error_message("empty_history"), icon="🕘")
else:
    table: list[dict[str, Any]] = [
        {
            "When": i.when.replace(" UTC", "")[:16],
            "Type": type_label(i),
            "Your goal": pl.goal_sentence(i.result) if i.result else i.goal_text,
            "Outcome": outcome(i),
        }
        for i in items
    ]
    st.caption("Click a row to open it.")
    event = st.dataframe(
        table,
        hide_index=True,
        column_config={
            "When": st.column_config.TextColumn(width="small"),
            "Type": st.column_config.TextColumn(width="medium"),
            "Your goal": st.column_config.TextColumn(width="large"),
            "Outcome": st.column_config.TextColumn(width="large"),
        },
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
            entry = saved_entry(item) if item.kind == "replayed" else None
            ui.plan_result(
                item.result,
                saved=item.kind == "replayed",
                saved_from=pl.saved_from(entry.goal_id, entry.seed) if entry else "",
                provenance=entry.provenance if entry else "",
            )
        else:
            st.info(f"**{outcome(item)}.** Press Find a plan on the Plan page to start again.",
                    icon="ℹ️")  # fmt: skip
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
