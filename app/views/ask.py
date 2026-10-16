"""Ask: a goal in plain English -> a twin-verified plan, or a replayed one.

Live mode runs ONE goal at a time and shows each node and the LLM calls used
against the per-goal cap as it goes. Every run ends with a status and exactly
one History entry -- finished, replayed or aborted. If the language model
cannot be reached, the closest replayed benchmark goal is shown instead,
labelled "replayed, not live".
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from twinmind.agents.contracts import AgentResult
from twinmind.ui import components as ui
from twinmind.ui import services as sv

NODE_TEXT = {
    "planner": "Planner read the goal",
    "explorer": "Explorer searched plans on the surrogate and shortlisted them on the twin",
    "evaluator": "Evaluator re-ran the plan and the current policy on the twin (30 reps)",
    "critic": "Critic checked the result",
    "explainer": "Explainer wrote and checked the explanation",
}

mode = sv.detect_mode()
entries = ui.demo_cache()
guard = ui.run_guard()
log = sv.history_path()

st.title("Ask")
st.caption(
    "Type a production-planning goal. The agent turns it into a precise objective, searches "
    "plans, and verifies the chosen one on the simulator against the current policy."
)


def _show(kind: str, res: AgentResult, provenance: str | None = None) -> None:
    st.session_state["ask_shown"] = {
        "kind": kind,
        "result": res.to_dict(),
        "provenance": provenance,
    }


def _replay(entry: sv.DemoEntry, asked: str | None, why: str) -> None:
    sv.append_history(log, "replayed", entry.goal_text, entry.result.status, why, entry.result)
    provenance = entry.provenance
    if asked:
        provenance = f"You asked: “{asked}”. {why} {entry.provenance}"
    _show("replayed", entry.result, provenance)


# --------------------------------------------------------------------------- #
# Live goal
# --------------------------------------------------------------------------- #
st.subheader("Your goal")
goal = st.text_area(
    "Goal",
    placeholder="e.g. Maximise profit while keeping unplanned failures to at most one this month.",
    key="ask_goal",
)
context_id = st.selectbox(
    "Planning context (a 28-day demand scenario)", list(range(20)), index=8, key="ask_ctx"
)
cfg: Any = ui.agent_config()
cap = int(cfg.budget.max_llm_calls_per_goal)

if mode.live:
    st.caption(
        f"Live: one goal at a time, at most {cap} language-model calls per goal. A run takes "
        "a few minutes; clicking elsewhere on the page while it runs stops it."
    )
    run = st.button("Run live", type="primary", disabled=guard.busy or not goal.strip())
else:
    st.caption(
        "Demo mode: live runs are off on this machine. The button below finds the closest "
        "replayed benchmark goal."
    )
    run = st.button("Find the closest replayed goal", disabled=not goal.strip())

if run and not mode.live:
    _replay(sv.closest_demo(goal, entries), goal, "Live mode is off, so this is a replay.")
elif run and not guard.try_acquire():
    st.warning("Another goal is running. One goal runs at a time; try again when it finishes.")
elif run:
    try:
        with st.status("Running the agent…", expanded=True) as status:
            counter = st.empty()
            counter.markdown(f"LLM calls: **0 / {cap}**")
            try:
                deps = ui.live_deps()
            except Exception as error:  # Groq or network unreachable: replay instead
                why = f"The language model could not be reached ({sv.safe_error(error)})."
                status.update(label="Language model unreachable: replaying", state="error")
                _replay(sv.closest_demo(goal, entries), goal, why)
            else:

                def on_step(node: str, used: int, call_cap: int) -> None:
                    st.write(f"✓ {NODE_TEXT.get(node, node)}")
                    counter.markdown(f"LLM calls: **{used} / {call_cap}**")

                from twinmind.agents.graph import run_agent_streaming

                try:
                    res = run_agent_streaming(goal, int(context_id), deps, on_step)
                except Exception as error:
                    detail = sv.safe_error(error)
                    sv.append_history(log, "aborted", goal, "aborted", detail)
                    status.update(label=f"Aborted: {detail}", state="error")
                    st.session_state.pop("ask_shown", None)
                    st.error(f"The run stopped and produced no result: {detail}")
                except BaseException:
                    # A rerun or stop (a click elsewhere on the page) interrupts
                    # the script. Record it before Streamlit unwinds.
                    sv.append_history(
                        log, "aborted", goal, "aborted", "interrupted: the page was rerun"
                    )
                    raise
                else:
                    if sv.is_llm_failure(res):
                        why = f"The language model could not be reached ({res.refusal})."
                        status.update(label="Language model unreachable: replaying", state="error")
                        _replay(sv.closest_demo(goal, entries), goal, why)
                    else:
                        sv.append_history(log, "live", goal, res.status, "", res)
                        status.update(label=f"Finished: {res.status}", state="complete")
                        _show("live", res)
    finally:
        guard.release()

# --------------------------------------------------------------------------- #
# Replayed benchmark goals
# --------------------------------------------------------------------------- #
st.subheader("Replayed benchmark goals")
st.caption(
    "Six goals from the agent benchmark, replayed from stored results: no new run and no "
    "language-model call. Each is labelled as a replay."
)
labels = {f"{e.goal_id} — {e.role}": e for e in entries}
picked = st.selectbox("Replayed goal", list(labels), key="ask_demo")
if st.button("Show this replay"):
    entry = labels[picked]
    _replay(entry, None, "Picked from the replayed benchmark goals.")

# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #
shown = st.session_state.get("ask_shown")
if shown:
    st.divider()
    ui.result(
        AgentResult.from_dict(shown["result"]),
        replayed=shown["kind"] == "replayed",
        provenance=shown["provenance"],
    )
