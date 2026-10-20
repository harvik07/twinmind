"""Plan: choose what to improve, set limits, press one button, read the answer.

Card goals become a goal directly (no AI needed to read them); "Describe it in
your own words" uses the AI planner. Live runs go to the single background
runner, so the page stays usable. Demo mode shows a saved example for a
similar goal, says so at the top, and greys out the limits.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from twinmind.agents.contracts import AgentResult
from twinmind.ui import components as ui
from twinmind.ui import plain as pl
from twinmind.ui import services as sv
from twinmind.ui.goals import (
    CARDS,
    DEFAULT_MONTH,
    OVERTIME_HELP,
    OWN_WORDS,
    OWN_WORDS_EXAMPLES,
    GoalFormError,
    build_spec,
    goal_text,
    is_related_only,
)
from twinmind.ui.jobs import JobStatus

mode = sv.detect_mode()
demo = not mode.live
runner = ui.job_runner()
log = sv.history_path()
state = st.session_state

st.title("Plan")
st.caption("Find better settings for your line, checked in a detailed simulation.")
if demo:
    st.warning(pl.error_message("demo"), icon="📁")

# --------------------------------------------------------------------------- #
# Welcome
# --------------------------------------------------------------------------- #
if not state.get("welcome_dismissed"):
    with st.container(border=True):
        st.markdown(
            "**Welcome.** Three steps:\n\n"
            "1. Choose what you want to improve.\n"
            "2. Set the limits you can live with, or skip them.\n"
            "3. Press **Find a plan**. It takes about 2–4 minutes."
        )
        if st.button("Got it", key="welcome_ok"):
            state["welcome_dismissed"] = True
            st.rerun()


# --------------------------------------------------------------------------- #
# Showing a result
# --------------------------------------------------------------------------- #
def show(kind: str, result: AgentResult, note: str = "", asked: str = "") -> None:
    state["plan_shown"] = {
        "kind": kind,
        "result": result.to_dict(),
        "note": note,
        "asked": asked,
    }


def show_saved(entry: sv.DemoEntry, card_id: str | None, asked: str, why: str) -> None:
    note = ""
    if card_id and is_related_only(card_id, entry.result.spec or {}):
        note = (
            "This saved example was for a related goal: "
            f"{goal_text(entry.result.spec or {})} Your settings were not used."
        )
    sv.append_history(log, "replayed", goal_text(entry.result.spec or {}), entry.result.status,
                      why, entry.result)  # fmt: skip
    show("saved", entry.result, note, asked)


# --------------------------------------------------------------------------- #
# Step 1: what to improve
# --------------------------------------------------------------------------- #
st.markdown("### 1 · What do you want to improve?")
chosen = state.get("plan_card")
columns = st.columns(3)
for index, card in enumerate([*CARDS.values(), None]):
    card_id = card.id if card else OWN_WORDS
    with columns[index % 3].container(border=True):
        if card:
            st.markdown(f"#### {card.icon} {card.title}")
            st.caption(card.example)
        else:
            st.markdown("#### ✏️ Describe it in your own words")
            st.caption("“Make more profit but keep breakdowns low”")
        selected = chosen == card_id
        if st.button(
            "✓ Selected" if selected else "Choose",
            key=f"card_{card_id}",
            width="stretch",
        ):
            state["plan_card"] = card_id
            st.rerun()

# --------------------------------------------------------------------------- #
# Step 2: limits (or own words)
# --------------------------------------------------------------------------- #
limits: dict[str, float | None] = {}
allow_overtime = True
own_words = ""
month = DEFAULT_MONTH
if chosen == OWN_WORDS:
    st.markdown("### 2 · Describe your goal")
    examples = st.columns(len(OWN_WORDS_EXAMPLES))
    for column, example in zip(examples, OWN_WORDS_EXAMPLES, strict=True):
        if column.button(example, key=f"ex_{example[:12]}", width="stretch"):
            state["own_words"] = example
            st.rerun()
    own_words = st.text_area(
        "Your goal in a sentence",
        key="own_words",
        placeholder="For example: make more profit but keep breakdowns to one a month.",
    )
elif chosen in CARDS:
    card = CARDS[chosen]
    st.markdown("### 2 · Any limits?")
    if demo:
        st.caption("Not used in demo mode: the saved example has its own limits.")
    else:
        st.caption("Optional. Tick a limit to use it.")
    for field, default_on in card.limit_fields:
        left, right = st.columns([2, 1])
        use = left.checkbox(
            field.label, value=default_on, key=f"use_{chosen}_{field.id}",
            help=field.help, disabled=demo,
        )  # fmt: skip
        value = right.number_input(
            field.label,
            min_value=field.min,
            max_value=field.max,
            value=field.default,
            step=field.step,
            key=f"val_{chosen}_{field.id}",
            label_visibility="collapsed",
            disabled=demo or not use,
        )
        limits[field.id] = float(value) if use else None
    if card.overtime_choice:
        allow_overtime = (
            st.radio(
                "Allow overtime?",
                ["Yes", "No"],
                horizontal=True,
                key=f"ot_{chosen}",
                help=OVERTIME_HELP,
                disabled=demo,
            )
            == "Yes"
        )
if chosen:
    with st.expander("More options"):
        month = (
            st.selectbox(
                "Which month of orders to plan for",
                list(range(20)),
                index=DEFAULT_MONTH,
                format_func=lambda m: f"Example month {m + 1}",
                key="plan_month",
                help="Each example month has a different pattern of customer orders.",
                disabled=demo,
            )
            or 0
        )

# --------------------------------------------------------------------------- #
# Step 3: one button
# --------------------------------------------------------------------------- #
st.markdown("### 3 · Get your plan")
busy = runner.busy
label = "Show a saved example" if demo else "Find a plan"
ready = bool(chosen) and (chosen != OWN_WORDS or bool(own_words.strip()))
pressed = st.button(label, type="primary", disabled=busy or not ready, key="find")
if demo:
    st.caption("Demo mode shows a saved result for a similar goal straight away.")
elif busy:
    st.caption(pl.error_message("busy"))
else:
    st.caption("Usually takes 2–4 minutes. You can keep using the page while it works.")


def _on_done(final: JobStatus) -> None:
    """Runs in the worker thread: write Past results, never touch Streamlit."""
    goal = final.goal_text
    if final.state == "finished" and isinstance(final.result, AgentResult):
        sv.append_history(log, "live", goal, final.result.status, "", final.result,
                          run_id=final.id)  # fmt: skip
    else:
        sv.append_history(log, "aborted", goal, final.state, final.error_detail or final.state,
                          run_id=final.id)  # fmt: skip


def _start_live() -> None:
    try:
        deps = ui.live_deps()
    except Exception as error:
        state["plan_error"] = {"kind": "ai_unreachable", "detail": sv.safe_error(error),
                               "goal": own_words or ""}  # fmt: skip
        return
    if chosen == OWN_WORDS:
        from twinmind.agents.graph import run_agent_streaming

        goal = own_words.strip()

        def work(progress: Any) -> AgentResult:
            result = run_agent_streaming(goal, int(month), deps, progress)
            if sv.is_llm_failure(result):
                raise ConnectionError(f"LLM unavailable: {result.refusal}")
            return result

        meta = {"card": False, "asked": goal}
    else:
        from twinmind.agents.graph import run_on_spec_streaming

        try:
            spec = build_spec(str(chosen), limits, allow_overtime, int(month))
        except GoalFormError as error:
            state["plan_error"] = {"kind": "form", "detail": str(error)}
            return
        goal = spec.name
        seed = int(ui.agent_config().explorer.seed)

        def work(progress: Any) -> AgentResult:
            return run_on_spec_streaming(spec, deps, seed, progress, explain_plan=True)

        meta = {"card": True, "asked": ""}
    started = runner.start(goal, work, _on_done, meta)
    if started is None:
        state["plan_error"] = {"kind": "busy", "detail": ""}
        return
    sv.append_history(log, "started", goal, "running", "", run_id=started.id)
    state["plan_job"] = started.id
    state.pop("plan_shown", None)


if pressed:
    state.pop("plan_error", None)
    if demo:
        if chosen == OWN_WORDS:
            entry = sv.closest_demo(own_words, ui.demo_cache())
            show_saved(entry, None, own_words.strip(), "Demo mode: closest saved example.")
        else:
            card = CARDS[str(chosen)]
            try:
                build_spec(card.id, {}, True, DEFAULT_MONTH)  # the card always makes a goal
            except GoalFormError as error:  # pragma: no cover -- every card is tested
                st.error(str(error))
            show_saved(ui.demo_by_id(card.demo_id), card.id, "", "Demo mode: saved example.")
    else:
        _start_live()

error = state.get("plan_error")
if error:
    if error["kind"] == "form":
        st.error(error["detail"], icon="✋")
    else:
        st.error(pl.error_message(error["kind"]), icon="⚠️")
        if error["kind"] == "ai_unreachable":
            entry = sv.closest_demo(error.get("goal") or "", ui.demo_cache())
            ui.plan_result(entry.result, saved=True)


# --------------------------------------------------------------------------- #
# Progress (polled) and the finished run
# --------------------------------------------------------------------------- #
@st.fragment(run_every=1.0)
def progress() -> None:
    job_id = state.get("plan_job")
    if not job_id:
        return
    status = runner.status()
    if status.id != job_id:
        return
    if status.active:
        ui.progress_panel(status, card_goal=bool(status.meta.get("card")))
    elif state.get("plan_job_done") != job_id:
        state["plan_job_done"] = job_id
        st.rerun(scope="app")


progress()

job_id = state.get("plan_job")
status = runner.status()
if job_id and status.id == job_id and not status.active and state.get("plan_seen") != job_id:
    state["plan_seen"] = job_id
    if status.state == "finished" and isinstance(status.result, AgentResult):
        show("live", status.result, asked=str(status.meta.get("asked", "")))
    else:
        state["plan_shown"] = None
        state["plan_failed"] = {"kind": status.error_kind or "failed", "detail":
                                status.error_detail, "goal": status.goal_text}  # fmt: skip

failed = state.get("plan_failed")
if failed and not state.get("plan_shown"):
    st.error(pl.error_message(failed["kind"]), icon="⚠️")
    if failed["kind"] == "ai_unreachable":
        entry = sv.closest_demo(failed.get("goal") or "", ui.demo_cache())
        ui.plan_result(entry.result, saved=True)
    if failed.get("detail"):
        with st.expander(ui.TECH):
            st.code(failed["detail"])

shown = state.get("plan_shown")
if shown:
    st.divider()
    ui.plan_result(
        AgentResult.from_dict(shown["result"]),
        saved=shown["kind"] == "saved",
        note=shown.get("note", ""),
        asked=shown.get("asked", ""),
    )

# --------------------------------------------------------------------------- #
# A saved example, any time
# --------------------------------------------------------------------------- #
st.divider()
st.markdown("##### No time? See a saved example result")
entries = {goal_text(e.result.spec or {}): e for e in ui.demo_cache()}
picked = st.selectbox("Saved example", list(entries), key="saved_pick")
if st.button("Show this saved example", key="saved_show") and picked:
    state.pop("plan_failed", None)
    show_saved(entries[picked], None, "", "Picked from the saved examples.")
    st.rerun()
