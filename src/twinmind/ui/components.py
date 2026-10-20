"""Streamlit rendering. The only ``twinmind.ui`` module that imports Streamlit.

The default view uses plain words only (``ui.plain``, ``ui.glossary``); the
exact technical labels from ``ui.viewmodel`` live in "Show technical details".
Heavy objects load once per process (``st.cache_resource``); agent runs go
through the single background ``JobRunner``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from twinmind.agents.contracts import AgentResult
from twinmind.ui import plain as pl
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm
from twinmind.ui.glossary import KPIS, SECTIONS, SETTINGS
from twinmind.ui.jobs import JobRunner

TECH = "Show technical details"
AI_EXPLANATION = "AI-written explanation (technical wording)"
CHECK_GAPS = "What this check does not catch"


# --------------------------------------------------------------------------- #
# Shared, cached once per process
# --------------------------------------------------------------------------- #
@st.cache_resource
def run_guard() -> sv.RunGuard:
    """One twin check on the Try changes page at a time, process-wide."""
    return sv.RunGuard()


@st.cache_resource
def job_runner() -> JobRunner:
    """The single background runner for agent goals, process-wide."""
    return JobRunner()


@st.cache_resource
def demo_cache() -> list[sv.DemoEntry]:
    return sv.load_demo_cache()


@st.cache_resource
def agent_config() -> Any:
    from twinmind.utils.config import load_config

    return load_config("agent")


@st.cache_resource(show_spinner="Loading the detailed simulation (first time only)…")
def twin_state() -> dict[str, Any]:
    """The quick-estimate model, familiarity scorer and simulation, loaded once."""
    from twinmind.agents.graph import _shared_state

    return _shared_state()


@st.cache_resource(show_spinner="Connecting to the AI service…")
def live_deps() -> Any:
    """The live agent's dependencies, built once."""
    from twinmind.agents.graph import AgentDeps, groq_client, twin_toolkit
    from twinmind.utils.paths import PROJECT_ROOT

    config = agent_config()
    twin_state()
    return AgentDeps(
        config=config,
        llm=groq_client(config),
        toolkit=twin_toolkit(config),
        run_log=PROJECT_ROOT / config.run_log,
    )


def demo_by_id(entry_id: str) -> sv.DemoEntry:
    return next(e for e in demo_cache() if e.id == entry_id)


# --------------------------------------------------------------------------- #
# Chrome: sidebar, running banner, glossary
# --------------------------------------------------------------------------- #
def sidebar(mode: sv.Mode) -> None:
    st.sidebar.markdown("### TwinMind")
    st.sidebar.caption(
        "Finds better production settings for your line and checks them in a detailed "
        "simulation before recommending anything."
    )
    if mode.live:
        st.sidebar.success("Live: plans are worked out now.", icon="🟢")
    else:
        st.sidebar.info("Demo mode: showing saved example results.", icon="📁")
    with st.sidebar.expander("Glossary"):
        glossary()
    if mode.reasons:
        with st.sidebar.expander(TECH):
            st.caption("Why demo mode is on (no key or file contents are shown):")
            st.markdown("\n".join(f"- {reason}" for reason in mode.reasons))


def glossary() -> None:
    st.markdown("**Numbers**")
    st.markdown(
        "\n".join(
            f"- **{t.name}** ({t.unit}, {t.better.lower()} is better): {t.meaning}"
            + (" *Rough guide.*" if t.rough else "")
            for t in KPIS.values()
        )
    )
    st.markdown("**Settings**")
    for section in SECTIONS:
        names = [s for s in SETTINGS.values() if s.section == section]
        st.markdown(
            f"*{section}*\n" + "\n".join(f"- **{s.name}** ({s.unit}): {s.meaning}" for s in names)
        )


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


@st.fragment(run_every=2.0)
def running_banner() -> None:
    """On every page while a plan is being worked out, with Cancel."""
    import time

    runner = job_runner()
    status = runner.status()
    if not status.active:
        return
    elapsed = _elapsed(status.elapsed(time.monotonic()))
    if status.state == "cancelling":
        text = "Stopping the plan: the current step finishes first (up to about a minute)."
    elif status.state == "stopping":
        text = "The plan hit the time limit and is stopping after the current step."
    else:
        text = f"A plan is being worked out ({elapsed} so far): “{status.goal_text}”"
    left, right = st.columns([5, 1])
    left.info(text, icon="⏳")
    if status.state == "running" and right.button("Cancel", key="cancel_run", width="stretch"):
        runner.cancel()
        st.rerun()


# --------------------------------------------------------------------------- #
# Pieces of a result
# --------------------------------------------------------------------------- #
def status_chip(chip: pl.Chip) -> None:
    st.badge(chip.text, icon=chip.icon, color=chip.color)  # type: ignore[arg-type]


def metric_cards(cards: Sequence[pl.KPICard]) -> None:
    """Headline numbers as small cards, two per row, so they stay readable on a
    tablet; the text wraps instead of being cut off."""
    for start in range(0, len(cards), 2):
        columns = st.columns(2)
        for column, card in zip(columns, cards[start : start + 2], strict=False):
            with column.container(border=True):
                st.markdown(f"**{card.label}**", help=card.help)
                st.subheader(card.after, anchor=False)
                if card.before is not None:
                    st.caption(f"Now: {card.before}")
                if card.change:
                    st.markdown(f":{card.color}[{card.line}]")
                if card.likely:
                    st.caption(card.likely)
                if card.rough:
                    st.caption("⚠ Rough guide: the model is less accurate here.")


def trust_block(trust: pl.Trust) -> None:
    st.markdown(f"{trust.dots} **{trust.label}**")
    st.caption(trust.conclusion)
    st.markdown("\n".join(f"- {c.icon} {c.text}" for c in trust.checks))


def watch_list(items: Sequence[pl.WatchItem]) -> None:
    if not items:
        st.markdown("Nothing to watch.")
        return
    st.markdown("\n".join(f"- {i.icon} **{i.severity}** · {i.text}" for i in items))


def plan_result(
    res: AgentResult,
    *,
    saved: bool,
    note: str = "",
    asked: str = "",
    unreachable: bool = False,
    saved_from: str = "",
    provenance: str = "",
) -> None:
    """The whole plain result: answer first, evidence after, technical last.

    Every safety item stays visible: the saved-example label and where it came
    from, the checked label and its 95% ranges, Before you act (side effects,
    flags, the rough-guide note), the automated-check caption, the trust level
    and the run cost. The exact technical labels are under Show technical
    details.
    """
    if saved:
        st.info(f"**{sv.REPLAY_LABEL}.**", icon="📁")
        st.caption(saved_from or "This result was worked out earlier and saved.")
    if note:
        st.info(note, icon="ℹ️")
    trust = pl.trust(res)
    top_left, top_right = st.columns([3, 2])
    with top_left:
        status_chip(pl.chip(res))
    if trust is not None:
        top_right.markdown(f"How much to trust this: {trust.dots} **{trust.label}**")
    if asked:
        st.caption(f"You asked: “{asked}”")
    st.caption(f"Goal: {pl.goal_sentence(res)}")
    st.markdown(f"### {pl.headline(res, unreachable=unreachable)}")

    options = pl.closest_lines(res)
    if pl.outcome(res) == "not_found" and options:
        st.markdown("#### Closest options")
        st.markdown(
            "\n".join(f"- {o}" for o in options)
            + "\n\nChange a limit and press Find a plan again, or keep your current setup."
        )

    changes = pl.setting_changes(res) if pl.outcome(res) == "recommended" else []
    if changes:
        st.markdown("#### What to change")
        st.markdown("\n".join(f"- {c.text}" for c in changes[:3]))
        if len(changes) > 3:
            with st.expander(f"{len(changes) - 3} more settings"):
                st.markdown("\n".join(f"- {c.text}" for c in changes[3:]))

    cards = pl.kpi_cards(res)
    if cards:
        title = "What you get" if pl.outcome(res) == "recommended" else "The best plan found"
        st.markdown(f"#### {title}")
        st.caption(pl.checked_label(res) + ", against your current setup.")
        metric_cards(cards)
        rows = pl.more_numbers(res)
        if rows:
            st.markdown("**More numbers**")
            st.caption(pl.checked_label(res) + ".")
            st.dataframe(rows, hide_index=True, width="stretch")

    if res.baseline_comparison:
        st.markdown("#### Before you act")
        watch_list(pl.things_to_watch(res))

    summary = pl.plain_summary(res)
    if summary:
        st.markdown("#### Why this plan" if pl.outcome(res) == "recommended" else "#### Summary")
        st.markdown(summary)
    if res.explanation:
        source = vm.explanation_source(res.explanation_mode, replayed=saved)
        caption = pl.check_caption(res.explanation_mode, saved=saved)
        if caption:
            st.caption(caption)
        with st.expander(AI_EXPLANATION):
            st.markdown(res.explanation)
            if source:
                st.caption(source)
        with st.expander(CHECK_GAPS):
            st.markdown("\n".join(f"- {gap}" for gap in vm.EXPLANATION_CHECK_GAPS))

    if trust is not None:
        st.markdown("#### How much to trust this")
        trust_block(trust)
        st.caption(pl.cost_line(res))

    with st.expander(TECH):
        technical_details(res, saved=saved, provenance=provenance)


# --------------------------------------------------------------------------- #
# Technical details: everything v1.0 showed, with the exact labels
# --------------------------------------------------------------------------- #
def _note(row: vm.KPIRow) -> str:
    return f"⚠️ {row.badge}" if row.badge else ""


def kpi_table(rows: Sequence[vm.KPIRow], *, with_baseline: bool) -> None:
    """A technical KPI table. Every row shares one source label."""
    if not rows:
        return
    labels = {r.source_label for r in rows}
    if len(labels) != 1:
        raise ValueError(f"a table mixes source labels: {sorted(labels)}")
    label = labels.pop()
    data: list[dict[str, str]] = []
    for r in rows:
        line = {"KPI": r.name}
        if with_baseline:
            line |= {"Current policy": r.baseline or "", "Plan": r.value}
            line |= {"Change": r.change or "", "Change 95% CI": r.change_ci or ""}
        else:
            line |= {"Value": r.value}
        if r.value_ci:
            line["Plan 95% CI" if with_baseline else "95% CI"] = r.value_ci
        line["Source"] = r.source_label
        line["Note"] = _note(r)
        data.append(line)
    st.caption(f"Numbers below: **{label}**.")
    st.dataframe(data, hide_index=True, width="stretch")
    names = " and ".join(vm.KPI_DISPLAY[k] for k in vm.WEAK_KPIS)
    st.caption(f"⚠️ {names}: {vm.WEAK_KPI_BADGE}.")


def technical_details(res: AgentResult, *, saved: bool, provenance: str = "") -> None:
    if saved:
        st.caption(f"{sv.TECH_REPLAY_LABEL}. {provenance}".strip())
    card = vm.recommendation_card(res, replayed=saved)
    st.markdown(
        f"Status `{res.status}` · {card.status_label} · {card.headline}  \n"
        f"Objective: {card.objective or '—'}  \n"
        f"Constraints: {'; '.join(card.constraints) or 'none'}  \n"
        f"Verdict reason: {card.reason or '—'}"
    )
    effects = vm.side_effects(res)
    if effects:
        st.markdown("\n".join(f"- {e.text} ({e.source_label})" for e in effects))
    rows = vm.kpi_table(res)
    if rows:
        kpi_table(rows, with_baseline=True)
    if card.plan_changes:
        st.markdown("Plan changes (scenario levers):  \n" + "  \n".join(card.plan_changes))
    flags = vm.flags_plain(res)
    if flags:
        st.markdown("\n".join(f"- `{f.code}`: {f.text}" for f in flags))
    if res.plan is not None:
        view = vm.novelty_view(res.plan.get("novelty"))
        score = "n/a" if view.score is None else f"{view.score:.3f}"
        st.markdown(f"Novelty score {score} ({view.band} band): {view.text}")
    status = vm.run_status(res)
    trace = res.trace or {}
    st.caption(
        f"{status.detail}; LLM calls {trace.get('llm_calls_total', 0)} of cap "
        f"{trace.get('llm_call_cap', '?')}; context {res.context_id}."
    )
    if card.explanation_source:
        st.caption(f"Explanation: {card.explanation_source}")


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
STEP_ICON = {"done": "✓", "now": "◌", "todo": "○"}


def progress_panel(status: Any, *, card_goal: bool) -> None:
    import time

    with st.container(border=True):
        st.markdown(f"**Finding a plan for:** {status.goal_text}")
        lines = []
        for text, state in status.steps:
            if card_goal and text == "Understanding your goal":
                state = "done"
            lines.append(f"{STEP_ICON[state]} {text}")
        st.markdown("  \n".join(lines))
        st.caption(
            f"Elapsed {_elapsed(status.elapsed(time.monotonic()))} · usually 2–4 minutes. "
            "You can keep using the page while it works."
        )
        if status.llm_cap:
            st.caption(f"AI calls used: {status.llm_used} of {status.llm_cap} allowed.")
