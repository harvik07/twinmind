"""Streamlit rendering of the view models. The only ``twinmind.ui`` module that
imports Streamlit; every word it shows comes from ``viewmodel`` or ``services``
or is a fixed string here, and all of it is scanned for overclaiming in tests.

Heavy objects (the surrogate, the twin's resources, the Groq client) are
loaded once per process with ``st.cache_resource``; nothing runs in the
background, and one agent goal or twin verification runs at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from twinmind.agents.contracts import AgentResult
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm

TONE_ICON = {"good": "✅", "warning": "⚠️", "info": "ℹ️", "error": "⛔"}  # noqa: RUF001


# --------------------------------------------------------------------------- #
# Shared, cached once per process
# --------------------------------------------------------------------------- #
@st.cache_resource
def run_guard() -> sv.RunGuard:
    """One heavy job (agent goal or twin verification) at a time, process-wide."""
    return sv.RunGuard()


@st.cache_resource
def demo_cache() -> list[sv.DemoEntry]:
    return sv.load_demo_cache()


@st.cache_resource
def agent_config() -> Any:
    from twinmind.utils.config import load_config

    return load_config("agent")


@st.cache_resource(show_spinner="Loading the surrogate and the twin (once per session)…")
def twin_state() -> dict[str, Any]:
    """The surrogate, novelty scorer and twin resources, loaded once."""
    from twinmind.agents.graph import _shared_state

    return _shared_state()


@st.cache_resource(show_spinner="Connecting to the language model…")
def live_deps() -> Any:
    """The live agent's dependencies: Groq client and the real toolkit, once."""
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


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #
def sidebar(mode: sv.Mode) -> None:
    """Mode and why, on every page. Never shows a key or an .env value."""
    st.sidebar.markdown("### TwinMind")
    st.sidebar.caption("An agentic digital twin of a five-machine production line.")
    if mode.live:
        st.sidebar.success(mode.label)
    else:
        st.sidebar.info(mode.label)
        for reason in mode.reasons:
            st.sidebar.caption(f"• {reason}")
    st.sidebar.caption(
        "Every number comes from the simulator or its surrogate; the language model only "
        "reads goals and writes explanations."
    )


def replay_banner(provenance: str | None = None) -> None:
    """The label every replayed result carries, visibly, above everything else."""
    st.warning(f"**{sv.REPLAY_LABEL.capitalize()}.**", icon="🔁")
    if provenance:
        st.caption(provenance)


def weak_kpi_note() -> None:
    names = " and ".join(vm.KPI_DISPLAY[k] for k in vm.WEAK_KPIS)
    st.caption(f"⚠️ {names}: {vm.WEAK_KPI_BADGE}.")


# --------------------------------------------------------------------------- #
# Tables: one source label per table, never mixed
# --------------------------------------------------------------------------- #
def _note(row: vm.KPIRow) -> str:
    return f"⚠️ {row.badge}" if row.badge else ""


def kpi_table(rows: Sequence[vm.KPIRow], *, with_baseline: bool) -> None:
    """A KPI table. Every row shares one source label, shown in the caption."""
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
    weak_kpi_note()


def novelty(view: vm.NoveltyView) -> None:
    icon = {"inside": "🟢", "edge": "🟡", "outside": "🔴"}.get(view.band, "⚪")
    st.markdown(f"{icon} **Training coverage:** {view.text}")


# --------------------------------------------------------------------------- #
# A whole result
# --------------------------------------------------------------------------- #
def result(res: AgentResult, *, replayed: bool, provenance: str | None = None) -> None:
    """Card, KPI table, flags, options, coverage, explanation, run status."""
    if replayed:
        replay_banner(provenance)
    card = vm.recommendation_card(res, replayed=replayed)
    icon = TONE_ICON.get(card.tone, "")
    st.markdown(f"#### {icon} {card.status_label}")
    st.markdown(f"> {card.goal_text}")
    st.markdown(f"**{card.headline}**")
    if card.question:
        st.info(f"The agent asks: {card.question}")
    if card.refusal:
        st.error(f"Reason: {card.refusal}")
    if card.objective:
        constraints = "; ".join(card.constraints) or "none"
        st.markdown(f"**Objective:** {card.objective}  \n**Constraints:** {constraints}")
    if card.reason:
        st.markdown(f"**Verdict reason:** {card.reason}")
    if card.plan_changes:
        with st.expander(f"Plan changes ({len(card.plan_changes)} levers)"):
            st.markdown("\n".join(f"- `{c}`" for c in card.plan_changes))

    rows = vm.kpi_table(res)
    if rows:
        st.markdown("##### Plan against the current policy")
        effects = vm.side_effects(res)
        if effects:
            st.warning(
                "\n".join(f"- {e.text}" for e in effects)
                + f"\n\n{effects[0].source_label}: the 95% interval of each change excludes "
                "the current policy's value, in the unfavourable direction. These KPIs are "
                "not part of the goal.",
                icon="⚠️",
            )
        kpi_table(rows, with_baseline=True)

    flags = vm.flags_plain(res)
    if flags:
        st.markdown("##### Critic flags")
        for f in flags:
            st.markdown(f"- **{f.code.replace('_', ' ')}**: {f.text}")

    options = vm.closest_options(res)
    if options:
        st.markdown("##### Closest achievable (options, not decisions)")
        for o in options:
            st.markdown(f"- {o.constraint}. {o.closest}. *{o.option}* — {o.note}")

    if res.plan is not None:
        novelty(vm.novelty_view(res.plan.get("novelty")))

    if card.explanation:
        st.markdown("##### Explanation")
        st.markdown(card.explanation)
        if card.explanation_source:
            st.caption(card.explanation_source)
        with st.expander("What this check does not catch"):
            st.markdown("\n".join(f"- {gap}" for gap in vm.EXPLANATION_CHECK_GAPS))

    status = vm.run_status(res)
    if replayed:
        twin_runs = (res.trace or {}).get("twin_runs", 0)
        st.caption(
            f"Replayed: the plan and its twin numbers are from the benchmark run "
            f"({twin_runs} twin runs); the explanation is from the benchmark's audit."
        )
    else:
        st.caption(f"Run: {status.label} — {status.detail}.")
