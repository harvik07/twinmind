"""Try changes: move settings, read a quick estimate, then check it properly.

The quick estimate (instant, rough, never with a range) and the detailed
simulation's numbers (with "likely between" ranges and the run count) are
separate blocks and never share a table.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from twinmind.agents.contracts import AgentResult
from twinmind.twin.runner import KPI_NAMES, Scenario
from twinmind.ui import components as ui
from twinmind.ui import plain as pl
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm
from twinmind.ui.glossary import KPIS, SECTIONS, SETTINGS
from twinmind.ui.goals import DEFAULT_MONTH, goal_text

mode = sv.detect_mode()
guard = ui.run_guard()
state = st.session_state
specs = {s.name: s for s in sv.slider_specs()}

st.title("Try changes")
st.caption("Move a setting and see roughly what happens. Then check it properly.")


# --------------------------------------------------------------------------- #
# Settings in display units (percent for speeds and pace)
# --------------------------------------------------------------------------- #
def to_display(code: str, value: float) -> float:
    return float(round(100.0 * value)) if SETTINGS[code].display == "percent" else float(value)


def from_display(code: str, value: float) -> float:
    return value / 100.0 if SETTINGS[code].display == "percent" else float(value)


def load(source: dict[str, Any]) -> None:
    for code, spec in specs.items():
        state[f"tc_{code}"] = to_display(code, float(source.get(code, spec.default)))


# --------------------------------------------------------------------------- #
# Start from
# --------------------------------------------------------------------------- #
starts: dict[str, tuple[dict[str, Any], AgentResult | None]] = {"Current setup": ({}, None)}
shown = state.get("plan_shown")
if shown:
    last = AgentResult.from_dict(shown["result"])
    if last.plan:
        tag = "saved example" if shown["kind"] == "saved" else "your latest plan"
        starts[f"Plan from the Plan page ({tag})"] = (last.plan["scenario"], last)
if not mode.twin_available:
    for entry in ui.demo_cache():
        if entry.result.plan and entry.result.plan.get("recommended"):
            label = f"Saved example: {goal_text(entry.result.spec or {})}"
            starts[label] = (entry.result.plan["scenario"], entry.result)

if "tc_shift_count" not in state:
    load({})


def _start_changed() -> None:
    scenario, _ = starts[state["tc_start"]]
    load(scenario)
    state.pop("tc_checked", None)


st.radio("Start from", list(starts), key="tc_start", on_change=_start_changed)
start_scenario, start_result = starts[state.get("tc_start", "Current setup")]

# --------------------------------------------------------------------------- #
# The 16 settings, in four sections
# --------------------------------------------------------------------------- #
FORMAT = {"percent": "%d%%", "hours": "%.1f h", "units": "%d", "count": "%d"}
for index, section in enumerate(SECTIONS):
    codes = [c for c, t in SETTINGS.items() if t.section == section]
    with st.expander(section, expanded=index == 0):
        for code in codes:
            term, spec = SETTINGS[code], specs[code]
            low, high = to_display(code, spec.min), to_display(code, spec.max)
            current = to_display(code, spec.default)
            changed = state.get(f"tc_{code}") != current
            st.slider(
                term.name + (" · changed" if changed else ""),
                min_value=low,
                max_value=high,
                step=1.0 if term.display in ("percent", "count") else spec.step,
                format="%d h" if code.startswith("pm_interval_") else FORMAT[term.display],
                key=f"tc_{code}",
                help=f"{term.help} Current setup: {current:g}.",
            )
if st.button("Reset to current setup"):
    load({})
    state["tc_start"] = "Current setup"
    st.rerun()

values: dict[str, Any] = {c: from_display(c, float(state[f"tc_{c}"])) for c in specs}
values["shift_count"] = int(round(values["shift_count"]))
scenario = Scenario(**values)
current_setup = Scenario()


# --------------------------------------------------------------------------- #
# Quick estimate
# --------------------------------------------------------------------------- #
st.markdown(f"### {pl.ESTIMATE_LABEL}")
if mode.twin_available:
    from twinmind.agents.graph import dev_context
    from twinmind.optim.evaluators import SurrogateEvaluator

    model = ui.twin_state()
    month = dev_context(DEFAULT_MONTH)
    now, new = SurrogateEvaluator(model["surrogate"], month).evaluate([current_setup, scenario])
    st.caption("Your current setup → these settings.")
    ui.metric_cards(pl.estimate_cards(now, new))
    with st.expander("More numbers"):
        st.caption(pl.ESTIMATE_LABEL + ".")
        st.dataframe(
            [
                {
                    "Number": KPIS[k].name,
                    "Current setup": pl.value_text(k, now[k]),
                    "These settings": pl.value_text(k, new[k]),
                    "Note": "⚠ rough guide" if KPIS[k].rough else "",
                }
                for k in KPI_NAMES
                if k not in pl.HEADLINE_KPIS
            ],
            hide_index=True,
            width="stretch",
        )
    band = vm.novelty_view(model["novelty"].score_scenario(scenario)).band
    familiar = {
        "inside": "Familiar territory: these settings are close to ones the model has seen.",
        "edge": "Less familiar territory: the model has seen few settings like these.",
        "outside": "Unfamiliar territory: treat the quick estimate with caution.",
    }.get(band, "")
    st.markdown(familiar)
    with st.expander(ui.TECH):
        ui.kpi_table(vm.estimate_table(new), with_baseline=False)
        st.caption(f"Novelty band: {band}. Context {DEFAULT_MONTH}.")
else:
    st.info(pl.error_message("no_quick_estimate"), icon="📁")
    unmoved = all(
        state[f"tc_{c}"] == to_display(c, float(start_scenario.get(c, specs[c].default)))
        for c in specs
    )
    if start_result is not None and unmoved:
        st.warning(f"**{sv.REPLAY_LABEL}.**", icon="📁")
        st.caption(pl.checked_label(start_result) + ", against the current setup.")
        ui.metric_cards(pl.kpi_cards(start_result))

# --------------------------------------------------------------------------- #
# Check with the detailed simulation
# --------------------------------------------------------------------------- #
st.markdown("### Check with the detailed simulation")
pressed = st.button(
    "Check with the detailed simulation",
    type="primary",
    disabled=not mode.twin_available or guard.busy or ui.job_runner().busy,
)
st.caption(
    f"Takes a few seconds ({sv.WHATIF_MAX_REPS} runs). "
    + ("Waiting for the plan being worked out to finish." if ui.job_runner().busy else "")
)
if pressed and guard.try_acquire():
    try:
        from twinmind.agents.graph import dev_context
        from twinmind.twin.runner import run_twin
        from twinmind.twin.streams import derive_seeds

        model = ui.twin_state()
        month = dev_context(DEFAULT_MONTH)
        with st.spinner("Running the detailed simulation…"):
            run = sv.verify_on_twin(
                lambda seed: run_twin(scenario, seed, month, model["resources"]).to_vector(),
                derive_seeds(0, sv.WHATIF_MAX_REPS),
                KPI_NAMES,
                sv.WHATIF_MAX_REPS,
            )
        state["tc_checked"] = {"run": run, "scenario": scenario.model_dump()}
    finally:
        guard.release()

checked = state.get("tc_checked")
if checked:
    run = checked["run"]
    if checked["scenario"] != scenario.model_dump():
        st.caption("You have moved settings since this check; these numbers are for before.")
    st.caption(pl.CHECKED_LABEL.format(n=run.n_reps) + ".")
    ui.metric_cards(
        [
            pl.KPICard(
                kpi=kpi,
                label=KPIS[kpi].name,
                after=pl.value_text(kpi, run.means[kpi]),
                before=None,
                change="",
                word="",
                help=f"{KPIS[kpi].help} Likely between {pl.value_text(kpi, run.lows[kpi])} "
                f"and {pl.value_text(kpi, run.highs[kpi])}.",
                rough=KPIS[kpi].rough,
            )
            for kpi in pl.HEADLINE_KPIS
        ]
    )
    note = f"Took {run.seconds:.1f} seconds."
    if run.slower_than_quiet:
        note += " Slower than usual: the computer may be busy."
    if run.stopped_by_time_cap:
        note += f" Stopped early after {run.n_reps} runs to save time."
    st.caption(note)
    with st.expander(ui.TECH):
        ui.kpi_table(vm.twin_table(run.means, run.lows, run.highs, run.n_reps), with_baseline=False)
