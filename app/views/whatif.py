"""What-if: move the 16 levers, read the surrogate's ESTIMATE, then verify on
the twin (at most 10 replications, under a wall-clock cap).

The estimate and the twin numbers are never in one table: an estimate has no
interval; a twin number always has its 95% CI and its replication count.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from twinmind.twin.runner import KPI_NAMES, Scenario
from twinmind.ui import components as ui
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm

mode = sv.detect_mode()
entries = ui.demo_cache()
guard = ui.run_guard()
specs = sv.slider_specs()

st.title("What-if")
st.caption(
    "Set the 16 levers the optimiser may move. The surrogate gives an instant estimate; "
    "the twin gives verified numbers with intervals."
)

# --------------------------------------------------------------------------- #
# Starting point
# --------------------------------------------------------------------------- #
CURRENT = "Current policy (defaults)"
presets: dict[str, sv.DemoEntry | None] = {CURRENT: None}
presets |= {f"Replayed plan: {e.goal_id} — {e.role}": e for e in entries if e.result.plan}

for spec in specs:
    st.session_state.setdefault(f"wi_{spec.name}", spec.default)
st.session_state.setdefault("wi_ctx", 8)

left, right = st.columns([3, 1])
preset_name = left.selectbox("Start from", list(presets), key="wi_preset")
if right.button("Load", width="stretch"):
    entry = presets[preset_name]
    source = entry.result.plan["scenario"] if entry and entry.result.plan else {}
    for spec in specs:
        value = float(source.get(spec.name, spec.default))
        st.session_state[f"wi_{spec.name}"] = min(max(value, spec.min), spec.max)
    if entry is not None:
        st.session_state["wi_ctx"] = entry.context_id
    st.session_state["wi_loaded"] = preset_name
    st.rerun()

context_id = int(
    st.selectbox("Planning context (a 28-day demand scenario)", list(range(20)), key="wi_ctx")
)

with st.expander("Levers", expanded=True):
    columns = st.columns(2)
    for index, spec in enumerate(specs):
        columns[index % 2].slider(
            spec.name,
            min_value=spec.min,
            max_value=spec.max,
            step=spec.step,
            key=f"wi_{spec.name}",
            format="%d" if spec.kind == "integer" else "%.2f",
            help=spec.description,
        )

values: dict[str, Any] = {spec.name: st.session_state[f"wi_{spec.name}"] for spec in specs}
values["shift_count"] = int(round(values["shift_count"]))
scenario = Scenario(**values)

# --------------------------------------------------------------------------- #
# Surrogate estimate
# --------------------------------------------------------------------------- #
st.subheader("Surrogate estimate")
if mode.twin_available:
    from twinmind.agents.graph import dev_context
    from twinmind.optim.evaluators import SurrogateEvaluator

    state = ui.twin_state()
    twin_context = dev_context(context_id)
    means = SurrogateEvaluator(state["surrogate"], twin_context).evaluate([scenario])[0]
    ui.kpi_table(vm.estimate_table(means), with_baseline=False)
    ui.novelty(vm.novelty_view(state["novelty"].score_scenario(scenario)))
else:
    st.info(
        "Surrogate estimates and twin runs are off in demo mode (see the sidebar for why). "
        "Load a replayed plan above to see its twin-verified numbers."
    )
    loaded = presets.get(st.session_state.get("wi_loaded", CURRENT))
    if loaded is not None:
        ui.replay_banner(loaded.provenance)
        ui.kpi_table(vm.kpi_table(loaded.result), with_baseline=True)

# --------------------------------------------------------------------------- #
# Verify on twin
# --------------------------------------------------------------------------- #
st.subheader("Verify on twin")
reps = st.slider("Replications", sv.WHATIF_MIN_REPS, sv.WHATIF_MAX_REPS, 5, key="wi_reps")
st.caption(
    f"At most {sv.WHATIF_MAX_REPS} replications, stopping after "
    f"{sv.WHATIF_TIME_CAP_SECONDS:.0f} s of twin time. One twin job runs at a time."
)
if st.button("Verify on twin", type="primary", disabled=not mode.twin_available or guard.busy):
    if not guard.try_acquire():
        st.warning("Another twin job is running. Try again when it finishes.")
    else:
        try:
            from twinmind.agents.graph import dev_context
            from twinmind.twin.runner import run_twin
            from twinmind.twin.streams import derive_seeds

            state = ui.twin_state()
            twin_context = dev_context(context_id)
            with st.spinner(f"Running the twin ({reps} replications)…"):
                run = sv.verify_on_twin(
                    lambda seed: run_twin(
                        scenario, seed, twin_context, state["resources"]
                    ).to_vector(),
                    derive_seeds(0, sv.WHATIF_MAX_REPS),
                    KPI_NAMES,
                    reps,
                )
            st.session_state["wi_verified"] = {
                "run": run,
                "scenario": scenario.model_dump(),
                "context_id": context_id,
            }
        finally:
            guard.release()

verified = st.session_state.get("wi_verified")
if verified:
    run = verified["run"]
    if verified["scenario"] != scenario.model_dump() or verified["context_id"] != context_id:
        st.caption(
            "The levers or context changed since this verification; these numbers are "
            "for the earlier settings."
        )
    ui.kpi_table(vm.twin_table(run.means, run.lows, run.highs, run.n_reps), with_baseline=False)
    st.caption(
        f"Measured: {run.n_reps} twin runs in {run.seconds:.1f} s "
        f"({run.seconds_per_run:.2f} s per run)."
    )
    if run.slower_than_quiet:
        st.info(
            f"The twin ran slower than the {sv.TWIN_SECONDS_PER_RUN_QUIET:.2f} s per run "
            "measured on a quiet, plugged-in machine; the laptop may be busy or warm. The "
            "numbers are unaffected, only the wait."
        )
    if run.stopped_by_time_cap:
        st.info(
            f"Stopped at the {sv.WHATIF_TIME_CAP_SECONDS:.0f} s cap after {run.n_reps} of "
            f"{run.requested} replications, so the intervals are wider."
        )
