"""Page parity with v1.0-dashboard: nothing honest may be dropped or hidden.

Each v1.0 page is kept as a fixture (tests/fixtures/v1_0_*_page.py, with the
v1.0 components they used) and rendered next to the page that replaced it, in
demo mode. For every guarantee the old page showed by default, the new page
must show the same content by default -- in plain words where the redesign
reworded it -- or, for the purely technical record, under "Show technical
details". A test fails if a guarantee is dropped or hidden.

    v1.0 Ask      -> Plan          (every saved example)
    v1.0 What-if  -> Try changes   (saved plan in demo mode; estimate and check with fakes)
    v1.0 History  -> Past results
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from twinmind.twin.runner import KPI_NAMES
from twinmind.ui import glossary as gl
from twinmind.ui import plain as pl
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm
from twinmind.ui.goals import goal_text
from twinmind.ui.jobs import JobRunner
from twinmind.utils.paths import PROJECT_ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
VIEWS = PROJECT_ROOT / "app" / "views"
TECH = "Show technical details"
SAVED = ["G07", "G01", "I01", "N01", "AA1", "G09", "G03", "G10"]


@pytest.fixture(autouse=True)
def demo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "dashboard_runs.jsonl"
    monkeypatch.setenv("TWINMIND_DEMO", "1")
    monkeypatch.setenv("TWINMIND_DASHBOARD_LOG", str(log))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from twinmind.ui import components

    runner = JobRunner()
    monkeypatch.setattr(components, "job_runner", lambda: runner)
    return log


def _render(path: Path) -> Any:
    at = AppTest.from_file(str(path), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _texts(node: Any, where: str = "visible") -> Iterator[tuple[str, str]]:
    """(where, text) for every string: "visible", or the enclosing expander."""
    if type(node).__name__ == "Expander":
        where = node.label
    seen: set[str] = set()
    for attr in ("value", "label", "body"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            yield where, value
        elif isinstance(value, pd.DataFrame):
            yield where, " | ".join(map(str, value.columns))
            for row in value.astype(str).to_numpy():
                yield where, " | ".join(row)
    for child in getattr(node, "children", {}).values():
        yield from _texts(child, where)


class Screen:
    def __init__(self, at: Any) -> None:
        self.items = list(_texts(at._tree))

    def visible(self, needle: str) -> bool:
        return any(w == "visible" and needle in t for w, t in self.items)

    def under(self, label: str, needle: str) -> bool:
        return any(w == label and needle in t for w, t in self.items)


def guarantee(old: Screen, new: Screen, name: str, old_needle: str, new_needle: str) -> None:
    assert old.visible(old_needle), f"fixture: v1.0 did not show {name}: {old_needle!r}"
    assert new.visible(new_needle), f"{name} dropped or hidden: {new_needle!r}"


# --------------------------------------------------------------------------- #
# v1.0 Ask -> Plan, every saved example
# --------------------------------------------------------------------------- #
def _old_ask(entry: sv.DemoEntry) -> Screen:
    at = _render(FIXTURES / "v1_0_ask_page.py")
    box = at.selectbox(key="ask_demo")
    box.set_value(next(o for o in box.options if o.startswith(entry.goal_id))).run()
    return Screen(next(b for b in at.button if b.label == "Show this replay").click().run())


def _new_plan(entry: sv.DemoEntry) -> Screen:
    at = _render(VIEWS / "plan.py")
    at.selectbox(key="saved_pick").set_value(goal_text(entry.result.spec or {})).run()
    at = at.button(key="saved_show").click().run()
    assert not at.exception
    return Screen(at)


@pytest.mark.parametrize("goal_id", SAVED)
def test_plan_keeps_every_guarantee_the_v1_ask_page_showed(goal_id: str) -> None:
    entry = next(e for e in sv.load_demo_cache() if e.goal_id == goal_id)
    result = entry.result
    old, new = _old_ask(entry), _new_plan(entry)

    guarantee(old, new, "saved-example label", "not a live run", "not a live run")
    guarantee(
        old,
        new,
        "where the saved example came from",
        f"goal {entry.goal_id}, seed {entry.seed}",
        f"goal {entry.goal_id}, run {entry.seed}",
    )
    assert new.under(TECH, entry.provenance), "exact provenance"
    guarantee(
        old, new, "checked label and run count", "verified on twin (30 reps)",
        "Checked with the detailed simulation (30 runs)",
    )  # fmt: skip

    evaluation = result.baseline_comparison or {}
    for kpi in KPI_NAMES:
        plan = evaluation["kpis"][kpi]["plan"]
        low = vm.format_value(kpi, plan["ci_low"])
        high = vm.format_value(kpi, plan["ci_high"])
        guarantee(old, new, f"95% range of {kpi}", f"{low} to {high}", f"{low} and {high}")

    guarantee(
        old, new, "weak-KPI warning", vm.WEAK_KPI_BADGE[:60], "Breakdown and downtime numbers"
    )
    assert new.under(TECH, vm.WEAK_KPI_BADGE), "exact badge kept"

    for effect in vm.side_effects(result):
        row = evaluation["kpis"][effect.kpi]
        plain_effect = (
            f"{gl.KPIS[effect.kpi].name} changes for the worse: "
            f"{vm.format_value(effect.kpi, row['baseline']['mean'])} → "
            f"{vm.format_value(effect.kpi, row['plan']['mean'])}"
        )
        guarantee(old, new, f"side effect on {effect.kpi}", effect.text, plain_effect)

    for flag in result.flags:
        guarantee(
            old, new, f"critic flag {flag['code']}", vm.FLAG_TEXT[flag["code"]][:50],
            pl.FLAG_WATCH[flag["code"]][1][:50],
        )  # fmt: skip

    for c in result.closest_achievable:
        closest = vm.format_value(c["kpi"], c["closest_achievable"])
        guarantee(old, new, "closest-achievable value", closest, closest)
        guarantee(old, new, "closest option", c["option"], "Option for you to decide on")

    band = vm.novelty_view((result.plan or {}).get("novelty"))
    trust = pl.trust(result)
    assert trust is not None
    guarantee(old, new, "familiar-territory band", band.text, trust.checks[2].text)

    caption = vm.explanation_source(result.explanation_mode, replayed=True) or ""
    guarantee(
        old, new, "automated-check caption", vm.EXPLANATION_CHECK,
        pl.check_caption(result.explanation_mode, saved=True),
    )  # fmt: skip
    assert "verif" not in caption.lower()
    gaps = "What this check does not catch"
    assert old.under(gaps, "Negation") and new.under(gaps, "Negation"), "check limits"
    assert new.under("AI-written explanation (technical wording)", (result.explanation or "")[:60])

    runs = int((result.trace or {}).get("twin_runs", 0))
    guarantee(old, new, "simulation cost", f"{runs} twin runs", f"{runs:,} runs")


# --------------------------------------------------------------------------- #
# v1.0 What-if -> Try changes
# --------------------------------------------------------------------------- #
def test_try_changes_keeps_the_saved_plan_guarantees_in_demo_mode() -> None:
    entry = next(e for e in sv.load_demo_cache() if e.goal_id == "G01")
    old_at = _render(FIXTURES / "v1_0_whatif_page.py")
    preset = next(o for o in old_at.selectbox(key="wi_preset").options if "G01" in o)
    old_at.selectbox(key="wi_preset").set_value(preset).run()
    old = Screen(next(b for b in old_at.button if b.label == "Load").click().run())

    new_at = _render(VIEWS / "try_changes.py")
    start = next(o for o in new_at.radio(key="tc_start").options if "breakdown" in o)
    new = Screen(new_at.radio(key="tc_start").set_value(start).run())

    guarantee(old, new, "saved-example label", "not a live run", "not a live run")
    guarantee(
        old, new, "checked label", "verified on twin (30 reps)",
        "Checked with the detailed simulation (30 runs)",
    )  # fmt: skip
    evaluation = entry.result.baseline_comparison or {}
    for kpi in pl.HEADLINE_KPIS:
        plan = evaluation["kpis"][kpi]["plan"]
        low, high = vm.format_value(kpi, plan["ci_low"]), vm.format_value(kpi, plan["ci_high"])
        guarantee(old, new, f"95% range of {kpi}", f"{low} to {high}", f"{low} and {high}")
    for kpi in KPI_NAMES:
        plan = evaluation["kpis"][kpi]["plan"]
        ci = vm.format_ci(kpi, plan["ci_low"], plan["ci_high"])
        assert new.under(TECH, ci), f"all eight ranges: {kpi}"
    guarantee(
        old, new, "weak-KPI warning", vm.WEAK_KPI_BADGE[:60], "Breakdown and downtime numbers"
    )
    assert new.under(TECH, vm.WEAK_KPI_BADGE)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake quick-estimate model and simulation, so the estimate and the check
    run in CI with no model files (both pages import the same functions)."""
    import twinmind.agents.graph as graph
    import twinmind.optim.evaluators as evaluators
    import twinmind.twin.runner as runner
    from twinmind.ui import components

    base = {
        "units_produced": 20000.0,
        "oee": 0.7,
        "unit_cost": 28.0,
        "profit": 300000.0,
        "stockout_probability": 0.2,
        "downtime_hours": 10.0,
        "unplanned_failures": 1.0,
        "on_time_delivery_rate": 0.8,
    }

    class FakeSurrogate:
        def __init__(self, model: Any, context: Any) -> None:
            pass

        def evaluate(self, scenarios: list[Any]) -> list[dict[str, float]]:
            return [{k: v * (1 + 0.01 * i) for k, v in base.items()} for i in range(len(scenarios))]

    class FakeNovelty:
        def score_scenario(self, scenario: Any) -> float:
            return 0.5

    class FakeRun:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def to_vector(self) -> list[float]:
            return [base[k] * (1 + 0.001 * (self.seed % 7)) for k in KPI_NAMES]

    state = {"surrogate": object(), "novelty": FakeNovelty(), "resources": object()}
    monkeypatch.setattr(evaluators, "SurrogateEvaluator", FakeSurrogate)
    monkeypatch.setattr(graph, "dev_context", lambda month: None)
    monkeypatch.setattr(runner, "run_twin", lambda sc, seed, ctx, res: FakeRun(seed))
    monkeypatch.setattr(components, "twin_state", lambda: state)
    monkeypatch.setattr(sv, "TWIN_SECONDS_PER_RUN_QUIET", 0.0)  # every run looks slow
    monkeypatch.setattr(
        sv, "detect_mode", lambda *a, **k: sv.Mode(live=False, twin_available=True, reasons=())
    )
    sys.path.insert(0, str(FIXTURES))
    import v1_0_components

    monkeypatch.setattr(v1_0_components, "twin_state", lambda: state)


def test_try_changes_keeps_the_estimate_and_check_guarantees(fake_model: None) -> None:
    old_at = _render(FIXTURES / "v1_0_whatif_page.py")
    old_at.slider(key="wi_reps").set_value(10).run()
    old = Screen(next(b for b in old_at.button if b.label == "Verify on twin").click().run())
    new_at = _render(VIEWS / "try_changes.py")
    new = Screen(
        next(b for b in new_at.button if b.label == "Check with the detailed simulation")
        .click()
        .run()
    )

    guarantee(old, new, "estimate label", "Numbers below: **estimate**", "Quick estimate")
    assert not any(
        "Likely between" in t for w, t in new.items if w == "visible" and "(rough)" in t
    ), "an estimate never carries a range"
    guarantee(old, new, "familiar-territory band", "Inside training coverage", "Familiar territory")
    guarantee(
        old, new, "checked label with the run count", "verified on twin (10 reps)",
        "Checked with the detailed simulation (10 runs)",
    )  # fmt: skip
    guarantee(old, new, "measured seconds", "twin runs in", "Took ")
    guarantee(old, new, "seconds per run", "s per run", "seconds per run")
    guarantee(old, new, "slower-than-usual note", "slower than the", "Slower than usual")
    guarantee(old, new, "95% ranges on checked numbers", "95% CI", "Likely between")
    guarantee(
        old, new, "weak-KPI warning", vm.WEAK_KPI_BADGE[:60], "Breakdown and downtime numbers"
    )


# --------------------------------------------------------------------------- #
# v1.0 History -> Past results
# --------------------------------------------------------------------------- #
def test_past_results_keep_every_history_guarantee(demo_env: Path) -> None:
    entry = sv.load_demo_cache()[0]
    sv.append_history(demo_env, "replayed", entry.goal_text, "ok", "picked", entry.result)
    sv.append_history(demo_env, "started", "a live goal", "running", run_id="r1")
    sv.append_history(demo_env, "live", "a live goal", "ok", "", entry.result, run_id="r1")
    sv.append_history(demo_env, "started", "a stopped goal", "running", run_id="r2")
    sv.append_history(
        demo_env, "aborted", "a stopped goal", "failed", "RuntimeError: twin exploded", run_id="r2"
    )
    old = Screen(_render(FIXTURES / "v1_0_history_page.py"))
    new_at = _render(VIEWS / "past_results.py")
    new = Screen(new_at)

    # The vendored v1.0 page reads today's labels, so its icons carry the meaning.
    guarantee(old, new, "live runs labelled", "🟢", "🟢 Live")
    guarantee(old, new, "saved examples labelled apart", "not a live run", "Saved example")
    guarantee(old, new, "a run that produced nothing", "⛔", "⏹ Stopped")
    guarantee(old, new, "why it stopped", "twin exploded", "something went wrong")
    assert len(new_at.dataframe[0].value) == 3, "every event is listed"
    assert new.under(TECH, "Agent run log"), "the agent's own run log"
