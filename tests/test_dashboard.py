"""The redesigned dashboard, rendered with AppTest, offline.

What a factory manager SEES is tested here: the jargon ban on the default
view, demo mode's banner and greyed-out limits, every goal card producing a
labelled saved example, the Prompt 10 honesty rules in their new plain
wording, Past results labels, the running banner with Cancel, and the live
path (with fakes) including "clicking elsewhere does not stop a run".

No network, no Groq key, no data/ or models/: TWINMIND_DEMO=1 forces demo
mode as on CI, and the history log goes to a temp file.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from twinmind.agents.contracts import AgentResult
from twinmind.ui import glossary as gl
from twinmind.ui import plain as pl
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm
from twinmind.ui.goals import CARDS, is_related_only
from twinmind.ui.jobs import JobRunner
from twinmind.utils.paths import PROJECT_ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
MAIN = str(PROJECT_ROOT / "app" / "main.py")
PAGES = [
    "views/plan.py",
    "views/try_changes.py",
    "views/reliability.py",
    "views/past_results.py",
]


@pytest.fixture(autouse=True)
def demo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "dashboard_runs.jsonl"
    monkeypatch.setenv("TWINMIND_DEMO", "1")
    monkeypatch.setenv("TWINMIND_DASHBOARD_LOG", str(log))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return log


@pytest.fixture(autouse=True)
def fresh_runner(monkeypatch: pytest.MonkeyPatch) -> JobRunner:
    """Each test gets its own background runner (no state leaks between tests)."""
    from twinmind.ui import components

    runner = JobRunner()
    monkeypatch.setattr(components, "job_runner", lambda: runner)
    return runner


def _app(page: str = "views/plan.py") -> Any:
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    if page != "views/plan.py":
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


# --------------------------------------------------------------------------- #
# Collecting what is on screen
# --------------------------------------------------------------------------- #
def _strings(node: Any) -> Iterator[str]:
    seen: set[str] = set()
    for attr in ("value", "label", "help", "placeholder", "body"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            yield value
        elif isinstance(value, pd.DataFrame):
            yield from (str(c) for c in value.columns)
            yield from (str(v) for v in value.astype(str).to_numpy().ravel())
    for option in getattr(node, "options", None) or []:
        yield str(option)


def _walk(node: Any, *, default_view: bool) -> Iterator[str]:
    """Every string under ``node``. With ``default_view`` the technical expanders
    (glossary.TECHNICAL_EXPANDERS) are skipped: that is the jargon ban's scope."""
    technical = getattr(node, "label", "") in gl.TECHNICAL_EXPANDERS
    if default_view and type(node).__name__ == "Expander" and technical:
        return
    yield from _strings(node)
    for child in getattr(node, "children", {}).values():
        yield from _walk(child, default_view=default_view)


def screen(at: Any, *, default_view: bool = True) -> list[str]:
    return list(_walk(at._tree, default_view=default_view))


def assert_plain(at: Any, where: str) -> None:
    for text in screen(at):
        found = gl.jargon_in(text)
        assert not found, f"{where}: banned {found} in {text[:160]!r}"


def _choose(at: Any, card_id: str) -> Any:
    return at.button(key=f"card_{card_id}").click().run()


def _find(at: Any) -> Any:
    return at.button(key="find").click().run()


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_in_demo_mode(page: str) -> None:
    at = _app(page)
    assert any("Demo mode" in str(i.value) for i in at.sidebar.info)


# --------------------------------------------------------------------------- #
# The jargon ban on the default view
# --------------------------------------------------------------------------- #
def test_the_walker_finds_a_planted_banned_word_and_respects_technical_expanders() -> None:
    def page() -> None:
        import streamlit as st

        st.markdown("Plain words, then the surrogate.")
        with st.expander("Show technical details"):
            st.markdown("verified on twin (30 reps), epistemic std")

    at = AppTest.from_function(page).run()
    found = [w for text in screen(at) for w in gl.jargon_in(text)]
    assert found == ["surrogate"]
    everything = [w for text in screen(at, default_view=False) for w in gl.jargon_in(text)]
    assert {"twin", "verified", "epistemic"} <= set(everything)


@pytest.mark.parametrize("page", PAGES)
def test_no_page_uses_jargon_in_its_default_view(page: str) -> None:
    assert_plain(_app(page), page)


@pytest.mark.parametrize("card_id", [*CARDS, "own_words"])
def test_each_goal_form_and_its_result_are_plain(card_id: str) -> None:
    at = _choose(_app(), card_id)
    assert_plain(at, f"form {card_id}")
    if card_id == "own_words":
        at.text_area(key="own_words").input("Make more profit with few breakdowns").run()
    at = _find(at)
    assert not at.exception
    assert_plain(at, f"result {card_id}")


@pytest.mark.parametrize("goal_id", ["G07", "G01", "I01", "N01", "AA1", "G09", "G03", "G10"])
def test_every_saved_example_result_is_plain(goal_id: str) -> None:
    at = _app()
    box = at.selectbox(key="saved_pick")
    entry = next(e for e in sv.load_demo_cache() if e.goal_id == goal_id)
    from twinmind.ui.goals import goal_text

    box.set_value(goal_text(entry.result.spec or {})).run()
    at = at.button(key="saved_show").click().run()
    assert not at.exception
    assert_plain(at, goal_id)


def test_try_changes_is_plain_from_every_starting_point() -> None:
    at = _app("views/try_changes.py")
    for option in at.radio(key="tc_start").options:
        at.radio(key="tc_start").set_value(option).run()
        assert not at.exception
        assert_plain(at, f"try changes from {option}")


def test_the_progress_view_and_running_banner_are_plain(fresh_runner: JobRunner) -> None:
    gate = threading.Event()
    fresh_runner.start(
        "Make more profit.",
        lambda progress: (progress("explorer", 0, 6), gate.wait(10)),
        meta={"card": True},
    )
    try:
        at = _app()
        assert any("A plan is being worked out" in str(i.value) for i in at.info)
        assert_plain(at, "running banner")
    finally:
        gate.set()


# --------------------------------------------------------------------------- #
# Demo mode must not look interactive
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("card_id", list(CARDS))
def test_demo_mode_shows_the_banner_and_greys_out_the_limits(card_id: str) -> None:
    at = _choose(_app(), card_id)
    assert str(at.warning[0].value) == pl.error_message("demo")
    assert at.checkbox and all(c.disabled for c in at.checkbox)
    assert at.number_input and all(n.disabled for n in at.number_input)
    assert all(r.disabled for r in at.radio)
    assert any("Not used in demo mode" in str(c.value) for c in at.caption)
    assert at.button(key="find").label == "Show a saved example"


@pytest.mark.parametrize("card_id", list(CARDS))
def test_every_card_gives_a_labelled_saved_example(card_id: str, demo_env: Path) -> None:
    at = _find(_choose(_app(), card_id))
    warnings = [str(w.value) for w in at.warning]
    assert f"**{sv.REPLAY_LABEL}.**" in warnings
    entry = next(e for e in sv.load_demo_cache() if e.id == CARDS[card_id].demo_id)
    related = any("related goal" in str(i.value) for i in at.info)
    assert related == is_related_only(card_id, entry.result.spec or {})
    (item,) = sv.read_history(demo_env)
    assert item.kind == "replayed" and item.label == "Saved example, not a live run"


# --------------------------------------------------------------------------- #
# Prompt 10 honesty rules, in the new wording
# --------------------------------------------------------------------------- #
def _saved(at: Any, goal_id: str) -> Any:
    from twinmind.ui.goals import goal_text

    entry = next(e for e in sv.load_demo_cache() if e.goal_id == goal_id)
    at.selectbox(key="saved_pick").set_value(goal_text(entry.result.spec or {})).run()
    return at.button(key="saved_show").click().run()


@pytest.mark.parametrize("goal_id", ["G07", "G01", "I01", "N01", "AA1", "G09", "G03", "G10"])
def test_no_forbidden_word_anywhere_and_never_verified_for_explanations(goal_id: str) -> None:
    at = _saved(_app(), goal_id)
    for text in screen(at, default_view=False):
        vm.assert_no_overclaim(text)
    captions = [str(c.value) for c in at.caption if vm.EXPLANATION_CHECK in str(c.value)]
    assert captions and all("verif" not in c.lower() for c in captions)
    (ai,) = [e for e in at.expander if e.label == "AI-written explanation (technical wording)"]
    gaps = " ".join(str(m.value) for m in ai.markdown)
    assert "Negation" in gaps and "plan changes" in gaps and "KPI swaps" in gaps


def test_checked_numbers_are_labelled_with_their_run_count_and_ranges() -> None:
    at = _saved(_app(), "G01")
    assert any("Checked with the detailed simulation (30 runs)" in str(c.value) for c in at.caption)
    helps = [str(m.help) for m in at.markdown if getattr(m, "help", "")]
    assert any("likely between" in h for h in helps)
    assert not any("Quick estimate" in t for t in screen(at)), "never mixed with estimates"


def test_estimates_never_carry_a_range_and_checked_numbers_always_do() -> None:
    now = {k: 1.0 for k in vm.KPI_DISPLAY}
    new = {k: 2.0 for k in vm.KPI_DISPLAY}
    for card in pl.estimate_cards(now, new):
        assert "likely between" not in card.help and card.line.endswith("(rough)")
    for card in pl.kpi_cards(sv.load_demo_cache()[0].result):
        assert "likely between" in card.help and "(rough)" not in card.line


def test_side_effects_and_the_weak_kpi_warning_render_in_plain_words() -> None:
    at = _saved(_app(), "G09")
    text = " ".join(screen(at))
    assert "Profit gets worse: £344,569 → −£243,182" in text
    assert gl.ROUGH_KPI_NOTE.split(":")[0] in text or "less reliable in this model" in text
    technical = " ".join(screen(at, default_view=False))
    assert vm.WEAK_KPI_BADGE in technical, "the exact badge survives in Technical details"
    assert "verified on twin (30 reps)" in technical


def test_the_weak_kpi_warning_appears_whenever_breakdowns_are_shown() -> None:
    at = _saved(_app(), "G07")
    text = " ".join(screen(at))
    assert "Breakdowns" in text and "Rough guide: the model is less accurate here." in text


def test_the_couldnt_find_wording_is_used_for_no_plan_found() -> None:
    at = _saved(_app(), "I01")
    text = " ".join(screen(at))
    assert "We couldn't find a plan that meets these limits. Closest options below." in text
    assert "can't be met" not in text


def test_the_reliability_page_shows_five_equal_cards_with_both_failures() -> None:
    at = _app("views/reliability.py")
    badges = [str(m.value) for m in at.markdown if "-badge[" in str(m.value)]
    assert len(badges) == 5
    assert sum("Failed" in b for b in badges) == 2 and sum("Passed" in b for b in badges) == 3
    exact = [e for e in at.expander if e.label == "Show the exact test"]
    assert len(exact) == 5
    text = " ".join(screen(at))
    assert "the improved version has not been retested" in text
    assert "3 of 5 tests passed and 2 failed" in text
    technical = " ".join(screen(at, default_view=False))
    assert sv.B5_NOTE in technical
    assert all(c in technical for c in sv.CHECKER_CAVEATS)


def test_past_results_label_live_runs_and_saved_examples_apart(demo_env: Path) -> None:
    entry = sv.load_demo_cache()[0]
    sv.append_history(demo_env, "replayed", "a saved goal", "ok", "", entry.result)
    sv.append_history(demo_env, "started", "a live goal", "running", run_id="r1")
    sv.append_history(demo_env, "live", "a live goal", "ok", "", entry.result, run_id="r1")
    sv.append_history(demo_env, "started", "a lost goal", "running", run_id="r2")
    at = _app("views/past_results.py")
    frame = at.dataframe[0].value
    assert list(frame["Type"]) == [
        "🟢 Live",
        "🟢 Live",
        "📁 Saved example, not a live run",
    ]
    assert list(frame["Outcome"])[0] == "⏹ Stopped before finishing"
    assert_plain(at, "past results")


# --------------------------------------------------------------------------- #
# Live mode with fakes: background run, banner, cancel, clicks elsewhere
# --------------------------------------------------------------------------- #
@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sv, "detect_mode", lambda *a, **k: sv.Mode(live=True, twin_available=False, reasons=())
    )


def test_a_finished_live_card_run_shows_the_plain_result(
    live: None, monkeypatch: pytest.MonkeyPatch, demo_env: Path, fresh_runner: JobRunner
) -> None:
    from twinmind.agents import graph
    from twinmind.ui import components

    sync = JobRunner(start_thread=lambda target: target())
    monkeypatch.setattr(components, "job_runner", lambda: sync)
    monkeypatch.setattr(components, "live_deps", lambda: object())
    saved = sv.load_demo_cache()[1].result  # G01: profit with a breakdown limit

    def fake(spec: Any, deps: Any, seed: int, on_step: Any, **kw: Any) -> AgentResult:
        for node in ("explorer", "evaluator", "critic", "explainer"):
            on_step(node, 1, 6)
        return saved

    monkeypatch.setattr(graph, "run_on_spec_streaming", fake)
    at = _choose(_app(), "profit")
    assert at.button(key="find").label == "Find a plan"
    at = _find(at).run()
    assert not at.exception
    assert not any(sv.REPLAY_LABEL in str(w.value) for w in at.warning)
    assert any("Recommended plan" in t for t in screen(at))
    items = sv.read_history(demo_env)
    assert len(items) == 1 and items[0].kind == "live" and items[0].label == "Live"


def test_a_live_run_survives_clicks_elsewhere_and_cancel_stops_it(
    live: None, monkeypatch: pytest.MonkeyPatch, demo_env: Path, fresh_runner: JobRunner
) -> None:
    from twinmind.agents import graph
    from twinmind.ui import components

    monkeypatch.setattr(components, "live_deps", lambda: object())
    gate, finished = threading.Event(), threading.Event()

    def fake(spec: Any, deps: Any, seed: int, on_step: Any, **kw: Any) -> AgentResult:
        on_step("explorer", 0, 6)
        gate.wait(10)
        on_step("evaluator", 0, 6)  # raises after Cancel
        return sv.load_demo_cache()[0].result

    monkeypatch.setattr(graph, "run_on_spec_streaming", fake)
    at = _find(_choose(_app(), "unit_cost"))
    assert fresh_runner.busy
    # Click around: dismiss the welcome, open other pages. The run keeps going.
    at.button(key="welcome_ok").click().run()
    at.switch_page("views/reliability.py").run()
    at.switch_page("views/past_results.py").run()
    assert fresh_runner.status().state == "running"
    assert any("A plan is being worked out" in str(i.value) for i in at.info)
    # A second press while running starts nothing.
    at.switch_page("views/plan.py").run()
    assert at.button(key="find").disabled
    # Cancel from the banner.
    at.button(key="cancel_run").click().run()
    assert fresh_runner.status().state == "cancelling"
    gate.set()
    for _ in range(50):
        if not fresh_runner.busy:
            break
        finished.wait(0.1)
    assert fresh_runner.status().state == "cancelled"
    (item,) = sv.read_history(demo_env)
    assert item.kind == "aborted" and item.status == "cancelled"


def test_an_unreachable_ai_service_explains_itself_and_shows_a_labelled_example(
    live: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from twinmind.ui import components

    def no_ai() -> Any:
        raise ConnectionError("network down; api_key=gsk_abcdef123")

    monkeypatch.setattr(components, "live_deps", no_ai)
    at = _find(_choose(_app(), "profit"))
    assert any(pl.error_message("ai_unreachable") in str(e.value) for e in at.error)
    assert any(sv.REPLAY_LABEL in str(w.value) for w in at.warning)
    assert not any("gsk_" in t for t in screen(at, default_view=False))
