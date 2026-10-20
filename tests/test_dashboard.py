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
FIXTURE_V1_EVIDENCE = PROJECT_ROOT / "tests" / "fixtures" / "v1_0_evidence_page.py"
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
    if page == "views/reliability.py":
        # The examiner's record below the summary keeps its technical terms.
        at = _app(page)
        for text in [*_walk(summary_block(at), default_view=True), *sidebar_text(at)]:
            assert not gl.jargon_in(text), (text[:160], gl.jargon_in(text))
        return
    assert_plain(_app(page), page)


def summary_block(at: Any) -> Any:
    """The reliability page's plain summary (a container keyed reliability_summary)."""

    def find(node: Any) -> Any:
        if str(getattr(getattr(node, "proto", None), "id", "")).endswith("reliability_summary"):
            return node
        for child in getattr(node, "children", {}).values():
            found = find(child)
            if found is not None:
                return found
        return None

    block = find(at._tree)
    assert block is not None, "the plain summary container is missing"
    return block


def sidebar_text(at: Any) -> list[str]:
    return list(_walk(at.sidebar, default_view=True))


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
    assert str(at.info[0].value) == pl.error_message("demo"), "the banner comes first"
    assert at.checkbox and all(c.disabled for c in at.checkbox)
    assert at.number_input and all(n.disabled for n in at.number_input)
    assert all(r.disabled for r in at.radio)
    assert any("Not used in demo mode" in str(c.value) for c in at.caption)
    assert at.button(key="find").label == "Show a saved example"


@pytest.mark.parametrize("card_id", list(CARDS))
def test_every_card_gives_a_labelled_saved_example(card_id: str, demo_env: Path) -> None:
    at = _find(_choose(_app(), card_id))
    labels = [str(i.value) for i in at.info]
    assert f"**{sv.REPLAY_LABEL}.**" in labels
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
    assert any(vm.EXPLANATION_CHECK in str(c.value) for c in ai.caption)
    visible = [str(c.value) for c in at.caption if vm.EXPLANATION_CHECK in str(c.value)]
    assert any(c.startswith("The AI-written explanation below") for c in visible), (
        "the check caption is visible beside the explanation, as in v1.0"
    )
    (limits,) = [e for e in at.expander if e.label == "What this check does not catch"]
    gaps = " ".join(str(m.value) for m in limits.markdown)
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
    assert "Profit changes for the worse: £344,569 → −£243,182" in text
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


def test_the_reliability_page_shows_both_failures_plainly_with_the_caveats_visible() -> None:
    at = _app("views/reliability.py")
    visible = " ".join(screen(at))  # default view: nothing behind an expander
    table = next(str(m.value) for m in at.markdown if "Pre-registered claim" in str(m.value))
    lines = table.splitlines()[2:]
    assert len(lines) == 5
    results = [line.split("|")[3].strip() for line in lines]
    registered = [line.split("|")[4].strip() for line in lines]
    assert results == [
        ":green[✓] **Met**",
        ":gray[○] **Not met**",
        ":green[✓] **Met**",
        ":green[✓] **Met**",
        ":gray[○] **Not met**",
    ], "Met and Not met rows are formatted identically, with no red"
    assert registered == ["PASS", "FAIL", "PASS", "PASS", "FAIL"]
    assert "3 of 5 met (PASS); 2 not met (FAIL)." in visible
    assert "**B2 · not met (FAIL).** *What it means:*" in visible
    assert "**B5 · not met (FAIL).** *What it means:*" in visible
    assert visible.count("*What it does not mean:*") == 2
    assert sv.B5_NOTE in visible
    assert all(c in visible for c in sv.CHECKER_CAVEATS)
    summary = " ".join(_walk(summary_block(at), default_view=True))
    assert "3 of 5 tests were met and 2 were not" in summary
    assert summary.index("What works well") < summary.index("Where it needs care")
    assert summary.count("**Not met**") == 2 and summary.count("**Met**") == 3
    assert "misses some mistakes" in summary, "the checker limits sit under Where it needs care"
    assert ":red" not in summary and "❌" not in summary
    assert "the improved version has not been retested" in summary
    collapsed = [e.label for e in at.main.get("expander")]  # the sidebar is not the page
    assert not collapsed, "nothing on this page is collapsed"
    glossary = [
        b for b in at.get("popover") if b.proto.popover.label == "Glossary of technical terms"
    ]
    assert glossary, "one-line link to the glossary of technical terms"
    terms = " ".join(_walk(glossary[0], default_view=False))
    assert "Robustly feasible" in terms and "Explanation checker v1 / v2" in terms


def test_the_reliability_page_drops_nothing_the_v1_evidence_page_showed() -> None:
    """Render the v1.0-dashboard Evidence page (kept verbatim in tests/fixtures)
    and the new page; every B1-B5 row, figure, model card, limitation and caveat
    of the old page must be on the new one, and visible."""
    old = AppTest.from_file(str(FIXTURE_V1_EVIDENCE), default_timeout=120).run()
    new = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / "reliability.py"),
                            default_timeout=120).run()  # fmt: skip
    assert not old.exception and not new.exception

    def b_rows(at: Any) -> set[tuple[str, str, str, str]]:
        table = next(str(m.value) for m in at.markdown if "Pre-registered claim" in str(m.value))
        out = set()
        for line in table.splitlines()[2:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            # v1.0: "**PASS**" in Verdict; now: the registered verdict has its own column.
            verdict = (cells[3] if len(cells) == 6 else cells[2]).replace("*", "").strip()
            out.add((cells[0], cells[1], verdict, cells[-1]))
        return out

    assert b_rows(old) == b_rows(new)
    assert len(b_rows(new)) == 5

    def options(at: Any, label: str) -> list[str]:
        return [str(o) for o in at.selectbox(key=label).options]

    assert options(old, "ev_fig") == options(new, "ev_fig"), "figures"
    assert options(old, "ev_card") == options(new, "ev_card"), "model cards"
    assert old.selectbox(key="ev_card").value == new.selectbox(key="ev_card").value

    old_text = " ".join(screen(old, default_view=False))
    new_visible = " ".join(screen(new))
    must_keep = [
        sv.B5_NOTE,
        *sv.CHECKER_CAVEATS,
        ("3 of 5 passed; 2 failed.", "3 of 5 met (PASS); 2 not met (FAIL)."),
        "gaps are lower bounds",
        "Limitations: where results underperformed the spec",
        "Neither checker catches negation",
        *(title for title, _ in sv.underperformed_sections()),
        *(body[:200] for _, body in sv.underperformed_sections()),
    ]
    for item in must_keep:
        # A pair is (v1.0 wording, new wording) for the same content: wording only.
        old_item, new_item = item if isinstance(item, tuple) else (item, item)
        assert old_item in old_text, f"fixture check: {old_item[:60]!r}"
        assert new_item in new_visible, f"dropped or hidden: {new_item[:60]!r}"
    old_frames = [d.value.to_dict() for d in old.dataframe]
    new_frames = [d.value.to_dict() for d in new.dataframe]
    assert old_frames == new_frames, "the post-benchmark checker table"
    assert len(new.get("imgs")) >= len(old.get("imgs")), "images (cards and figure)"


# --------------------------------------------------------------------------- #
# Honesty rules carried over from v1.0 (restored after the Stage B rewrite)
# --------------------------------------------------------------------------- #
def test_the_table_renderer_refuses_mixed_labels() -> None:
    from twinmind.ui import components

    result = sv.load_demo_cache()[0].result
    mixed = [*vm.kpi_table(result)[:4], *vm.estimate_table({k: 1.0 for k in vm.KPI_DISPLAY})[4:]]
    with pytest.raises(ValueError, match="mixes source labels"):
        components.kpi_table(mixed, with_baseline=True)


def test_no_app_source_cached_text_or_model_card_uses_a_forbidden_word() -> None:
    files = [
        *sorted((PROJECT_ROOT / "app").rglob("*.py")),
        *sorted((PROJECT_ROOT / "app" / "demo_cache").glob("*.json")),
        *sorted((PROJECT_ROOT / "reports" / "model_cards").glob("*.md")),
    ]
    for path in files:
        vm.assert_no_overclaim(path.read_text(encoding="utf-8"))


def test_model_card_f_states_the_checker_caveats() -> None:
    card = (PROJECT_ROOT / "reports" / "model_cards" / "f_agent_benchmark.md").read_text(
        encoding="utf-8"
    )
    assert "Development results, not validation" in card
    assert "neither a lower nor" in card and "verdict is unchanged: FAIL, 35/40" in card


@pytest.mark.parametrize("goal_id", ["G07", "G01", "I01", "N01", "AA1", "G09", "G03", "G10"])
def test_technical_tables_never_mix_labels_and_badge_only_the_weak_kpis(goal_id: str) -> None:
    at = _saved(_app(), goal_id)
    technical = [d.value for d in at.dataframe if "Source" in d.value.columns]
    assert technical, "the technical KPI table is rendered"
    for frame in technical:
        assert set(frame["Source"]) == {"verified on twin (30 reps)"}
        noted = set(frame.loc[frame["Note"] != "", "KPI"])
        assert noted == {"Downtime", "Unplanned failures"}
        assert all(vm.WEAK_KPI_BADGE in n for n in frame.loc[frame["Note"] != "", "Note"])


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
    assert list(frame["Outcome"])[0] == (
        "⏹ Stopped: the app was closed or restarted before it finished"
    )
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
    assert not any(sv.REPLAY_LABEL in t for t in screen(at))
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
    assert any(pl.error_message("ai_unreachable") in str(i.value) for i in at.info)
    assert not at.error, "calm styling: no red error box"
    assert any(sv.REPLAY_LABEL in str(i.value) for i in at.info)
    assert not any("gsk_" in t for t in screen(at, default_view=False))


# --------------------------------------------------------------------------- #
# v1.0 guarantees the Stage B rewrite lost, restored on the new pages
# --------------------------------------------------------------------------- #
def _every_text(at: Any) -> list[str]:
    texts = screen(at, default_view=False)
    for box in at.selectbox:
        texts += [box.label, *map(str, box.options)]
    return texts


@pytest.mark.parametrize("page", PAGES)
def test_demo_mode_says_why_without_showing_any_secret(page: str) -> None:
    at = _app(page)
    (tech,) = [e for e in at.sidebar.expander if e.label == "Show technical details"]
    assert "TWINMIND_DEMO=1 is set" in " ".join(str(m.value) for m in tech.markdown)
    for text in _every_text(at):
        assert "gsk_" not in text and "GROQ_API_KEY=" not in text, text[:80]
        assert "C:\\" not in text and "/Users/" not in text, text[:80]


@pytest.mark.parametrize("goal_id", ["G07", "G01", "I01", "N01", "AA1", "G09", "G03", "G10"])
def test_every_saved_example_is_labelled_and_says_where_it_came_from(goal_id: str) -> None:
    entry = next(e for e in sv.load_demo_cache() if e.goal_id == goal_id)
    at = _saved(_app(), goal_id)
    assert f"**{sv.REPLAY_LABEL}.**" in [str(i.value) for i in at.info]
    assert pl.saved_from(entry.goal_id, entry.seed) in [str(c.value) for c in at.caption]


def test_a_typed_goal_in_demo_mode_shows_a_labelled_saved_example_and_is_logged(
    demo_env: Path,
) -> None:
    at = _choose(_app(), "own_words")
    at.text_area(key="own_words").input("Maximise OEE with few unplanned failures").run()
    at = _find(at)
    assert f"**{sv.REPLAY_LABEL}.**" in [str(i.value) for i in at.info]
    assert any("You asked" in str(c.value) for c in at.caption)
    (item,) = sv.read_history(demo_env)
    assert item.kind == "replayed" and item.label == "Saved example, not a live run"


def test_no_page_renders_a_forbidden_word_anywhere() -> None:
    at = _app()
    texts = _every_text(at)
    for goal_id in ["G07", "G01", "I01", "N01", "AA1", "G09", "G03", "G10"]:
        texts += _every_text(_saved(at, goal_id))
    at.switch_page("views/try_changes.py").run()
    for option in at.radio(key="tc_start").options:
        at.radio(key="tc_start").set_value(option).run()
        texts += _every_text(at)
    at.switch_page("views/reliability.py").run()
    for card in at.selectbox(key="ev_card").options:
        at.selectbox(key="ev_card").set_value(card).run()
        texts += _every_text(at)
    at.switch_page("views/past_results.py").run()
    texts += _every_text(at)
    assert not at.exception
    for text in texts:
        vm.assert_no_overclaim(text)


def test_g07_renders_only_its_oee_side_effect() -> None:
    at = _saved(_app(), "G07")
    effects = [t for t in screen(at) if "changes for the worse" in t]
    assert len(effects) == 1 and effects[0].count("changes for the worse") == 1
    assert "Equipment efficiency (OEE) changes for the worse: 73.3% → 70.6%" in effects[0]


def test_an_unreachable_ai_service_is_logged_without_any_key(
    live: None, monkeypatch: pytest.MonkeyPatch, demo_env: Path
) -> None:
    from twinmind.ui import components

    def no_ai() -> Any:
        raise ConnectionError("network down; api_key=gsk_abcdef123")

    monkeypatch.setattr(components, "live_deps", no_ai)
    _find(_choose(_app(), "profit"))
    (item,) = sv.read_history(demo_env)
    assert item.kind == "replayed" and "could not be reached" in item.detail
    assert "gsk_" not in demo_env.read_text(encoding="utf-8")


def test_a_crashed_live_run_is_shown_as_stopped_calmly_and_recorded(
    live: None, monkeypatch: pytest.MonkeyPatch, demo_env: Path
) -> None:
    from twinmind.agents import graph
    from twinmind.ui import components

    sync = JobRunner(start_thread=lambda target: target())
    monkeypatch.setattr(components, "job_runner", lambda: sync)
    monkeypatch.setattr(components, "live_deps", lambda: object())

    def crash(*args: Any, **kwargs: Any) -> AgentResult:
        raise RuntimeError("twin exploded")

    monkeypatch.setattr(graph, "run_on_spec_streaming", crash)
    at = _find(_choose(_app(), "profit")).run()
    assert not at.error, "calm styling: a stopped run is not a red error"
    assert any(str(i.value).startswith("**Stopped:**") for i in at.info)
    (item,) = sv.read_history(demo_env)
    assert item.kind == "aborted" and "twin exploded" in item.detail
    assert not sync.busy, "the one-run lock is released"


def test_live_progress_shows_the_ai_calls_used_against_the_cap(
    live: None, monkeypatch: pytest.MonkeyPatch, fresh_runner: JobRunner
) -> None:
    from twinmind.agents import graph
    from twinmind.ui import components

    monkeypatch.setattr(components, "live_deps", lambda: object())
    gate = threading.Event()

    def fake(spec: Any, deps: Any, seed: int, on_step: Any, **kw: Any) -> AgentResult:
        on_step("explorer", 1, 6)
        gate.wait(10)
        return sv.load_demo_cache()[0].result

    monkeypatch.setattr(graph, "run_on_spec_streaming", fake)
    try:
        at = _find(_choose(_app(), "profit"))
        for _ in range(40):
            if fresh_runner.status().nodes:
                break
            threading.Event().wait(0.05)
        at.run()
        assert any("AI calls used: 1 of 6 allowed." in str(c.value) for c in at.caption)
    finally:
        gate.set()


# --------------------------------------------------------------------------- #
# Calm tone: no red or error styling for normal outcomes
# --------------------------------------------------------------------------- #
def _no_red(at: Any, where: str) -> None:
    assert not at.error, f"{where}: st.error used for a normal outcome"
    for text in screen(at, default_view=False):
        assert ":red[" not in text and ":red-badge[" not in text, (where, text[:80])
        assert "❌" not in text and "⛔" not in text, (where, text[:80])


@pytest.mark.parametrize(
    ("goal_id", "chip"),
    [
        ("G07", "Recommended plan"),
        ("N01", "Your current setup is already close to the best we could find"),
        ("I01", "We couldn't find a plan that meets these limits. Closest options below."),
        ("G09", "Recommended plan"),
    ],
)
def test_normal_outcomes_are_calm(goal_id: str, chip: str) -> None:
    at = _saved(_app(), goal_id)
    _no_red(at, goal_id)
    badges = [str(m.value) for m in at.markdown if "-badge[" in str(m.value)]
    assert any(chip in b for b in badges)
    explanation = next(e for e in sv.load_demo_cache() if e.goal_id == goal_id).result.explanation
    for text in screen(at):  # the default view, outside the AI's own words
        if text != explanation:
            assert "!" not in text.replace("![", ""), (goal_id, text[:80])
    if goal_id == "I01":
        text = " ".join(screen(at))
        assert text.index("Closest options") < text.index("Before you act")


def test_stopped_runs_and_the_reliability_page_are_calm(demo_env: Path) -> None:
    sv.append_history(demo_env, "started", "a lost goal", "running", run_id="r1")
    sv.append_history(demo_env, "aborted", "a stopped goal", "cancelled", "cancelled", run_id="r2")
    at = _app("views/past_results.py")
    _no_red(at, "past results")
    outcomes = list(at.dataframe[0].value["Outcome"])
    assert outcomes == [
        "⏹ Stopped: you cancelled it",
        "⏹ Stopped: the app was closed or restarted before it finished",
    ]
    at = _app("views/reliability.py")
    _no_red(at, "reliability")
    visible = " ".join(screen(at))
    assert visible.count("**Not met**") >= 2 and "B2 · not met (FAIL)" in visible
    assert "B5 · not met (FAIL)" in visible
    assert all(c in visible for c in sv.CHECKER_CAVEATS)
