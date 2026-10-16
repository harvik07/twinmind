"""The dashboard, offline: AppTest smoke tests of every page in demo mode, the
honesty rules on what is RENDERED (labels, badge, forbidden words), the live
path's fallback/abort/finish handling with fakes, and the services.

No network, no Groq key, no data/ or models/ files: TWINMIND_DEMO=1 forces
demo mode, as on CI, and the history log goes to a temp file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from twinmind.agents.contracts import AgentResult
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm
from twinmind.utils.paths import PROJECT_ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
MAIN = str(PROJECT_ROOT / "app" / "main.py")
PAGES = ["views/ask.py", "views/whatif.py", "views/evidence.py", "views/history.py"]


@pytest.fixture(autouse=True)
def demo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "dashboard_runs.jsonl"
    monkeypatch.setenv("TWINMIND_DEMO", "1")
    monkeypatch.setenv("TWINMIND_DASHBOARD_LOG", str(log))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return log


def _app() -> Any:
    return AppTest.from_file(MAIN, default_timeout=120).run()


def _texts(at: Any) -> list[str]:
    """Every string the page put on screen."""
    out: list[str] = []
    for kind in ("title", "subheader", "markdown", "caption", "warning", "info", "error"):
        out += [str(e.value) for e in getattr(at, kind)]
    for table in [*at.table, *at.dataframe]:
        frame = table.value
        out += [str(c) for c in frame.columns]
        out += [str(v) for v in frame.astype(str).to_numpy().ravel()]
    out += [b.label for b in at.button]
    for box in at.selectbox:
        out += [box.label, *map(str, box.options)]
    return out


def _replay(at: Any, goal_id: str) -> Any:
    box = at.selectbox(key="ask_demo")
    label = next(o for o in box.options if o.startswith(goal_id))
    box.set_value(label).run()
    return next(b for b in at.button if b.label == "Show this replay").click().run()


# --------------------------------------------------------------------------- #
# Smoke: every page renders in demo mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_in_demo_mode(page: str) -> None:
    at = _app()
    if page != "views/ask.py":
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Demo mode" in str(i.value) for i in at.sidebar.info)


def test_demo_mode_says_why_without_showing_any_secret() -> None:
    at = _app()
    text = " ".join(_texts(at)) + " ".join(str(c.value) for c in at.sidebar.caption)
    assert "TWINMIND_DEMO=1 is set" in text
    assert "gsk_" not in text and "C:\\" not in text and "/Users/" not in text


# --------------------------------------------------------------------------- #
# Replays are labelled, everywhere
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("goal_id", ["G07", "G01", "I01", "N01", "AA1", "G09"])
def test_every_replay_carries_the_visible_replay_label(goal_id: str) -> None:
    at = _replay(_app(), goal_id)
    assert not at.exception
    warnings = [str(w.value) for w in at.warning]
    assert any(sv.REPLAY_LABEL.capitalize() in w for w in warnings), warnings


def test_the_closest_replay_for_a_typed_goal_is_labelled_and_logged(demo_env: Path) -> None:
    at = _app()
    at.text_area(key="ask_goal").input("Maximise OEE with few unplanned failures").run()
    at.button[0].click().run()
    assert any(sv.REPLAY_LABEL.capitalize() in str(w.value) for w in at.warning)
    assert any("You asked" in str(c.value) for c in at.caption)
    (item,) = sv.read_history(demo_env)
    assert item.kind == "replayed" and item.label == sv.REPLAY_LABEL


def test_history_labels_replays_and_live_runs_differently(demo_env: Path) -> None:
    entry = sv.load_demo_cache()[0]
    sv.append_history(demo_env, "replayed", entry.goal_text, "ok", "", entry.result)
    sv.append_history(demo_env, "live", "a live goal", "ok", "", entry.result)
    sv.append_history(demo_env, "aborted", "a stopped goal", "aborted", "RuntimeError: x")
    at = _app()
    at.switch_page("views/history.py").run()
    assert not at.exception
    kinds = list(at.dataframe[0].value["Kind"])
    assert kinds == [
        "⛔ aborted, no result",
        "🟢 live run",
        f"🔁 {sv.REPLAY_LABEL}",
    ]


# --------------------------------------------------------------------------- #
# What is rendered: badge, labels, forbidden words
# --------------------------------------------------------------------------- #
def test_the_weak_kpi_badge_is_rendered_on_both_weak_kpis_only() -> None:
    at = _replay(_app(), "G09")
    frame = at.dataframe[0].value
    noted = set(frame.loc[frame["Note"] != "", "KPI"])
    assert noted == {"Downtime", "Unplanned failures"}
    assert all(vm.WEAK_KPI_BADGE in n for n in frame.loc[frame["Note"] != "", "Note"])
    assert any(vm.WEAK_KPI_BADGE in str(c.value) for c in at.caption)


@pytest.mark.parametrize("goal_id", ["G07", "I01"])
def test_no_rendered_table_mixes_estimate_and_verified(goal_id: str) -> None:
    at = _replay(_app(), goal_id)
    for table in at.dataframe:
        frame = table.value
        if "Source" in frame:
            labels = set(frame["Source"])
            assert len(labels) == 1, labels
            assert labels == {"verified on twin (30 reps)"}


def test_the_table_renderer_refuses_mixed_labels() -> None:
    from twinmind.ui import components

    result = sv.load_demo_cache()[0].result
    mixed = [*vm.kpi_table(result)[:4], *vm.estimate_table({k: 1.0 for k in vm.KPI_DISPLAY})[4:]]
    with pytest.raises(ValueError, match="mixes source labels"):
        components.kpi_table(mixed, with_baseline=True)


def test_no_page_renders_a_forbidden_word() -> None:
    at = _app()
    texts = _texts(at)
    for goal_id in ["G07", "G01", "I01", "N01", "AA1", "G09"]:
        texts += _texts(_replay(at, goal_id))
    at.switch_page("views/whatif.py").run()
    for option in at.selectbox(key="wi_preset").options:
        at.selectbox(key="wi_preset").set_value(option).run()
        at.button[0].click().run()
        texts += _texts(at)
    at.switch_page("views/evidence.py").run()
    for card in at.selectbox(key="ev_card").options:
        at.selectbox(key="ev_card").set_value(card).run()
        texts += _texts(at)
    at.switch_page("views/history.py").run()
    texts += _texts(at)
    assert not at.exception
    for text in texts:
        vm.assert_no_overclaim(text)


def test_no_app_source_or_cached_text_uses_a_forbidden_word() -> None:
    files = [
        *sorted((PROJECT_ROOT / "app").rglob("*.py")),
        *sorted((PROJECT_ROOT / "app" / "demo_cache").glob("*.json")),
        *sorted((PROJECT_ROOT / "reports" / "model_cards").glob("*.md")),
    ]
    for path in files:
        vm.assert_no_overclaim(path.read_text(encoding="utf-8"))


def test_the_evidence_page_shows_both_fails_and_the_b5_note() -> None:
    at = _app()
    at.switch_page("views/evidence.py").run()
    table = next(str(m.value) for m in at.markdown if "Pre-registered claim" in str(m.value))
    verdicts = [line.split("|")[3].strip() for line in table.splitlines()[2:]]
    assert verdicts == ["**PASS**", "**FAIL**", "**PASS**", "**PASS**", "**FAIL**"]
    assert any(sv.B5_NOTE in str(w.value) for w in at.warning)
    assert any("3 of 5 passed; 2 failed" in str(m.value) for m in at.markdown)


# --------------------------------------------------------------------------- #
# Live mode, with fakes: fallback, abort, finish -- always one status
# --------------------------------------------------------------------------- #
@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sv, "detect_mode", lambda *a, **k: sv.Mode(live=True, twin_available=False, reasons=())
    )


def _run_live(goal: str = "Maximise profit with few failures") -> Any:
    at = _app()
    at.text_area(key="ask_goal").input(goal).run()
    return next(b for b in at.button if b.label == "Run live").click().run()


def test_unreachable_llm_falls_back_to_a_labelled_replay(
    live: None, monkeypatch: pytest.MonkeyPatch, demo_env: Path
) -> None:
    from twinmind.ui import components

    def no_groq() -> Any:
        raise ConnectionError("network down; api_key=gsk_abcdef123")

    monkeypatch.setattr(components, "live_deps", no_groq)
    at = _run_live()
    assert any(sv.REPLAY_LABEL.capitalize() in str(w.value) for w in at.warning)
    (item,) = sv.read_history(demo_env)
    assert item.kind == "replayed" and "could not be reached" in item.detail
    assert "gsk_" not in demo_env.read_text(encoding="utf-8")


def test_a_crashed_live_run_is_recorded_as_aborted(
    live: None, monkeypatch: pytest.MonkeyPatch, demo_env: Path
) -> None:
    from twinmind.agents import graph
    from twinmind.ui import components

    monkeypatch.setattr(components, "live_deps", lambda: object())

    def crash(*args: Any, **kwargs: Any) -> AgentResult:
        raise RuntimeError("twin exploded")

    monkeypatch.setattr(graph, "run_agent_streaming", crash)
    at = _run_live()
    assert any("produced no result" in str(e.value) for e in at.error)
    (item,) = sv.read_history(demo_env)
    assert item.kind == "aborted" and "twin exploded" in item.detail
    assert not components.run_guard().busy, "the one-run lock must be released"


def test_a_finished_live_run_shows_progress_the_budget_and_no_replay_label(
    live: None, monkeypatch: pytest.MonkeyPatch, demo_env: Path
) -> None:
    from twinmind.agents import graph
    from twinmind.ui import components

    monkeypatch.setattr(components, "live_deps", lambda: object())
    replayed = sv.load_demo_cache()[0].result

    def fake_run(goal: str, ctx: int, deps: Any, on_step: Any) -> AgentResult:
        for used, node in enumerate(["planner", "explorer", "evaluator", "critic"]):
            on_step(node, min(used, 1), 6)
        on_step("explainer", 2, 6)
        return AgentResult.from_dict({**replayed.to_dict(), "goal_text": goal})

    monkeypatch.setattr(graph, "run_agent_streaming", fake_run)
    at = _run_live()
    assert not at.exception
    assert any("LLM calls: **2 / 6**" in str(m.value) for m in at.markdown)
    assert not any(sv.REPLAY_LABEL.capitalize() in str(w.value) for w in at.warning)
    (item,) = sv.read_history(demo_env)
    assert item.kind == "live" and item.label == "live run"


# --------------------------------------------------------------------------- #
# Services
# --------------------------------------------------------------------------- #
def test_detect_mode_without_files_or_key_is_demo(tmp_path: Path) -> None:
    mode = sv.detect_mode({}, root=tmp_path)
    assert not mode.live and not mode.twin_available
    assert any("GROQ_API_KEY" in r for r in mode.reasons)
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_secret_value\n", encoding="utf-8")
    mode = sv.detect_mode({}, root=tmp_path)
    assert not any("GROQ_API_KEY" in r for r in mode.reasons)
    assert "gsk_" not in repr(mode), "the mode never holds the key"


def test_safe_error_redacts_key_like_text() -> None:
    text = sv.safe_error(RuntimeError("bad key gsk_ABC123 and api_key=xyz"))
    assert "gsk_" not in text and "xyz" not in text and text.startswith("RuntimeError")


def test_closest_demo_and_llm_failure_detection() -> None:
    entries = sv.load_demo_cache()
    assert sv.closest_demo("Maximise OEE, at most 1.5 unplanned failures", entries).goal_id == "G09"
    failed = AgentResult(
        goal_text="x", context_id=8, status="spec_invalid", refusal="LLM unavailable: timeout"
    )
    assert sv.is_llm_failure(failed)
    assert not sv.is_llm_failure(entries[0].result)


def test_verify_on_twin_caps_replications_and_time() -> None:
    names = ["a", "b"]
    seeds = list(range(20))

    def clock() -> Any:
        """Each run takes one second (the clock is read after each run)."""
        return iter(range(1000)).__next__

    run = sv.verify_on_twin(lambda s: [float(s), 1.0], seeds, names, 50, 1e9, clock())
    assert run.n_reps == sv.WHATIF_MAX_REPS and not run.stopped_by_time_cap
    run = sv.verify_on_twin(lambda s: [float(s), 1.0], seeds, names, 10, 2.5, clock())
    assert run.n_reps == 3 and run.stopped_by_time_cap and run.seconds == 3.0
    assert run.lows["a"] < run.means["a"] < run.highs["a"]
    assert run.lows["b"] == run.highs["b"] == 1.0
    run = sv.verify_on_twin(lambda s: [0.0, 0.0], seeds, names, 1, 0.0, clock())
    assert run.n_reps == sv.WHATIF_MIN_REPS, "never fewer than two: no interval from one run"


def test_a_slow_twin_is_reported_not_failed() -> None:
    clock = iter([0.0, 0.9, 1.8]).__next__
    run = sv.verify_on_twin(lambda s: [1.0], [1, 2], ["a"], 2, 60.0, clock)
    assert run.n_reps == 2 and run.slower_than_quiet


def test_twin_table_is_verified_with_intervals_and_needs_two_reps() -> None:
    means = {k: 1.0 for k in vm.KPI_DISPLAY}
    rows = vm.twin_table(means, means, means, 5)
    assert {r.source_label for r in rows} == {"verified on twin (5 reps)"}
    assert all(r.value_ci for r in rows)
    with pytest.raises(ValueError):
        vm.twin_table(means, means, means, 1)


def test_sixteen_sliders_with_searchable_bounds() -> None:
    specs = sv.slider_specs()
    assert len(specs) == 16
    assert all(s.min <= s.default <= s.max and s.step > 0 for s in specs)


def test_benchmark_rows_keep_both_fails() -> None:
    analysis = json.loads(
        (sv.RESULTS_DIR / "agent_bench_analysis.json").read_text(encoding="utf-8")
    )
    rows = sv.benchmark_rows(analysis)
    assert [(r.id, r.verdict) for r in rows] == [
        ("B1", "PASS"),
        ("B2", "FAIL"),
        ("B3", "PASS"),
        ("B4", "PASS"),
        ("B5", "FAIL"),
    ]
    assert "35/40" in rows[4].deciding and "22/23" in rows[1].deciding


def test_limitations_come_from_the_underperformed_sections() -> None:
    sections = dict(sv.underperformed_sections())
    assert "B5 FAIL" in sections["Model F — the agent benchmark"]


def test_split_markdown_resolves_local_images(tmp_path: Path) -> None:
    (tmp_path / "fig.png").write_bytes(b"\x89PNG")
    parts = sv.split_markdown("Before ![a](fig.png) after ![gone](missing.png).", tmp_path)
    assert parts[0] == ("md", "Before ")
    assert parts[1] == ("img", str((tmp_path / "fig.png").resolve()))
    assert ("md", "*gone*") in parts


# --------------------------------------------------------------------------- #
# The streaming runner is run_agent plus progress
# --------------------------------------------------------------------------- #
def test_run_agent_streaming_matches_run_agent_and_reports_each_node(tmp_path: Path) -> None:
    from twinmind.agents.graph import AgentDeps, run_agent, run_agent_streaming
    from twinmind.agents.llm_client import FakeLLM

    loader = importlib.util.spec_from_file_location(
        "test_agent_fakes_dashboard", PROJECT_ROOT / "tests" / "test_agent.py"
    )
    assert loader and loader.loader
    ta: Any = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = ta
    loader.loader.exec_module(ta)

    def deps(log: Path) -> AgentDeps:
        llm = FakeLLM([ta.spec_reply(), ta.grounded_explainer])
        return AgentDeps(config=ta._config(), llm=llm, toolkit=ta._happy_toolkit(), run_log=log)

    steps: list[tuple[str, int, int]] = []
    streamed = run_agent_streaming("x", 0, deps(tmp_path / "a.jsonl"), lambda *s: steps.append(s))
    plain = run_agent("x", 0, deps(tmp_path / "b.jsonl"))
    assert [s[0] for s in steps] == streamed.trace["nodes_run"]
    assert steps[0] == ("planner", 1, 6) and steps[-1][1] == 2
    a, b = streamed.to_dict(), plain.to_dict()
    for d in (a, b):
        d.pop("created_at")
        d["trace"].pop("wall_seconds")
        for call in d["trace"]["llm_calls"]:
            call.pop("latency_seconds", None)
        d["trace"].pop("llm_latency_seconds")
    assert a == b
    assert (tmp_path / "a.jsonl").read_text(encoding="utf-8").count("\n") == 1
