"""Dashboard services, offline: mode detection, saved examples, history, the
twin check's caps, evidence helpers, and the agent's read-only progress hooks
(each returns exactly what its non-streaming twin returns).
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


def _agent_fakes(name: str) -> Any:
    loader = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "tests" / "test_agent.py")
    assert loader and loader.loader
    ta: Any = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = ta
    loader.loader.exec_module(ta)
    return ta


def _comparable(result: AgentResult) -> dict[str, Any]:
    d = result.to_dict()
    d.pop("created_at")
    d["trace"].pop("wall_seconds")
    d["trace"].pop("llm_latency_seconds")
    for call in d["trace"]["llm_calls"]:
        call.pop("latency_seconds", None)
    return d


@pytest.mark.parametrize("explain_plan", [False, True])
def test_run_on_spec_streaming_returns_exactly_what_run_on_spec_returns(
    explain_plan: bool,
) -> None:
    """The dashboard's goal cards use the streaming entry point. For the same
    seed it must return exactly run_on_spec's result (timestamps aside): it is a
    read-only progress hook, and the benchmark's run_on_spec is untouched."""
    from twinmind.agents.graph import AgentDeps, run_on_spec, run_on_spec_streaming
    from twinmind.agents.llm_client import FakeLLM
    from twinmind.optim.contracts import Constraint, ObjectiveSpec

    ta = _agent_fakes("test_agent_fakes_on_spec")
    spec = ObjectiveSpec(
        name="gold",
        objective={"profit": 1.0},
        constraints=(Constraint("stockout_probability", "<=", 0.1),),
        context_id=8,
    )

    def deps() -> AgentDeps:
        fragile = {**ta.BASE, "profit": 290_000.0, "stockout_probability": 0.095}
        explored = ta._exploration(
            ta._candidate(1.1, 290_000.0, robust=False),
            [ta._candidate(1.2, 270_000.0, robust=True)],
        )
        toolkit = ta.FakeToolkit(
            [explored], {1.1: fragile, 1.2: ta.GOOD_PLAN}, noise={"stockout_probability": 0.05}
        )
        llm = FakeLLM([ta.grounded_explainer] if explain_plan else [])
        return AgentDeps(config=ta._config(), llm=llm, toolkit=toolkit)

    steps: list[str] = []
    streamed = run_on_spec_streaming(
        spec, deps(), 2, lambda node, used, cap: steps.append(node), explain_plan=explain_plan
    )
    plain = run_on_spec(spec, deps(), 2, explain_plan=explain_plan)
    assert _comparable(streamed) == _comparable(plain)
    assert steps == streamed.trace["nodes_run"]
    assert steps[0] == "explorer" and ("explainer" in steps) == explain_plan


def test_a_progress_hook_that_raises_stops_the_run_between_steps() -> None:
    from twinmind.agents.graph import AgentDeps, run_on_spec_streaming
    from twinmind.agents.llm_client import FakeLLM
    from twinmind.optim.contracts import ObjectiveSpec

    ta = _agent_fakes("test_agent_fakes_on_spec_stop")
    deps = AgentDeps(config=ta._config(), llm=FakeLLM([]), toolkit=ta._happy_toolkit())
    seen: list[str] = []

    def stop_after_explorer(node: str, used: int, cap: int) -> None:
        seen.append(node)
        if node == "explorer":
            raise RuntimeError("stop")

    spec = ObjectiveSpec(name="g", objective={"profit": 1.0}, context_id=8)
    with pytest.raises(RuntimeError, match="stop"):
        run_on_spec_streaming(spec, deps, 1, stop_after_explorer)
    assert seen == ["explorer"]


# --------------------------------------------------------------------------- #
# Past results: run ids, running and lost runs
# --------------------------------------------------------------------------- #
def test_history_folds_started_events_and_shows_lost_runs(tmp_path: Path) -> None:
    log = tmp_path / "h.jsonl"
    result = sv.load_demo_cache()[0].result
    sv.append_history(log, "started", "finished goal", "running", run_id="a")
    sv.append_history(log, "live", "finished goal", "ok", "", result, run_id="a")
    sv.append_history(log, "started", "lost goal", "running", run_id="b")
    sv.append_history(log, "started", "current goal", "running", run_id="c")
    sv.append_history(log, "replayed", "saved goal", "ok", "", result)
    items = sv.read_history(log, running_id="c")
    by_goal = {i.goal_text: i for i in items}
    assert len(items) == 4
    assert by_goal["finished goal"].kind == "live" and by_goal["finished goal"].result
    assert by_goal["lost goal"].kind == "aborted" and by_goal["lost goal"].status == "stopped"
    assert by_goal["current goal"].status == "running"
    assert by_goal["saved goal"].label == "Saved example, not a live run"
    assert by_goal["finished goal"].label == "Live"
