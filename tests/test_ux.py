"""The plain-language layer, offline: glossary, goal cards, the plain view
model (headline, summary, things to watch), the trust rule, and the single
background runner -- all without Streamlit.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from typing import Any

import pytest

from twinmind.agents.contracts import AgentResult
from twinmind.optim.contracts import CONTROLLABLE_DIMS, ObjectiveSpec
from twinmind.twin.runner import KPI_NAMES
from twinmind.ui import glossary as gl
from twinmind.ui import goals as gs
from twinmind.ui import plain as pl
from twinmind.ui import services as sv
from twinmind.ui import viewmodel as vm
from twinmind.ui.jobs import JobRunner, JobStatus, classify_error


def _saved(goal_id: str) -> AgentResult:
    return next(e for e in sv.load_demo_cache() if e.goal_id == goal_id).result


ALL_SAVED = ["G07", "G01", "I01", "N01", "AA1", "G09", "G03", "G10"]


def _with(result: AgentResult, **changes: Any) -> AgentResult:
    return AgentResult.from_dict({**result.to_dict(), **changes})


# --------------------------------------------------------------------------- #
# Glossary
# --------------------------------------------------------------------------- #
def test_every_kpi_and_setting_has_a_glossary_entry() -> None:
    gl.check_complete()
    assert set(gl.KPIS) == set(KPI_NAMES)
    assert set(gl.SETTINGS) == set(CONTROLLABLE_DIMS)
    for term in [*gl.KPIS.values(), *gl.SETTINGS.values()]:
        assert term.name and term.unit and term.meaning
        assert not gl.jargon_in(term.name + " " + term.help), term.code
    assert {s.section for s in gl.SETTINGS.values()} == set(gl.SECTIONS)


def test_rough_kpis_are_exactly_the_weak_ones() -> None:
    assert {k for k, t in gl.KPIS.items() if t.rough} == set(vm.WEAK_KPIS)
    assert gl.ROUGH_KPI_NOTE in gl.KPIS["downtime_hours"].help


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("The surrogate says so", ["surrogate"]),
        ("95% CI from 3 to 4", ["CI"]),
        ("OEE rose", ["OEE without its explanation"]),
        ("Equipment efficiency (OEE) rose", []),
        ("TwinMind circuits and decisions", []),
        ("The plan is infeasible", ["infeasible"]),
        ("verified on twin", ["twin", "verified"]),
    ],
)
def test_jargon_detector(text: str, found: list[str]) -> None:
    assert sorted(gl.jargon_in(text)) == sorted(found)


# --------------------------------------------------------------------------- #
# Goal cards -> goal specs
# --------------------------------------------------------------------------- #
def _combinations(card: gs.GoalCard) -> list[dict[str, float | None]]:
    out = []
    fields = [f for f, _ in card.limit_fields]
    for picks in itertools.product([None, "default", "low", "high"], repeat=len(fields)):
        limits: dict[str, float | None] = {}
        for field, pick in zip(fields, picks, strict=True):
            limits[field.id] = {
                None: None,
                "default": field.default,
                "low": field.min,
                "high": field.max,
            }[pick]
        out.append(limits)
    return out


@pytest.mark.parametrize("card_id", list(gs.CARDS))
def test_every_card_and_limit_combination_makes_a_valid_goal(card_id: str) -> None:
    card = gs.CARDS[card_id]
    for limits in _combinations(card):
        for overtime in [True, False] if card.overtime_choice else [True]:
            spec = gs.build_spec(card_id, limits, overtime, 8)
            again = ObjectiveSpec.from_dict(spec.to_dict())
            assert again.objective == {card.kpi: 1.0} and again.sense == card.sense
            assert len(again.constraints) == sum(v is not None for v in limits.values())
            free = set(again.free_dims)
            assert ("overtime_hours" in free) == overtime
            expected = set(CONTROLLABLE_DIMS) - (set() if overtime else {"overtime_hours"})
            assert free == expected
            assert not gl.jargon_in(spec.name)


def test_bad_limits_give_plain_messages_not_exceptions() -> None:
    with pytest.raises(gs.GoalFormError, match="must be between"):
        gs.build_spec("profit", {"max_breakdowns": 99.0})
    with pytest.raises(gs.GoalFormError, match="isn't a limit for this goal"):
        gs.build_spec("profit", {"min_profit": 1.0})


def test_percent_limits_become_fractions() -> None:
    spec = gs.build_spec("unit_cost", {"min_on_time": 85.0, "max_short_days": 20.0})
    assert {(c.kpi, c.threshold) for c in spec.constraints} == {
        ("on_time_delivery_rate", 0.85),
        ("stockout_probability", 0.2),
    }
    assert spec.name == (
        "Lower the cost per unit, with on-time delivery of at least 85% and short of stock on "
        "at most 20% of days."
    )


@pytest.mark.parametrize(
    ("card_id", "related"),
    [("profit", False), ("breakdowns", True), ("on_time", True), ("unit_cost", False),
     ("stock", True)],
)  # fmt: skip
def test_each_card_has_a_saved_example_and_says_when_it_is_only_related(
    card_id: str, related: bool
) -> None:
    card = gs.CARDS[card_id]
    entry = next(e for e in sv.load_demo_cache() if e.id == card.demo_id)
    assert gs.is_related_only(card_id, entry.result.spec or {}) is related


# --------------------------------------------------------------------------- #
# Plain view model
# --------------------------------------------------------------------------- #
CAUSAL = re.compile(
    r"\b(because|therefore|thanks to|due to|leads? to|led to|as a result|caus\w*|so that|"
    r"so|hence|results? in)\b",
    re.I,
)


@pytest.mark.parametrize("goal_id", ALL_SAVED)
def test_every_plain_string_is_jargon_free_and_honest(goal_id: str) -> None:
    result = _saved(goal_id)
    texts = [
        pl.headline(result),
        pl.plain_summary(result),
        pl.chip(result).text,
        pl.goal_sentence(result),
        pl.checked_label(result),
        *(c.text for c in pl.setting_changes(result)),
        *(f"{c.label} {c.after} {c.before} {c.line} {c.help}" for c in pl.kpi_cards(result)),
        *(" ".join(r.values()) for r in pl.more_numbers(result)),
        *(w.text for w in pl.things_to_watch(result)),
        *pl.closest_lines(result),
    ]
    trust = pl.trust(result)
    assert trust is not None
    texts += [trust.label, trust.conclusion, *(c.text for c in trust.checks)]
    for text in texts:
        assert not gl.jargon_in(text), (text, gl.jargon_in(text))
        vm.assert_no_overclaim(text)
        assert "verif" not in text.lower()


@pytest.mark.parametrize("goal_id", ALL_SAVED)
def test_the_plain_summary_states_numbers_and_never_causes(goal_id: str) -> None:
    summary = pl.plain_summary(_saved(goal_id))
    assert summary
    assert not CAUSAL.search(summary), CAUSAL.search(summary)


@pytest.mark.parametrize(
    ("goal_id", "outcome", "chip"),
    [
        ("G07", "recommended", "Recommended plan"),
        ("N01", "no_better", "No better plan found: your current setup is already close"),
        (
            "I01",
            "not_found",
            "We couldn't find a plan that meets these limits. Closest options below.",
        ),
    ],
)
def test_status_chips(goal_id: str, outcome: str, chip: str) -> None:
    result = _saved(goal_id)
    assert pl.outcome(result) == outcome and pl.chip(result).text == chip
    assert "can't be met" not in pl.headline(result)


def test_headline_answers_first() -> None:
    assert pl.headline(_saved("G01")) == (
        "Change 16 settings: Profit goes from £297,771 to £381,879 (+£84,109) over 4 weeks, "
        "with your limits met on average."
    )
    assert pl.headline(_saved("N01")).startswith("Keep your current setup.")
    assert pl.headline(_saved("I01")).startswith("None of the plans we checked met all")


def test_kpi_cards_say_better_or_worse_in_words() -> None:
    cards = {c.kpi: c for c in pl.kpi_cards(_saved("G09"))}
    assert list(cards)[:2] == ["oee", "unplanned_failures"], "the goal's own numbers first"
    assert cards["profit"].word == "worse" and cards["profit"].color == "red"
    assert cards["unplanned_failures"].rough
    assert all(c.word in ("better", "worse", "no clear change") for c in cards.values())


def test_side_effects_are_things_to_watch_in_plain_words() -> None:
    watch = [w.text for w in pl.things_to_watch(_saved("G09"))]
    assert any(t.startswith("Profit gets worse: £344,569 → −£243,182") for t in watch)
    assert pl.things_to_watch(_saved("G09"))[0].severity == "High"


def test_what_to_change_uses_friendly_names_and_units() -> None:
    lines = [c.text for c in pl.setting_changes(_saved("G01"))]
    assert len(lines) == 16
    assert "Overtime: 3.9 hours a day instead of none" in lines
    assert any(
        line.startswith("Service the forming machine (M2) every 549 running hours instead of 168")
        for line in lines
    )
    assert not any("_" in line for line in lines), "no code names"


# --------------------------------------------------------------------------- #
# The trust rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("goal_id", "level"),
    [
        ("G07", "checked"),
        ("AA1", "checked"),
        ("N01", "checked"),
        ("I01", "checked"),
        ("G03", "checked"),
        ("G10", "partly"),
        ("G01", "rough"),
        ("G09", "rough"),
    ],
)
def test_trust_levels_on_the_saved_examples(goal_id: str, level: str) -> None:
    trust = pl.trust(_saved(goal_id))
    assert trust is not None and trust.level == level
    assert len(trust.checks) == 4


def _synthetic(robust: bool, novelty: float, weak: bool, reps: int = 30) -> AgentResult:
    base = _saved("AA1")  # recommended, with a stock-out limit
    evaluation = json.loads(json.dumps(base.baseline_comparison))
    evaluation["feasible_robust"] = robust
    evaluation["n_reps"] = reps
    spec = json.loads(json.dumps(base.spec))
    if weak:
        spec["constraints"].append({"kpi": "unplanned_failures", "op": "<=", "threshold": 9.0})
    plan = {**(base.plan or {}), "novelty": novelty}
    return _with(base, baseline_comparison=evaluation, spec=spec, plan=plan)


@pytest.mark.parametrize(
    ("robust", "novelty", "weak", "level"),
    [
        (True, 0.5, False, "checked"),
        (False, 0.5, False, "partly"),
        (True, 1.05, False, "partly"),
        (True, 0.5, True, "partly"),
        (False, 1.05, False, "rough"),
        (False, 0.5, True, "rough"),
        (True, 1.5, True, "rough"),
        (False, 1.5, True, "rough"),
    ],
)
def test_trust_rule_every_branch(robust: bool, novelty: float, weak: bool, level: str) -> None:
    trust = pl.trust(_synthetic(robust, novelty, weak))
    assert trust is not None and trust.level == level
    failed = [c for c in trust.checks if c.passed is False]
    assert len(failed) == (not robust) + (novelty > vm.NOVELTY_T1_P95) + weak


def test_trust_without_the_detailed_simulation_is_a_rough_guide() -> None:
    trust = pl.trust(_synthetic(True, 0.5, False, reps=1))
    assert trust is not None and trust.level == "rough"
    assert trust.checks[0].passed is False


def test_trust_limits_check_is_not_applicable_without_a_recommendation_or_limits() -> None:
    no_better = pl.trust(_saved("N01"))
    assert no_better is not None and no_better.checks[1].passed is None
    assert no_better.conclusion.startswith("We searched and checked")
    assert "none beat your current setup" in no_better.conclusion
    not_found = pl.trust(_saved("I01"))
    assert not_found is not None and "none met all your limits" in not_found.conclusion
    unlimited = _with(_saved("AA1"), spec={**(_saved("AA1").spec or {}), "constraints": []})
    trust = pl.trust(unlimited)
    assert trust is not None and trust.checks[1].passed is None


def test_trust_wording_for_breakdown_goals_is_not_a_verdict_on_the_plan() -> None:
    trust = pl.trust(_saved("G10"))
    assert trust is not None
    assert trust.checks[3].text == (
        "Breakdown and downtime numbers are less reliable in this model, so treat this as a "
        "rough guide."
    )


@pytest.mark.parametrize("goal_id", ALL_SAVED)
def test_trust_makes_no_probability_claim(goal_id: str) -> None:
    trust = pl.trust(_saved(goal_id))
    assert trust is not None
    text = " ".join([trust.label, trust.conclusion, *(c.text for c in trust.checks)])
    assert "%" not in text and "probab" not in text.lower() and "chance" not in text.lower()


# --------------------------------------------------------------------------- #
# The background runner, with fakes
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _sync() -> tuple[JobRunner, list[JobStatus], Clock]:
    clock = Clock()
    done: list[JobStatus] = []
    return JobRunner(60.0, clock, start_thread=lambda target: target()), done, clock


def test_a_run_that_succeeds_ends_finished_and_logs_once() -> None:
    runner, done, _ = _sync()

    def work(progress: Any) -> str:
        for node in ("explorer", "evaluator", "critic", "explainer"):
            progress(node, 1, 6)
        return "result"

    runner.start("goal", work, done.append)
    status = runner.status()
    assert status.state == "finished" and status.result == "result" and not status.active
    assert status.nodes == ("explorer", "evaluator", "critic", "explainer")
    assert len(done) == 1 and done[0].state == "finished"
    assert not runner.busy


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ValueError("bad"), "failed"),
        (ConnectionError("network down"), "ai_unreachable"),
        (RuntimeError("Error code: 429 rate limit"), "ai_busy"),
    ],
)
def test_a_run_that_raises_ends_failed_in_plain_words(error: Exception, kind: str) -> None:
    runner, done, _ = _sync()

    def work(progress: Any) -> None:
        raise error

    runner.start("goal", work, done.append)
    status = runner.status()
    assert status.state == "failed" and status.error_kind == kind
    assert type(error).__name__ in status.error_detail
    assert done[0].state == "failed" and not runner.busy


def test_the_error_detail_never_holds_a_key() -> None:
    runner, done, _ = _sync()

    def work(progress: Any) -> None:
        raise RuntimeError("401 bad key gsk_abcdefghijk")

    runner.start("goal", work, done.append)
    assert "gsk_" not in runner.status().error_detail
    assert classify_error(RuntimeError("Error code: 429")) == "ai_busy"


def test_cancel_stops_the_run_at_the_next_step() -> None:
    runner = JobRunner(600.0)
    gate, finished = threading.Event(), threading.Event()
    done: list[JobStatus] = []

    def work(progress: Any) -> str:
        progress("explorer", 0, 6)
        gate.wait(5)
        progress("evaluator", 0, 6)  # raises: cancelled
        return "never"

    def on_done(status: JobStatus) -> None:
        done.append(status)
        finished.set()

    runner.start("goal", work, on_done)
    runner.cancel()
    assert runner.status().state == "cancelling" and runner.busy
    gate.set()
    assert finished.wait(5)
    status = runner.status()
    assert status.state == "cancelled" and status.result is None and not runner.busy
    assert done[0].error_kind == "cancelled"


def test_the_hard_time_limit_stops_the_run_and_shows_stopping_meanwhile() -> None:
    clock = Clock()
    runner = JobRunner(10.0, clock)
    gate, finished = threading.Event(), threading.Event()
    done: list[JobStatus] = []

    def work(progress: Any) -> str:
        gate.wait(5)
        progress("explorer", 0, 6)  # raises: over the limit
        return "never"

    def on_done(status: JobStatus) -> None:
        done.append(status)
        finished.set()

    runner.start("goal", work, on_done)
    clock.now = 11.0
    assert runner.status().state == "stopping"
    gate.set()
    assert finished.wait(5)
    assert runner.status().state == "timed_out" and done[0].error_kind == "timed_out"


def test_a_second_start_while_running_is_refused_and_starts_no_thread() -> None:
    threads: list[Any] = []

    def start_thread(target: Any) -> None:
        thread = threading.Thread(target=target, daemon=True)
        threads.append(thread)
        thread.start()

    runner = JobRunner(600.0, start_thread=start_thread)
    gate = threading.Event()

    def work(progress: Any) -> str:
        gate.wait(5)
        return "ok"

    first = runner.start("goal", work)
    second = runner.start("goal again", work)  # a double click or a rerun
    assert first is not None and second is None
    assert len(threads) == 1
    gate.set()
    threads[0].join(5)
    assert runner.status().state == "finished"
    assert runner.start("next goal", lambda progress: "ok") is not None


def test_a_failing_logger_cannot_leave_a_run_without_a_status() -> None:
    runner, _, _ = _sync()

    def broken_logger(status: JobStatus) -> None:
        raise OSError("disk full")

    runner.start("goal", lambda progress: "ok", broken_logger)
    assert runner.status().state == "finished" and not runner.busy


def test_progress_steps_are_plain_and_ordered() -> None:
    status = JobStatus(id="x", state="running", nodes=("explorer",), started=1.0)
    steps = status.steps
    assert [s for s, _ in steps] == [text for _, text in __import__(
        "twinmind.ui.jobs", fromlist=["STEPS"]).STEPS]  # fmt: skip
    assert dict(steps)["Searching for plans"] == "done"
    assert dict(steps)["Checking the best candidates in the detailed simulation (30 runs)"] == "now"
    assert all(not gl.jargon_in(text) for text, _ in steps)
