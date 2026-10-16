"""Explanation checker v2, regression-tested on the REAL benchmark failure texts.

Source: the first-attempt failure reasons stored in
reports/results/agent_bench_audit.json (Prompt 9). Each reason keeps only the
first 80 characters of its sentence, so where the sentence was cut the tail is
completed minimally -- marked ``# tail reconstructed`` -- with wording v1 does
not strip (v1 strips "higher is better" itself, so the real text must have
used another form). Deltas are the real cells' deltas.

Four stored reasons are NOT used: AA2 s1, G02 s2, G03 s2 and G04 s2. Their
flagged word lies past the 80-character cut, so which KPI it described cannot
be known, and a test built on a guess would prove nothing.

Each false-flag case shows v1 flagging it and v2 not. The real errors
("downtime hours fall from 14.69 to 20.05" while downtime ROSE) stay flagged
by BOTH versions.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from twinmind.agents.explanation_v2 import (
    FORBIDDEN_WORDS,
    audit_v2,
    build_facts_v2,
    direction_errors_v2,
    explain_v2,
    forbidden_word_errors,
    parse_numbers,
    passes,
    ungrounded_numbers_v2,
)
from twinmind.agents.llm_client import CallBudget, FakeLLM
from twinmind.agents.tools import direction_errors, template_explanation, ungrounded_numbers


def _facts(**deltas: float) -> dict[str, Any]:
    return {
        "kpi_table": [{"kpi": k, "delta": v} for k, v in deltas.items()],
        "recommendation_reason": "",
    }


# Real cells' deltas (from data/processed/agent_bench/full, rebuilt with the v1
# facts builder exactly as the benchmark explainer saw them).
G01_S2 = _facts(oee=-0.009269, profit=55100.0, unit_cost=-2.501)
G02_S1 = _facts(downtime_hours=1.362, profit=-113100.0, units_produced=10020.0)
G03_S1 = _facts(oee=-0.08811, profit=-34200.0, unplanned_failures=0.6667)
G10_S1 = _facts(
    units_produced=-862.0,
    oee=-0.125,
    stockout_probability=0.02593,
    on_time_delivery_rate=-0.01426,
)
AA2_S2 = _facts(downtime_hours=5.361, profit=-284800.0, units_produced=16120.0)

FALSE_FLAGS = [
    pytest.param(
        "OEE falls from 0.6982 to 0.689, a decrease of –0.009269, which is worse "
        "when higher values are better.",  # tail reconstructed
        G01_S2,
        id="G01-s2-oee",
    ),
    pytest.param(
        "OEE falls from 0.7416 to 0.6535, a decrease of –0.08811, which is worse "
        "when higher values are better.",  # tail reconstructed
        G03_S1,
        id="G03-s1-oee",
    ),
    pytest.param(
        "Profit falls from 377100.0 to 342900.0, a decrease of –34200.0, which is "
        "worse when higher values are better.",  # tail reconstructed
        G03_S1,
        id="G03-s1-profit",
    ),
    pytest.param(
        "Unplanned failures rise from 0.2667 to 0.9333, an increase of 0.6667, which is "
        "worse when lower values are better.",  # tail reconstructed
        G03_S1,
        id="G03-s1-failures",
    ),
    pytest.param(
        "**Units produced** fall to 21730.0 from 22590.0, a negative delta of –862.0, "
        "which is worse when higher values are better.",  # tail reconstructed
        G10_S1,
        id="G10-s1-units",
    ),
    pytest.param(
        "**OEE** falls to 0.5655 from 0.6905, a negative delta of –0.125, which is "
        "worse when higher values are better.",  # tail reconstructed
        G10_S1,
        id="G10-s1-oee",
    ),
    pytest.param(
        "**Stockout probability** rises to 0.2753 from 0.2494, a positive delta of "
        "0.02593, which is worse when lower values are better.",  # tail reconstructed
        G10_S1,
        id="G10-s1-stockout",
    ),
    pytest.param(
        "**On-time delivery rate** falls to 0.7696 from 0.7839, a negative delta of "
        "–0.01426, which is worse when higher values are better.",  # tail reconstructed
        G10_S1,
        id="G10-s1-otd",
    ),
    pytest.param(
        # Complete as stored: "the better_when is lower" is a convention, and the
        # increase it is attached to is correct (downtime delta +1.362).
        "The better_when is lower, but the delta is positive, so downtime hours increase.",
        G02_S1,
        id="G02-s1-better-when",
    ),
]


@pytest.mark.parametrize(("text", "facts"), FALSE_FLAGS)
def test_v1_flags_the_real_false_flags_and_v2_does_not(text: str, facts: dict[str, Any]) -> None:
    assert direction_errors(text, facts), "negative control: v1 must flag this"
    assert direction_errors_v2(text, facts) == [], direction_errors_v2(text, facts)


REAL_ERRORS = [
    pytest.param("Downtime hours fall from 14.69 to 20.05, a delta of 5.361.", AA2_S2, id="AA2-s2"),
    pytest.param("Downtime hours fall from 10.66 to 12.02.", G02_S1, id="G02-s1"),
]


@pytest.mark.parametrize(("text", "facts"), REAL_ERRORS)
def test_real_direction_errors_stay_flagged_by_both_versions(
    text: str, facts: dict[str, Any]
) -> None:
    assert direction_errors(text, facts)
    errors = direction_errors_v2(text, facts)
    assert errors and "downtime_hours" in errors[0]


def test_n02_scientific_notation_v1_splits_it_v2_reads_one_number() -> None:
    """N02 seed 1: v1 facts carried OTD's delta as -1.208e-05 and its CI as
    -3.678e-05 .. 1.262e-05; the model copied them and v1 called "1.208" and
    "05" ungrounded."""
    facts = {
        "kpi_table": [
            {
                "kpi": "on_time_delivery_rate",
                "delta": -1.208e-05,
                "delta_ci95": [-3.678e-05, 1.262e-05],
            }
        ]
    }
    text = "On-time delivery changes by -1.208e-05 (CI -3.678e-05 to 1.262e-05)."
    v1 = {lit.lstrip("-−") for lit in ungrounded_numbers(text, facts)}
    assert v1 >= {"1.208", "05"}, "negative control: v1 splits the exponent off"
    assert ungrounded_numbers_v2(text, facts) == []


def test_parse_numbers_reads_exponents_suffixes_and_skips_names() -> None:
    parsed = {lit: val for lit, val, _ in parse_numbers("1.208e-05, £257k, 12.5% and M3")}
    assert parsed["1.208e-05"] == pytest.approx(1.208e-05)
    assert parsed["£257k"] == pytest.approx(257_000)
    assert parsed["12.5%"] == pytest.approx(12.5)
    assert all(lit != "3" for lit in parsed), "M3 is a machine name, not a number"


@pytest.mark.parametrize(
    ("text", "kpi"),
    [
        ("The trade-offs: higher units produced and lower profit.", None),
        ("It means higher profit.", "profit"),
        ("These changes collectively lower unit cost while increasing output.", None),
        ("These changes keep costs flat while increasing profit.", "profit"),
    ],
)
def test_a_direction_word_describes_what_it_modifies(text: str, kpi: str | None) -> None:
    facts = _facts(units_produced=100.0, profit=-50.0, unit_cost=-1.0)
    errors = direction_errors_v2(text, facts)
    if kpi is None:
        assert errors == [], errors
    else:
        assert errors and kpi in errors[0]


# --------------------------------------------------------------------------- #
# Found by the post-benchmark mutation test (scripts/mutation_test_explanations.py):
# corruptions v1 caught and the first v2 missed, and a real error both missed.
# --------------------------------------------------------------------------- #
AA1_S1 = _facts(units_produced=352.3, oee=-0.04514, profit=40960.0, downtime_hours=-5.341)
G07_S2 = _facts(profit=39010.0, unit_cost=-1.858, downtime_hours=-1.669, unplanned_failures=-0.2)
I01_S1 = _facts(stockout_probability=-0.04074, units_produced=6664.0)


@pytest.mark.parametrize(
    ("text", "facts"),
    [
        # A trailing ", a decrease of ..." clause continues the KPI before it.
        ("OEE falls from 0.7148 to 0.6697, an increase of 0.04514.", AA1_S1),
        ("Profit rises from 318100.0 to 359000.0, a decrease of 40960.0.", AA1_S1),
        # A subject-less verb applies to the list that follows it.
        (
            "These adjustments collectively improve profit and key service metrics while "
            "increasing costs, downtime, and failures.",
            G07_S2,
        ),
    ],
)
def test_mutants_v1_caught_are_caught_by_v2(text: str, facts: dict[str, Any]) -> None:
    assert direction_errors(text, facts), "negative control: v1 catches this mutant"
    assert direction_errors_v2(text, facts)


def test_nothing_carries_across_while() -> None:
    """The real G07 s2 sentence: "reducing" must not inherit "profit"."""
    text = (
        "These adjustments collectively improve profit and key service metrics while "
        "reducing costs, downtime, and failures."
    )
    assert direction_errors_v2(text, G07_S2) == []


def test_snake_case_kpi_keys_are_anchors() -> None:
    """The real I01 s1 benchmark text says stockout probability ROSE; it fell
    (0.2309 -> 0.1901). v1 cannot see "stockout_probability" as a KPI, so the
    benchmark passed it; v2 flags it."""
    text = "- stockout_probability rises to 0.1901 from 0.2309."
    assert direction_errors(text, I01_S1) == [], "negative control: v1 misses snake_case"
    errors = direction_errors_v2(text, I01_S1)
    assert errors and "stockout_probability" in errors[0]
    assert direction_errors_v2("- stockout_probability falls to 0.1901.", I01_S1) == []


def test_forbidden_words_are_caught_in_any_case() -> None:
    for word in FORBIDDEN_WORDS:
        assert forbidden_word_errors(f"This plan is {word.upper()}.") == [word]
    assert forbidden_word_errors("A verified plan, improved over the current policy.") == []


# --------------------------------------------------------------------------- #
# v2 facts: plain decimals with units
# --------------------------------------------------------------------------- #
def _evaluation(tiny: float) -> dict[str, Any]:
    from twinmind.twin.runner import KPI_NAMES

    def s(mean: float) -> dict[str, float]:
        return {
            "mean": mean,
            "ci_low": mean - abs(tiny) - 1e-6,
            "ci_high": mean + abs(tiny),
            "sem": 0.0,
        }

    return {
        "kpis": {k: {"plan": s(0.8), "baseline": s(0.8), "diff": s(tiny)} for k in KPI_NAMES},
        "objective": {"plan": s(1.0), "baseline": s(1.0), "diff": s(tiny)},
        "feasible_point": True,
        "feasible_robust": True,
        "n_reps": 30,
        "confidence_level": 0.95,
    }


def test_v2_facts_never_contain_scientific_notation_and_carry_units() -> None:
    from twinmind.optim.contracts import ObjectiveSpec
    from twinmind.twin.runner import Scenario

    spec = ObjectiveSpec(name="g", objective={"on_time_delivery_rate": 1.0})
    facts = build_facts_v2(
        spec,
        {"scenario": Scenario().model_dump()},
        _evaluation(-1.208e-05),
        [],
        [],
        False,
        "no improvement over current policy",
    )
    dumped = json.dumps(facts)
    assert "e-0" not in dumped and "e-1" not in dumped, dumped[:300]
    row = next(r for r in facts["kpi_table"] if r["kpi"] == "on_time_delivery_rate")
    assert row["delta"] == 0.0 and row["unit"] == "fraction 0-1"
    template = template_explanation(facts)
    assert passes(audit_v2(template, facts)), audit_v2(template, facts)


def test_agent_result_and_run_log_keep_every_attempt(tmp_path: Any) -> None:
    """The benchmark stored only first-attempt reasons, so 2 of 25 failures
    could not be judged. The AgentResult and the run log now keep them all."""
    import importlib.util
    import sys

    from twinmind.agents.contracts import AgentResult
    from twinmind.utils.paths import PROJECT_ROOT

    loader = importlib.util.spec_from_file_location(
        "test_agent_fakes_v2", PROJECT_ROOT / "tests" / "test_agent.py"
    )
    assert loader and loader.loader
    ta: Any = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = ta
    loader.loader.exec_module(ta)

    liar = "This plan saves £1,234,567 and is guaranteed."
    log = tmp_path / "runs.jsonl"
    result = ta.run(
        "x",
        FakeLLM([ta.spec_reply(), liar, ta.grounded_explainer]),
        ta._happy_toolkit(),
        run_log=log,
    )
    record = result.trace["explanation"]
    assert record["checker_version"] == 2 and record["attempts"] == 2
    texts = [a["text"] for a in record["attempt_log"]]
    assert texts[0] == liar and len(texts) == 2
    assert record["attempt_log"][0]["audit"]["forbidden_words"] == ["guaranteed"]
    logged = AgentResult.from_dict(json.loads(log.read_text(encoding="utf-8")))
    assert logged.trace["explanation"]["attempt_log"][0]["text"] == liar


# --------------------------------------------------------------------------- #
# Every attempt is recorded
# --------------------------------------------------------------------------- #
def test_explain_v2_records_every_attempt_including_the_failed_ones() -> None:
    facts = {
        "kpi_table": [{"kpi": "profit", "delta": -10.0, "plan": 90.0}],
        "recommendation_reason": "no improvement over current policy",
        "recommended": False,
        "plan_changes": [],
        "flags": [],
        "closest_achievable": [],
        "objective_change": {"change": -10.0, "change_ci95": [-12.0, -8.0]},
        "replications": 30,
        "confidence_level_pct": 95,
        "goal": "g",
    }
    first = "Profit rises to 90.0. It is the optimal plan."
    second = "Profit falls to 90.0; not recommended: no improvement over current policy."
    llm = FakeLLM([first, second])
    result = explain_v2(facts, CallBudget(llm, 6), max_regenerations=1, max_words=100)
    assert result.mode == "llm" and result.text == second
    assert [a["text"] for a in result.attempts] == [first, second]
    failed = result.attempts[0]["audit"]
    assert failed["direction_errors"] and failed["forbidden_words"] == ["optimal"]
    assert not any(result.attempts[1]["audit"].values())
