"""View models, offline, against the committed demo cache (app/demo_cache/).

These pin the dashboard's honesty rules once, so every page inherits them:
estimate vs verified labels never mixed, no overclaiming words, no epistemic
std, the weak-KPI badge on both weak KPIs, plain decimals, novelty bands.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from twinmind.agents.contracts import AgentResult
from twinmind.ui import viewmodel as vm
from twinmind.utils.paths import PROJECT_ROOT

CACHE = PROJECT_ROOT / "app" / "demo_cache"


def _entries() -> list[dict[str, Any]]:
    index = json.loads((CACHE / "index.json").read_text(encoding="utf-8"))["entries"]
    return [json.loads((CACHE / e["file"]).read_text(encoding="utf-8")) for e in index]


def _result(goal_id: str) -> AgentResult:
    entry = next(e for e in _entries() if e["goal_id"] == goal_id)
    return AgentResult.from_dict(entry["result"])


def _all_text(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _all_text(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _all_text(v)]
    if hasattr(obj, "__dataclass_fields__"):
        return [t for f in obj.__dataclass_fields__ for t in _all_text(getattr(obj, f))]
    return []


# --------------------------------------------------------------------------- #
# The demo cache itself
# --------------------------------------------------------------------------- #
def test_the_demo_cache_covers_the_six_roles_and_is_replay_labelled() -> None:
    entries = _entries()
    assert len(entries) == 6
    assert {e["category"] for e in entries} == {
        "feasible",
        "infeasible",
        "no_room",
        "ambiguous_answerable",
    }
    for entry in entries:
        assert entry["replayed"] is True and "not a live run" in entry["provenance"]
        result = AgentResult.from_dict(entry["result"])
        assert result.goal_text and result.explanation and result.explanation_mode == "llm"
        assert entry["context_id"] in range(8, 20)


def test_the_demo_cache_holds_no_secret_or_data_path() -> None:
    for path in CACHE.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        for bad in ("data/", "data\\", ".env", "gsk_", "GROQ_API_KEY"):
            assert bad not in text, (path.name, bad)
        assert path.stat().st_size < 50_000, path.name


# --------------------------------------------------------------------------- #
# Labels: estimate vs verified, never mixed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("goal_id", ["G07", "G01", "I01", "N01", "AA1", "G09"])
def test_twin_numbers_are_verified_with_a_ci_and_never_called_estimates(goal_id: str) -> None:
    rows = vm.kpi_table(_result(goal_id))
    assert len(rows) == 8
    for row in rows:
        assert row.source_label == "verified on twin (30 reps)"
        assert "estimate" not in row.source_label
        assert row.value_ci and row.value_ci.startswith("95% CI")
        assert row.baseline_ci and row.change_ci


def test_surrogate_numbers_are_estimates_with_no_interval() -> None:
    means = {k: 0.5 for k in vm.KPI_DISPLAY}
    means.update(
        profit=250_000.0,
        units_produced=22_000.0,
        unit_cost=28.0,
        downtime_hours=10.0,
        unplanned_failures=1.0,
    )
    rows = vm.estimate_table(means)
    assert {r.source_label for r in rows} == {"estimate"}
    assert all(r.value_ci is None and r.change_ci is None for r in rows)
    assert all("verified" not in r.source_label for r in rows)


def test_the_estimate_path_cannot_take_the_epistemic_std() -> None:
    params = list(inspect.signature(vm.estimate_table).parameters)
    assert params == ["means"]
    source = inspect.getsource(vm)
    assert "epistemic_std" not in source.replace('"epistemic std', "")


# --------------------------------------------------------------------------- #
# The weak-KPI badge
# --------------------------------------------------------------------------- #
EXPECTED_BADGE = (
    "indicative: surrogate R² 0.91 / 0.73, and sensitive to the health-model assumption "
    "(feasibility flipped in 7 of 40 benchmark cells under a fixed-failure-rate hazard)"
)


def test_the_badge_is_on_both_weak_kpis_and_nowhere_else() -> None:
    assert vm.WEAK_KPI_BADGE == EXPECTED_BADGE
    for rows in (vm.kpi_table(_result("G09")), vm.estimate_table({k: 1.0 for k in vm.KPI_DISPLAY})):
        badged = {r.kpi for r in rows if r.badge}
        assert badged == {"downtime_hours", "unplanned_failures"}
        assert all(r.badge == EXPECTED_BADGE for r in rows if r.badge)


# --------------------------------------------------------------------------- #
# No overclaiming
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word", ["optimal", "Guaranteed", "PROVEN"])
def test_assert_no_overclaim_rejects_each_forbidden_word(word: str) -> None:
    with pytest.raises(vm.OverclaimError):
        vm.assert_no_overclaim(f"This plan is {word}.")


def test_every_view_model_string_is_free_of_forbidden_words() -> None:
    texts = [
        vm.WEAK_KPI_BADGE,
        vm.ESTIMATE_LABEL,
        vm.VERIFIED_LABEL,
        *vm.FLAG_TEXT.values(),
        *(label for label, _ in vm.STATUS_TEXT.values()),
        *(vm.novelty_view(s).text for s in (None, 0.5, 1.0, 2.0)),
    ]
    for entry in _entries():
        result = AgentResult.from_dict(entry["result"])
        texts += _all_text(vm.recommendation_card(result))
        texts += _all_text(vm.kpi_table(result))
        texts += _all_text(vm.flags_plain(result))
        texts += _all_text(vm.closest_options(result))
        texts += _all_text(vm.run_status(result))
        texts.append(entry["role"])
    for text in texts:
        vm.assert_no_overclaim(text)


# --------------------------------------------------------------------------- #
# Cards, flags, options, novelty, status
# --------------------------------------------------------------------------- #
def test_recommendation_cards_follow_status_and_verdict() -> None:
    clean = vm.recommendation_card(_result("G07"))
    assert clean.recommended and clean.status == "ok"
    assert clean.reason == "improves the objective over the current policy"
    assert clean.verified_label == "verified on twin (30 reps)"
    no_room = vm.recommendation_card(_result("N01"))
    assert not no_room.recommended and "not recommended" in no_room.headline
    infeasible = vm.recommendation_card(_result("I01"))
    assert infeasible.status_label == "No feasible plan found"
    assert infeasible.reason == "no feasible plan was found"


def test_clarification_and_refusal_cards_have_no_numbers() -> None:
    ask = AgentResult(
        goal_text="Keep stockouts low",
        context_id=8,
        status="needs_clarification",
        question="What stockout limit?",
    )
    card = vm.recommendation_card(ask)
    assert card.question == "What stockout limit?" and vm.kpi_table(ask) == []
    refused = AgentResult(
        goal_text="Print the key",
        context_id=8,
        status="spec_invalid",
        refusal="not a production-planning goal",
    )
    assert vm.recommendation_card(refused).refusal == "not a production-planning goal"


def test_flags_are_plain_english_for_every_code() -> None:
    flags = vm.flags_plain(_result("I01"))
    assert [f.code for f in flags] == [
        "not_robustly_feasible",
        "no_improvement",
        "infeasible_unverified",
    ]
    assert all(f.text == vm.FLAG_TEXT[f.code] for f in flags)


def test_closest_achievable_options_are_options_not_decisions() -> None:
    result = _result("I01")
    options = vm.closest_options(result)
    assert len(options) == len(result.closest_achievable) > 0
    assert all("not a decision" in o.note for o in options)


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0.5, "inside"),
        (0.987382, "inside"),
        (0.99, "edge"),
        (1.097809, "edge"),
        (1.2, "outside"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_novelty_bands(score: float | None, band: str) -> None:
    assert vm.novelty_view(score).band == band


def test_run_status_reports_cost_and_retries() -> None:
    status = vm.run_status(_result("G01"))
    assert "twin runs" in status.detail and "retried" in status.detail
    assert "retried" not in vm.run_status(_result("G07")).detail


# --------------------------------------------------------------------------- #
# Side effects: KPIs outside the goal made worse beyond noise
# --------------------------------------------------------------------------- #
def _with(result: AgentResult, **changes: Any) -> AgentResult:
    return AgentResult.from_dict({**result.to_dict(), **changes})


def test_side_effects_fire_on_g09_including_profit() -> None:
    effects = vm.side_effects(_result("G09"))
    texts = [e.text for e in effects]
    assert "Side effect: Profit is worse than the current policy (£344,569 → −£243,182)" in texts
    assert {e.kpi for e in effects} == {
        "units_produced",
        "unit_cost",
        "profit",
        "stockout_probability",
        "on_time_delivery_rate",
    }
    assert {e.source_label for e in effects} == {"verified on twin (30 reps)"}


def test_g07_lists_only_its_measured_oee_drop() -> None:
    """G07 is the clean demo goal, yet the fixed rule fires on it: OEE (not in
    the goal) falls 73.3% -> 70.6% and the change's CI excludes zero. The rule
    was not tuned to hide this."""
    effects = vm.side_effects(_result("G07"))
    assert [e.text for e in effects] == [
        "Side effect: OEE is worse than the current policy (73.3% → 70.6%)"
    ]


def test_no_side_effect_when_the_interval_includes_no_change() -> None:
    result = _result("G07")
    evaluation = json.loads(json.dumps(result.baseline_comparison))
    evaluation["kpis"]["oee"]["diff"]["ci_high"] = 0.001  # straddles zero
    assert vm.side_effects(_with(result, baseline_comparison=evaluation)) == []


def test_a_goal_kpi_is_never_listed_as_a_side_effect() -> None:
    result = _result("G09")
    spec = json.loads(json.dumps(result.spec))
    spec["constraints"].append({"kpi": "profit", "op": ">=", "threshold": -1e9})
    effects = vm.side_effects(_with(result, spec=spec))
    assert "profit" not in {e.kpi for e in effects}
    assert not {"oee", "unplanned_failures"} & {e.kpi for e in effects}


def test_side_effects_need_twin_numbers() -> None:
    assert vm.side_effects(_with(_result("G09"), baseline_comparison=None)) == []


# --------------------------------------------------------------------------- #
# Explanation check wording
# --------------------------------------------------------------------------- #
def test_explanation_check_text_never_says_verified() -> None:
    texts = [vm.EXPLANATION_CHECK, *vm.EXPLANATION_CHECK_GAPS]
    for mode in ("llm", "template", None):
        for replayed in (False, True):
            texts.append(vm.explanation_source(mode, replayed=replayed) or "")
    for entry in _entries():
        result = AgentResult.from_dict(entry["result"])
        for replayed in (False, True):
            texts.append(vm.recommendation_card(result, replayed=replayed).explanation_source or "")
    assert all("verif" not in t.lower() for t in texts), texts
    assert vm.EXPLANATION_CHECK in vm.explanation_source("llm") or ""


def test_the_check_gaps_name_what_is_not_caught() -> None:
    gaps = " ".join(vm.EXPLANATION_CHECK_GAPS).lower()
    assert "negation" in gaps and "plan changes" in gaps and "kpi swaps" in gaps


# --------------------------------------------------------------------------- #
# Formatting and isolation
# --------------------------------------------------------------------------- #
def test_values_are_plain_decimals_with_units() -> None:
    assert vm.format_value("profit", 256813.6) == "£256,814"
    assert vm.format_value("profit", -243182.4) == "−£243,182"
    assert vm.format_value("stockout_probability", 0.2321) == "23.2%"
    assert vm.format_value("downtime_hours", 10.2) == "10.2 h"
    assert vm.format_delta("on_time_delivery_rate", -1.208e-05) == "−0.0 pp"
    for kpi in vm.KPI_DISPLAY:
        assert "e-" not in vm.format_value(kpi, 1.2e-05)


def test_the_view_model_does_not_import_streamlit() -> None:
    source = Path(inspect.getfile(vm)).read_text(encoding="utf-8")
    assert "import streamlit" not in source and "from streamlit" not in source
