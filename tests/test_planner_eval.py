"""The planner-evaluation dataset and its scoring rules, offline.

The labels are checked against the goal library itself: every paraphrase's
expected spec must match the goal YAML it paraphrases under the same
spec-match rule the live evaluation uses. A mislabelled item would otherwise
score the model against a wrong answer.
"""

from __future__ import annotations

from collections import Counter

import pytest

from twinmind.agents.planner_eval import THRESHOLDS, aggregate, load_items, score_item, spec_match
from twinmind.optim.contracts import Constraint, ObjectiveSpec
from twinmind.optim.goals import load_goals


def test_composition_is_40_items_split_12_dev_28_test() -> None:
    items = load_items()
    assert len(items) == 40
    assert len({i.id for i in items}) == 40
    counts = Counter((i.split, i.category) for i in items)
    assert counts == {
        ("dev", "paraphrase"): 8,
        ("dev", "ambiguous"): 2,
        ("dev", "adversarial"): 2,
        ("test", "paraphrase"): 17,
        ("test", "ambiguous"): 6,
        ("test", "adversarial"): 5,
    }


def test_every_goal_yaml_is_paraphrased_at_least_once() -> None:
    sources = {i.source for i in load_items() if i.category == "paraphrase"}
    assert sources == {g.name for g in load_goals()}


@pytest.mark.parametrize("item", [i for i in load_items() if i.category == "paraphrase"], ids=str)
def test_each_label_matches_its_source_goal(item: object) -> None:
    goal = next(g for g in load_goals() if g.name == item.source)  # type: ignore[attr-defined]
    ok, reasons = spec_match(goal.spec, item.expected)  # type: ignore[attr-defined]
    assert ok, reasons


def _spec(**kwargs: object) -> ObjectiveSpec:
    payload: dict[str, object] = {"name": "t", "objective": {"profit": 1.0}}
    payload.update(kwargs)
    return ObjectiveSpec(**payload)  # type: ignore[arg-type]


EXPECTED = {
    "objective": {"profit": 1.0},
    "sense": "maximise",
    "constraints": [{"kpi": "stockout_probability", "op": "<=", "threshold": 0.1}],
    "free_dims": None,
    "fixed": {},
}


def test_spec_match_accepts_equivalent_forms_and_rounding() -> None:
    within = _spec(constraints=(Constraint("stockout_probability", "<=", 0.104),))
    assert spec_match(within, EXPECTED)[0]
    flipped = _spec(
        objective={"profit": -2.0},
        sense="minimise",
        constraints=(Constraint("stockout_probability", "<=", 0.1),),
    )
    assert spec_match(flipped, EXPECTED)[0], "minimise -profit is maximise profit"


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_spec(constraints=(Constraint("stockout_probability", "<=", 0.11),)), "missing"),
        (_spec(constraints=(Constraint("stockout_probability", ">=", 0.1),)), "missing"),
        (
            _spec(
                objective={"profit": 1.0},
                sense="minimise",
                constraints=(Constraint("stockout_probability", "<=", 0.1),),
            ),
            "direction",
        ),
        (
            _spec(
                constraints=(
                    Constraint("stockout_probability", "<=", 0.1),
                    Constraint("oee", ">=", 0.5),
                )
            ),
            "extra",
        ),
        (
            _spec(
                constraints=(Constraint("stockout_probability", "<=", 0.1),),
                free_dims=("safety_stock",),
            ),
            "free dims",
        ),
        (
            _spec(
                constraints=(Constraint("stockout_probability", "<=", 0.1),),
                fixed={"supplier_choice": 2.0},
            ),
            "fixed",
        ),
    ],
)
def test_spec_match_rejects_real_differences(spec: ObjectiveSpec, reason: str) -> None:
    ok, reasons = spec_match(spec, EXPECTED)
    assert not ok and any(reason in r for r in reasons)


def test_scoring_and_thresholds() -> None:
    items = {i.id: i for i in load_items()}
    scores = [
        score_item(items["A01"], "clarify", None),
        score_item(items["A02"], "spec", _spec()),
        score_item(items["X01"], "invalid", None),
        score_item(items["X03"], "spec", _spec()),
        score_item(items["P01"], "spec", _spec()),
        score_item(items["P03"], "refuse", None),
    ]
    metrics = aggregate(scores)
    assert metrics["clarification"]["rate"] == 0.5
    assert metrics["refused_or_bounded"]["rate"] == 1.0
    assert metrics["adversarial_refused"]["rate"] == 0.5
    assert metrics["spec_match"]["hits"] == 1 and metrics["spec_match"]["n"] == 2
    # A refusal on a non-adversarial item is not a valid outcome.
    assert metrics["spec_validity"]["hits"] == 3 and metrics["spec_validity"]["n"] == 4
    assert THRESHOLDS == {
        "spec_validity": 0.95,
        "spec_match": 0.85,
        "clarification": 0.75,
        "refused_or_bounded": 1.00,
    }
