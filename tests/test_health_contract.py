"""The A1 contract the twin depends on.

Written against the Protocol, so a stub satisfies them before the real model
exists -- which is the point of freezing the interface first.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from twinmind.models.health import ConstantHealthModel, HealthModel, RULPrediction


def _valid(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "rul_mean": 42.0,
        "rul_std": 3.5,
        "p_fail_within_h": 0.12,
        "horizon_h": 30,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# RULPrediction
# --------------------------------------------------------------------------- #
def test_valid_prediction_round_trips() -> None:
    prediction = RULPrediction(**_valid())  # type: ignore[arg-type]
    assert prediction.rul_mean == 42.0
    assert RULPrediction(**prediction.model_dump()) == prediction


@pytest.mark.parametrize(
    "field,value",
    [
        ("rul_mean", -1.0),
        ("rul_std", -0.1),
        ("p_fail_within_h", -0.01),
        ("p_fail_within_h", 1.01),
        ("horizon_h", 0),
        ("horizon_h", -5),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        RULPrediction(**_valid(**{field: value}))  # type: ignore[arg-type]


def test_probability_bounds_are_inclusive() -> None:
    """0 and 1 are legitimate: a brand-new engine and a certain failure."""
    assert RULPrediction(**_valid(p_fail_within_h=0.0)).p_fail_within_h == 0.0  # type: ignore[arg-type]
    assert RULPrediction(**_valid(p_fail_within_h=1.0)).p_fail_within_h == 1.0  # type: ignore[arg-type]


def test_zero_rul_is_allowed() -> None:
    """RUL reaches exactly zero at the failure cycle."""
    assert RULPrediction(**_valid(rul_mean=0.0)).rul_mean == 0.0  # type: ignore[arg-type]


def test_prediction_is_frozen_and_closed() -> None:
    prediction = RULPrediction(**_valid())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        prediction.rul_mean = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RULPrediction(**_valid(), confidence=0.9)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The Protocol
# --------------------------------------------------------------------------- #
def test_stub_satisfies_the_protocol() -> None:
    """A stub must satisfy the contract before the real model exists."""
    assert isinstance(ConstantHealthModel(), HealthModel)


def test_predict_accepts_a_single_window() -> None:
    model: HealthModel = ConstantHealthModel(rul_mean=100.0, horizon_h=30)
    prediction = model.predict(np.zeros((30, 14), dtype="float32"))

    assert isinstance(prediction, RULPrediction)
    assert prediction.rul_mean == 100.0
    assert prediction.horizon_h == 30


def test_predict_batch_returns_one_prediction_per_row() -> None:
    model: HealthModel = ConstantHealthModel()
    predictions = model.predict_batch(np.zeros((7, 30, 14), dtype="float32"))

    assert len(predictions) == 7
    assert all(isinstance(p, RULPrediction) for p in predictions)


def test_wrong_rank_is_rejected_on_both_methods() -> None:
    """[T, C] for predict and [B, T, C] for predict_batch -- confusing them is easy."""
    model = ConstantHealthModel()
    with pytest.raises(ValueError, match=r"\[T, C\]"):
        model.predict(np.zeros((4, 30, 14)))
    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        model.predict_batch(np.zeros((30, 14)))


def test_batch_of_one_is_not_the_same_shape_as_a_single_window() -> None:
    model = ConstantHealthModel()
    assert len(model.predict_batch(np.zeros((1, 30, 14)))) == 1
    with pytest.raises(ValueError):
        model.predict_batch(np.zeros((30, 14)))


def test_the_contract_surface_is_exactly_two_methods() -> None:
    """The twin calls these and nothing else; widening this is a design decision."""
    public = {name for name in dir(HealthModel) if not name.startswith("_")}
    assert public == {"predict", "predict_batch"}
