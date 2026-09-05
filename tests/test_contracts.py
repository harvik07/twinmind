"""The twin's data contracts are the interface everything else depends on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from twinmind.twin.runner import (
    KPI_NAMES,
    SCENARIO_DIMS,
    CIEstimate,
    KPIVector,
    KPIWithCI,
    Scenario,
)


def _kpis(**overrides: float) -> dict[str, float]:
    base: dict[str, float] = {
        "units_produced": 10000.0,
        "oee": 0.72,
        "unit_cost": 31.5,
        "profit": 125000.0,
        "stockout_probability": 0.02,
        "downtime_hours": 48.0,
        "unplanned_failures": 3,
        "on_time_delivery_rate": 0.94,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# KPIVector
# --------------------------------------------------------------------------- #
def test_valid_kpi_vector_round_trips() -> None:
    kpis = KPIVector(**_kpis())
    assert kpis.to_vector() == [getattr(kpis, name) for name in KPI_NAMES]
    assert KPIVector.from_vector(kpis.to_vector()) == kpis


def test_kpi_vector_rejects_negative_unit_cost() -> None:
    with pytest.raises(ValidationError) as exc:
        KPIVector(**_kpis(unit_cost=-1.0))
    assert "unit_cost" in str(exc.value)


def test_kpi_vector_rejects_oee_above_one() -> None:
    with pytest.raises(ValidationError) as exc:
        KPIVector(**_kpis(oee=1.01))
    assert "oee" in str(exc.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("units_produced", -1.0),
        ("stockout_probability", 1.5),
        ("downtime_hours", -0.1),
        ("unplanned_failures", -1),
        ("on_time_delivery_rate", 1.2),
    ],
)
def test_kpi_vector_rejects_out_of_range_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        KPIVector(**_kpis(**{field: value}))


def test_profit_may_be_negative() -> None:
    """A loss-making scenario is a legitimate result, not a validation error."""
    assert KPIVector(**_kpis(profit=-5000.0)).profit == -5000.0


def test_kpi_vector_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        KPIVector(**_kpis(), throughput=1.0)


def test_kpi_vector_from_vector_length_is_checked() -> None:
    with pytest.raises(ValueError, match="expected 8 values"):
        KPIVector.from_vector([1.0, 2.0])


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #
def test_scenario_defaults_round_trip_through_a_vector() -> None:
    scenario = Scenario()
    assert len(scenario.to_vector()) == 18
    assert Scenario.from_vector(scenario.to_vector()) == scenario


def test_scenario_vector_order_matches_the_declared_dims() -> None:
    assert tuple(Scenario.model_fields) == SCENARIO_DIMS


def test_scenario_rejects_impossible_values() -> None:
    with pytest.raises(ValidationError):
        Scenario(shift_count=0)
    with pytest.raises(ValidationError):
        Scenario(throughput_multiplier=0.0)
    with pytest.raises(ValidationError):
        Scenario(supplier_choice=-1)


def test_scenario_from_vector_rounds_integer_dims() -> None:
    vector = Scenario().to_vector()
    vector[SCENARIO_DIMS.index("shift_count")] = 2.6
    assert Scenario.from_vector(vector).shift_count == 3


def test_scenario_is_frozen() -> None:
    with pytest.raises(ValidationError):
        Scenario().shift_count = 3  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# CIs
# --------------------------------------------------------------------------- #
def test_ci_estimate_requires_ordering() -> None:
    with pytest.raises(ValidationError):
        CIEstimate(mean=5.0, lower=6.0, upper=7.0)


def test_kpi_with_ci_exposes_means_and_lookup() -> None:
    estimates = {name: CIEstimate(mean=1.0, lower=0.5, upper=1.5) for name in KPI_NAMES}
    aggregated = KPIWithCI(n_reps=20, **estimates)

    assert aggregated.means() == dict.fromkeys(KPI_NAMES, 1.0)
    assert aggregated.get("profit").width == pytest.approx(1.0)
    with pytest.raises(KeyError):
        aggregated.get("not_a_kpi")


def test_kpi_with_ci_requires_positive_reps() -> None:
    estimates = {name: CIEstimate(mean=1.0, lower=1.0, upper=1.0) for name in KPI_NAMES}
    with pytest.raises(ValidationError):
        KPIWithCI(n_reps=0, **estimates)
