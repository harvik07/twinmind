"""Deterministic cost and KPI accounting for the twin.

Every number a result table can show is computed here, in plain Python. The
agent never touches this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from twinmind.twin.runner import KPIVector
from twinmind.utils.config import CostConfig


@dataclass
class CostAccumulator:
    """Running totals collected during a replication."""

    maintenance_preventive: float = 0.0
    maintenance_corrective: float = 0.0
    downtime: float = 0.0
    labour: float = 0.0
    holding: float = 0.0
    stockout: float = 0.0
    late_delivery: float = 0.0
    raw_material: float = 0.0
    energy: float = 0.0
    overhead: float = 0.0
    revenue: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Sum of every cost bucket, excluding revenue."""
        raise NotImplementedError("CostAccumulator.total_cost is not implemented yet")


def compute_oee(
    planned_hours: float, downtime_hours: float, actual_units: float, theoretical_units: float,
    good_units: float,
) -> float:
    """Availability x Performance x Quality, clipped to [0, 1]."""
    raise NotImplementedError("compute_oee is not implemented yet")


def build_kpi_vector(accumulator: CostAccumulator, costs: CostConfig, **counters: float
                     ) -> KPIVector:
    """Turn raw counters into the canonical :class:`KPIVector`."""
    raise NotImplementedError("build_kpi_vector is not implemented yet")
