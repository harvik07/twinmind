"""Deterministic cost and KPI accounting for the twin.

Every number a result table can show is computed here, in plain Python. The
agent never touches this module: the LLM proposes scenario parameters and writes
prose, and every KPI, comparison and constraint check is this code.

The profit identity is the thing to get right. ``profit = revenue - total_cost``
is tested as an identity rather than asserted in a comment, because a cost line
that is accumulated but left out of the sum is invisible -- the number still
looks plausible -- and it would quietly change which scenario the optimiser
prefers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from twinmind.twin.runner import KPIVector
from twinmind.utils.config import CostConfig

#: The cost buckets that make up ``total_cost``. Derived from the dataclass
#: fields rather than hand-listed, so adding a bucket cannot silently omit it
#: from the total; ``revenue`` and ``extras`` are handled separately.
_NON_COST_FIELDS = frozenset({"revenue", "extras"})


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
        """Sum of every cost bucket, excluding revenue.

        Enumerates the dataclass fields so a newly added bucket is included
        automatically. ``extras`` is summed too, so a cost parked there still
        reaches the total and the profit identity still holds.
        """
        total = sum(
            float(getattr(self, f.name)) for f in fields(self) if f.name not in _NON_COST_FIELDS
        )
        return total + sum(float(v) for v in self.extras.values())

    @property
    def profit(self) -> float:
        """Revenue less every cost. The identity the test pins."""
        return float(self.revenue) - self.total_cost

    def cost_lines(self) -> dict[str, float]:
        """Each bucket by name, for the cost-identity test and the model card."""
        lines = {
            f.name: float(getattr(self, f.name))
            for f in fields(self)
            if f.name not in _NON_COST_FIELDS
        }
        lines.update({f"extra_{k}": float(v) for k, v in self.extras.items()})
        return lines


def compute_oee(
    planned_hours: float,
    downtime_hours: float,
    actual_units: float,
    theoretical_units: float,
    good_units: float,
) -> float:
    """Availability x Performance x Quality, clipped to [0, 1].

    * availability = uptime / planned hours
    * performance  = units actually made / units the bottleneck could have made
                     in the uptime
    * quality      = good units / units made

    Performance is measured against what was achievable **in the uptime**, not
    in the planned hours. Dividing by planned hours instead would double-count
    the downtime -- it is already in availability -- and OEE would fall as the
    square of availability, which is the single most common way this metric is
    reported wrongly.
    """
    if planned_hours <= 0.0:
        return 0.0
    uptime = max(planned_hours - downtime_hours, 0.0)
    availability = uptime / planned_hours

    if theoretical_units <= 0.0 or actual_units <= 0.0:
        return 0.0
    performance = min(actual_units / theoretical_units, 1.0)
    quality = min(good_units / actual_units, 1.0) if actual_units > 0 else 0.0
    return float(min(max(availability * performance * quality, 0.0), 1.0))


def build_kpi_vector(
    accumulator: CostAccumulator, costs: CostConfig, **counters: float
) -> KPIVector:
    """Turn raw counters into the canonical :class:`KPIVector`.

    ``unit_cost`` when nothing was produced is a genuine divide-by-zero. It
    reports total cost rather than infinity or zero: a line that spent money and
    made nothing has a very high unit cost, and zero would make a catastrophic
    scenario look like the cheapest one in the search.
    """
    del costs  # every rate has already been applied into the accumulator
    units = float(counters.get("units_produced", 0.0))
    good = float(counters.get("good_units", units))
    planned = float(counters.get("planned_hours", 0.0))
    downtime = float(counters.get("downtime_hours", 0.0))
    theoretical = float(counters.get("theoretical_units", 0.0))

    unit_cost = accumulator.total_cost / units if units > 0 else accumulator.total_cost
    demanded = float(counters.get("units_demanded", 0.0))
    delivered = float(counters.get("units_delivered", 0.0))
    on_time = min(delivered / demanded, 1.0) if demanded > 0 else 1.0

    return KPIVector(
        units_produced=units,
        oee=compute_oee(planned, downtime, units, theoretical, good),
        unit_cost=max(unit_cost, 0.0),
        profit=accumulator.profit,
        stockout_probability=min(max(float(counters.get("stockout_probability", 0.0)), 0.0), 1.0),
        downtime_hours=max(downtime, 0.0),
        unplanned_failures=int(counters.get("unplanned_failures", 0)),
        on_time_delivery_rate=min(max(on_time, 0.0), 1.0),
    )
