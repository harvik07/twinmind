"""Raw-material inventory, supplier replenishment, and the shift calendar.

The (s, Q) policy reviews the **inventory position** -- on hand plus on order --
not just what is on the shelf. Reviewing on-hand alone is the classic error: it
reorders every review period while the first order is still in transit, so a
21-day offshore lead time produces a pile of duplicate orders and the supplier
comparison becomes meaningless.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import simpy

from twinmind.utils.config import ShiftConfig, SupplierConfig


class ShiftCalendar:
    """Which calendar hours the line is staffed.

    Precomputed as two float arrays of window starts and ends, then queried by
    binary search. The stations ask "am I staffed now?" at every lot boundary,
    roughly 22,000 times per replication, so this is on the hot path and a
    datetime calculation per call would dominate the run.

    Weekends are unstaffed: ``working_days_per_week`` is 5, so the calendar has
    20 windows over a 28-day horizon, not 28.
    """

    def __init__(self, shifts: ShiftConfig, shift_count: int, overtime_hours: float, days: int):
        self.hours_per_day = shift_count * shifts.hours_per_shift + overtime_hours
        starts: list[float] = []
        ends: list[float] = []
        for day in range(days):
            # Day 0 is a Monday by construction; weekends are days 5 and 6 mod 7.
            if day % 7 >= shifts.working_days_per_week:
                continue
            starts.append(day * 24.0)
            ends.append(day * 24.0 + self.hours_per_day)
        self.starts = np.asarray(starts, dtype="float64")
        self.ends = np.asarray(ends, dtype="float64")
        self.staffed_hours = float(self.ends.sum() - self.starts.sum())
        self.horizon_hours = days * 24.0

    def hours_until_staffed(self, now: float) -> float:
        """0.0 if staffed right now, else the wait until the next window opens.

        Returns a large number past the end of the calendar, which parks the
        station processes harmlessly rather than spinning them: the environment
        is run to a fixed horizon, so a process waiting beyond it simply never
        resumes.
        """
        index = int(np.searchsorted(self.ends, now, side="right"))
        if index >= self.starts.size:
            return self.horizon_hours * 2.0
        start = float(self.starts[index])
        return 0.0 if now >= start else start - now

    def next_shift_end(self, now: float) -> float:
        """The end of the current or next staffed window.

        Early-warning maintenance is scheduled here rather than immediately,
        because taking a machine down mid-shift to act on a probabilistic
        warning is not what a plant does.
        """
        index = int(np.searchsorted(self.ends, now, side="right"))
        if index >= self.ends.size:
            return self.horizon_hours * 2.0
        return float(self.ends[index])


class InventoryPolicy:
    """Continuous-review (s, Q) policy with a safety-stock floor."""

    def __init__(self, reorder_point: float, order_quantity: float, safety_stock: float) -> None:
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
        self.safety_stock = safety_stock

    def should_order(self, on_hand: float, on_order: float) -> bool:
        """True when the inventory *position* has fallen to the reorder point.

        ``safety_stock`` raises the trigger rather than sitting untouched at the
        bottom of the bin. A floor that can never be consumed is not stock, it
        is a smaller warehouse; as a trigger offset it does what safety stock is
        for -- ordering earlier so demand during the lead time is covered.
        """
        return (on_hand + on_order) <= (self.reorder_point + self.safety_stock)


@dataclass
class WarehouseState:
    """Totals the warehouse accumulates over a replication."""

    orders_placed: int = 0
    orders_short: int = 0
    units_ordered: float = 0.0
    units_received: float = 0.0
    units_consumed: float = 0.0
    shortfall_units: float = 0.0
    material_cost: float = 0.0
    holding_unit_days: float = 0.0
    on_order: float = 0.0
    raw_level_trace: list[float] = field(default_factory=list)


class Warehouse:
    """Tracks on-hand stock, outstanding orders, stockouts and holding cost."""

    def __init__(
        self,
        env: simpy.Environment,
        policy: InventoryPolicy,
        supplier: SupplierConfig,
        initial_stock: float,
        *,
        container: simpy.Container,
        state: WarehouseState,
        rng: np.random.Generator,
        lead_time_multiplier: float = 1.0,
        short_ship_fraction: float = 0.7,
        review_interval_hours: float = 24.0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.supplier = supplier
        self.initial_stock = initial_stock
        self.container = container
        self.state = state
        self.rng = rng
        self.lead_time_multiplier = lead_time_multiplier
        self.short_ship_fraction = short_ship_fraction
        self.review_interval = review_interval_hours

    def run(self) -> Generator[simpy.Event, Any, Any]:
        """Daily review loop that places replenishment orders."""
        while True:
            if self.policy.should_order(self.container.level, self.state.on_order):
                quantity = self.policy.order_quantity
                self.state.orders_placed += 1
                self.state.units_ordered += quantity
                self.state.on_order += quantity
                self.env.process(self._deliver(quantity))
            # Holding cost accrues on what is actually on the shelf.
            self.state.holding_unit_days += self.container.level * (self.review_interval / 24.0)
            self.state.raw_level_trace.append(float(self.container.level))
            yield self.env.timeout(self.review_interval)

    def _deliver(self, quantity: float) -> Generator[simpy.Event, Any, Any]:
        """One order in transit: lead time, then a possibly short delivery."""
        mean = self.supplier.lead_time_days_mean * self.lead_time_multiplier
        sd = self.supplier.lead_time_days_sd
        days = float(np.clip(self.rng.normal(mean, sd), 0.25, 4.0 * mean))
        yield self.env.timeout(days * 24.0)

        delivered = quantity
        if self.rng.random() > self.supplier.reliability:
            delivered = quantity * self.short_ship_fraction
            self.state.orders_short += 1

        self.state.on_order -= quantity
        # Never overfill: the container's capacity is the physical yard.
        room = self.container.capacity - self.container.level
        accepted = max(min(delivered, room), 0.0)
        if accepted > 0:
            yield self.container.put(accepted)
        self.state.units_received += accepted
        # Paid for what was delivered, at the chosen supplier's price.
        self.state.material_cost += delivered * self.supplier.unit_price
