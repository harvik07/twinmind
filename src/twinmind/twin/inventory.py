"""Raw-material inventory and supplier replenishment."""

from __future__ import annotations

from collections.abc import Iterator

import simpy

from twinmind.utils.config import SupplierConfig


class InventoryPolicy:
    """Continuous-review (s, Q) policy with a safety-stock floor."""

    def __init__(self, reorder_point: float, order_quantity: float, safety_stock: float) -> None:
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
        self.safety_stock = safety_stock

    def should_order(self, on_hand: float, on_order: float) -> bool:
        """True when the inventory position has fallen to the reorder point."""
        raise NotImplementedError("InventoryPolicy.should_order is not implemented yet")


class Warehouse:
    """Tracks on-hand stock, outstanding orders, stockouts and holding cost."""

    def __init__(
        self,
        env: simpy.Environment,
        policy: InventoryPolicy,
        supplier: SupplierConfig,
        initial_stock: float,
    ) -> None:
        self.env = env
        self.policy = policy
        self.supplier = supplier
        self.initial_stock = initial_stock

    def run(self) -> Iterator[simpy.Event]:
        """Review loop that places replenishment orders."""
        raise NotImplementedError("Warehouse.run is not implemented yet")

    def withdraw(self, units: float) -> float:
        """Consume units, returning the shortfall (0.0 when fully satisfied)."""
        raise NotImplementedError("Warehouse.withdraw is not implemented yet")
