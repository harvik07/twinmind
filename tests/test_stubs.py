"""Every package must import cleanly, and every stub must fail loudly, not silently."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import twinmind

MODULES = [
    "twinmind.data.download",
    "twinmind.data.cmapss_prep",
    "twinmind.data.kaggle_auth",
    "twinmind.data.loaders",
    "twinmind.data.windowing",
    "twinmind.data.features",
    "twinmind.models.a1_cnn_bilstm",
    "twinmind.models.a1_cv",
    "twinmind.models.a1_train",
    "twinmind.models.ai4i",
    "twinmind.models.blended_health",
    "twinmind.models.forest_health",
    "twinmind.models.health",
    "twinmind.models.demand",
    "twinmind.models.demand_features",
    "twinmind.models.demand_lgbm",
    "twinmind.models.surrogate",
    "twinmind.twin.machines",
    "twinmind.twin.inventory",
    "twinmind.twin.demand_proc",
    "twinmind.twin.costs",
    "twinmind.twin.runner",
    "twinmind.twin.context",
    "twinmind.twin.health_replay",
    "twinmind.twin.prng",
    "twinmind.twin.streams",
    "twinmind.optimise.random_search",
    "twinmind.optimise.grid",
    "twinmind.optimise.optuna_tpe",
    "twinmind.optimise.oracle",
    "twinmind.agents.graph",
    "twinmind.agents.nodes",
    "twinmind.agents.tools",
    "twinmind.agents.memory",
    "twinmind.agents.llm_client",
    "twinmind.eval.figures",
    "twinmind.eval.demand_metrics",
    "twinmind.eval.metrics",
    "twinmind.eval.registry",
    "twinmind.eval.ablations",
    "twinmind.eval.faithfulness",
    "twinmind.utils.paths",
    "twinmind.utils.config",
    "twinmind.utils.seed",
    "twinmind.utils.logging",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_every_module_on_disk_is_covered() -> None:
    """Guard against adding a module and forgetting to import-test it."""
    discovered = {
        info.name
        for info in pkgutil.walk_packages(twinmind.__path__, prefix="twinmind.")
        if not info.ispkg
    }
    assert discovered == set(MODULES), f"MODULES is out of sync: {discovered ^ set(MODULES)}"


def test_stubs_raise_not_implemented() -> None:
    """The remaining stubs must fail loudly rather than returning a wrong number.

    ``InventoryPolicy.should_order`` used to be on this list and is now
    implemented, so it moved to the test below. The twin's modules are no longer
    stubs at all; what is left here belongs to the surrogate (week 4) and the
    optimiser oracle (week 5).
    """
    from twinmind.eval.metrics import surrogate_fidelity
    from twinmind.optimise.oracle import regret

    with pytest.raises(NotImplementedError):
        surrogate_fidelity(None, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        regret(None, None)  # type: ignore[arg-type]


def test_inventory_policy_is_implemented() -> None:
    """(s, Q) reviews the inventory POSITION, so on-order counts toward it."""
    from twinmind.twin.inventory import InventoryPolicy

    policy = InventoryPolicy(reorder_point=1000.0, order_quantity=500.0, safety_stock=200.0)
    assert policy.should_order(on_hand=900.0, on_order=0.0) is True
    # Already covered by stock in transit: reviewing on-hand alone would
    # reorder here and pile up duplicate orders against a long lead time.
    assert policy.should_order(on_hand=900.0, on_order=800.0) is False
