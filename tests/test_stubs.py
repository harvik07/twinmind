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
    "twinmind.models.a1_train",
    "twinmind.models.ai4i",
    "twinmind.models.health",
    "twinmind.models.demand",
    "twinmind.models.surrogate",
    "twinmind.twin.machines",
    "twinmind.twin.inventory",
    "twinmind.twin.demand_proc",
    "twinmind.twin.costs",
    "twinmind.twin.runner",
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
    from twinmind.eval.metrics import surrogate_fidelity
    from twinmind.optimise.oracle import regret
    from twinmind.twin.inventory import InventoryPolicy

    with pytest.raises(NotImplementedError):
        surrogate_fidelity(None, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        regret(None, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        InventoryPolicy(1.0, 1.0, 1.0).should_order(0.0, 0.0)
