"""The state the twin starts from, and the state vector the surrogate reads.

A scenario says what to *do*; a context says what the plant currently *is* --
which engine trajectory sits in each machine and how worn it is, what stock is
on hand, and which demand series and forecast origin apply. The two are
separate because the optimiser varies the decision at a fixed state, while the
surrogate has to generalise across states.

``state_vector`` is the surrogate's input, so its layout is frozen here in week
3 rather than invented in week 4. It deliberately carries the health model's
*beliefs* (predicted RUL mean and std) and not the replayed ground truth: the
surrogate stands where a plant manager stands, and a plant manager sees
predictions, not failure cycles. Feeding it the true remaining cycles would
leak the answer and the trained surrogate would be useless on a real line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from twinmind.twin.health_replay import HealthCache

#: Layout of ``state_vector()``. Frozen, and pinned by a test: the surrogate
#: regresses on these positions, so reordering them silently invalidates a
#: trained surrogate.
STATE_FIELDS: tuple[str, ...] = (
    "rul_mean_M1",
    "rul_std_M1",
    "rul_mean_M2",
    "rul_std_M2",
    "rul_mean_M3",
    "rul_std_M3",
    "rul_mean_M4",
    "rul_std_M4",
    "rul_mean_M5",
    "rul_std_M5",
    "raw_inventory",
    "finished_inventory",
    "demand_q10_mean",
    "demand_q50_mean",
    "demand_q90_mean",
)


class MachineInit(BaseModel):
    """One machine's starting condition: which trajectory, and how far into it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_id: int = Field(ge=0, description="C-MAPSS engine id from the replay pool.")
    age_cycles: float = Field(ge=0.0, description="Cycles consumed since the last renewal.")


class TwinContext(BaseModel):
    """Immutable starting state for a replication.

    Frozen so a replication cannot mutate the state its siblings start from --
    the bug that would make replications silently dependent and the confidence
    intervals wrong.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    series_id: str
    origin: pd.Timestamp
    machines: list[MachineInit] = Field(min_length=1)
    raw_inventory: float = Field(ge=0.0)
    finished_inventory: float = Field(ge=0.0)

    #: Filled by ``sample_context`` from the cache; held here so ``state_vector``
    #: needs no I/O and stays usable in a worker process.
    rul_beliefs: list[tuple[float, float]] = Field(default_factory=list)
    demand_summary: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def state_vector(self) -> NDArray[np.float64]:
        """The surrogate's conditioning input, in ``STATE_FIELDS`` order."""
        values: list[float] = []
        for index in range(len(self.machines)):
            if index < len(self.rul_beliefs):
                mean, std = self.rul_beliefs[index]
            else:
                mean, std = 0.0, 0.0
            values.extend([float(mean), float(std)])
        values.extend([float(self.raw_inventory), float(self.finished_inventory)])
        values.extend(float(v) for v in self.demand_summary)
        return np.asarray(values, dtype="float64")


def default_context(
    series_id: str = "1_GROCERY I",
    origin: pd.Timestamp | None = None,
    n_machines: int = 5,
) -> TwinContext:
    """The documented context a ``context=None`` call uses.

    Every machine starts a third of the way through the pool's *mean* lifetime
    rather than at zero. A line whose five machines are all brand new is a
    degenerate state: nothing can fail early, so maintenance policy cannot
    matter and the twin would look insensitive to the levers that are the whole
    point of the exercise.
    """
    from twinmind.twin.health_replay import PoolSpec

    del PoolSpec  # imported only to keep the dependency explicit in the docstring
    resolved_origin = pd.Timestamp("2017-08-15") if origin is None else pd.Timestamp(origin)
    # Engine ids and the 68-cycle start are the documented defaults; they are
    # overwritten by sample_context whenever a cache is available.
    return TwinContext(
        series_id=series_id,
        origin=resolved_origin,
        machines=[MachineInit(trajectory_id=0, age_cycles=68.0) for _ in range(n_machines)],
        raw_inventory=3000.0,
        finished_inventory=0.0,
        rul_beliefs=[(125.0, 0.0)] * n_machines,
        demand_summary=(0.0, 0.0, 0.0),
    )


def sample_context(
    seed: int,
    cache: HealthCache | None = None,
    series_id: str = "1_GROCERY I",
    origin: pd.Timestamp | None = None,
    n_machines: int = 5,
    raw_inventory_range: tuple[float, float] = (1500.0, 4500.0),
    demand_summary: tuple[float, float, float] | None = None,
) -> TwinContext:
    """Draw a random but reachable starting state.

    Ages are drawn uniform on ``[0, 0.8 x failure_cycle]`` of the engine
    actually assigned, so a machine is never initialised past its own failure
    point and the distribution of "how worn is the line" is wide enough for the
    surrogate to learn from. The 0.8 cap keeps a little life in every machine;
    a context where a machine fails in hour one carries no information about the
    decision.

    Uses its own ``SeedSequence`` child rather than the replication streams, so
    sampling a context does not consume randomness the simulation needs.
    """
    if cache is None:
        base = default_context(series_id, origin, n_machines)
        return base

    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    slots = rng.integers(0, cache.n_engines, size=n_machines)
    machines: list[MachineInit] = []
    beliefs: list[tuple[float, float]] = []
    for slot in slots:
        failure_cycle = int(cache.failure_cycles[slot])
        age = float(rng.uniform(0.0, 0.8 * failure_cycle))
        machines.append(MachineInit(trajectory_id=int(cache.engine_ids[slot]), age_cycles=age))
        mean, std, _ = cache.predictions_at(int(slot), int(age))
        beliefs.append((mean, std))

    return TwinContext(
        series_id=series_id,
        origin=pd.Timestamp("2017-08-15") if origin is None else pd.Timestamp(origin),
        machines=machines,
        raw_inventory=float(rng.uniform(*raw_inventory_range)),
        finished_inventory=0.0,
        rul_beliefs=beliefs,
        demand_summary=demand_summary or (0.0, 0.0, 0.0),
    )
