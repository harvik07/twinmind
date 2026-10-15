"""Seed streams and the two evaluators the optimisers are allowed to call.

THE SEED FIREWALL, WHICH IS THE POINT OF THIS MODULE.

An optimiser that could choose its own seeds could, deliberately or by accident,
evaluate its final answer on the same seeds it searched with. The reported
number would then be a maximum over noise rather than an estimate of quality --
the winner's curse, baked in and invisible.

So there are exactly two named streams, derived from one root by
``SeedSequence.spawn``:

* **"search"** -- a small fixed seed set, reused across candidates within a goal.
  Reusing them is deliberate: common random numbers make candidate-to-candidate
  comparisons far less noisy, which is what an optimiser needs.
* **"reeval"** -- used *only* for the final independent re-evaluation, and never
  reachable from an optimiser.

The firewall is structural, not a convention. ``Evaluator.evaluate`` takes no
seed argument at all, so an optimiser has nothing to pass; the reeval seeds live
on a separate object that optimisers are never handed. A test asserts the two
seed sets are disjoint and that the optimiser-facing evaluator exposes no route
to the reeval stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from twinmind.optim.contracts import ObjectiveSpec
from twinmind.twin.runner import KPI_NAMES, Scenario
from twinmind.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from twinmind.twin.context import TwinContext

logger = get_logger(__name__)

#: Stream names. Append-only, like the twin's STREAM_NAMES, and for the same
#: reason: inserting one renumbers the spawn order and silently changes every
#: stored result.
STREAM_NAMES: tuple[str, ...] = ("search", "reeval")

#: Root entropy. Distinct from the rollout generator's roots (20250101 and
#: 77770001) so optimisation seeds cannot collide with training or test ones.
OPTIM_SEED_ROOT = 31_415_926


def stream_seeds(stream: str, goal_seed: int, n: int) -> list[int]:
    """``n`` seeds from a named stream, for one goal and optimiser seed.

    ``SeedSequence((root, stream_index, goal_seed))`` hashes the triple, so the
    "search" and "reeval" sets for the same goal are unrelated rather than
    offset -- which is what makes the disjointness real instead of nominal.
    """
    if stream not in STREAM_NAMES:
        raise KeyError(f"unknown stream {stream!r}; expected one of {STREAM_NAMES}")
    index = STREAM_NAMES.index(stream)
    children = np.random.SeedSequence((OPTIM_SEED_ROOT, index, int(goal_seed))).spawn(n)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


@dataclass
class EvalCost:
    """What an optimisation run spent."""

    surrogate_evals: int = 0
    twin_runs: int = 0
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0

    def merge(self, other: EvalCost) -> EvalCost:
        """Combine two cost records."""
        return EvalCost(
            surrogate_evals=self.surrogate_evals + other.surrogate_evals,
            twin_runs=self.twin_runs + other.twin_runs,
            cpu_seconds=self.cpu_seconds + other.cpu_seconds,
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )

    def to_dict(self) -> dict[str, float]:
        """JSON-ready form."""
        return {
            "surrogate_evals": float(self.surrogate_evals),
            "twin_runs": float(self.twin_runs),
            "cpu_seconds": self.cpu_seconds,
            "wall_seconds": self.wall_seconds,
        }


class Evaluator(Protocol):
    """What an optimiser may call. Note there is no seed argument."""

    def evaluate(self, scenarios: Sequence[Scenario]) -> list[dict[str, float]]:
        """KPI means for a batch of scenarios."""
        ...

    @property
    def cost(self) -> EvalCost:
        """What has been spent so far."""
        ...


@dataclass
class SurrogateEvaluator:
    """Batch surrogate evaluation, counting calls.

    Carries no seeds at all: the surrogate is deterministic, so there is nothing
    to seed. That is also why surrogate search is free of the winner's curse in
    the *noise* sense -- its error is bias, not variance, which is exactly why
    the chosen plan still has to be verified on the twin.
    """

    surrogate: Any
    context: TwinContext
    _cost: EvalCost = field(default_factory=EvalCost)

    def evaluate(self, scenarios: Sequence[Scenario]) -> list[dict[str, float]]:
        """Predicted KPI means, one dict per scenario."""
        if not scenarios:
            return []
        started = time.perf_counter()
        output = self.surrogate.predict(list(scenarios), self.context)
        self._cost.surrogate_evals += len(scenarios)
        self._cost.wall_seconds += time.perf_counter() - started
        return [
            {kpi: float(output.mean[row, index]) for index, kpi in enumerate(KPI_NAMES)}
            for row in range(len(scenarios))
        ]

    def predict_full(self, scenarios: Sequence[Scenario]) -> Any:
        """The full :class:`SurrogateOutput`, for the uncertainty fields."""
        self._cost.surrogate_evals += len(scenarios)
        return self.surrogate.predict(list(scenarios), self.context)

    @property
    def cost(self) -> EvalCost:
        """What has been spent so far."""
        return self._cost


@dataclass
class TwinEvaluator:
    """Twin evaluation on a named stream, counting runs and CPU-seconds.

    The stream is fixed at construction. An optimiser receives one built on
    "search" and has no way to reach "reeval": there is no seed argument on
    ``evaluate`` and no setter for the stream.
    """

    resources: Any
    context: TwinContext
    goal_seed: int
    n_reps: int = 10
    stream: str = "search"
    workers: int = 1
    _cost: EvalCost = field(default_factory=EvalCost)
    _seeds: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.stream not in STREAM_NAMES:
            raise KeyError(f"unknown stream {self.stream!r}; expected one of {STREAM_NAMES}")
        # A FIXED seed set for the whole goal: every candidate is evaluated on
        # the same replications, which is common random numbers and is what
        # makes candidate comparisons comparable.
        self._seeds = stream_seeds(self.stream, self.goal_seed, self.n_reps)

    @property
    def seeds(self) -> tuple[int, ...]:
        """The replication seeds in use, for the disjointness test."""
        return tuple(self._seeds)

    def evaluate(self, scenarios: Sequence[Scenario]) -> list[dict[str, float]]:
        """Twin KPI means over ``n_reps`` replications, per scenario."""
        return [self._evaluate_one(s)[0] for s in scenarios]

    def evaluate_with_spread(
        self, scenarios: Sequence[Scenario]
    ) -> list[tuple[dict[str, float], dict[str, float]]]:
        """``(mean, standard error)`` per scenario, for robust feasibility."""
        return [self._evaluate_one(s) for s in scenarios]

    def _evaluate_one(self, scenario: Scenario) -> tuple[dict[str, float], dict[str, float]]:
        """One scenario over the fixed seed set."""
        from twinmind.twin.runner import run_twin

        started = time.perf_counter()
        rows = np.array(
            [
                run_twin(scenario, seed, self.context, self.resources).to_vector()
                for seed in self._seeds
            ],
            dtype="float64",
        )
        elapsed = time.perf_counter() - started
        self._cost.twin_runs += len(self._seeds)
        self._cost.cpu_seconds += elapsed
        self._cost.wall_seconds += elapsed

        mean = {kpi: float(rows[:, i].mean()) for i, kpi in enumerate(KPI_NAMES)}
        standard_error = {
            kpi: float(rows[:, i].std(ddof=1) / np.sqrt(len(self._seeds)))
            if len(self._seeds) > 1
            else 0.0
            for i, kpi in enumerate(KPI_NAMES)
        }
        return mean, standard_error

    @property
    def cost(self) -> EvalCost:
        """What has been spent so far."""
        return self._cost


def feasible_robust(
    spec: ObjectiveSpec, mean: dict[str, float], standard_error: dict[str, float]
) -> bool:
    """Is the whole 95% interval inside the feasible region?

    Stricter than point feasibility and the honest thing to report alongside it:
    a plan whose mean clears a constraint by less than its own standard error
    has not really been shown to satisfy it.
    """
    for constraint in spec.constraints:
        if not constraint.hard:
            continue
        value = float(mean[constraint.kpi])
        half = 1.96 * float(standard_error.get(constraint.kpi, 0.0))
        worst = value + half if constraint.op == "<=" else value - half
        if not constraint.satisfied(worst):
            return False
    return True
