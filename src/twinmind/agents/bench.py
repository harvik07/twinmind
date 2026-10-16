"""Agent-benchmark machinery that must be testable offline.

The ``twin_only`` arm runs CMA-ES on the twin for EXACTLY the wall-clock the
agent spent on the same goal and seed. CMA-ES has an evaluation budget, not a
time budget, so it is driven through :class:`TimeBudgetEvaluator`, which raises
:class:`TimeUp` once the deadline passes; if CMA-ES converges before the
deadline it is restarted with a new seed, so the arm uses all of its time.
Every evaluated point is recorded, and the best by feasibility-first order is
the arm's plan -- no point is lost to the interruption.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from twinmind.optim.contracts import ObjectiveSpec
from twinmind.twin.runner import Scenario


class TimeUp(Exception):  # noqa: N818 - a control-flow signal, not an error
    """The arm's wall-clock budget is spent."""


@dataclass
class TimeBudgetEvaluator:
    """Wraps an evaluator; refuses work after ``deadline`` and records every point."""

    inner: Any
    deadline: float
    clock: Callable[[], float] = time.monotonic
    points: list[tuple[Scenario, dict[str, float]]] = field(default_factory=list)

    def evaluate(self, scenarios: Sequence[Scenario]) -> list[dict[str, float]]:
        """Evaluate unless the time is up."""
        if self.clock() >= self.deadline:
            raise TimeUp
        rows = self.inner.evaluate(scenarios)
        self.points.extend(zip(scenarios, rows, strict=True))
        return rows

    @property
    def cost(self) -> Any:
        """Passthrough, for the optimisers' bookkeeping."""
        return self.inner.cost


def best_point(
    spec: ObjectiveSpec, points: Sequence[tuple[Scenario, dict[str, float]]]
) -> tuple[Scenario, dict[str, float]] | None:
    """Feasibility-first: feasible beats infeasible, then smaller violation, then score."""
    if not points:
        return None
    return min(
        points,
        key=lambda p: (
            0 if spec.feasible(p[1]) else 1,
            spec.total_violation(p[1]),
            -spec.score(p[1]),
        ),
    )


@dataclass(frozen=True)
class TwinOnlyOutcome:
    """What a time-boxed twin-only search produced."""

    scenario: Scenario | None
    search_kpis: dict[str, float] | None
    feasible_on_search: bool
    evaluations: int
    restarts: int
    seconds: float
    note: str = ""


def twin_only_timeboxed(
    spec: ObjectiveSpec,
    evaluator: Any,
    seconds: float,
    seed: int,
    clock: Callable[[], float] = time.monotonic,
    search: Callable[..., Any] | None = None,
) -> TwinOnlyOutcome:
    """CMA-ES on ``evaluator`` for ``seconds`` of wall-clock, restarting on convergence."""
    if search is None:
        from twinmind.optim.optimisers import cmaes_search

        search = cmaes_search
    from twinmind.optim.contracts import SpecError
    from twinmind.optim.optimisers import free_dimensions

    started = clock()
    wrapper = TimeBudgetEvaluator(evaluator, deadline=started + seconds, clock=clock)
    try:
        free_dimensions(spec)
    except SpecError:
        # Nothing to search: the only admissible plan is the current one. It is
        # evaluated once on the search stream so "recommended" (feasible on the
        # arm's own search stream) still has an answer. ObjectiveSpec refuses
        # such a goal, so this is a guard, not a path the benchmark takes.
        base = spec.base_scenario()
        kpis = wrapper.evaluate([base])[0]
        return TwinOnlyOutcome(
            scenario=base,
            search_kpis=kpis,
            feasible_on_search=bool(spec.feasible(kpis)),
            evaluations=1,
            restarts=0,
            seconds=clock() - started,
            note="no free dimensions: the current policy is the only plan",
        )
    restarts = 0
    while clock() < wrapper.deadline:
        before = len(wrapper.points)
        try:
            search(spec, wrapper, 10**9, seed * 1000 + restarts)
        except TimeUp:
            break
        restarts += 1
        if len(wrapper.points) == before:  # pragma: no cover - a search that did nothing
            break
    best = best_point(spec, wrapper.points)
    return TwinOnlyOutcome(
        scenario=best[0] if best else None,
        search_kpis=best[1] if best else None,
        feasible_on_search=bool(best and spec.feasible(best[1])),
        evaluations=len(wrapper.points),
        restarts=restarts,
        seconds=clock() - started,
    )
