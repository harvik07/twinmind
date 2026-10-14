"""The optimisers, behind one interface: ``optimise(spec, evaluator, budget, seed)``.

All four maximise ``spec.score`` under feasibility-first ordering, and none of
them sees a seed stream — they get an :class:`Evaluator` and call it.

FEASIBILITY-FIRST, NOT A PENALTY. Feasible beats infeasible always; among
infeasible, smaller total violation wins. The alternative, a weighted penalty,
needs a weight, and the right weight depends on units nobody has calibrated:
how many pounds of profit is one point of stockout probability worth? The
lexicographic order sidesteps that, and it also means a goal's reported
objective never silently mixes in a penalty term.

THE SURROGATE SAFETY MARGIN. When searching on the surrogate, a constraint is
tightened by ``margin_std`` times the surrogate's own predicted std for that
KPI. It exists because the surrogate is biased, not just noisy: a plan sitting
exactly on a constraint boundary in surrogate space lands on the wrong side of
it about half the time on the twin. The multiplier is a hyperparameter and is
**tuned on DEV goals only**.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from twinmind.models.surrogate import CATEGORICAL_DIMS, _categorical_levels, scenario_bounds
from twinmind.optim.contracts import EXOGENOUS_DIMS, ObjectiveSpec, SpecError
from twinmind.optim.evaluators import Evaluator
from twinmind.twin.runner import SCENARIO_DIMS, Scenario
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Grid search is only honest when the free space is small. Above this many free
#: controllables the method reports "n/a" rather than silently becoming
#: something else.
GRID_MAX_DIMS = 4

#: Integer-valued Scenario fields.
INTEGER_DIMS: frozenset[str] = frozenset({"shift_count", "supplier_choice"})

#: Candidates every optimiser keeps. ``best_known`` verifies the top 100 of
#: these and ``surrogate_first_verify`` shortlists 20, so 200 is ample.
TOP_K = 200

#: Rows per Sobol draw. A power of two (so the first draw is a balanced
#: ``random_base2`` block) and a multiple of the 2,048-row evaluation batch (so
#: the surrogate sees exactly the batches it saw before chunking). 2**15 rows
#: over all 16 controllables is 4 MiB, against 64 MiB for one 2**19 block.
SOBOL_CHUNK = 2**15


@dataclass
class Candidate:
    """One evaluated point."""

    scenario: Scenario
    kpis: dict[str, float]
    score: float
    violation: float

    @property
    def feasible(self) -> bool:
        """Does it satisfy every hard constraint?"""
        return self.violation <= 0.0


def rank_key(candidate: Candidate) -> tuple[float, float, float]:
    """Sort key implementing feasibility-first ordering (smaller is better)."""
    return (0.0 if candidate.feasible else 1.0, candidate.violation, -candidate.score)


class TopK:
    """The best ``k`` candidates seen so far, in bounded memory.

    Each candidate is tagged with its global evaluation index, and the order is
    ``(rank_key, index)``: feasibility-first, then the earlier evaluation on a
    tie. That is a strict total order, so the top ``k`` of everything is the top
    ``k`` of (the top ``k`` so far, plus what came after) -- trimming as we go
    returns exactly what sorting the full list once would have, and is what
    the old code did implicitly through a stable sort.

    Trimming waits until the buffer holds ``2k`` so a one-at-a-time caller
    (TPE) does not re-sort on every trial.
    """

    def __init__(self, k: int = TOP_K) -> None:
        self.k = k
        self.seen = 0
        self._items: list[tuple[tuple[float, float, float], int, Candidate]] = []

    def add(self, candidates: Sequence[Candidate]) -> None:
        """Merge a batch, in evaluation order."""
        for candidate in candidates:
            self._items.append((rank_key(candidate), self.seen, candidate))
            self.seen += 1
        if len(self._items) >= 2 * self.k:
            self._trim()

    def _trim(self) -> None:
        self._items.sort(key=lambda item: (item[0], item[1]))
        del self._items[self.k :]

    def result(self) -> list[Candidate]:
        """The best ``k``, best first."""
        self._trim()
        return [candidate for _, _, candidate in self._items]


def sobol_chunks(
    d: int, budget: int, seed: int, chunk: int = SOBOL_CHUNK
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """The first ``budget`` points of one scrambled Sobol sequence, ``chunk`` rows at a time.

    Yields ``(global_start, points)``. The concatenation is bitwise identical
    to ``Sobol(d, scramble=True, seed=seed).random_base2(m)[:budget]`` with
    ``2**m >= budget`` -- the single-block draw this replaces -- without ever
    holding more than ``chunk`` rows. A regression test pins that identity.

    WHY THIS DOES NOT TRIP SCIPY'S BALANCE CHECK. ``random_base2(m)`` refuses
    unless the running total ``num_generated + 2**m`` is a power of two, which is
    why the original per-batch code crashed, and why repeated ``random_base2``
    calls cannot be used here either (2**15 + 2**15 + 2**15 is not a power of
    two). ``random(n)`` checks the power-of-two property ONLY when nothing has
    been generated yet. So the first draw is ``random_base2`` -- balanced, and
    silent -- and every later draw is ``random(chunk)``, which continues the same
    sequence from where it stopped. The last chunk is drawn in full and
    truncated, so ``n`` is a power of two on every call regardless.

    A budget that fits in one chunk takes the old path exactly: one block of
    ``2**ceil(log2(budget))``, truncated.
    """
    if chunk <= 0 or chunk & (chunk - 1):
        raise ValueError(f"chunk must be a positive power of two, got {chunk}")
    sampler = stats.qmc.Sobol(d=d, scramble=True, seed=seed)
    m = max(int(math.ceil(math.log2(max(budget, 2)))), 1)
    if 2**m <= chunk:
        yield 0, sampler.random_base2(m=m)[:budget]
        return
    drawn = 0
    while drawn < budget:
        if drawn == 0:
            block = sampler.random_base2(m=int(math.log2(chunk)))
        else:
            block = sampler.random(chunk)
        block = block[: budget - drawn]
        yield drawn, block
        drawn += block.shape[0]


def free_dimensions(spec: ObjectiveSpec) -> list[str]:
    """The dimensions this goal actually searches, in canonical order."""
    free = [d for d in SCENARIO_DIMS if d in spec.free_dims and d not in spec.fixed]
    if not free:
        raise SpecError(
            f"goal {spec.name!r} has no free dimensions left after applying 'fixed'. "
            "Free at least one controllable dimension."
        )
    return free


def build_scenario(spec: ObjectiveSpec, free: Sequence[str], unit: NDArray[np.float64]) -> Scenario:
    """Map a point in the unit cube over ``free`` to a full Scenario.

    Dimensions this goal does not search take their value from ``fixed`` if the
    goal pins them, and otherwise from the Scenario default -- which is the
    current operating policy, so an unsearched dimension means "leave it as it
    is" rather than "set it to something arbitrary".
    """
    bounds = scenario_bounds()
    payload: dict[str, Any] = dict(spec.fixed)
    for index, dim in enumerate(free):
        low, high = bounds[dim]
        value = low + float(np.clip(unit[index], 0.0, 1.0)) * (high - low)
        payload[dim] = int(round(value)) if dim in INTEGER_DIMS else value
    for dim in INTEGER_DIMS:
        if dim in payload:
            payload[dim] = int(round(float(payload[dim])))
    return Scenario(**payload)


def _score_batch(
    spec: ObjectiveSpec,
    scenarios: list[Scenario],
    evaluator: Evaluator,
    margins: dict[str, float] | None = None,
) -> list[Candidate]:
    """Evaluate and wrap a batch, applying any surrogate safety margin."""
    results = evaluator.evaluate(scenarios)
    out: list[Candidate] = []
    for scenario, kpis in zip(scenarios, results, strict=True):
        adjusted = dict(kpis)
        if margins:
            # Tighten the KPI in the direction that makes a constraint harder,
            # so a plan sitting on the boundary in surrogate space is rejected.
            for constraint in spec.constraints:
                if not constraint.hard:
                    continue
                shift = margins.get(constraint.kpi, 0.0)
                if shift:
                    adjusted[constraint.kpi] = (
                        kpis[constraint.kpi] + shift
                        if constraint.op == "<="
                        else kpis[constraint.kpi] - shift
                    )
        out.append(
            Candidate(
                scenario=scenario,
                kpis=kpis,
                score=spec.score(kpis),
                violation=spec.total_violation(adjusted),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Random search (Sobol)
# --------------------------------------------------------------------------- #
def random_search(
    spec: ObjectiveSpec,
    evaluator: Evaluator,
    budget: int,
    seed: int,
    batch_size: int = 2048,
    margins: dict[str, float] | None = None,
    chunk: int | None = None,
) -> list[Candidate]:
    """Scrambled Sobol over the free dimensions, in bounded memory.

    Sobol rather than uniform random: it covers the space far more evenly at the
    same budget, which matters most at the small budgets the twin-only arms use.

    THE SOBOL DRAW HAPPENS EXACTLY ONCE, and that is load-bearing rather than
    stylistic. An earlier version created one sampler and called
    ``random_base2`` once per evaluation batch, which scipy refuses:

        ValueError: The balance properties of Sobol' points require n to be a
        power of 2. 4096 points have been previously generated, then:
        n=4096+2**11=6144.

    The refusal is correct. Sobol's low-discrepancy guarantee holds for prefixes
    of ``2**m`` from the *start* of the sequence; resuming with a second block
    of a different size breaks the balance, so scipy will not do it silently.
    Drawing one block of ``2**ceil(log2(budget))`` and keeping the first
    ``budget`` points preserves the property, because a prefix of a balanced
    block is exactly what the design wants.

    Evaluation is still chunked by ``batch_size`` — that is about not handing
    the surrogate a 500,000-row call, and is independent of how the points were
    drawn.

    This only bit above ``batch_size`` (2,048), so the smoke run, whose largest
    budget was 1,000, passed while the production run died in seconds.

    BOUNDED MEMORY. The one-block draw then held all ``2**19`` points of the
    500,000 ``best_known`` budget at once -- 64 MiB over 16 dimensions -- and six
    ``best_known`` cells doing that concurrently on an 8 GB machine ran it out
    of memory. The SAME sequence is now drawn ``chunk`` rows at a time by
    :func:`sobol_chunks` (bitwise identical, see there), evaluated in the same
    ``batch_size`` batches as before, and merged into a running :class:`TopK`.
    Peak memory is one chunk plus ``TOP_K`` candidates, whatever the budget.
    Results are unchanged: same points, same batches, same order.
    """
    chunk = SOBOL_CHUNK if chunk is None else chunk
    if chunk % batch_size and budget > chunk:
        # Batches must not straddle chunks, or the surrogate would see
        # different batch compositions than the single-block code did.
        raise ValueError(f"chunk ({chunk}) must be a multiple of batch_size ({batch_size})")
    free = free_dimensions(spec)
    best = TopK()
    for _, points in sobol_chunks(len(free), budget, seed, chunk):
        for start in range(0, points.shape[0], batch_size):
            block = points[start : start + batch_size]
            scenarios = [build_scenario(spec, free, row) for row in block]
            best.add(_score_batch(spec, scenarios, evaluator, margins))
    return best.result()


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def grid_search(
    spec: ObjectiveSpec,
    evaluator: Evaluator,
    budget: int,
    seed: int,
    margins: dict[str, float] | None = None,
    batch_size: int = 2048,
) -> list[Candidate] | None:
    """A full factorial grid, or ``None`` when the free space is too large.

    Returning ``None`` rather than quietly falling back to random is deliberate:
    the benchmark reports "n/a" for that goal, so a reader can see the method did
    not apply instead of seeing a number produced by a different algorithm under
    the grid's name.
    """
    del seed
    free = free_dimensions(spec)
    if len(free) > GRID_MAX_DIMS:
        logger.info(
            "grid: n/a for %r -- %d free controllables exceeds the limit of %d",
            spec.name,
            len(free),
            GRID_MAX_DIMS,
        )
        return None

    per_dim = max(2, int(round(budget ** (1.0 / len(free)))))
    axis = np.linspace(0.0, 1.0, per_dim)
    shape = (per_dim,) * len(free)
    total = per_dim ** len(free)
    # The grid is walked in C order -- the order ``meshgrid(indexing="ij")``
    # ravels to -- one batch at a time, rather than materialising the whole
    # mesh and scoring it in a single call.
    best = TopK()
    for start in range(0, total, batch_size):
        index = np.arange(start, min(start + batch_size, total))
        mesh = axis[np.stack(np.unravel_index(index, shape), axis=1)]
        scenarios = [build_scenario(spec, free, row) for row in mesh]
        best.add(_score_batch(spec, scenarios, evaluator, margins))
    return best.result()


# --------------------------------------------------------------------------- #
# Optuna TPE
# --------------------------------------------------------------------------- #
def tpe_search(
    spec: ObjectiveSpec,
    evaluator: Evaluator,
    budget: int,
    seed: int,
    margins: dict[str, float] | None = None,
) -> list[Candidate]:
    """Optuna's TPE over the free dimensions.

    Optuna is told a single scalar that already encodes feasibility-first: a
    feasible point scores its objective, an infeasible one scores far below any
    feasible point. That keeps the lexicographic ordering without handing TPE a
    constrained problem it would otherwise need a penalty weight for.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    free = free_dimensions(spec)
    found = TopK()

    def objective(trial: Any) -> float:
        unit = np.array([trial.suggest_float(dim, 0.0, 1.0) for dim in free], dtype="float64")
        scenario = build_scenario(spec, free, unit)
        candidate = _score_batch(spec, [scenario], evaluator, margins)[0]
        found.add([candidate])
        # Infeasible points are pushed below every feasible one by a constant
        # offset plus their violation, which preserves the ordering without a
        # tunable weight.
        return candidate.score if candidate.feasible else -1e12 - candidate.violation

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return found.result()


# --------------------------------------------------------------------------- #
# CMA-ES
# --------------------------------------------------------------------------- #
def cmaes_search(
    spec: ObjectiveSpec,
    evaluator: Evaluator,
    budget: int,
    seed: int,
    margins: dict[str, float] | None = None,
) -> list[Candidate]:
    """CMA-ES over the continuous free dimensions.

    THE CATEGORICAL. CMA-ES searches a continuous space and has no notion of an
    unordered category, so when ``supplier_choice`` is free the budget is
    **split evenly across the three suppliers** and a separate CMA-ES runs with
    each pinned. That is more expensive than relaxing it to a continuous
    coordinate and rounding, and it is correct rather than nearly-correct: a
    rounded relaxation lets the optimiser sit between two suppliers, where the
    objective is a fiction.

    ``shift_count`` stays in the continuous space and is rounded, because 1, 2,
    3 shifts genuinely is ordered and the rounded point is a real plan.
    """
    import cma

    free = free_dimensions(spec)
    categorical = [d for d in free if d in CATEGORICAL_DIMS]
    if categorical:
        results: list[Candidate] = []
        levels = _categorical_levels(categorical[0])
        share = max(budget // len(levels), 1)
        for level in levels:
            sub = ObjectiveSpec(
                name=f"{spec.name}::{categorical[0]}={level}",
                objective=spec.objective,
                sense=spec.sense,
                constraints=spec.constraints,
                free_dims=tuple(d for d in spec.free_dims if d != categorical[0]),
                fixed={**spec.fixed, categorical[0]: float(level)},
                context_id=spec.context_id,
            )
            results.extend(cmaes_search(sub, evaluator, share, seed + level, margins))
        results.sort(key=rank_key)
        return results[:200]

    # A running top-K, not a list of every evaluation: at the 20,000 sf_cmaes
    # budget the full list was 20,000 Candidates, each a Scenario plus a KPI dict.
    found = TopK()
    options = {
        "seed": int(seed) % (2**31 - 1) + 1,
        "bounds": [0.0, 1.0],
        "verbose": -9,
        "maxfevals": budget,
    }
    strategy = cma.CMAEvolutionStrategy([0.5] * len(free), 0.25, options)
    while not strategy.stop() and found.seen < budget:
        points = strategy.ask()
        scenarios = [build_scenario(spec, free, np.asarray(p)) for p in points]
        batch = _score_batch(spec, scenarios, evaluator, margins)
        found.add(batch)
        # CMA-ES minimises, so the sign is flipped; infeasible points get the
        # same lexicographic treatment as everywhere else.
        strategy.tell(
            points,
            [(-c.score if c.feasible else 1e12 + c.violation) for c in batch],
        )
    return found.result()


#: Name -> callable, for the benchmark and the tests.
OPTIMISERS: dict[str, Callable[..., Any]] = {
    "random": random_search,
    "grid": grid_search,
    "tpe": tpe_search,
    "cmaes": cmaes_search,
}


def exogenous_in_free(spec: ObjectiveSpec) -> list[str]:
    """Exogenous dimensions this goal has explicitly freed, for the record."""
    return [d for d in spec.free_dims if d in EXOGENOUS_DIMS]
