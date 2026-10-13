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
from collections.abc import Callable, Sequence
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
) -> list[Candidate]:
    """Scrambled Sobol over the free dimensions.

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
    """
    free = free_dimensions(spec)
    sampler = stats.qmc.Sobol(d=len(free), scramble=True, seed=seed)
    m = max(int(math.ceil(math.log2(max(budget, 2)))), 1)
    points = sampler.random_base2(m=m)[:budget]

    best: list[Candidate] = []
    for start in range(0, len(points), batch_size):
        block = points[start : start + batch_size]
        scenarios = [build_scenario(spec, free, row) for row in block]
        best.extend(_score_batch(spec, scenarios, evaluator, margins))
        best.sort(key=rank_key)
        best = best[:200]
    return best


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def grid_search(
    spec: ObjectiveSpec,
    evaluator: Evaluator,
    budget: int,
    seed: int,
    margins: dict[str, float] | None = None,
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
    axes = [np.linspace(0.0, 1.0, per_dim) for _ in free]
    mesh = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    scenarios = [build_scenario(spec, free, row) for row in mesh]
    candidates = _score_batch(spec, scenarios, evaluator, margins)
    candidates.sort(key=rank_key)
    return candidates[:200]


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
    found: list[Candidate] = []

    def objective(trial: Any) -> float:
        unit = np.array([trial.suggest_float(dim, 0.0, 1.0) for dim in free], dtype="float64")
        scenario = build_scenario(spec, free, unit)
        candidate = _score_batch(spec, [scenario], evaluator, margins)[0]
        found.append(candidate)
        # Infeasible points are pushed below every feasible one by a constant
        # offset plus their violation, which preserves the ordering without a
        # tunable weight.
        return candidate.score if candidate.feasible else -1e12 - candidate.violation

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    found.sort(key=rank_key)
    return found[:200]


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

    found: list[Candidate] = []
    options = {
        "seed": int(seed) % (2**31 - 1) + 1,
        "bounds": [0.0, 1.0],
        "verbose": -9,
        "maxfevals": budget,
    }
    strategy = cma.CMAEvolutionStrategy([0.5] * len(free), 0.25, options)
    while not strategy.stop() and len(found) < budget:
        points = strategy.ask()
        scenarios = [build_scenario(spec, free, np.asarray(p)) for p in points]
        batch = _score_batch(spec, scenarios, evaluator, margins)
        found.extend(batch)
        # CMA-ES minimises, so the sign is flipped; infeasible points get the
        # same lexicographic treatment as everywhere else.
        strategy.tell(
            points,
            [(-c.score if c.feasible else 1e12 + c.violation) for c in batch],
        )
    found.sort(key=rank_key)
    return found[:200]


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
