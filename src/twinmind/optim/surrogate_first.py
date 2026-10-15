"""Surrogate-first search with twin verification — the method the agent calls.

THE SHAPE OF THE PROBLEM. The surrogate is about 59,000x faster than a 30-rep
twin evaluation and reproduces the twin's *ranking* almost perfectly (median
Spearman 0.9815 over 40 contexts, against a pessimistic 0.9865 ceiling). But it
is biased, and two of its eight KPIs are materially wrong -- ``unplanned_failures``
at R2 0.733 and ``downtime_hours`` at 0.907. So it is trusted to **shortlist**
and never to **decide**.

Hence: search wide and cheap on the surrogate, then spend the twin budget only
on a short list, then re-evaluate the single chosen plan on an independent seed
stream so the number reported is not a maximum over noise.

THE THREE THINGS THAT ARE EASY TO GET WRONG HERE.

**The shortlist must be diverse.** An optimiser's top 20 are usually 20 nearly
identical points in one basin. Verifying them spends the entire twin budget
learning the same thing twenty times. ``min_distance`` enforces a minimum
separation in normalised scenario space, so the shortlist covers distinct plans.

**Infeasibility must never pass silently.** If no verified plan is feasible, the
search retries with a wider surrogate safety margin, and if that still fails it
returns ``status="infeasible_unverified"`` with the best effort attached. It
never returns a plan that looks fine because the constraint check was skipped.

**The final number must come from a stream the search never touched.** The
chosen plan is re-evaluated at 30 reps on "reeval". The gap between what the
surrogate predicted and what that re-evaluation found is reported per KPI, and
that gap is the signal a future agent uses to decide whether to trust the
surrogate in a region.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from twinmind.models.surrogate import KPI_ORDER
from twinmind.optim.contracts import ObjectiveSpec, Plan, rank_plans
from twinmind.optim.evaluators import (
    EvalCost,
    SurrogateEvaluator,
    TwinEvaluator,
    feasible_robust,
)
from twinmind.optim.novelty import NoveltyScorer, normalise_scenarios
from twinmind.optim.optimisers import OPTIMISERS, Candidate, free_dimensions
from twinmind.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from twinmind.twin.context import TwinContext
    from twinmind.twin.runner import Scenario

logger = get_logger(__name__)

#: Shortlist size sent to the twin.
DEFAULT_K = 20

#: Minimum separation between shortlisted plans, in normalised scenario units.
#: Without it an optimiser's top 20 are 20 near-identical points in one basin
#: and the twin budget is spent learning the same thing twenty times.
MIN_DISTANCE = 0.15

#: Extra rounds allowed when nothing verified feasible.
MAX_ROUNDS = 2

#: Surrogate safety margin in predicted-std units. TUNED ON DEV GOALS ONLY.
DEFAULT_MARGIN_STD = 1.0


@dataclass
class VerifyConfig:
    """Knobs for :func:`surrogate_first_verify`."""

    optimiser: str = "tpe"
    budget: int = 3000
    k: int = DEFAULT_K
    min_distance: float = MIN_DISTANCE
    max_rounds: int = MAX_ROUNDS
    margin_std: float = DEFAULT_MARGIN_STD
    verify_reps: int = 10
    final_reps: int = 30
    novelty_active: bool = True
    #: False hands the chosen plan back with its SEARCH-stream verification and
    #: skips step 6; the caller then owns the independent re-evaluation. The
    #: agent layer does, pairing it with the baseline on the same seeds. The
    #: benchmark leaves this True.
    reevaluate: bool = True


def select_diverse(candidates: list[Candidate], k: int, min_distance: float) -> list[Candidate]:
    """Greedy farthest-point selection down the ranked list.

    Walks the candidates best-first and keeps one only if it is at least
    ``min_distance`` from everything already kept. If that yields fewer than
    ``k``, the gap is filled from the remaining best — a short, diverse list is
    better than a long, redundant one, but an empty slot helps nobody.
    """
    if not candidates:
        return []
    kept: list[Candidate] = []
    kept_points: list[np.ndarray] = []
    for candidate in candidates:
        point = normalise_scenarios(np.array([candidate.scenario.to_vector()]))[0]
        if kept_points:
            distance = min(float(np.linalg.norm(point - p)) for p in kept_points)
            if distance < min_distance:
                continue
        kept.append(candidate)
        kept_points.append(point)
        if len(kept) >= k:
            break

    if len(kept) < k:
        chosen = {id(c) for c in kept}
        for candidate in candidates:
            if id(candidate) in chosen:
                continue
            kept.append(candidate)
            if len(kept) >= k:
                break
    return kept[:k]


def surrogate_margins(
    spec: ObjectiveSpec,
    surrogate_eval: SurrogateEvaluator,
    scenarios: list[Scenario],
    margin_std: float,
) -> dict[str, float]:
    """Per-KPI safety margin, in the surrogate's own predicted std units.

    Measured on a sample of the search space rather than assumed, because the
    right margin for ``stockout_probability`` (std of order 0.03) and for
    ``profit`` (std of order £17,000) are not the same number.
    """
    if margin_std <= 0 or not scenarios:
        return {}
    output = surrogate_eval.predict_full(scenarios)
    total = np.sqrt(output.aleatoric_std**2 + output.epistemic_std**2)
    return {
        constraint.kpi: margin_std * float(np.median(total[:, KPI_ORDER.index(constraint.kpi)]))
        for constraint in spec.constraints
        if constraint.hard
    }


def surrogate_first_verify(
    spec: ObjectiveSpec,
    context: TwinContext,
    surrogate: Any,
    resources: Any,
    seed: int,
    config: VerifyConfig | None = None,
    novelty: NoveltyScorer | None = None,
) -> tuple[list[Plan], dict[str, Any]]:
    """Search on the surrogate, verify on the twin, re-evaluate independently.

    Returns ``(ranked plans, trace)``. The first plan is the recommendation; its
    ``status`` says whether it is verified-feasible.
    """
    cfg = config or VerifyConfig()
    started = time.perf_counter()
    surrogate_eval = SurrogateEvaluator(surrogate=surrogate, context=context)
    trace: dict[str, Any] = {"rounds": [], "optimiser": cfg.optimiser, "budget": cfg.budget}

    free = free_dimensions(spec)
    search = OPTIMISERS[cfg.optimiser]

    chosen: Plan | None = None
    all_plans: list[Plan] = []
    margin_std = cfg.margin_std
    # Every twin evaluator's own measured cost, merged. The run COUNT used to be
    # rebuilt by hand from the rounds and the CPU-seconds were dropped, so the
    # benchmark's surrogate-first cells all reported cpu_seconds = 0.
    twin_cost = EvalCost()

    for round_index in range(cfg.max_rounds + 1):
        # 1. Wide, cheap search on the surrogate.
        probe = search(spec, surrogate_eval, min(cfg.budget, 256), seed + round_index)
        margins = surrogate_margins(
            spec, surrogate_eval, [c.scenario for c in (probe or [])[:64]], margin_std
        )
        candidates = search(spec, surrogate_eval, cfg.budget, seed + round_index, margins=margins)
        if candidates is None:
            raise RuntimeError(f"optimiser {cfg.optimiser!r} returned no candidates")

        # 2. A DISTINCT shortlist, not 20 points in the same basin.
        shortlist = select_diverse(candidates, cfg.k, cfg.min_distance)

        # 3. Novelty, advisory or active depending on the A0 gate.
        novelty_scores = (
            novelty.score(np.array([c.scenario.to_vector() for c in shortlist]))
            if novelty is not None and shortlist
            else np.full(len(shortlist), np.nan)
        )

        # 4. Verify the whole shortlist on the twin, on the SEARCH stream.
        twin_eval = TwinEvaluator(
            resources=resources,
            context=context,
            goal_seed=seed + round_index,
            n_reps=cfg.verify_reps,
            stream="search",
        )
        verified = twin_eval.evaluate_with_spread([c.scenario for c in shortlist])
        twin_cost = twin_cost.merge(twin_eval.cost)

        plans: list[Plan] = []
        for candidate, (mean, sem), score in zip(shortlist, verified, novelty_scores, strict=True):
            margins_out = {
                f"{c.kpi}{c.op}{c.threshold}": c.margin(float(mean[c.kpi]))
                for c in spec.constraints
                if c.hard
            }
            plans.append(
                Plan(
                    scenario=candidate.scenario,
                    surrogate_pred=candidate.kpis,
                    twin_verified={"mean": mean, "sem": sem},
                    feasible_point=spec.feasible(mean),
                    feasible_robust=feasible_robust(spec, mean, sem),
                    constraint_margins=margins_out,
                    surrogate_twin_gap={kpi: candidate.kpis[kpi] - mean[kpi] for kpi in KPI_ORDER},
                    novelty=float(score),
                    objective_value=spec.score(mean),
                    status="ok" if spec.feasible(mean) else "infeasible_unverified",
                )
            )

        plans = rank_plans(plans)
        all_plans = plans
        feasible = [p for p in plans if p.feasible_point]
        trace["rounds"].append(
            {
                "round": round_index,
                "margin_std": margin_std,
                "n_candidates": len(candidates),
                "n_shortlist": len(shortlist),
                "n_feasible": len(feasible),
                "margins": margins,
            }
        )
        logger.info(
            "  round %d (margin %.2f std): %d shortlisted, %d verified feasible",
            round_index,
            margin_std,
            len(shortlist),
            len(feasible),
        )

        if feasible:
            chosen = feasible[0]
            break
        # 5. Nothing feasible: widen the margin and try again, never pass silently.
        margin_std += 1.0

    if chosen is None:
        best_effort = all_plans[0] if all_plans else None
        if best_effort is None:
            return [], {**trace, "status": "no_candidate"}
        logger.info(
            "  no verified-feasible plan after %d rounds -> status=infeasible_unverified",
            cfg.max_rounds + 1,
        )
        cost = surrogate_eval.cost.merge(twin_cost)
        cost.wall_seconds = time.perf_counter() - started
        return (
            [
                Plan(
                    scenario=best_effort.scenario,
                    surrogate_pred=best_effort.surrogate_pred,
                    twin_verified=best_effort.twin_verified,
                    feasible_point=False,
                    feasible_robust=False,
                    constraint_margins=best_effort.constraint_margins,
                    surrogate_twin_gap=best_effort.surrogate_twin_gap,
                    novelty=best_effort.novelty,
                    cost=cost.to_dict(),
                    status="infeasible_unverified",
                    objective_value=best_effort.objective_value,
                ),
                *all_plans[1:],
            ],
            {**trace, "status": "infeasible_unverified"},
        )

    if not cfg.reevaluate:
        cost = surrogate_eval.cost.merge(twin_cost)
        cost.wall_seconds = time.perf_counter() - started
        trace["status"] = "ok"
        trace["free_dims"] = free
        trace["reevaluated"] = False
        rest = [p for p in all_plans if p is not chosen]
        return [Plan(**{**chosen.__dict__, "cost": cost.to_dict()}), *rest], trace

    # 6. Independent re-evaluation of the chosen plan, on a stream the search
    #    never touched. Without this the reported value is a maximum over the
    #    search seeds -- the winner's curse.
    final_eval = TwinEvaluator(
        resources=resources,
        context=context,
        goal_seed=seed,
        n_reps=cfg.final_reps,
        stream="reeval",
    )
    (final_mean, final_sem) = final_eval.evaluate_with_spread([chosen.scenario])[0]

    cost = surrogate_eval.cost.merge(twin_cost.merge(final_eval.cost))
    cost.wall_seconds = time.perf_counter() - started

    final_plan = Plan(
        scenario=chosen.scenario,
        surrogate_pred=chosen.surrogate_pred,
        twin_verified={"mean": final_mean, "sem": final_sem},
        feasible_point=spec.feasible(final_mean),
        feasible_robust=feasible_robust(spec, final_mean, final_sem),
        constraint_margins={
            f"{c.kpi}{c.op}{c.threshold}": c.margin(float(final_mean[c.kpi]))
            for c in spec.constraints
            if c.hard
        },
        surrogate_twin_gap={kpi: chosen.surrogate_pred[kpi] - final_mean[kpi] for kpi in KPI_ORDER},
        novelty=chosen.novelty,
        cost=cost.to_dict(),
        status="ok" if spec.feasible(final_mean) else "infeasible_unverified",
        objective_value=spec.score(final_mean),
    )
    trace["status"] = final_plan.status
    trace["free_dims"] = free
    logger.info(
        "  chosen plan: objective %.1f, feasible %s (robust %s), profit gap %.0f",
        final_plan.objective_value,
        final_plan.feasible_point,
        final_plan.feasible_robust,
        final_plan.surrogate_twin_gap.get("profit", float("nan")),
    )
    return [final_plan, *all_plans[1:]], trace


def surrogate_only(
    spec: ObjectiveSpec,
    context: TwinContext,
    surrogate: Any,
    resources: Any,
    seed: int,
    config: VerifyConfig | None = None,
) -> tuple[list[Plan], dict[str, Any]]:
    """The ablation: take the surrogate's single best plan with NO verification.

    Exists to measure what verification is worth. Its chosen plan is still
    re-evaluated on "reeval" -- otherwise there would be no honest number to
    compare -- but the *choice* was made on the surrogate alone, so the
    surrogate-minus-twin gap on that plan is the winner's curse, isolated.
    """
    cfg = config or VerifyConfig()
    started = time.perf_counter()
    surrogate_eval = SurrogateEvaluator(surrogate=surrogate, context=context)
    search = OPTIMISERS[cfg.optimiser]
    candidates = search(spec, surrogate_eval, cfg.budget, seed)
    if not candidates:
        return [], {"status": "no_candidate", "method": "surrogate_only"}

    best = candidates[0]
    final_eval = TwinEvaluator(
        resources=resources,
        context=context,
        goal_seed=seed,
        n_reps=cfg.final_reps,
        stream="reeval",
    )
    (mean, sem) = final_eval.evaluate_with_spread([best.scenario])[0]

    cost = surrogate_eval.cost.merge(final_eval.cost)
    cost.wall_seconds = time.perf_counter() - started
    plan = Plan(
        scenario=best.scenario,
        surrogate_pred=best.kpis,
        twin_verified={"mean": mean, "sem": sem},
        feasible_point=spec.feasible(mean),
        feasible_robust=feasible_robust(spec, mean, sem),
        constraint_margins={
            f"{c.kpi}{c.op}{c.threshold}": c.margin(float(mean[c.kpi]))
            for c in spec.constraints
            if c.hard
        },
        surrogate_twin_gap={kpi: best.kpis[kpi] - mean[kpi] for kpi in KPI_ORDER},
        cost=cost.to_dict(),
        status="ok" if spec.feasible(mean) else "infeasible_unverified",
        objective_value=spec.score(mean),
    )
    return [plan], {"status": plan.status, "method": "surrogate_only"}
