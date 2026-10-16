"""Critic: fixed rules, no LLM. Thresholds live in configs/agent.yaml.

Flags, each independent:

(a) ``weak_kpi_constraint`` -- a hard constraint is on a KPI the surrogate
    predicts poorly (``unplanned_failures`` R^2 0.733, ``downtime_hours`` 0.907),
    so the search was steered by its weakest predictions.
(b) ``not_robustly_feasible`` -- the re-evaluated plan's 95% interval is not
    wholly inside the feasible region.
(c) ``novel_plan`` -- novelty above the 95th percentile of the T1 scenarios
    (calibrated on T1, not the OOD set): the surrogate is extrapolating.
(d) ``no_improvement`` -- the paired objective difference's CI does not lie
    above 0 (and, if configured, the plan does not restore feasibility the
    baseline lacked). The plan is then NOT presented as a recommendation.
(e) ``infeasible_unverified`` -- no shortlisted plan verified feasible.

Retry rule (at most ``max_retries`` = 1 per goal): if the plan is not robustly
feasible, first swap to the best already-verified candidate whose verification
CI lies inside the feasible region -- no new search or verification runs; the
swapped plan is re-evaluated so its reported numbers stay independent. If no
such candidate exists, one more explorer round with the margin widened by
``retry_margin_increment_std``. Otherwise accept with the flags.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from twinmind.agents.tools import best_robust_candidate
from twinmind.optim.contracts import ObjectiveSpec

#: The message for rule (d), verbatim from the spec.
NO_IMPROVEMENT = "no improvement over current policy"


@dataclass(frozen=True)
class CriticVerdict:
    """Flags plus what to do next."""

    flags: list[dict[str, Any]]
    action: str  # "accept" | "swap" | "explore_again"
    recommended: bool
    swap_to: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    recommendation_reason: str = ""


def _flag(code: str, message: str, **detail: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "detail": detail}


def flags_for(
    spec: ObjectiveSpec,
    evaluation: dict[str, Any],
    exploration_status: str,
    novelty: float,
    weak_kpis: Sequence[str],
    novelty_threshold: float,
    restoration_counts: bool,
) -> list[dict[str, Any]]:
    """Rules (a)-(e). Pure function of its inputs."""
    flags = []
    weak = [c for c in spec.constraints if c.hard and c.kpi in weak_kpis]
    if weak:
        names = ", ".join(f"{c.kpi} {c.op} {c.threshold:g}" for c in weak)
        flags.append(
            _flag(
                "weak_kpi_constraint",
                f"constraint relies on a weakly predicted KPI ({names})",
                kpis=[c.kpi for c in weak],
            )
        )
    if not evaluation["feasible_robust"]:
        flags.append(
            _flag(
                "not_robustly_feasible",
                "the plan's 95% interval is not wholly inside the feasible region",
                feasible_point=evaluation["feasible_point"],
            )
        )
    if novelty > novelty_threshold:
        flags.append(
            _flag(
                "novel_plan",
                f"plan novelty {novelty:.3f} is above the T1 95th percentile "
                f"{novelty_threshold:.3f}; the surrogate is extrapolating",
                novelty=novelty,
                threshold=novelty_threshold,
            )
        )
    diff = evaluation["objective"]["diff"]
    beats = diff["ci_low"] > 0
    restores = (
        restoration_counts
        and evaluation["feasible_point"]
        and not evaluation["baseline_feasible_point"]
    )
    if not (beats or restores):
        flags.append(
            _flag(
                "no_improvement",
                NO_IMPROVEMENT,
                objective_diff=diff["mean"],
                ci_low=diff["ci_low"],
                ci_high=diff["ci_high"],
            )
        )
    if exploration_status == "infeasible_unverified":
        flags.append(
            _flag(
                "infeasible_unverified",
                "no shortlisted plan verified feasible on the twin",
            )
        )
    return flags


#: The closed set of reasons, so the explainer's stated reason can be checked.
REASON_NO_FEASIBLE = "no feasible plan was found"
REASON_IMPROVES = "improves the objective over the current policy"
REASON_RESTORES = "restores {constraints} that the current policy breaks"


def recommendation_reason(spec: ObjectiveSpec, evaluation: dict[str, Any], codes: set[str]) -> str:
    """Why the plan is, or is not, recommended -- decided here, never by the LLM.

    Precedence: no feasible plan; no improvement; a significant objective gain;
    otherwise the plan restores constraints the baseline violates (the only
    other way rule (d) lets a plan through).
    """
    if not evaluation["feasible_point"]:
        return REASON_NO_FEASIBLE
    if "no_improvement" in codes:
        return NO_IMPROVEMENT
    if evaluation["objective"]["diff"]["ci_low"] > 0:
        return REASON_IMPROVES
    base = {k: v["baseline"]["mean"] for k, v in evaluation["kpis"].items()}
    broken = [
        f"the {c.kpi} {c.op} {c.threshold:g} constraint"
        for c in spec.constraints
        if c.hard and not c.satisfied(base[c.kpi])
    ]
    return REASON_RESTORES.format(constraints=" and ".join(broken) or "feasibility")


def critique(
    spec: ObjectiveSpec,
    evaluation: dict[str, Any],
    chosen: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    exploration_status: str,
    novelty: float,
    retries_used: int,
    config: Any,
) -> CriticVerdict:
    """Flags, the recommendation decision, and the retry action."""
    flags = flags_for(
        spec,
        evaluation,
        exploration_status,
        novelty,
        config.weak_kpis,
        config.novelty_threshold,
        config.feasibility_restoration_counts_as_improvement,
    )
    codes = {f["code"] for f in flags}
    recommended = (
        evaluation["feasible_point"]
        and "no_improvement" not in codes
        and "infeasible_unverified" not in codes
    )
    reason = recommendation_reason(spec, evaluation, codes)
    if not evaluation["feasible_robust"] and retries_used < config.max_retries:
        swap = best_robust_candidate(candidates, exclude=chosen)
        if swap is not None:
            return CriticVerdict(
                flags,
                "swap",
                recommended,
                swap_to=swap,
                recommendation_reason=reason,
                notes=["swapped to the best verified candidate with a robust verification CI"],
            )
        return CriticVerdict(
            flags,
            "explore_again",
            recommended,
            notes=["no robust verified candidate; one more explorer round, margin widened"],
            recommendation_reason=reason,
        )
    return CriticVerdict(flags, "accept", recommended, recommendation_reason=reason)
