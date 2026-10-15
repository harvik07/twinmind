"""Deterministic tools for the agent graph. None of these consults an LLM.

* :class:`TwinToolkit` -- the real explorer and evaluator: surrogate-first
  search with twin verification, then a PAIRED 30-rep re-evaluation of the
  plan and the current-policy baseline on the independent "reeval" stream.
* :func:`closest_achievable` -- for an infeasible goal, how close the verified
  candidates came to each violated constraint, offered as options.
* :func:`build_facts`, :func:`ungrounded_numbers`, :func:`template_explanation`
  -- the explainer's input, its numeric grounding check, and its fallback.

The toolkit is a Protocol so the graph can be tested offline with a fake.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from twinmind.optim.contracts import ObjectiveSpec, Plan
from twinmind.twin.runner import KPI_NAMES, SCENARIO_DIMS, Scenario


# --------------------------------------------------------------------------- #
# Exploration and evaluation records
# --------------------------------------------------------------------------- #
def candidate_record(plan: Plan, origin: str) -> dict[str, Any]:
    """A verified candidate with its verification CI (search stream)."""
    mean = plan.twin_verified.get("mean", {})
    sem = plan.twin_verified.get("sem", {})
    return {
        "scenario": plan.scenario.model_dump(),
        "objective": float(plan.objective_value),
        "verification_mean": {k: float(v) for k, v in mean.items()},
        "verification_sem": {k: float(v) for k, v in sem.items()},
        "feasible_point": bool(plan.feasible_point),
        "feasible_robust": bool(plan.feasible_robust),
        "status": plan.status,
        "surrogate_pred": {k: float(v) for k, v in plan.surrogate_pred.items()},
        "origin": origin,
    }


@dataclass
class Exploration:
    """What one explorer round produced."""

    chosen: dict[str, Any]
    candidates: list[dict[str, Any]]
    status: str
    margin_std: float
    twin_runs: int = 0
    surrogate_evals: int = 0
    wall_seconds: float = 0.0


class Toolkit(Protocol):
    """What the graph's deterministic nodes need."""

    def context_summary(self, context_id: int) -> str:
        """A short description of the context, for the planner prompt."""
        ...

    def explore(self, spec: ObjectiveSpec, margin_std: float, seed: int) -> Exploration:
        """Surrogate-first search with twin verification (search stream)."""
        ...

    def evaluate(self, spec: ObjectiveSpec, scenario: dict[str, Any]) -> dict[str, Any]:
        """Paired 30-rep re-evaluation of ``scenario`` and the baseline (reeval stream)."""
        ...

    def novelty(self, scenario: dict[str, Any]) -> float:
        """NoveltyScorer score of a scenario."""
        ...


def _t_half_width(values: np.ndarray, confidence: float) -> float:
    from scipy import stats

    n = len(values)
    if n < 2:
        return float("nan")
    sem = float(np.std(values, ddof=1) / math.sqrt(n))
    return float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * sem)


def _summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    mean = float(np.mean(values))
    half = _t_half_width(values, confidence)
    return {
        "mean": mean,
        "sem": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


def paired_evaluation(
    spec: ObjectiveSpec,
    plan_rows: np.ndarray,
    baseline_rows: np.ndarray,
    confidence: float,
    goal_seed: int,
) -> dict[str, Any]:
    """KPI CIs for plan and baseline, and their PAIRED difference, from per-rep rows.

    Row ``i`` of both arrays is the same replication seed, so the difference is
    taken per replication: common random numbers cancel most of the noise, and
    the CI on the difference is what decides "beats the baseline".
    """
    from twinmind.optim.evaluators import feasible_robust

    kpis: dict[str, Any] = {}
    for index, kpi in enumerate(KPI_NAMES):
        p, b = plan_rows[:, index], baseline_rows[:, index]
        kpis[kpi] = {
            "plan": _summary(p, confidence),
            "baseline": _summary(b, confidence),
            "diff": _summary(p - b, confidence),
        }
    plan_obj = np.array([spec.score(dict(zip(KPI_NAMES, r, strict=True))) for r in plan_rows])
    base_obj = np.array([spec.score(dict(zip(KPI_NAMES, r, strict=True))) for r in baseline_rows])
    plan_mean = {k: kpis[k]["plan"]["mean"] for k in KPI_NAMES}
    plan_sem = {k: kpis[k]["plan"]["sem"] for k in KPI_NAMES}
    base_mean = {k: kpis[k]["baseline"]["mean"] for k in KPI_NAMES}
    base_sem = {k: kpis[k]["baseline"]["sem"] for k in KPI_NAMES}
    return {
        "n_reps": int(len(plan_rows)),
        "stream": "reeval",
        "goal_seed": int(goal_seed),
        "confidence_level": confidence,
        "kpis": kpis,
        "objective": {
            "plan": _summary(plan_obj, confidence),
            "baseline": _summary(base_obj, confidence),
            "diff": _summary(plan_obj - base_obj, confidence),
        },
        "constraint_margins": {
            f"{c.kpi}{c.op}{c.threshold:g}": c.margin(plan_mean[c.kpi])
            for c in spec.constraints
            if c.hard
        },
        "feasible_point": bool(spec.feasible(plan_mean)),
        "feasible_robust": bool(feasible_robust(spec, plan_mean, plan_sem)),
        "baseline_feasible_point": bool(spec.feasible(base_mean)),
        "baseline_feasible_robust": bool(feasible_robust(spec, base_mean, base_sem)),
    }


@dataclass
class TwinToolkit:
    """The real tools: the twin, the surrogate and the novelty scorer.

    The baseline's replications are computed once per spec and reused when the
    critic swaps or re-explores, so a retry never pays for the baseline twice.
    """

    resources: Any
    surrogate: Any
    novelty_scorer: Any
    contexts: Any  # callable: context_id -> TwinContext
    optimiser: str = "tpe"
    budget: int = 600
    k: int = 20
    min_distance: float = 0.15
    verify_reps: int = 10
    max_rounds: int = 0
    eval_reps: int = 30
    confidence: float = 0.95
    goal_seed: int = 1
    twin_runs: int = 0
    _baseline_cache: dict[str, np.ndarray] = field(default_factory=dict)

    def context_summary(self, context_id: int) -> str:
        context = self.contexts(context_id)
        low, mid, high = (float(v) for v in context.demand_summary)
        machines = len(context.machines)
        return (
            f"dev context {context_id}; 28-day horizon; {machines} machines in series; "
            f"forecast demand summary (low/mid/high) {low:.0f}/{mid:.0f}/{high:.0f}; "
            "current policy = the listed current values."
        )

    def explore(self, spec: ObjectiveSpec, margin_std: float, seed: int) -> Exploration:
        from twinmind.optim.surrogate_first import VerifyConfig, surrogate_first_verify

        started = time.perf_counter()
        plans, trace = surrogate_first_verify(
            spec,
            self.contexts(spec.context_id),
            self.surrogate,
            self.resources,
            seed,
            VerifyConfig(
                optimiser=self.optimiser,
                budget=self.budget,
                k=self.k,
                min_distance=self.min_distance,
                max_rounds=self.max_rounds,
                margin_std=margin_std,
                verify_reps=self.verify_reps,
                reevaluate=False,
            ),
            None,
        )
        if not plans:
            raise RuntimeError(f"explorer produced no candidate for {spec.name!r}")
        cost = plans[0].cost or {}
        runs = int(cost.get("twin_runs", 0))
        self.twin_runs += runs
        origin = f"explorer margin {margin_std:g} std, seed {seed}"
        return Exploration(
            chosen=candidate_record(plans[0], origin),
            candidates=[candidate_record(p, origin) for p in plans],
            status=str(trace.get("status", plans[0].status)),
            margin_std=margin_std,
            twin_runs=runs,
            surrogate_evals=int(cost.get("surrogate_evals", 0)),
            wall_seconds=time.perf_counter() - started,
        )

    def _rows(self, spec: ObjectiveSpec, scenario: Scenario) -> np.ndarray:
        from twinmind.optim.evaluators import stream_seeds
        from twinmind.twin.runner import run_twin

        context = self.contexts(spec.context_id)
        seeds = stream_seeds("reeval", self.goal_seed, self.eval_reps)
        rows = np.array(
            [run_twin(scenario, s, context, self.resources).to_vector() for s in seeds],
            dtype="float64",
        )
        self.twin_runs += len(seeds)
        return rows

    def evaluate(self, spec: ObjectiveSpec, scenario: dict[str, Any]) -> dict[str, Any]:
        key = f"{spec.context_id}:{sorted(spec.fixed.items())}"
        if key not in self._baseline_cache:
            self._baseline_cache[key] = self._rows(spec, spec.base_scenario())
        plan_rows = self._rows(spec, Scenario(**scenario))
        return paired_evaluation(
            spec, plan_rows, self._baseline_cache[key], self.confidence, self.goal_seed
        )

    def novelty(self, scenario: dict[str, Any]) -> float:
        vector = np.asarray([Scenario(**scenario).to_vector()], dtype="float64")
        return float(self.novelty_scorer.score(vector)[0])


# --------------------------------------------------------------------------- #
# Critic helpers
# --------------------------------------------------------------------------- #
def best_robust_candidate(
    candidates: Sequence[dict[str, Any]], exclude: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The best verified candidate whose verification CI lies inside the feasible region."""
    robust = [
        c
        for c in candidates
        if c["feasible_robust"] and (exclude is None or c["scenario"] != exclude["scenario"])
    ]
    return max(robust, key=lambda c: c["objective"]) if robust else None


def closest_achievable(
    spec: ObjectiveSpec, evaluation: dict[str, Any], candidates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """For each constraint the final plan violates, the best value anything reached.

    Sources: every verified candidate (search stream, ``verify_reps``) and the
    re-evaluated plan (reeval stream). Each entry is an OPTION for the user --
    relaxing a threshold is their decision -- and values for different
    constraints may come from different plans, so they are not jointly
    achievable unless a single source is named for all of them.
    """
    out = []
    plan_means = {k: v["plan"]["mean"] for k, v in evaluation["kpis"].items()}
    for c in spec.constraints:
        if not c.hard or c.satisfied(plan_means[c.kpi]):
            continue
        points = [("re-evaluated plan (30 reps)", plan_means[c.kpi])] + [
            ("verified candidate (search stream)", cand["verification_mean"][c.kpi])
            for cand in candidates
            if c.kpi in cand["verification_mean"]
        ]
        better = min if c.op == "<=" else max
        source, value = better(points, key=lambda p: p[1])
        out.append(
            {
                "constraint": f"{c.kpi} {c.op} {c.threshold:g}",
                "kpi": c.kpi,
                "op": c.op,
                "threshold": c.threshold,
                "closest_achievable": float(value),
                "source": source,
                "option": f"relax to {c.kpi} {c.op} {fmt(_round_sig(value, 3))}",
                "note": "an option for you to decide on, not a decision",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Explainer facts, grounding, template
# --------------------------------------------------------------------------- #
def _round_sig(x: float, sig: int = 4) -> float:
    if x == 0 or not math.isfinite(x):
        return float(x)
    return float(round(x, sig - int(math.floor(math.log10(abs(x)))) - 1))


def build_facts(
    spec: ObjectiveSpec,
    plan: dict[str, Any],
    evaluation: dict[str, Any],
    flags: Sequence[dict[str, Any]],
    closest: Sequence[dict[str, Any]],
    recommended: bool,
) -> dict[str, Any]:
    """The ONLY input the explainer sees. Every number is pre-rounded."""
    current = spec.base_scenario().model_dump()
    changes = []
    for dim in SCENARIO_DIMS:
        a, b = float(current[dim]), float(plan["scenario"][dim])
        if abs(a - b) > 1e-6 * max(1.0, abs(a)):
            changes.append({"dimension": dim, "current": _round_sig(a), "plan": _round_sig(b)})
    table = []
    for kpi, row in evaluation["kpis"].items():
        base = row["baseline"]["mean"]
        table.append(
            {
                "kpi": kpi,
                "plan": _round_sig(row["plan"]["mean"]),
                "plan_ci95": [
                    _round_sig(row["plan"]["ci_low"]),
                    _round_sig(row["plan"]["ci_high"]),
                ],
                "current": _round_sig(base),
                "current_ci95": [
                    _round_sig(row["baseline"]["ci_low"]),
                    _round_sig(row["baseline"]["ci_high"]),
                ],
                "change": _round_sig(row["diff"]["mean"]),
                "change_ci95": [
                    _round_sig(row["diff"]["ci_low"]),
                    _round_sig(row["diff"]["ci_high"]),
                ],
                "change_pct": _round_sig(100.0 * row["diff"]["mean"] / base) if base else None,
            }
        )
    objective = evaluation["objective"]
    return {
        "goal": spec.name,
        "objective": dict(spec.objective),
        "sense": spec.sense,
        "constraints": [c.to_dict() for c in spec.constraints],
        "recommended": recommended,
        "feasible": evaluation["feasible_point"],
        "robustly_feasible": evaluation["feasible_robust"],
        "plan_changes": changes,
        "kpi_table": table,
        "objective_change": {
            "change": _round_sig(objective["diff"]["mean"]),
            "change_ci95": [
                _round_sig(objective["diff"]["ci_low"]),
                _round_sig(objective["diff"]["ci_high"]),
            ],
        },
        "flags": [{"code": f["code"], "message": f["message"]} for f in flags],
        "closest_achievable": [
            {
                "constraint": c["constraint"],
                "closest_achievable": _round_sig(c["closest_achievable"]),
                "option": c["option"],
            }
            for c in closest
        ],
        "replications": evaluation["n_reps"],
        "confidence_level_pct": round(100 * evaluation["confidence_level"]),
    }


_NUMBER = re.compile(
    r"(?<![A-Za-z_\d.])[-+−]?(?:£|\$)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|k\b|K\b|m\b|M\b)?"
)


def _fact_numbers(value: Any, out: list[float]) -> list[float]:
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            out.append(float(value))
    elif isinstance(value, str):
        for match in _NUMBER.finditer(value):
            out.append(float((match.group(1) + (match.group(2) or "")).replace(",", "")))
    elif isinstance(value, dict):
        for v in value.values():
            _fact_numbers(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _fact_numbers(v, out)
    return out


def ungrounded_numbers(text: str, facts: dict[str, Any]) -> list[str]:
    """Numbers in ``text`` that appear nowhere in ``facts``, allowing rounding.

    A number written with ``d`` decimals matches a fact if the fact rounds to
    it at ``d`` decimals (after a k/m suffix is applied). A percentage may also
    match a fraction in the facts times 100. Signs are ignored ("down 5" and
    "-5" state the same fact). Digits glued to letters ("M3", "pm_interval_M1")
    are names, not numbers, and are skipped.
    """
    facts_abs = [abs(v) for v in _fact_numbers(facts, [])]
    bad = []
    for match in _NUMBER.finditer(text):
        whole, frac, suffix = match.group(1), match.group(2) or "", match.group(3) or ""
        value = float((whole + frac).replace(",", ""))
        decimals = len(frac) - 1 if frac else 0
        scale = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}.get(suffix.strip(), 1.0)
        tolerance = 0.5 * 10.0 ** (-decimals) * scale * (1 + 1e-9)
        target = value * scale
        candidates = list(facts_abs)
        if suffix.strip() == "%":
            candidates += [100.0 * f for f in facts_abs if f <= 1.0]
        if not any(abs(target - f) <= tolerance for f in candidates):
            bad.append(match.group(0).strip())
    return bad


def fmt(x: float) -> str:
    """A plain decimal, never scientific notation (which the grounding check
    would misread as two numbers)."""
    text = f"{float(x):.10f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def template_explanation(facts: dict[str, Any]) -> str:
    """A deterministic explanation built only from the facts (grounded by construction)."""
    lines = []
    verdict = (
        "This plan is recommended." if facts["recommended"] else "This plan is NOT recommended."
    )
    lines.append(f"Goal {facts['goal']}: {verdict}")
    if facts["plan_changes"]:
        changes = "; ".join(
            f"{c['dimension']} {fmt(c['current'])} -> {fmt(c['plan'])}"
            for c in facts["plan_changes"]
        )
        lines.append(f"Changes from the current policy: {changes}.")
    else:
        lines.append("The plan keeps the current policy unchanged.")
    obj = facts["objective_change"]
    lines.append(
        f"Objective change vs current policy: {fmt(obj['change'])} "
        f"({facts['confidence_level_pct']}% CI {fmt(obj['change_ci95'][0])} to "
        f"{fmt(obj['change_ci95'][1])}), over {facts['replications']} paired replications."
    )
    for row in facts["kpi_table"]:
        lines.append(
            f"- {row['kpi']}: {fmt(row['plan'])} (CI {fmt(row['plan_ci95'][0])} to "
            f"{fmt(row['plan_ci95'][1])}) vs current {fmt(row['current'])}."
        )
    for flag in facts["flags"]:
        lines.append(f"Flag: {flag['message']}")
    for c in facts["closest_achievable"]:
        lines.append(
            f"Closest achievable for {c['constraint']}: {fmt(c['closest_achievable'])}; "
            f"option: {c['option']}."
        )
    return "\n".join(lines)
