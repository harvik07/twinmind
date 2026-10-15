"""Stage B: aggregate the optimisation benchmark under the Amendment 2 design.

    python scripts/analyse_optim_benchmark.py

Reads ONLY the design's 104 cells, through
:func:`twinmind.optim.design.load_design_cells`; the seed-3 cells and anything
from a reserve goal are listed by filename and never opened. Writes
``reports/results/optim_benchmark_analysis.json`` and three figures.

The one twin computation here is the baseline -- the default Scenario on each
core goal's context, 30 reps on the "reeval" stream -- because no cell stores
it. 8 goals x 30 runs, single process, cached in
``data/processed/optim_benchmark/baselines.json``.

ANALYSIS CONVENTIONS. Fixed before any held-out number was looked at, and
stated because the pre-registration is silent on them:

1. **A cell's result is its chosen plan's 30-rep re-evaluation** on the
   "reeval" stream. Every method has one, except a surrogate-first cell whose
   shortlist never verified feasible: that path returns the best 10-rep
   SEARCH-stream plan with ``status="infeasible_unverified"`` and no
   re-evaluation. Such a cell delivered no verified plan.
2. **A delivered plan that is infeasible on re-evaluation, or was never
   re-evaluated, scores gap = 1** -- the same as keeping the current policy.
   A plan that violates a hard constraint is not an answer, and scoring its
   raw objective would reward ignoring the constraint. The raw-objective gap
   is reported as a sensitivity.
3. **The best_known reference** for a goal is, as pre-registered, the best
   point-feasible re-evaluated plan across the 500,000-point sweep's verified
   plans AND every method's chosen plan. Under that definition no method can
   have a negative gap. The sweep-only reference, under which one can, is
   reported as a sensitivity.
4. **The baseline** is the default Scenario (with the goal's fixed values) on
   the goal's context, re-evaluated at 30 reps on the reeval stream with
   goal seed 1 -- the same replication seeds as the best_known sweep.
5. **"Surrogate-first" in H1, H2 and H4 is ``sf_tpe``**: the pre-registered
   ``surrogate_first_verify`` default optimiser, and the arm Amendment 1
   matched ``surrogate_only`` to. ``sf_random`` and ``sf_cmaes`` are reported
   alongside.
6. **Per goal, a method's gap is the mean over its 2 seeds**; H1 is the median
   of those per-goal means. Bootstraps resample GOALS (10,000 resamples,
   percentile intervals, fixed seed).
7. **"Matches or beats"** (H2): the paired bootstrap CI of the difference in
   median gap (surrogate-first minus twin-only) has upper bound <= 0 for
   "beats", contains 0 for "matches"; lower bound > 0 is "worse". **"Worse"**
   (H3) requires surrogate_only's CI of the same difference to lie wholly
   above 0; a tie is not "worse".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_optim_benchmark as bench  # noqa: E402

from twinmind.eval.registry import results_dir  # noqa: E402
from twinmind.optim.design import BenchmarkDesign, load_design_cells  # noqa: E402
from twinmind.optim.goals import Goal, load_goals  # noqa: E402
from twinmind.utils.logging import get_logger  # noqa: E402
from twinmind.utils.paths import PROJECT_ROOT  # noqa: E402

logger = get_logger("analyse_optim_benchmark")

CELLS = bench.RESULTS_DIR / "full"
BASELINES = bench.RESULTS_DIR / "baselines.json"
FIGURES = PROJECT_ROOT / "reports" / "figures"
OUT = results_dir() / "optim_benchmark_analysis.json"

SURROGATE_FIRST = ("sf_tpe", "sf_random", "sf_cmaes")
PRIMARY_SF = "sf_tpe"
TWIN_ONLY = ("twin_tpe", "twin_cmaes")
COMPARED = (*SURROGATE_FIRST, "surrogate_only", *TWIN_ONLY)
N_BOOT = 10_000

#: Pre-registered thresholds (reports/preregistration_optim.md).
H1_MAX_MEDIAN_GAP = 0.05
H2_MAX_CPU_RATIO = 1 / 5
H4_MIN_FEASIBILITY = 0.90

#: From the A0 check (reports/results/optim_a0_checks.json): mean novelty of
#: the bound-corner OOD set. There is no pre-registered "novel" threshold and
#: the optimiser defines no flag; this is a descriptive reference only.
A0_OOD_MEAN_NOVELTY = 1.006184362538678
A0_T1_MEAN_NOVELTY = 0.8720506581184284


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def compute_baselines(goals: list[Goal]) -> dict[str, Any]:
    """Default Scenario per goal, 30 reps on the reeval stream, goal seed 1."""
    from twinmind.optim.evaluators import TwinEvaluator

    state = bench._worker_context()
    out: dict[str, Any] = {}
    for goal in goals:
        context = bench._context_for_goal(goal, state)
        evaluator = TwinEvaluator(
            resources=state["resources"], context=context, goal_seed=1, n_reps=30, stream="reeval"
        )
        mean, sem = evaluator.evaluate_with_spread([goal.spec.base_scenario()])[0]
        out[goal.name] = {
            "objective": goal.spec.score(mean),
            "feasible_point": goal.spec.feasible(mean),
            "mean": mean,
            "sem": sem,
            "cpu_seconds": evaluator.cost.cpu_seconds,
        }
        logger.info(
            "baseline %-30s objective %.1f feasible %s",
            goal.name,
            out[goal.name]["objective"],
            out[goal.name]["feasible_point"],
        )
    return out


# --------------------------------------------------------------------------- #
# Cells -> rows
# --------------------------------------------------------------------------- #
def has_reeval(cell: dict[str, Any]) -> bool:
    """Was the chosen plan re-evaluated on the reeval stream? (convention 1)

    Surrogate-first's re-evaluation path is the only one that records
    ``free_dims`` in its trace; its infeasible-at-search exit does not, and
    its twin-run count is then exactly shortlist x verify reps, with no 30.
    """
    if not cell["method"].startswith("sf_"):
        return True
    reeval = "free_dims" in cell["trace"]
    shortlisted = sum(r["n_shortlist"] for r in cell["trace"]["rounds"])
    runs = int(cell["plans"][0]["cost"]["twin_runs"])
    expected = shortlisted * 10 + (30 if reeval else 0)
    if runs != expected:
        raise ValueError(f"{cell['goal']}/{cell['method']}: twin_runs {runs} != {expected}")
    return reeval


def to_row(cell: dict[str, Any]) -> dict[str, Any]:
    """The fields the analysis uses from one cell's chosen plan."""
    plan = cell["plans"][0]
    reeval = has_reeval(cell)
    row: dict[str, Any] = {
        "goal": cell["goal"],
        "method": cell["method"],
        "seed": cell["seed"],
        "status": plan["status"],
        "reeval": reeval,
        "objective": plan["objective_value"] if reeval else None,
        "feasible_point": bool(plan["feasible_point"]) if reeval else False,
        "feasible_robust": bool(plan["feasible_robust"]) if reeval else False,
        "kpis": plan["twin_verified"]["mean"] if reeval else None,
        "surrogate_pred": plan.get("surrogate_pred") or {},
        "profit_gap": (plan.get("surrogate_twin_gap") or {}).get("profit") if reeval else None,
        "novelty": plan.get("novelty"),
        "surrogate_evals": float(plan["cost"]["surrogate_evals"]),
        "twin_runs": float(plan["cost"]["twin_runs"]),
        "cpu_seconds_recorded": float(plan["cost"]["cpu_seconds"]),
        "wall_seconds": float(plan["cost"]["wall_seconds"]),
        "anytime": cell["trace"].get("anytime"),
        "rounds": cell["trace"].get("rounds"),
        # For an unverified surrogate-first exit: the SEARCH-stream values,
        # kept for the record and never used as a result.
        "search_stream_kpis": None if reeval else plan["twin_verified"]["mean"],
    }
    return row


def reference_pool(cells: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Every re-evaluated plan per goal: the sweep's verified plans + chosen plans."""
    pool: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        plans = cell["plans"] if cell["method"] == "best_known" else cell["plans"][:1]
        if cell["method"] != "best_known" and not has_reeval(cell):
            continue
        for plan in plans:
            pool.setdefault(cell["goal"], []).append(
                {
                    "source": cell["method"],
                    "seed": cell["seed"],
                    "objective": plan["objective_value"],
                    "feasible_point": bool(plan["feasible_point"]),
                    "kpis": plan["twin_verified"]["mean"],
                }
            )
    return pool


def best_feasible(plans: list[dict[str, Any]]) -> float | None:
    values = [p["objective"] for p in plans if p["feasible_point"]]
    return max(values) if values else None


def gap(objective: float, reference: float, baseline: float) -> float | None:
    """(reference - objective) / (reference - baseline); None when undefined."""
    denominator = reference - baseline
    if denominator <= 0:
        return None
    return (reference - objective) / denominator


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def boot_median(values: Sequence[Any], seed: int = 0) -> dict[str, Any]:
    v = np.asarray(values, dtype="float64")
    rng = np.random.default_rng(seed)
    draws = np.median(v[rng.integers(0, len(v), (N_BOOT, len(v)))], axis=1)
    return {
        "median": float(np.median(v)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_goals": len(v),
    }


def paired_median_diff(a: Sequence[Any], b: Sequence[Any], seed: int = 0) -> dict[str, Any]:
    """median(a) - median(b), resampling goals jointly."""
    x, y = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    draws = np.median(x[idx], axis=1) - np.median(y[idx], axis=1)
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    point = float(np.median(x) - np.median(y))
    verdict = "beats" if hi <= 0 else ("worse" if lo > 0 else "matches (tie within CI)")
    return {"difference": point, "ci95": [lo, hi], "verdict": verdict, "n_goals": len(x)}


def boot_mean(values: Sequence[Any], seed: int = 0) -> dict[str, Any]:
    v = np.asarray(values, dtype="float64")
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n": len(v),
    }


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def analyse(recompute_baselines: bool = False) -> dict[str, Any]:
    design = BenchmarkDesign.load()
    found = load_design_cells(CELLS, design, bench.METHODS, bench.budgets_hash())
    if found.missing or found.stale:
        raise SystemExit(f"design incomplete: missing {found.missing}, stale {found.stale}")
    cells = found.cells
    goals = {g.name: g for g in load_goals(split="held_out") if g.name in design.core_goals}
    feasible_goals = [n for n in design.core_goals if not goals[n].spec.infeasible_by_design]
    infeasible_goals = [n for n in design.core_goals if goals[n].spec.infeasible_by_design]

    if recompute_baselines or not BASELINES.exists():
        bench.write_atomic(compute_baselines([goals[n] for n in design.core_goals]), BASELINES)
    baselines = json.loads(BASELINES.read_text(encoding="utf-8"))

    rows = [to_row(c) for c in cells]
    pool = reference_pool(cells)
    ref_pooled = {g: best_feasible(pool.get(g, [])) for g in design.core_goals}
    ref_sweep = {
        g: best_feasible([p for p in pool.get(g, []) if p["source"] == "best_known"])
        for g in design.core_goals
    }
    any_feasible_source = {
        g: sorted({f"{p['source']}/s{p['seed']}" for p in pool.get(g, []) if p["feasible_point"]})
        for g in design.core_goals
    }

    # Per-cell gaps.
    for row in rows:
        base = baselines[row["goal"]]["objective"]
        for label, refs in (("", ref_pooled), ("_sweep_ref", ref_sweep)):
            ref = refs[row["goal"]]
            if ref is None:
                row["gap" + label] = None
                row["gap_raw" + label] = None
                continue
            raw = gap(row["objective"], ref, base) if row["objective"] is not None else None
            row["gap_raw" + label] = raw
            if raw is None and row["objective"] is not None:
                row["gap" + label] = None  # undefined denominator
            else:
                row["gap" + label] = raw if (row["reeval"] and row["feasible_point"]) else 1.0

    def per_goal(method: str, key: str = "gap", goal_list: list[str] | None = None) -> dict:
        out = {}
        for goal in goal_list or feasible_goals:
            values = [
                r[key]
                for r in rows
                if r["goal"] == goal and r["method"] == method and r[key] is not None
            ]
            out[goal] = float(np.mean(values)) if values else None
        return out

    gap_goals = [g for g in feasible_goals if ref_pooled[g] is not None]
    no_feasible_plan = [g for g in feasible_goals if ref_pooled[g] is None]

    per_goal_gap = {m: per_goal(m, goal_list=gap_goals) for m in COMPARED}
    aggregates = {
        m: boot_median([v for v in per_goal_gap[m].values() if v is not None]) for m in COMPARED
    }
    sensitivities = {
        "raw_objective_gap": {
            m: boot_median([v for v in per_goal(m, "gap_raw", gap_goals).values() if v is not None])
            for m in COMPARED
        },
        "sweep_only_reference": {
            m: boot_median(
                [
                    v
                    for v in per_goal(
                        m, "gap_sweep_ref", [g for g in gap_goals if ref_sweep[g] is not None]
                    ).values()
                    if v is not None
                ]
            )
            for m in COMPARED
        },
    }

    # Feasibility, per method, over the feasible goals (with and without goals
    # on which no method found any feasible plan).
    def feas(method: str, goal_list: list[str], key: str) -> dict[str, Any]:
        sel = [r for r in rows if r["method"] == method and r["goal"] in goal_list]
        return {"rate": float(np.mean([r[key] for r in sel])) if sel else None, "n_cells": len(sel)}

    feasibility = {
        m: {
            "point_all_feasible_goals": feas(m, feasible_goals, "feasible_point"),
            "robust_all_feasible_goals": feas(m, feasible_goals, "feasible_robust"),
            "point_excluding_no_feasible_plan_goals": feas(m, gap_goals, "feasible_point"),
            "robust_excluding_no_feasible_plan_goals": feas(m, gap_goals, "feasible_robust"),
        }
        for m in (*COMPARED, "best_known")
    }

    # Twin CPU-seconds. Recorded for twin-only and best_known; NOT recorded for
    # surrogate-first or surrogate_only (cost bug: only the run COUNT was kept),
    # so for those it is estimated as runs x the same goal's measured twin-only
    # CPU-seconds per run.
    cpu_per_run = {}
    for goal in design.core_goals:
        measured = [r for r in rows if r["goal"] == goal and r["method"] in TWIN_ONLY]
        cpu_per_run[goal] = sum(r["cpu_seconds_recorded"] for r in measured) / sum(
            r["twin_runs"] for r in measured
        )
    for row in rows:
        recorded = row["cpu_seconds_recorded"]
        row["twin_cpu_seconds"] = (
            recorded if recorded > 0 else row["twin_runs"] * cpu_per_run[row["goal"]]
        )
        row["twin_cpu_estimated"] = recorded <= 0

    cost = {}
    for m in (*COMPARED, "best_known"):
        sel = [r for r in rows if r["method"] == m]
        cost[m] = {
            "cells": len(sel),
            "surrogate_evals_mean": float(np.mean([r["surrogate_evals"] for r in sel])),
            "twin_runs_mean": float(np.mean([r["twin_runs"] for r in sel])),
            "twin_cpu_seconds_mean": float(np.mean([r["twin_cpu_seconds"] for r in sel])),
            "twin_cpu_estimated": bool(any(r["twin_cpu_estimated"] for r in sel)),
            "wall_seconds_mean": float(np.mean([r["wall_seconds"] for r in sel])),
        }

    hypotheses = evaluate_hypotheses(
        rows,
        per_goal_gap,
        aggregates,
        feasibility,
        gap_goals,
        feasible_goals,
        infeasible_goals,
        no_feasible_plan,
    )
    hypotheses["H2"]["cost_excluding_no_feasible_plan_goals"] = {
        twin: {
            "twin_runs_ratio_measured": sum(
                r["twin_runs"] for r in rows if r["method"] == PRIMARY_SF and r["goal"] in gap_goals
            )
            / sum(r["twin_runs"] for r in rows if r["method"] == twin and r["goal"] in gap_goals),
            "cpu_seconds_ratio_estimated": sum(
                r["twin_cpu_seconds"]
                for r in rows
                if r["method"] == PRIMARY_SF and r["goal"] in gap_goals
            )
            / sum(
                r["twin_cpu_seconds"]
                for r in rows
                if r["method"] == twin and r["goal"] in gap_goals
            ),
        }
        for twin in TWIN_ONLY
    }
    return {
        "design_hash": design.hash(),
        "cells_read": len(cells),
        "excluded_files_not_opened": found.excluded,
        "feasible_goals": feasible_goals,
        "infeasible_by_design_goals": infeasible_goals,
        "goals_with_no_feasible_plan_found": no_feasible_plan,
        "feasible_plan_sources": any_feasible_source,
        "gap_goals": gap_goals,
        "baselines": baselines,
        "reference_pooled": ref_pooled,
        "reference_sweep_only": ref_sweep,
        "per_goal_gap": per_goal_gap,
        "median_gap": aggregates,
        "sensitivities": sensitivities,
        "feasibility": feasibility,
        "cost": cost,
        "cpu_per_twin_run_by_goal": cpu_per_run,
        "hypotheses": hypotheses,
        "diagnostics": diagnostics(rows, goals, design, gap_goals, no_feasible_plan),
        "rows": rows,
    }


def diagnostics(
    rows: list[dict[str, Any]],
    goals: dict[str, Goal],
    design: BenchmarkDesign,
    gap_goals: list[str],
    no_feasible_plan: list[str],
) -> dict[str, Any]:
    """Items beyond the pre-registered hypotheses. Labelled per entry."""
    out: dict[str, Any] = {}

    # Verification on the 10-rep SEARCH stream vs the independent 30-rep
    # re-evaluation. Every surrogate-first cell that reached re-evaluation had
    # a chosen plan verified feasible on the search stream (that is the only
    # way to reach it), so each infeasible re-evaluation is a reversal.
    reversal: dict[str, Any] = {}
    for goal in design.core_goals:
        per = {}
        for method in SURROGATE_FIRST:
            sel = [r for r in rows if r["goal"] == goal and r["method"] == method and r["reeval"]]
            per[method] = {
                "verified_feasible_on_search_stream": len(sel),
                "feasible_on_reeval": sum(r["feasible_point"] for r in sel),
                "robust_on_reeval": sum(r["feasible_robust"] for r in sel),
                "unverified_exits": sum(
                    1
                    for r in rows
                    if r["goal"] == goal and r["method"] == method and not r["reeval"]
                ),
            }
        reversal[goal] = per
    total_verified = sum(
        v["verified_feasible_on_search_stream"] for g in reversal.values() for v in g.values()
    )
    total_held = sum(v["feasible_on_reeval"] for g in reversal.values() for v in g.values())
    out["verify_to_reeval"] = {
        "label": "pre-registered quantities (feasibility on re-evaluation), cut by verification",
        "by_goal": reversal,
        "verified_feasible_total": total_verified,
        "still_feasible_on_reeval_total": total_held,
    }

    # Constrained-KPI surrogate bias on chosen plans (EXPLORATORY).
    bias: dict[str, Any] = {}
    for goal in design.core_goals:
        spec = goals[goal].spec
        per_kpi = {}
        for c in spec.constraints:
            if not c.hard:
                continue
            entries = []
            for r in rows:
                if r["goal"] != goal or not r["surrogate_pred"] or c.kpi not in r["surrogate_pred"]:
                    continue
                twin = r["kpis"] if r["reeval"] else r["search_stream_kpis"]
                entries.append(
                    {
                        "method": r["method"],
                        "seed": r["seed"],
                        "surrogate": r["surrogate_pred"][c.kpi],
                        "twin": twin[c.kpi],
                        "stream": "reeval" if r["reeval"] else "search (10 reps)",
                    }
                )
            achieved = [
                r["kpis"][c.kpi] for r in rows if r["goal"] == goal and r["reeval"] and r["kpis"]
            ]
            best = (min(achieved) if c.op == "<=" else max(achieved)) if achieved else None
            per_kpi[f"{c.kpi}{c.op}{c.threshold}"] = {
                "mean_surrogate_minus_twin": float(
                    np.mean([e["surrogate"] - e["twin"] for e in entries])
                )
                if entries
                else None,
                "best_achieved_on_reeval_by_any_chosen_plan": best,
                "threshold": c.threshold,
                "entries": entries,
            }
        bias[goal] = per_kpi
    out["constrained_kpi_bias"] = {"label": "EXPLORATORY", "by_goal": bias}

    # Did twin-only search ever SEE a feasible point (5-rep search stream)?
    seen = {}
    for goal in design.core_goals:
        seen[goal] = {
            f"{r['method']}/s{r['seed']}": r["anytime"][-1]["best_feasible_score"]
            for r in rows
            if r["goal"] == goal and r["method"] in TWIN_ONLY
        }
    out["twin_only_search_saw_feasible"] = {
        "label": "search stream, 5 reps, best feasible score seen during search; None = never",
        "by_goal": seen,
    }

    # Verification rounds used by surrogate-first.
    rounds = {}
    for goal in design.core_goals:
        rounds[goal] = {
            f"{r['method']}/s{r['seed']}": [
                {"margin_std": x["margin_std"], "n_feasible_of_shortlist": x["n_feasible"]}
                for x in r["rounds"]
            ]
            for r in rows
            if r["goal"] == goal and r["method"] in SURROGATE_FIRST
        }
    out["verification_rounds"] = rounds

    # Novelty on the chosen plans (descriptive; no flag exists).
    chosen = [r for r in rows if r["method"] in SURROGATE_FIRST and r["novelty"] is not None]
    values = [float(r["novelty"]) for r in chosen]
    out["novelty"] = {
        "label": "computed and stored only; never read by ranking or selection; the "
        "optimiser defines no 'novel' flag",
        "n_chosen_plans": len(values),
        "flagged_by_the_optimiser": 0,
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "exploratory_above_a0_t1_mean": int(sum(v > A0_T1_MEAN_NOVELTY for v in values)),
        "exploratory_above_a0_ood_mean": int(sum(v > A0_OOD_MEAN_NOVELTY for v in values)),
        "a0_t1_mean": A0_T1_MEAN_NOVELTY,
        "a0_ood_mean": A0_OOD_MEAN_NOVELTY,
    }

    # Twin-only evaluations actually used: (runs - the 30-rep re-evaluation) / 5.
    # CMA-ES stops on its own convergence criteria and can use fewer than 300.
    out["twin_only_evaluations_below_budget"] = {
        f"{r['goal']}/{r['method']}/s{r['seed']}": int((r["twin_runs"] - 30) / 5)
        for r in rows
        if r["method"] in TWIN_ONLY and (r["twin_runs"] - 30) / 5 < 300
    }
    out["goals_without_feasible_plan"] = no_feasible_plan
    out["gap_goals"] = gap_goals
    return out


def evaluate_hypotheses(
    rows: list[dict[str, Any]],
    per_goal_gap: dict[str, dict[str, float | None]],
    aggregates: dict[str, Any],
    feasibility: dict[str, Any],
    gap_goals: list[str],
    feasible_goals: list[str],
    infeasible_goals: list[str],
    no_feasible_plan: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # H1: median normalised gap of surrogate-first <= 5%.
    h1 = aggregates[PRIMARY_SF]
    out["H1"] = {
        "statement": "median normalised gap to best_known over held-out feasible goals <= 5%",
        "method": PRIMARY_SF,
        "median_gap": h1["median"],
        "ci95": h1["ci95"],
        "n_goals": h1["n_goals"],
        "goals_excluded_no_feasible_plan": no_feasible_plan,
        "verdict": "PASS" if h1["median"] <= H1_MAX_MEDIAN_GAP else "FAIL",
        "other_sf_arms": {m: aggregates[m] for m in SURROGATE_FIRST if m != PRIMARY_SF},
    }

    # H2: matches or beats twin-only TPE and CMA-ES, at <= 1/5 of the twin CPU-s.
    sf = [per_goal_gap[PRIMARY_SF][g] for g in gap_goals]
    comparisons = {}
    for twin in TWIN_ONLY:
        comparisons[twin] = paired_median_diff(sf, [per_goal_gap[twin][g] for g in gap_goals])
    quality_ok = all(c["verdict"] != "worse" for c in comparisons.values())
    sf_rows = [r for r in rows if r["method"] == PRIMARY_SF and r["goal"] in feasible_goals]
    cpu = {}
    for twin in TWIN_ONLY:
        t_rows = [r for r in rows if r["method"] == twin and r["goal"] in feasible_goals]
        cpu[twin] = {
            "twin_runs_ratio_measured": sum(r["twin_runs"] for r in sf_rows)
            / sum(r["twin_runs"] for r in t_rows),
            "cpu_seconds_ratio_estimated": sum(r["twin_cpu_seconds"] for r in sf_rows)
            / sum(r["twin_cpu_seconds"] for r in t_rows),
        }
    cost_ok = all(
        c["twin_runs_ratio_measured"] <= H2_MAX_CPU_RATIO
        and c["cpu_seconds_ratio_estimated"] <= H2_MAX_CPU_RATIO
        for c in cpu.values()
    )
    out["H2"] = {
        "statement": "surrogate-first matches or beats twin-only TPE and CMA-ES at 300 "
        "evaluations while using <= 1/5 of their twin CPU-seconds",
        "method": PRIMARY_SF,
        "quality": comparisons,
        "cost": cpu,
        "cost_caveat": "surrogate-first twin CPU-seconds were not recorded (cost bug: only "
        "the twin-run count was kept). The measured twin-run ratio and an estimate "
        "(runs x the same goal's measured twin-only CPU-s per run) are both required "
        "to clear 1/5.",
        "verdict": "PASS" if (quality_ok and cost_ok) else "FAIL",
        "quality_half": "PASS" if quality_ok else "FAIL",
        "cost_half": "PASS" if cost_ok else "FAIL",
    }

    # H3: surrogate_only worse than surrogate-first, AND positive mean profit gap.
    diff = paired_median_diff(
        [per_goal_gap["surrogate_only"][g] for g in gap_goals],
        [per_goal_gap[PRIMARY_SF][g] for g in gap_goals],
    )
    clause1 = diff["verdict"] == "worse"
    so_gaps = [
        r["profit_gap"]
        for r in rows
        if r["method"] == "surrogate_only" and r["goal"] in feasible_goals
    ]
    so_gaps_all = [r["profit_gap"] for r in rows if r["method"] == "surrogate_only"]
    profit = boot_mean(so_gaps)
    clause2 = profit["mean"] > 0
    out["H3"] = {
        "statement": "surrogate_only's verified objective is worse than surrogate-first's AND "
        "its mean surrogate-minus-twin profit gap on the chosen plan is positive",
        "clause1_gap_difference_so_minus_sf": diff,
        "clause1": "PASS" if clause1 else "FAIL",
        "clause2_mean_profit_gap_feasible_goals": profit,
        "clause2_mean_profit_gap_all_core_goals": boot_mean(so_gaps_all),
        "clause2": "PASS" if clause2 else "FAIL",
        "verdict": "PASS" if (clause1 and clause2) else "FAIL",
    }

    # H4: feasibility >= 90%, and the infeasible goal reports infeasible_unverified.
    rate = feasibility[PRIMARY_SF]["point_all_feasible_goals"]
    rate_excl = feasibility[PRIMARY_SF]["point_excluding_no_feasible_plan_goals"]
    status_rows = [
        {"method": r["method"], "seed": r["seed"], "status": r["status"]}
        for r in rows
        if r["goal"] in infeasible_goals
    ]
    sf_status_ok = all(
        r["status"] == "infeasible_unverified"
        for r in status_rows
        if r["method"] in SURROGATE_FIRST
    )
    out["H4"] = {
        "statement": "feasibility rate >= 90% on held-out feasible goals (point, on re-evaluation) "
        "AND the infeasible-by-design goal returns infeasible_unverified",
        "method": PRIMARY_SF,
        "feasibility_all_feasible_goals": rate,
        "feasibility_excluding_no_feasible_plan_goals": rate_excl,
        "infeasible_goal_statuses": status_rows,
        "infeasible_clause_n_goals": len(infeasible_goals),
        "clause1_all_goals": "PASS" if rate["rate"] >= H4_MIN_FEASIBILITY else "FAIL",
        "clause1_excluding": "PASS"
        if rate_excl["rate"] is not None and rate_excl["rate"] >= H4_MIN_FEASIBILITY
        else "FAIL",
        "clause2": "PASS" if sf_status_ok else "FAIL",
        "verdict": "PASS" if (rate["rate"] >= H4_MIN_FEASIBILITY and sf_status_ok) else "FAIL",
        "verdict_excluding_no_feasible_plan_goals": "PASS"
        if (
            rate_excl["rate"] is not None
            and rate_excl["rate"] >= H4_MIN_FEASIBILITY
            and sf_status_ok
        )
        else "FAIL",
    }
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
#: Reference palette (dataviz skill), light mode; first three slots only, which
#: validate all-pairs for scatter forms. Colour encodes method FAMILY; every
#: method is also direct-labelled, so identity is never colour alone.
FAMILY = {"surrogate-first": "#2a78d6", "twin-only": "#eb6834", "reference / ablation": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _family(method: str) -> str:
    if method in SURROGATE_FIRST:
        return "surrogate-first"
    if method in TWIN_ONLY:
        return "twin-only"
    return "reference / ablation"


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(result: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    gap_goals = result["gap_goals"]
    written = []

    # (a) Cost-quality frontier.
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    _style(ax)
    for method in COMPARED:
        sel = [r for r in rows if r["method"] == method and r["goal"] in gap_goals]
        x = float(np.mean([r["twin_cpu_seconds"] for r in sel]))
        y = result["median_gap"][method]["median"]
        lo, hi = result["median_gap"][method]["ci95"]
        colour = FAMILY[_family(method)]
        estimated = any(r["twin_cpu_estimated"] for r in sel)
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=colour, elinewidth=1.2)
        ax.scatter(
            x,
            y,
            s=70,
            color="white" if estimated else colour,
            edgecolors=colour,
            linewidths=2,
            zorder=3,
        )
        offset = {"sf_tpe": (8, -14), "sf_cmaes": (8, 4)}.get(method, (8, 6))
        ax.annotate(
            method, (x, y), textcoords="offset points", xytext=offset, fontsize=9, color=INK
        )
    # Twin-only anytime curves: search-stream best-so-far, converted to a gap
    # with the same reference and baseline. Optimistic by construction.
    for method in TWIN_ONLY:
        points = []
        for checkpoint in bench.ANYTIME_CHECKPOINTS:
            gaps, cpus = [], []
            for goal in gap_goals:
                ref = result["reference_pooled"][goal]
                base = result["baselines"][goal]["objective"]
                values = []
                for r in rows:
                    if r["method"] != method or r["goal"] != goal:
                        continue
                    # CMA-ES can stop on its own convergence criteria before
                    # 300 evaluations (it did on the 3- and 5-dim goals); its
                    # last snapshot then stands for every later checkpoint.
                    reached = [a for a in r["anytime"] if a["evaluations"] <= checkpoint]
                    snap = reached[-1] if reached else r["anytime"][-1]
                    score = snap["best_feasible_score"]
                    g = gap(score, ref, base) if score is not None else None
                    values.append(1.0 if g is None else g)
                    cpus.append(snap["cpu_seconds"])
                gaps.append(float(np.mean(values)))
            points.append((float(np.mean(cpus)), float(np.median(gaps))))
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=FAMILY["twin-only"], linewidth=2, linestyle="--", alpha=0.7)
        ax.annotate(
            f"{method} anytime",
            (xs[0], ys[0]),
            textcoords="offset points",
            xytext=(-10, 8),
            fontsize=8,
            color=MUTED,
            ha="right",
        )
        for (x, y), checkpoint in zip(points, bench.ANYTIME_CHECKPOINTS, strict=True):
            ax.scatter(x, y, s=24, color=FAMILY["twin-only"], zorder=3)
            ax.annotate(
                f"{checkpoint}",
                (x, y),
                textcoords="offset points",
                xytext=(4, -12),
                fontsize=7,
                color=MUTED,
            )
    ax.set_xscale("log")
    ax.set_xlabel("twin CPU-seconds per cell (log; hollow = estimated, not recorded)", color=INK)
    ax.set_ylabel("median normalised gap to best_known (lower is better)", color=INK)
    ax.set_title(
        "Cost-quality frontier: median gap over goals, 95% bootstrap CI\n"
        "dashed: twin-only anytime best-so-far at 50/150/300 evaluations "
        "(search stream, optimistic)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=c, label=f, markersize=8)
        for f, c in FAMILY.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    path = FIGURES / "d_optim_cost_quality_frontier.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # (b) Per-goal gap, small multiples by method (no colour needed for identity).
    methods = [*COMPARED]
    fig, axes = plt.subplots(1, len(methods), figsize=(14, 4.2), dpi=150, sharey=True)
    goals = result["feasible_goals"]
    for ax, method in zip(axes, methods, strict=True):
        _style(ax)
        colour = FAMILY[_family(method)]
        for yi, goal in enumerate(goals):
            for r in rows:
                if r["method"] != method or r["goal"] != goal:
                    continue
                if r["gap"] is None:
                    ax.scatter(1.05, yi, marker="x", color=MUTED, s=30)
                else:
                    filled = r["reeval"] and r["feasible_point"]
                    ax.scatter(
                        min(r["gap"], 1.05),
                        yi,
                        s=34,
                        color=colour if filled else "white",
                        edgecolors=colour,
                        linewidths=1.5,
                        zorder=3,
                    )
        ax.axvline(H1_MAX_MEDIAN_GAP, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlim(-0.05, 1.12)
        ax.set_title(method, fontsize=9, color=INK)
        ax.set_yticks(range(len(goals)))
        ax.set_yticklabels([g.removeprefix("held_") for g in goals], fontsize=8)
    fig.suptitle(
        "Normalised gap per goal and seed (hollow = no feasible re-evaluated plan, scored 1; "
        "x = no feasible plan found by any method; gaps > 1 drawn at the right edge; "
        "dotted = 5%)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    path = FIGURES / "d_optim_per_goal_gap.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # (c) Winner's curse: surrogate-predicted vs twin-verified profit.
    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=150)
    _style(ax)
    series = {
        "surrogate_only (no verification)": ("surrogate_only", FAMILY["twin-only"]),
        "surrogate-first (verified)": (SURROGATE_FIRST, FAMILY["surrogate-first"]),
    }
    lo, hi = np.inf, -np.inf
    for label, (methods_sel, colour) in series.items():
        sel = [
            r
            for r in rows
            if (
                r["method"] in methods_sel
                if isinstance(methods_sel, tuple)
                else r["method"] == methods_sel
            )
            and r["reeval"]
            and r["surrogate_pred"]
        ]
        predicted = [r["surrogate_pred"]["profit"] for r in sel]
        verified = [r["kpis"]["profit"] for r in sel]
        ax.scatter(
            predicted,
            verified,
            s=34,
            color=colour,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=f"{label}, n={len(sel)}",
        )
        if predicted:
            lo = min(lo, *predicted, *verified)
            hi = max(hi, *predicted, *verified)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        "y = x", (hi, hi), textcoords="offset points", xytext=(-30, -12), fontsize=8, color=MUTED
    )
    ax.set_xlabel("surrogate-predicted profit (GBP)", color=INK)
    ax.set_ylabel("twin profit, 30-rep re-evaluation (GBP)", color=INK)
    ax.set_title(
        "Winner's curse on the chosen plans: below the line = over-promised",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = FIGURES / "d_optim_winners_curse.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate the optimisation benchmark.")
    parser.add_argument("--recompute-baselines", action="store_true")
    args = parser.parse_args()
    result = analyse(recompute_baselines=args.recompute_baselines)
    paths = figures(result)
    bench.write_atomic(result, OUT)
    logger.info("wrote %s and %d figures", OUT, len(paths))
    for name, h in result["hypotheses"].items():
        logger.info("%s: %s", name, h["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
