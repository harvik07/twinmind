"""Stage B of the agent benchmark: B1-B5 exactly as pre-registered, plus figures.

    python scripts/analyse_agent_bench.py

Reads the 80 cells in data/processed/agent_bench/full and the NL audit in
reports/results/agent_bench_audit.json. Writes
reports/results/agent_bench_analysis.json and three figures. Applies the rules
in reports/preregistration_agent_bench.md (including Amendment A.1); nothing
here was tuned after the results were seen. Anything not pre-registered is
labelled POST-HOC and feeds no verdict.

ARMS per (goal, seed), all re-evaluated at 30 reps on the reeval stream:
``agent`` (the critic's final plan), ``agent_nocritic`` (the explorer's first
plan, from the same trace), ``twin_only`` (CMA-ES at equal wall-clock) and
``baseline`` (the current policy, from the agent's paired evaluation).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from twinmind.agents.bench_goals import load_bench_goals
from twinmind.eval.registry import results_dir
from twinmind.optim.contracts import ObjectiveSpec
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("analyse_agent_bench")

CELLS = PROJECT_ROOT / "data" / "processed" / "agent_bench" / "full"
AUDIT = results_dir() / "agent_bench_audit.json"
OUT = results_dir() / "agent_bench_analysis.json"
FIGURES = PROJECT_ROOT / "reports" / "figures"
SEEDS = (1, 2)
ARMS = ("agent", "agent_nocritic", "twin_only", "baseline")
N_BOOT = 10_000
B1_MARGIN = 0.10
B5_SPEC_MATCH = 0.85


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def _beats(evaluation: dict[str, Any]) -> bool:
    """Paired objective CI above 0, or restores feasibility the baseline lacks."""
    return bool(
        evaluation["objective"]["diff"]["ci_low"] > 0
        or (evaluation["feasible_point"] and not evaluation["baseline_feasible_point"])
    )


def arms_for(goal_id: str, seed: int) -> dict[str, dict[str, Any]]:
    """The four arms' re-evaluated outcomes for one (goal, seed)."""
    agent = json.loads((CELLS / f"{goal_id}__agent__seed{seed}.json").read_text(encoding="utf-8"))
    twin = json.loads(
        (CELLS / f"{goal_id}__twin_only__seed{seed}.json").read_text(encoding="utf-8")
    )
    result = agent["result"]
    final = result["baseline_comparison"]
    first = result["trace"]["history"][0]
    out = {
        "agent": {
            "evaluation": final,
            "recommended": bool(result["plan"]["recommended"]),
            "status": result["status"],
            "flags": [f["code"] for f in result["flags"]],
            "closest": result["closest_achievable"],
            "reason": result["plan"]["recommendation_reason"],
            "wall_seconds": agent["agent_wall_seconds"],
            "critic_actions": result["trace"]["critic_actions"],
        },
        "agent_nocritic": {
            "evaluation": first["evaluation"],
            # Pre-registered: recommends the first-round plan whenever it
            # verified feasible on the search stream.
            "recommended": first["status"] == "ok",
        },
        "twin_only": {
            "evaluation": twin["evaluation"],
            "recommended": bool(twin["feasible_on_search"]),
            "wall_seconds": twin["search_seconds"],
            "evaluations": twin["evaluations"],
            "restarts": twin["restarts"],
        },
    }
    obj = final["objective"]["baseline"]
    out["baseline"] = {
        "objective": obj["mean"],
        "objective_ci": [obj["ci_low"], obj["ci_high"]],
        "feasible": bool(final["baseline_feasible_point"]),
    }
    # The twin_only cell's baseline must be the same paired replications.
    twin_base = twin["evaluation"]["objective"]["baseline"]["mean"]
    if abs(twin_base - obj["mean"]) > 1e-6 * max(1.0, abs(obj["mean"])):
        raise ValueError(f"{goal_id} seed {seed}: baselines differ between arms")
    for arm in ("agent", "agent_nocritic", "twin_only"):
        ev = out[arm]["evaluation"]
        out[arm].update(
            {
                "objective": ev["objective"]["plan"]["mean"],
                "feasible": bool(ev["feasible_point"]),
                "robust": bool(ev["feasible_robust"]),
                "beats": _beats(ev),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def boot_mean(values: list[float], seed: int = 0) -> dict[str, Any]:
    v = np.asarray(values, dtype="float64")
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n": len(v),
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyse() -> dict[str, Any]:
    goals = load_bench_goals()
    data = {g.id: {s: arms_for(g.id, s) for s in SEEDS} for g in goals}

    # Reference: best re-evaluated point-feasible plan across all arms and both
    # seeds (a lower bound on the true gap, not an oracle).
    reference: dict[str, float | None] = {}
    for g in goals:
        values = []
        for s in SEEDS:
            arms = data[g.id][s]
            values += [
                arms[a]["objective"]
                for a in ("agent", "agent_nocritic", "twin_only")
                if arms[a]["feasible"]
            ]
            if arms["baseline"]["feasible"]:
                values.append(arms["baseline"]["objective"])
        reference[g.id] = max(values) if values else None

    gaps: dict[str, dict[str, Any]] = {}
    unstable: list[dict[str, Any]] = []
    gap_goals = []
    for g in goals:
        if g.category not in ("feasible", "ambiguous_answerable"):
            continue
        ref = reference[g.id]
        per_arm: dict[str, list[float]] = {a: [] for a in ("agent", "agent_nocritic", "twin_only")}
        for s in SEEDS:
            arms = data[g.id][s]
            base = arms["baseline"]["objective"]
            half = (arms["baseline"]["objective_ci"][1] - arms["baseline"]["objective_ci"][0]) / 2
            # POST-HOC flag (feeds no verdict): a denominator under twice the
            # baseline's own 95% CI half-width means the reference beats the
            # current policy by little more than noise, so the gap is unstable.
            # A zero denominator (the reference IS this seed's baseline) is
            # excluded by the pre-registered rule and recorded here too.
            denominator = (ref - base) if ref is not None else 0.0
            if denominator < 2 * half:
                unstable.append(
                    {
                        "goal": g.id,
                        "seed": s,
                        "denominator": denominator,
                        "baseline_ci_half_width": half,
                        "ratio": denominator / half if half else None,
                        "excluded_by_rule": ref is None or ref <= base,
                    }
                )
            if ref is None or ref <= base:
                continue
            for a in per_arm:
                per_arm[a].append(
                    (ref - arms[a]["objective"]) / denominator if arms[a]["feasible"] else 1.0
                )
        if all(per_arm[a] for a in per_arm):
            gap_goals.append(g.id)
            gaps[g.id] = {a: float(np.mean(v)) for a, v in per_arm.items()}

    # B1
    diffs = [gaps[g]["agent"] - gaps[g]["twin_only"] for g in gap_goals]
    b1 = boot_mean(diffs)
    b1.update(
        {
            "verdict": "PASS" if b1["ci95"][1] < B1_MARGIN else "FAIL",
            "superiority": b1["ci95"][1] < 0,
            "agent_mean_gap": float(np.mean([gaps[g]["agent"] for g in gap_goals])),
            "twin_only_mean_gap": float(np.mean([gaps[g]["twin_only"] for g in gap_goals])),
            "agent_nocritic_mean_gap": float(
                np.mean([gaps[g]["agent_nocritic"] for g in gap_goals])
            ),
            "gap_goals": gap_goals,
        }
    )

    # B2: robust-feasibility rate of RECOMMENDED plans, all goals x seeds.
    def robust_rate(arm: str) -> dict[str, Any]:
        rec = [
            data[g.id][s][arm] for g in goals for s in SEEDS if data[g.id][s][arm]["recommended"]
        ]
        hits = sum(r["robust"] for r in rec)
        return {"robust": hits, "recommended": len(rec), "rate": hits / len(rec) if rec else None}

    b2_agent, b2_twin = robust_rate("agent"), robust_rate("twin_only")
    b2 = {
        "agent": b2_agent,
        "twin_only": b2_twin,
        "agent_nocritic": robust_rate("agent_nocritic"),
        "verdict": "PASS" if (b2_agent["rate"] or 0) >= (b2_twin["rate"] or 0) else "FAIL",
    }

    # B3: bad = recommended AND (not robust OR not beating the baseline).
    def bad(arm: str) -> dict[str, Any]:
        rows = []
        for g in goals:
            for s in SEEDS:
                r = data[g.id][s][arm]
                if r["recommended"] and (not r["robust"] or not r["beats"]):
                    rows.append(
                        {"goal": g.id, "seed": s, "robust": r["robust"], "beats": r["beats"]}
                    )
        return {"count": len(rows), "cells": rows}

    b3: dict[str, Any] = {"agent": bad("agent"), "agent_nocritic": bad("agent_nocritic")}
    b3["verdict"] = "PASS" if b3["agent"]["count"] < b3["agent_nocritic"]["count"] else "FAIL"

    # B4
    infeasible = [g for g in goals if g.category == "infeasible"]
    no_room = [g for g in goals if g.category == "no_room"]
    b4_inf = {
        g.id: {
            s: {
                "status": data[g.id][s]["agent"]["status"],
                "options": len(data[g.id][s]["agent"]["closest"]),
            }
            for s in SEEDS
        }
        for g in infeasible
    }
    inf_ok = {
        gid: all(v["status"] == "infeasible" and v["options"] > 0 for v in per.values())
        for gid, per in b4_inf.items()
    }
    b4_nr = {
        g.id: {
            s: {
                "flagged": "no_improvement" in data[g.id][s]["agent"]["flags"],
                "recommended": data[g.id][s]["agent"]["recommended"],
            }
            for s in SEEDS
        }
        for g in no_room
    }
    nr_ok = {
        gid: all(v["flagged"] and not v["recommended"] for v in per.values())
        for gid, per in b4_nr.items()
    }
    b4 = {
        "infeasible": b4_inf,
        "infeasible_ok": inf_ok,
        "no_room": b4_nr,
        "no_room_ok": nr_ok,
        "verdict": "PASS" if all(inf_ok.values()) and sum(nr_ok.values()) >= 2 else "FAIL",
        "free_dims": {g.id: list(g.spec.free_dims) for g in no_room},
    }

    # B5
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    planner = audit["planner"]
    match = sum(bool(v["match"]) for v in planner.values())
    explainer = audit["explainer"]
    first = sum(
        1
        for v in explainer.values()
        if v["first_attempt_audit"] and not any(v["first_attempt_audit"].values())
    )
    llm_pass = sum(bool(v["llm_passed"]) for v in explainer.values())
    from twinmind.agents.critic import recommendation_reason
    from twinmind.agents.tools import audit_explanation, audit_passes, build_facts

    including_template = 0
    for key, v in explainer.items():
        goal_id, _, seed = key.split("__")
        result = json.loads((CELLS / f"{key}.json").read_text(encoding="utf-8"))["result"]
        spec = ObjectiveSpec.from_dict(result["spec"])
        ev = result["baseline_comparison"]
        facts = build_facts(
            spec,
            result["plan"],
            ev,
            result["flags"],
            result["closest_achievable"],
            result["plan"]["recommended"],
            recommendation_reason(spec, ev, {f["code"] for f in result["flags"]}),
        )
        including_template += audit_passes(audit_explanation(v["text"], facts))
    n_expl = len(explainer)
    b5 = {
        "planner_spec_match": {"hits": match, "n": len(planner), "rate": match / len(planner)},
        "planner_kinds": {
            k: sum(v["kind"] == k for v in planner.values())
            for k in ("spec", "clarify", "refuse", "invalid")
        },
        "planner_failures": {k: v["reasons"] for k, v in planner.items() if not v["match"]},
        "explainer": {
            "n": n_expl,
            "first_attempt_pass": first,
            "llm_text_pass_within_two_attempts": llm_pass,
            "pass_including_template": including_template,
            "criterion": "llm_text_pass_within_two_attempts (template fallback counts as a "
            "failure)",
        },
        "calls_used": audit["calls_used"],
        "call_cap": audit["cap"],
    }
    b5["planner_clause"] = "PASS" if match / len(planner) >= B5_SPEC_MATCH else "FAIL"
    b5["explainer_clause"] = "PASS" if llm_pass == n_expl else "FAIL"
    b5["verdict"] = (
        "PASS" if b5["planner_clause"] == "PASS" and b5["explainer_clause"] == "PASS" else "FAIL"
    )

    # POST-HOC (feeds no verdict): the checker's first-attempt failure reasons
    # for every explanation that needed a retry or fell back, plus a mechanical
    # test for the negation false flag predicted in Amendment A.1 -- a
    # direction word next to a delta of EXACTLY zero. The human reading of each
    # reason is in the model card; only the first attempt's reasons were stored.
    exact_zero = re.compile(r"whose delta is 0 \(")
    negation = re.compile(r"\b(?:not|no|never|without|n't|unchanged|remain\w*|stay\w*|same)\b")
    posthoc = []
    for key, v in explainer.items():
        audit_first = v["first_attempt_audit"] or {}
        if v["mode"] == "llm" and v["attempts"] == 1:
            continue
        for kind, items in audit_first.items():
            for item in items:
                zero = bool(exact_zero.search(item))
                posthoc.append(
                    {
                        "cell": key,
                        "kind": kind,
                        "mode": v["mode"],
                        "exact_zero_delta": zero,
                        "negation_false_flag": zero
                        and bool(negation.search(item.split("(", 1)[-1])),
                        "item": item,
                    }
                )

    # Twin-only search effort (describes N01-N03's "many short restarts").
    effort = {
        g.id: {
            "evaluations": [data[g.id][s]["twin_only"]["evaluations"] for s in SEEDS],
            "restarts": [data[g.id][s]["twin_only"]["restarts"] for s in SEEDS],
        }
        for g in goals
    }
    flags = {g.id: {s: data[g.id][s]["agent"]["flags"] for s in SEEDS} for g in goals}
    return {
        "reference": reference,
        "gaps": gaps,
        "unstable_denominators_posthoc": unstable,
        "B1": b1,
        "B2": b2,
        "B3": b3,
        "B4": b4,
        "B5": b5,
        "explainer_failures_posthoc": posthoc,
        "twin_only_effort": effort,
        "agent_flags": flags,
        "categories": {g.id: g.category for g in goals},
        "wall_seconds": {
            g.id: {s: data[g.id][s]["agent"]["wall_seconds"] for s in SEEDS} for g in goals
        },
        "per_cell": {
            f"{g.id}__seed{s}": {
                a: {k: v for k, v in data[g.id][s][a].items() if k != "evaluation"} for a in ARMS
            }
            for g in goals
            for s in SEEDS
        },
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
#: Reference palette (dataviz skill), first three slots; colour = arm, and every
#: arm is also labelled, so identity is never colour alone.
COLOURS = {"agent": "#2a78d6", "twin_only": "#eb6834", "agent_nocritic": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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
    written = []
    gap_goals = result["B1"]["gap_goals"]

    # (a) cost-quality: gap vs wall-clock, agent and twin_only (equal wall by design).
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    _style(ax)
    for arm in ("agent", "twin_only"):
        xs, ys = [], []
        for g in gap_goals:
            xs.append(float(np.mean(list(result["wall_seconds"][g].values()))))
            ys.append(result["gaps"][g][arm])
        ax.scatter(
            xs,
            ys,
            s=40,
            color=COLOURS[arm],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=f"{arm} (mean gap {np.mean(ys):.3f})",
        )
    for g in gap_goals:
        x = float(np.mean(list(result["wall_seconds"][g].values())))
        ax.plot(
            [x, x],
            [result["gaps"][g]["agent"], result["gaps"][g]["twin_only"]],
            color=GRID,
            linewidth=1,
            zorder=0,
        )
        ax.annotate(
            g,
            (x, max(result["gaps"][g]["agent"], result["gaps"][g]["twin_only"])),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=7,
            color=MUTED,
        )
    ax.set_xlabel("wall-clock per cell, s (twin_only is given the agent's)", color=INK)
    ax.set_ylabel("normalised gap to reference (lower is better)", color=INK)
    ax.set_title(
        "Cost-quality per goal (mean of 2 seeds); grey lines pair the arms",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "f_bench_cost_quality.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # (b) per-goal gap by arm.
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    _style(ax)
    order = sorted(gap_goals)
    for offset, arm in zip((-0.2, 0.0, 0.2), ("agent", "agent_nocritic", "twin_only"), strict=True):
        ax.scatter(
            [result["gaps"][g][arm] for g in order],
            [i + offset for i in range(len(order))],
            s=36,
            color=COLOURS[arm],
            label=arm,
            zorder=3,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel(
        "normalised gap (0 = the reference; 1 = no better than the current policy)", color=INK
    )
    ax.set_title("Per-goal gap by arm, gap goals only", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = FIGURES / "f_bench_per_goal_gap.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    # (c) critic flags table as a figure (goal x seed).
    codes = [
        "weak_kpi_constraint",
        "not_robustly_feasible",
        "novel_plan",
        "no_improvement",
        "infeasible_unverified",
    ]
    goals = sorted(result["agent_flags"])
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=150)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, g in enumerate(goals):
        for s_index, s in enumerate(("1", "2")):
            flags = result["agent_flags"][g].get(s, result["agent_flags"][g].get(int(s), []))
            for j, code in enumerate(codes):
                if code in flags:
                    ax.scatter(
                        j + (s_index - 0.5) * 0.25,
                        i,
                        marker="s",
                        s=60,
                        color=COLOURS["agent"] if s_index == 0 else "#4a3aa7",
                    )
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in codes], fontsize=8, color=INK)
    ax.set_yticks(range(len(goals)))
    ax.set_yticklabels([f"{g} ({result['categories'][g]})" for g in goals], fontsize=8)
    ax.set_xlim(-0.6, len(codes) - 0.4)
    ax.invert_yaxis()
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_title(
        "Agent critic flags per goal: left square seed 1, right square seed 2",
        fontsize=10,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    path = FIGURES / "f_bench_critic_flags.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)
    return written


def main() -> int:
    result = analyse()
    OUT.write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    paths = figures(result)
    for b in ("B1", "B2", "B3", "B4", "B5"):
        logger.info("%s: %s", b, result[b]["verdict"])
    logger.info("wrote %s and %d figures", OUT, len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
