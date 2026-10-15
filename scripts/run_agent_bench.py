"""The Prompt 9 agent benchmark. Pre-registered in reports/preregistration_agent_bench.md.

    python scripts/run_agent_bench.py --workers 3          # 20 goals x 2 seeds, no LLM
    python scripts/run_agent_bench.py --audit              # the NL audit (LLM, capped)
    python scripts/run_agent_bench.py --smoke --workers 2  # 2 DEV goals, seed 1

ARMS, all on the GOLD spec so optimisation is isolated from parsing:

* ``agent`` -- the graph from the explorer on (sf_tpe, 600 TPE trials, one
  round; evaluator; critic with its one retry). No LLM.
* ``agent_nocritic`` -- the explorer's FIRST chosen plan, before the critic had
  any say. Read from the agent cell's trace: same search, no extra cost.
* ``baseline`` -- the current policy, from the agent cell's paired evaluation.
* ``twin_only`` -- CMA-ES on the twin, 5 reps per evaluation, for EXACTLY the
  agent cell's measured wall-clock on that goal and seed; one worker.

Every final plan is re-evaluated at 30 reps on the independent "reeval" stream,
paired by replication with the baseline (goal seed = optimiser seed).

RESUMABLE: one atomic JSON per (goal, arm, seed) cell, plus a manifest; a
twin_only cell is dispatched as soon as its agent cell exists. A failing cell
is written to failures/ and retried on the next run. --max-hours stops starting
cells. Spawn workers, at most 3 by default for this 8 GB laptop.

THE NL AUDIT (--audit) is the only LLM use: planner v3 on the 20 NL goals
(spec match against gold) and the explainer on every agent result (numbers,
directions, reason), sequential and capped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import signal
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from twinmind.agents.bench_goals import BenchGoal, bench_goals_hash, load_bench_goals
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("run_agent_bench")

RESULTS = PROJECT_ROOT / "data" / "processed" / "agent_bench"
PREREG = PROJECT_ROOT / "reports" / "preregistration_agent_bench.md"
SEEDS = (1, 2)
ARMS_RUN = ("agent", "twin_only")
AUDIT_CALL_CAP = 120
#: Smoke: two DEV goals on their own dev contexts, never a benchmark goal.
SMOKE_GOALS = ("dev_max_profit", "dev_pm_only")


# --------------------------------------------------------------------------- #
# Goals, cells, manifest
# --------------------------------------------------------------------------- #
def goals_for(args: argparse.Namespace) -> list[BenchGoal]:
    """Benchmark goals, or the two dev goals for --smoke."""
    if args.smoke:
        from twinmind.optim.goals import load_goals

        out: list[BenchGoal] = []
        for goal in load_goals(split="dev"):
            if goal.name in SMOKE_GOALS:
                spec = goal.spec.to_dict()
                gold = {k: spec[k] for k in ("name", "objective", "sense", "constraints")}
                gold.update({"free_dims": spec["free_dims"], "fixed": spec["fixed"]})
                out.append(
                    BenchGoal(
                        id=goal.name,
                        category="smoke",
                        context_id=goal.spec.context_id,
                        text=goal.name,
                        gold=gold,
                    )
                )
        return out
    goals = load_bench_goals()
    return [g for g in goals if not args.goals or g.id in args.goals]


def cell_name(goal: str, arm: str, seed: int) -> str:
    """Deterministic cell filename."""
    return f"{goal}__{arm}__seed{seed}.json"


def write_atomic(payload: dict[str, Any], path: Path) -> None:
    """Temp file then os.replace, so an interrupted write never looks finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(
        json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temp, path)


def cell_done(path: Path) -> bool:
    """A cell counts only if it parses and says it succeeded."""
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("ok"))
    except json.JSONDecodeError:
        return False


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that invalidates existing cells if it changes."""
    from twinmind.utils.config import load_config

    agent: Any = load_config("agent")
    twin: Any = load_config("twin")
    surrogate = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "models" / "surrogate").glob("member_*.pt")):
        surrogate.update(path.read_bytes())
    behaviour = {
        "explorer": agent.explorer.model_dump(),
        "evaluator": agent.evaluator.model_dump(),
        "critic": agent.critic.model_dump(),
    }
    return {
        "preregistration_hash": _sha(PREREG.read_bytes()) if PREREG.exists() else None,
        "goals_hash": bench_goals_hash(),
        "agent_config_hash": _sha(json.dumps(behaviour, sort_keys=True).encode()),
        "twin_config_hash": _sha(
            json.dumps(twin.model_dump(mode="json"), sort_keys=True, default=str).encode()
        ),
        "surrogate_hash": surrogate.hexdigest()[:16],
        "seeds": list(args.seeds),
        "arms": list(ARMS_RUN),
        "smoke": bool(args.smoke),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


INVALIDATING = (
    "preregistration_hash",
    "goals_hash",
    "agent_config_hash",
    "twin_config_hash",
    "surrogate_hash",
    "seeds",
)


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #
def _init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass


def _toolkit(seed: int) -> Any:
    from twinmind.agents.graph import twin_toolkit
    from twinmind.utils.config import load_config

    config: Any = load_config("agent")
    toolkit = twin_toolkit(config)
    toolkit.goal_seed = seed
    return config, toolkit


def _agent_cell(task: dict[str, Any]) -> dict[str, Any]:
    from twinmind.agents.graph import AgentDeps, run_on_spec
    from twinmind.agents.llm_client import FakeLLM

    config, toolkit = _toolkit(task["seed"])
    spec = _goal(task).spec if task["category"] != "smoke" else _smoke_spec(task)
    started = time.perf_counter()
    # No LLM in a benchmark cell: an empty FakeLLM fails loudly if one is attempted.
    deps = AgentDeps(config=config, llm=FakeLLM([]), toolkit=toolkit)
    result = run_on_spec(spec, deps, task["seed"])
    return {
        "result": result.to_dict(),
        "agent_wall_seconds": time.perf_counter() - started,
    }


def _twin_only_cell(task: dict[str, Any]) -> dict[str, Any]:
    from twinmind.agents.bench import twin_only_timeboxed
    from twinmind.agents.graph import dev_context
    from twinmind.optim.evaluators import TwinEvaluator

    _, toolkit = _toolkit(task["seed"])
    spec = _goal(task).spec if task["category"] != "smoke" else _smoke_spec(task)
    search = TwinEvaluator(
        resources=toolkit.resources,
        context=dev_context(spec.context_id),
        goal_seed=task["seed"],
        n_reps=5,
        stream="search",
    )
    outcome = twin_only_timeboxed(spec, search, task["budget_seconds"], task["seed"])
    scenario = (outcome.scenario or spec.base_scenario()).model_dump()
    evaluation = toolkit.evaluate(spec, scenario)
    return {
        "scenario": scenario,
        "evaluation": evaluation,
        "feasible_on_search": outcome.feasible_on_search,
        "search_kpis": outcome.search_kpis,
        "evaluations": outcome.evaluations,
        "restarts": outcome.restarts,
        "search_seconds": outcome.seconds,
        "budget_seconds": task["budget_seconds"],
        "twin_runs_search": int(search.cost.twin_runs),
    }


def _goal(task: dict[str, Any]) -> BenchGoal:
    return next(g for g in load_bench_goals() if g.id == task["goal"])


def _smoke_spec(task: dict[str, Any]) -> Any:
    from twinmind.optim.goals import load_goals

    return next(g.spec for g in load_goals(split="dev") if g.name == task["goal"])


def run_cell(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point. Never raises: a failure comes back as ok=False."""
    import traceback

    from twinmind.utils.peak_memory import PeakMemory

    memory = PeakMemory().start()
    started = time.perf_counter()
    try:
        body = _agent_cell(task) if task["arm"] == "agent" else _twin_only_cell(task)
        out = {**task, "ok": True, **body}
    except BaseException as error:  # noqa: BLE001 - reported, never raised
        out = {
            **task,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    out["cell_seconds"] = time.perf_counter() - started
    out.update(memory.stop())
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _failed(sink: queue.Queue[dict[str, Any]], task: dict[str, Any], error: BaseException) -> None:
    """A worker that died before returning a result still reports a failure."""
    sink.put({**task, "ok": False, "error": repr(error)})


def run_benchmark(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    out = RESULTS / ("smoke" if args.smoke else "full")
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not args.force_new_manifest:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        conflicts = [k for k in INVALIDATING if existing.get(k) != manifest.get(k)]
        if conflicts:
            raise SystemExit(
                f"refusing to resume: the manifest in {out} disagrees on {conflicts}. "
                "Move the old cells or pass --force-new-manifest."
            )
    else:
        write_atomic(manifest, manifest_path)

    goals = goals_for(args)
    seeds = [1] if args.smoke else list(args.seeds)
    workers = max(1, min(args.workers, 3))
    pending_agent, pending_twin = [], []
    for goal in goals:
        for seed in seeds:
            base = {"goal": goal.id, "category": goal.category, "seed": seed}
            agent_path = out / cell_name(goal.id, "agent", seed)
            if not cell_done(agent_path):
                pending_agent.append({**base, "arm": "agent"})
            elif not cell_done(out / cell_name(goal.id, "twin_only", seed)):
                wall = json.loads(agent_path.read_text(encoding="utf-8"))["agent_wall_seconds"]
                pending_twin.append({**base, "arm": "twin_only", "budget_seconds": wall})
    total = len(goals) * len(seeds) * len(ARMS_RUN)
    logger.info(
        "%d goals x %d seeds x %d arms = %d cells; %d agent + %d twin_only outstanding; %d workers",
        len(goals),
        len(seeds),
        len(ARMS_RUN),
        total,
        len(pending_agent),
        len(pending_twin),
        workers,
    )
    _estimate(pending_agent, pending_twin, workers)

    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else None
    finished: queue.Queue[dict[str, Any]] = queue.Queue()
    pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, maxtasksperchild=4)
    in_flight = done = failed = 0
    started = time.perf_counter()

    def submit(task: dict[str, Any]) -> None:
        nonlocal in_flight
        in_flight += 1
        pool.apply_async(
            run_cell,
            (task,),
            callback=finished.put,
            error_callback=partial(_failed, finished, task),
        )

    # Agent cells first (they set the twin_only budgets); twin_only cells whose
    # agent cell already exists can start straight away.
    for task in [*pending_twin, *pending_agent]:
        submit(task)
    try:
        while in_flight:
            try:
                result = finished.get(timeout=1.0)
            except queue.Empty:
                continue
            in_flight -= 1
            name = cell_name(result["goal"], result["arm"], result["seed"])
            if result.get("ok"):
                write_atomic(result, out / name)
                done += 1
                logger.info(
                    "[%3d] %-26s %-9s seed %d  %6.1fs  peak rss %.2f GB, commit %.2f GB",
                    done,
                    result["goal"],
                    result["arm"],
                    result["seed"],
                    result["cell_seconds"],
                    result.get("peak_rss_gb", float("nan")),
                    result.get("peak_private_gb", float("nan")),
                )
                if result["arm"] == "agent":
                    if deadline is not None and time.monotonic() >= deadline:
                        logger.warning("--max-hours reached: not starting %s twin_only", name)
                        continue
                    submit(
                        {
                            "goal": result["goal"],
                            "category": result["category"],
                            "seed": result["seed"],
                            "arm": "twin_only",
                            "budget_seconds": result["agent_wall_seconds"],
                        }
                    )
            else:
                failed += 1
                write_atomic(result, out / "failures" / name)
                logger.error("FAILED %s: %s", name, result.get("error"))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pool.terminate()
        pool.join()
        raise
    pool.close()
    pool.join()
    logger.info(
        "finished: %d cells written, %d failed, %.1f min",
        done,
        failed,
        (time.perf_counter() - started) / 60,
    )
    write_atomic(
        {
            "written": done,
            "failed": failed,
            "manifest": manifest,
            "elapsed_s": time.perf_counter() - started,
        },
        out / "summary.json",
    )
    return 1 if failed else 0


#: Seconds per agent cell. Measured: the smoke run's agent cells took 76.9 s and
#: 105.5 s on 2 dev goals (2 workers), and the Prompt 8
#: demos 102-205 s, the slow end being a critic retry -- which the 4
#: infeasible goals will trigger. 150 s is the conservative middle. twin_only
#: takes the agent's wall plus its final 30-rep re-evaluation.
MEASURED_AGENT_SECONDS = 150.0
FINAL_EVAL_SECONDS = 12.0


def _estimate(pending_agent: list[Any], pending_twin: list[Any], workers: int) -> None:
    agent = len(pending_agent) * MEASURED_AGENT_SECONDS
    twin = (len(pending_agent) + len(pending_twin)) * (MEASURED_AGENT_SECONDS + FINAL_EVAL_SECONDS)
    for n in sorted({2, 3, workers}):
        hours = (agent + twin) / n / 3600
        logger.info(
            "estimate on %d workers: %.2f h cool, %.2f h at the measured 1.8x hot slowdown",
            n,
            hours,
            1.8 * hours,
        )


# --------------------------------------------------------------------------- #
# NL audit (the only LLM use)
# --------------------------------------------------------------------------- #
def run_audit(args: argparse.Namespace) -> int:
    from twinmind.agents.critic import recommendation_reason
    from twinmind.agents.explainer import explain
    from twinmind.agents.graph import groq_client
    from twinmind.agents.llm_client import CallBudget
    from twinmind.agents.planner import plan_goal, static_context_summary
    from twinmind.agents.planner_eval import spec_match
    from twinmind.agents.tools import audit_explanation, audit_passes, build_facts
    from twinmind.eval.registry import results_dir
    from twinmind.optim.contracts import ObjectiveSpec
    from twinmind.utils.config import load_config

    config: Any = load_config("agent")
    client = groq_client(config)
    out = RESULTS / "full"
    report_path = results_dir() / "agent_bench_audit.json"
    report: dict[str, Any] = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {"planner": {}, "explainer": {}, "calls_used": 0, "cap": args.audit_cap}
    )

    def remaining() -> int:
        return int(args.audit_cap - report["calls_used"])

    for goal in load_bench_goals():
        if goal.id in report["planner"] or remaining() <= 0:
            continue
        budget = CallBudget(client, min(1 + config.budget.max_spec_repairs, remaining()))
        outcome = plan_goal(
            goal.text,
            goal.context_id,
            static_context_summary(goal.context_id),
            budget,
            min(config.budget.max_spec_repairs, budget.max_calls - 1),
            config.planner.max_objective_terms,
            config.planner.max_constraints,
        )
        report["calls_used"] += budget.used
        ok, reasons = (
            spec_match(outcome.spec, goal.gold)
            if outcome.spec is not None
            else (False, [f"planner ended with {outcome.kind!r}"])
        )
        report["planner"][goal.id] = {
            "kind": outcome.kind,
            "match": ok,
            "reasons": reasons,
            "calls": budget.used,
            "spec": outcome.spec.to_dict() if outcome.spec else None,
            "question": outcome.question,
        }
        logger.info("planner %-4s %-8s match=%s", goal.id, outcome.kind, ok)
        write_atomic(report, report_path)

    for path in sorted(out.glob("*__agent__seed*.json")):
        key = path.stem
        if key in report["explainer"] or remaining() <= 0:
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        if not cell.get("ok"):
            continue
        result = cell["result"]
        spec = ObjectiveSpec.from_dict(result["spec"])
        evaluation = result["baseline_comparison"]
        codes = {f["code"] for f in result["flags"]}
        facts = build_facts(
            spec,
            result["plan"],
            evaluation,
            result["flags"],
            result["closest_achievable"],
            result["plan"]["recommended"],
            recommendation_reason(spec, evaluation, codes),
        )
        budget = CallBudget(client, min(1 + config.budget.max_explainer_regenerations, remaining()))
        explanation = explain(
            facts,
            budget,
            min(config.budget.max_explainer_regenerations, budget.max_calls - 1),
            config.explainer.max_words,
        )
        report["calls_used"] += budget.used
        final = audit_explanation(explanation.text, facts)
        report["explainer"][key] = {
            "mode": explanation.mode,
            "attempts": explanation.attempts,
            "first_attempt_audit": explanation.audits[0] if explanation.audits else None,
            "llm_passed": explanation.mode == "llm" and audit_passes(final),
            "calls": budget.used,
            "text": explanation.text,
        }
        logger.info(
            "explainer %-30s mode=%s attempts=%d", key, explanation.mode, explanation.attempts
        )
        write_atomic(report, report_path)
    logger.info("audit calls used %d of %d", report["calls_used"], args.audit_cap)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="The agent benchmark.")
    parser.add_argument("--workers", type=int, default=3, help="At most 3 on this laptop.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--goals", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true", help="2 dev goals, seed 1.")
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--audit", action="store_true", help="Run the NL audit (LLM).")
    parser.add_argument("--audit-cap", type=int, default=AUDIT_CALL_CAP)
    parser.add_argument("--force-new-manifest", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return run_audit(args) if args.audit else run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
