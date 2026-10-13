"""Generate the surrogate's training rollouts and its test splits.

    python scripts/generate_rollouts.py --measure          # time this machine first
    python scripts/generate_rollouts.py --smoke            # 64 scenarios, 2 reps
    python scripts/generate_rollouts.py --contexts 512 --scenarios-per-context 64
    python scripts/generate_rollouts.py --test-splits

DESIGN: A CROSSED BLOCK, NOT A FLAT RANDOM SAMPLE.

The agent's job is to rank scenarios **within one plant state**, so the training
data must contain many scenarios per context. A flat sample of (context,
scenario) pairs with one scenario each would teach the surrogate the marginal
effect of a scenario averaged over states and almost nothing about which plan is
better *here*. So: ``contexts x scenarios_per_context``, 512 x 64 by default.

Scenarios come from ONE scrambled Sobol sequence over the 18 dimensions in
``SCENARIO_DIMS`` order, and context block ``k`` consumes points
``[64k, 64k + 64)``. This matters for two reasons beyond tidiness. Sobol is
balanced in prefixes of ``2^m``, so **any prefix of complete blocks is itself a
valid space-filling design** — which makes the job stoppable at any point and
makes the 2^11..2^15 learning curve trainable from nested prefixes rather than
from five separate samples. And a scenario's global Sobol index is a stable id,
so train/test disjointness can be asserted on ids rather than on floating-point
comparisons.

REPLICATION: INDEPENDENT SEEDS, DELIBERATELY NOT CRN.

Each scenario runs ``n_reps`` replications with seeds from
``SeedSequence((global_seed, scenario_index))``. Common random numbers are right
for *comparing* scenarios and wrong for *training data*: under CRN every
scenario shares the same luck, the surrogate sees that shared noise as a
reproducible function of the inputs, and it learns it as signal. The label is
the mean over reps and the stored std is what the noise-aware evaluation needs.

CHRONOLOGICAL SPLIT. Training contexts take origins strictly before
``TRAIN_ORIGIN_END``; test contexts take later ones. Contexts differ in machine
wear, stock and the demand forecast, so a random split would let the surrogate
see the same forecast origin in both halves.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from twinmind.data.rollouts import (
    LAST_ORIGIN,
    ROLLOUT_DIR,
    TEST_ORIGIN_START,
    TEST_SCRAMBLE,
    TEST_SEED_ROOT,
    TRAIN_SCRAMBLE,
    TRAIN_SEED_ROOT,
    SobolDesign,
    build_manifest,
    manifest_conflicts,
    ood_design,
    origin_for_block,
    rep_seeds,
    shard_path,
    valid_shard,
    write_shard,
)
from twinmind.models.surrogate import KPI_ORDER
from twinmind.twin.context import STATE_FIELDS, TwinContext, sample_context
from twinmind.twin.runner import SCENARIO_DIMS, Scenario, build_resources, run_twin
from twinmind.utils.logging import get_logger

logger = get_logger("generate_rollouts")

_WORKER_STATE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def measure_machine(resources: Any, n_runs: int = 50) -> dict[str, Any]:
    """Time this machine, so the projection below is its number and not mine."""
    physical = _physical_cores()
    logical = os.cpu_count() or 1
    scenario = Scenario()
    context = sample_context(0, cache=getattr(resources.lookup, "cache", None))
    run_twin(scenario, 0, context, resources)

    durations = np.empty(n_runs, dtype="float64")
    for index in range(n_runs):
        started = time.perf_counter()
        run_twin(scenario, index, context, resources)
        durations[index] = time.perf_counter() - started

    result = {
        "physical_cores": physical,
        "logical_cores": logical,
        "n_runs": n_runs,
        "mean_s": float(durations.mean()),
        "p50_s": float(np.percentile(durations, 50)),
        "p95_s": float(np.percentile(durations, 95)),
    }
    logger.info(
        "machine: %s physical cores (%d logical); %.1f ms mean, %.1f p50, %.1f p95 over %d runs",
        physical if physical else "unknown",
        logical,
        1000 * result["mean_s"],
        1000 * result["p50_s"],
        1000 * result["p95_s"],
        n_runs,
    )
    return result


def _physical_cores() -> int | None:
    """Physical core count, or None when it cannot be determined."""
    try:
        import psutil

        return int(psutil.cpu_count(logical=False) or 0) or None
    except ImportError:
        pass
    try:  # pragma: no cover - Windows-specific fallback
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor"
                " | Measure-Object -Property NumberOfCores -Sum).Sum",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out) if out.isdigit() else None
    except Exception:  # noqa: BLE001
        return None


def project(
    measurement: dict[str, Any],
    n_scenarios: int,
    n_reps: int,
    workers: int,
    runs_per_second: float | None = None,
) -> dict[str, Any]:
    """Projected wall-clock for a job of this size on this machine.

    The parallel speed-up is **measured, not assumed**. An efficiency constant
    picked out of the air was the first version here and it was wrong in both
    directions at once: it ignored that this machine has 6 physical cores behind
    12 logical ones, and it ignored that the pool was memory-bound rather than
    CPU-bound until the demand grids were precomputed.

    Measured on this machine (6 physical / 12 logical, 7.3 GB), 5,120-run jobs:

        6 workers, no grid cache   31.9 runs/s   (from the 32-block dev run)
        6 workers + grid cache     27.5 runs/s
        10 workers + grid cache    38.8 runs/s   <- best

    38.8 runs/s against a 7.46 runs/s single-core rate is an effective
    parallelism of 5.2x, which is about what 6 physical cores with
    hyperthreading delivers. The default below is that measured figure; pass
    ``runs_per_second`` to override it on another machine.
    """
    measured_runs_per_second = 38.8
    rate = runs_per_second if runs_per_second else measured_runs_per_second
    core_seconds = n_scenarios * n_reps * measurement["mean_s"]
    total_runs = n_scenarios * n_reps
    wall = total_runs / max(rate, 1e-9)
    return {
        "n_scenarios": n_scenarios,
        "n_reps": n_reps,
        "total_runs": total_runs,
        "workers": workers,
        "measured_runs_per_second": rate,
        "effective_parallelism": rate * measurement["mean_s"],
        "core_hours": core_seconds / 3600.0,
        "projected_wall_hours": wall / 3600.0,
    }


# --------------------------------------------------------------------------- #
# Workers
# --------------------------------------------------------------------------- #
#: Default location of the precomputed demand grids, auto-detected when present.
DEFAULT_DEMAND_CACHE = ROLLOUT_DIR / "demand_grids.npz"

#: Measured on this machine. See reports/data_cards/rollouts.md section 4.
FULL_FRAME_GB_PER_WORKER = 1.33
CACHED_GB_PER_WORKER = 0.15


def _worker_resources(demand_cache: str | None = None) -> Any:
    """Build (once per process) the twin resources a worker needs.

    With ``demand_cache`` the worker loads precomputed quantile grids instead of
    the featurised Favorita frame and the LightGBM boosters. Measured: 1.33 GB
    per worker becomes 0.15 GB, which on a 7.3 GB machine is the difference
    between a four-worker pool and a ten-worker one.

    A cache that was asked for and cannot be read **raises**. Falling back here
    would quietly reinstate the 1.33 GB path the cache exists to avoid, and the
    only symptom would be the pool dying of memory exhaustion several minutes
    later with a numpy allocation error -- which is exactly how the first
    test-splits run failed.
    """
    cached = _WORKER_STATE.get("resources")
    if cached is None:
        if demand_cache:
            from twinmind.twin.demand_proc import CachedDemandModel

            try:
                model = CachedDemandModel.load(Path(demand_cache))
            except Exception as error:  # noqa: BLE001 - re-raised with context
                raise RuntimeError(
                    f"could not load the demand grid cache at {demand_cache!r}: {error}. "
                    "Rebuild it with: python scripts/build_demand_cache.py . "
                    "Refusing to fall back to the full Favorita frame, which costs "
                    f"{FULL_FRAME_GB_PER_WORKER:.2f} GB per worker."
                ) from error
            cached = build_resources(demand_model=model, series_id=model.series_ids[0])
        else:
            cached = build_resources()
        _WORKER_STATE["resources"] = cached
    return cached


def resolve_demand_cache(requested: str | None, workers: int, allow_full_frame: bool) -> str | None:
    """Decide what cache the workers will use, and refuse unsafe combinations.

    Three guards, each for a failure that actually happened:

    * A cache **asked for but unreadable** stops the job here, before a pool is
      started, naming the file.
    * **Many workers with no cache** is refused outright. An eleven-worker pool
      on the full frame died with a numpy allocation error after several
      minutes; failing in the first second with an explanation is better.
    * The **default path is auto-detected** when it exists, so forgetting the
      flag costs nothing.
    """
    from twinmind.twin.demand_proc import CachedDemandModel

    path = requested
    if path is None and DEFAULT_DEMAND_CACHE.exists():
        path = str(DEFAULT_DEMAND_CACHE)
        logger.info("auto-detected demand grid cache at %s", path)

    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise SystemExit(
                f"no demand grid cache at {resolved}. "
                "Build it first: python scripts/build_demand_cache.py"
            )
        try:
            model = CachedDemandModel.load(resolved)
        except Exception as error:  # noqa: BLE001 - re-raised as a clear exit
            raise SystemExit(
                f"the demand grid cache at {resolved} could not be read: {error}. "
                "Rebuild it: python scripts/build_demand_cache.py"
            ) from error
        logger.info(
            "demand grid cache: %d grids, series %s (~%.2f GB/worker)",
            len(model.grids),
            ", ".join(model.series_ids),
            CACHED_GB_PER_WORKER,
        )
        return str(resolved)

    if workers > 1 and not allow_full_frame:
        raise SystemExit(
            f"refusing to start {workers} workers without a demand grid cache. "
            f"Each would rebuild the featurised Favorita frame and hold about "
            f"{FULL_FRAME_GB_PER_WORKER:.2f} GB, against "
            f"{CACHED_GB_PER_WORKER:.2f} GB with the cache; {workers} workers need "
            f"roughly {workers * FULL_FRAME_GB_PER_WORKER:.1f} GB and will die of "
            "memory exhaustion partway through. "
            "Either build the cache (python scripts/build_demand_cache.py) "
            "or pass --allow-full-frame if you really mean it."
        )
    logger.warning(
        "running without a demand grid cache: ~%.2f GB per worker",
        FULL_FRAME_GB_PER_WORKER,
    )
    return None


def _init_worker() -> None:
    """Make workers ignore Ctrl-C so the parent can shut the pool down cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_block(task: dict[str, Any]) -> dict[str, Any]:
    """Simulate one context block and return its rows.

    Module-level and returning plain data, because under "spawn" everything
    crossing the process boundary is pickled: a closure would not survive, and
    returning a DataFrame would pickle the index machinery for no benefit.
    """
    started = time.perf_counter()
    resources = _worker_resources(task.get("demand_cache"))
    design: SobolDesign = task["design"]
    block: int = task["block"]
    per_block: int = task["scenarios_per_context"]
    n_reps: int = task["n_reps"]
    seed_root: int = task["seed_root"]

    context = _context_for(task, resources)
    state = np.asarray(context.state_vector(), dtype="float64")

    rows: list[dict[str, Any]] = []
    for position in range(per_block):
        scenario_index = task["index_offset"] + position
        if scenario_index >= len(design):
            break
        scenario = design.scenario(scenario_index)
        seeds = rep_seeds(seed_root, scenario_index, n_reps)

        values = np.empty((n_reps, len(KPI_ORDER)), dtype="float64")
        for rep, seed in enumerate(seeds):
            values[rep] = run_twin(scenario, seed, context, resources).to_vector()

        row: dict[str, Any] = {
            "scenario_index": scenario_index,
            "context_block": block,
            "origin": str(task["origin"]),
            "n_reps": n_reps,
            "seeds": json.dumps(seeds),
        }
        row.update(dict(zip(SCENARIO_DIMS, scenario.to_vector(), strict=True)))
        row.update({f"state_{f}": v for f, v in zip(STATE_FIELDS, state, strict=True)})
        # KPI columns carry a "kpi_" prefix because three STATE fields are named
        # demand_q10_mean / q50 / q90, so a loader selecting targets by
        # `endswith("_mean")` would silently pull three context features in as
        # KPIs. Prefixing makes the target set unambiguous by construction.
        row.update(
            {f"kpi_{k}_mean": float(v) for k, v in zip(KPI_ORDER, values.mean(axis=0), strict=True)}
        )
        row.update(
            {
                f"kpi_{k}_std": float(v)
                for k, v in zip(KPI_ORDER, values.std(axis=0, ddof=1), strict=True)
            }
        )
        rows.append(row)

    return {
        "block": block,
        "rows": rows,
        "duration_s": time.perf_counter() - started,
        "worker_rss_gb": _worker_rss_gb(),
    }


def _worker_rss_gb() -> float:
    """This worker's resident memory, so the pool's footprint is observable.

    Reported per shard because the failure this guards against -- every worker
    silently rebuilding the 0.71 GB Favorita frame -- is invisible until the
    machine runs out of memory. A number in the log makes it obvious in the
    first shard instead of the fiftieth.
    """
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / 1e9)
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        return float("nan")


def _context_for(task: dict[str, Any], resources: Any) -> TwinContext:
    """The context a block runs at, rebuilt identically in every process.

    Rebuilt from its seed rather than pickled across the boundary: a TwinContext
    carries numpy arrays and a Timestamp, and reconstructing it from an integer
    is both cheaper and immune to a pickle round-trip changing a float.
    """
    return sample_context(
        task["context_seed"],
        cache=getattr(resources.lookup, "cache", None),
        series_id=resources.series_id,
        origin=pd.Timestamp(task["origin"]),
        n_machines=len(resources.config.machines),
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunSettings:
    """Everything every split shares: workers, replications and the demand cache.

    This exists because of a real bug, not for tidiness. ``demand_cache`` used
    to be an optional keyword on ``generate``, and the three test-split call
    sites simply did not pass it -- so every test-split worker silently rebuilt
    the 0.71 GB Favorita frame and the pool died of memory exhaustion, while the
    training path, which did pass it, worked fine.

    Bundling the shared settings into one **required** argument makes that
    divergence impossible to express: a call site either passes the settings, in
    which case it passes the cache, or it fails with a TypeError.
    """

    workers: int
    n_reps: int
    demand_cache: str | None


def build_tasks(
    directory: Path,
    design: SobolDesign,
    n_blocks: int,
    scenarios_per_context: int,
    seed_root: int,
    settings: RunSettings,
    test: bool,
    context_seed_offset: int,
) -> list[dict[str, Any]]:
    """The outstanding blocks as worker tasks. The ONLY place a task is built.

    Both the training path and the test-splits path come through here, so the
    cache cannot reach one and not the other.
    """
    tasks: list[dict[str, Any]] = []
    for block in range(n_blocks):
        if valid_shard(shard_path(directory, block), scenarios_per_context):
            continue
        tasks.append(
            {
                "block": block,
                "design": design,
                "index_offset": block * scenarios_per_context,
                "scenarios_per_context": scenarios_per_context,
                "n_reps": settings.n_reps,
                "seed_root": seed_root,
                "origin": str(origin_for_block(block, n_blocks, test=test).date()),
                "context_seed": context_seed_offset + block,
                "demand_cache": settings.demand_cache,
            }
        )
    return tasks


def generate(
    directory: Path,
    design: SobolDesign,
    n_blocks: int,
    scenarios_per_context: int,
    seed_root: int,
    manifest: dict[str, Any],
    settings: RunSettings,
    test: bool = False,
    context_seed_offset: int = 0,
) -> dict[str, Any]:
    """Run every outstanding block, in parallel, writing one shard each.

    Resumable by construction: a block whose shard already reads back with the
    right row count is skipped, so an interrupted multi-hour job restarts where
    it stopped rather than from zero.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        conflicts = manifest_conflicts(existing, manifest)
        if conflicts:
            raise SystemExit(
                "refusing to resume: the manifest in "
                f"{directory} disagrees on {conflicts}.\n"
                "The twin or its assumptions changed since those shards were written, so "
                "mixing them would train on two different simulators.\n"
                "Pass --force-new-manifest to overwrite, after deleting or moving the old shards."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )

    tasks = build_tasks(
        directory,
        design,
        n_blocks,
        scenarios_per_context,
        seed_root,
        settings,
        test,
        context_seed_offset,
    )
    done = n_blocks - len(tasks)
    if done:
        logger.info("resuming: %d of %d blocks already complete", done, n_blocks)
    if not tasks:
        logger.info("nothing to do; all %d blocks present", n_blocks)
        return {"blocks": n_blocks, "written": 0, "skipped": done}

    logger.info(
        "%d blocks x %d scenarios x %d reps = %d twin runs on %d workers (%s)",
        len(tasks),
        scenarios_per_context,
        settings.n_reps,
        len(tasks) * scenarios_per_context * settings.n_reps,
        settings.workers,
        "grid cache" if settings.demand_cache else "FULL FRAME, ~1.33 GB/worker",
    )

    started = time.perf_counter()
    written = 0
    durations: list[float] = []
    worker_rss: list[float] = []
    context = mp.get_context("spawn")
    pool = context.Pool(settings.workers, initializer=_init_worker)
    try:
        for result in pool.imap_unordered(run_block, tasks):
            write_shard(result["rows"], shard_path(directory, result["block"]))
            written += 1
            durations.append(result["duration_s"])
            rss = result.get("worker_rss_gb")
            if rss == rss and rss is not None:  # not NaN
                worker_rss.append(float(rss))
            _report_progress(written, len(tasks), durations, started)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        logger.warning("interrupted; %d shards written and valid, rerun to resume", written)
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    elapsed = time.perf_counter() - started
    logger.info(
        "wrote %d shards in %.1f min (%.1f s/shard)",
        written,
        elapsed / 60.0,
        elapsed / max(written, 1),
    )
    if worker_rss:
        peak = max(worker_rss)
        logger.info(
            "worker memory: %.2f GB mean, %.2f GB peak per worker; "
            "~%.1f GB total across %d workers (%s)",
            float(np.mean(worker_rss)),
            peak,
            peak * settings.workers,
            settings.workers,
            "as expected with the grid cache"
            if peak < 0.5
            else "HIGH -- is the grid cache reaching the workers?",
        )
    return {
        "blocks": n_blocks,
        "written": written,
        "skipped": done,
        "elapsed_s": elapsed,
        "mean_shard_s": float(np.mean(durations)) if durations else 0.0,
        "worker_rss_gb_mean": float(np.mean(worker_rss)) if worker_rss else float("nan"),
        "worker_rss_gb_peak": float(max(worker_rss)) if worker_rss else float("nan"),
    }


def _report_progress(done: int, total: int, durations: list[float], started: float) -> None:
    """A progress line with a measured ETA, every shard or every 10."""
    if done != total and done % max(1, total // 50) and done % 10:
        return
    elapsed = time.perf_counter() - started
    rate = done / max(elapsed, 1e-9)
    remaining = (total - done) / max(rate, 1e-9)
    filled = int(30 * done / max(total, 1))
    bar = "#" * filled + "-" * (30 - filled)
    logger.info(
        "[%s] %d/%d shards  %.0f%%  elapsed %s  eta %s  (%.1f s/shard)",
        bar,
        done,
        total,
        100.0 * done / max(total, 1),
        _hms(elapsed),
        _hms(remaining),
        float(np.mean(durations[-20:])),
    )


def _hms(seconds: float) -> str:
    """Seconds as H:MM:SS."""
    seconds = max(int(seconds), 0)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# --------------------------------------------------------------------------- #
# Test splits
# --------------------------------------------------------------------------- #
def typical_context_seed(resources: Any, candidates: int = 64) -> int:
    """Pick the T2 context: the one closest to the median state.

    Documented choice, not an arbitrary one. T2 measures ranking within a single
    plant state, so that state should be representative rather than extreme --
    a context with every machine nearly dead would make the maintenance
    dimensions dominate and flatter the surrogate on an unusual problem. The
    median is taken over the standardised state vector, and the seed whose state
    is nearest it in Euclidean distance wins.
    """
    cache = getattr(resources.lookup, "cache", None)
    states = []
    for seed in range(candidates):
        context = sample_context(
            seed, cache=cache, series_id=resources.series_id, origin=TEST_ORIGIN_START
        )
        states.append(context.state_vector())
    matrix = np.asarray(states, dtype="float64")
    centre = np.median(matrix, axis=0)
    spread = np.where(matrix.std(axis=0) < 1e-12, 1.0, matrix.std(axis=0))
    distance = np.linalg.norm((matrix - centre) / spread, axis=1)
    chosen = int(np.argmin(distance))
    logger.info(
        "T2 context: seed %d of %d candidates, distance %.3f from the median state "
        "(worst candidate %.3f)",
        chosen,
        candidates,
        distance[chosen],
        distance.max(),
    )
    return chosen


def generate_test_splits(
    resources: Any,
    args: argparse.Namespace,
    bounds: list[tuple[float, float]],
    settings: RunSettings,
) -> dict[str, Any]:
    """T1, T2 and OOD, all with later origins and disjoint seeds.

    ``settings`` is required and carries the demand cache to every split. It
    used to be an optional keyword that each call site passed individually, and
    all three of the ones below forgot it.
    """
    manifest = build_manifest(resources, args)
    manifest.update(
        {
            "split": "test",
            "scramble_seed": TEST_SCRAMBLE,
            "seed_root": TEST_SEED_ROOT,
            "test_origin_window": [str(TEST_ORIGIN_START.date()), str(LAST_ORIGIN.date())],
            "n_reps": args.test_reps,
        }
    )
    summary: dict[str, Any] = {}
    # A smoke run writes to its own directories. Sharing them with the real
    # splits would either clash with a stored manifest (different n_reps) or,
    # worse, leave 2-rep shards sitting where 30-rep ones belong.
    prefix = "smoke_" if getattr(args, "smoke", False) else ""

    # T1: i.i.d. accuracy over many contexts.
    t1_dir = ROLLOUT_DIR / f"{prefix}test_t1"
    t1_design = SobolDesign.build(args.t1_contexts * args.t1_scenarios, TEST_SCRAMBLE, bounds)
    t1_manifest = {**manifest, "scenarios_per_context": args.t1_scenarios, "split_name": "T1"}
    summary["T1"] = generate(
        t1_dir,
        t1_design,
        args.t1_contexts,
        args.t1_scenarios,
        TEST_SEED_ROOT,
        t1_manifest,
        settings,
        test=True,
        context_seed_offset=10_000,
    )

    # T2: ranking within ONE typical context, two independent 30-rep estimates
    # so the twin-vs-twin agreement gives the ceiling a perfect surrogate hits.
    chosen_seed = typical_context_seed(resources)
    t2_design = SobolDesign.build(args.t2_scenarios, TEST_SCRAMBLE + 1, bounds)
    for replicate_id, root in enumerate((TEST_SEED_ROOT + 500, TEST_SEED_ROOT + 900)):
        t2_dir = ROLLOUT_DIR / f"{prefix}test_t2_estimate{replicate_id + 1}"
        t2_manifest = {
            **manifest,
            "scenarios_per_context": args.t2_scenarios,
            "split_name": f"T2_estimate{replicate_id + 1}",
            "seed_root": root,
            "context_seed": chosen_seed,
        }
        summary[f"T2_estimate{replicate_id + 1}"] = generate(
            t2_dir,
            t2_design,
            1,
            args.t2_scenarios,
            root,
            t2_manifest,
            settings,
            test=True,
            context_seed_offset=chosen_seed,
        )

    # OOD: at least 3 dimensions at a bound.
    ood_dir = ROLLOUT_DIR / f"{prefix}test_ood"
    design = ood_design(args.ood_scenarios, TEST_SCRAMBLE + 2, bounds)
    per_block = min(args.t1_scenarios, args.ood_scenarios)
    n_blocks = max(args.ood_scenarios // per_block, 1)
    ood_manifest = {**manifest, "scenarios_per_context": per_block, "split_name": "OOD"}
    summary["OOD"] = generate(
        ood_dir,
        design,
        n_blocks,
        per_block,
        TEST_SEED_ROOT + 1300,
        ood_manifest,
        settings,
        test=True,
        context_seed_offset=20_000,
    )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Generate surrogate rollouts.")
    parser.add_argument("--contexts", type=int, default=512)
    parser.add_argument("--scenarios-per-context", type=int, default=64)
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 1")
    parser.add_argument(
        "--out", default=None, help="Output directory (default data/processed/rollouts/train)."
    )
    parser.add_argument("--smoke", action="store_true", help="1 context x 64 scenarios x 2 reps.")
    parser.add_argument("--dev", action="store_true", help="32 contexts x 64 scenarios (2,048).")
    parser.add_argument(
        "--measure", action="store_true", help="Time the machine and project, then exit."
    )
    parser.add_argument("--test-splits", action="store_true", help="Generate T1, T2 and OOD.")
    parser.add_argument("--test-reps", type=int, default=30)
    parser.add_argument("--t1-contexts", type=int, default=40)
    parser.add_argument("--t1-scenarios", type=int, default=25)
    parser.add_argument("--t2-scenarios", type=int, default=1000)
    parser.add_argument("--ood-scenarios", type=int, default=300)
    parser.add_argument("--force-new-manifest", action="store_true")
    parser.add_argument(
        "--allow-full-frame",
        action="store_true",
        help="Permit a multi-worker run with no demand cache (~1.33 GB/worker).",
    )
    parser.add_argument(
        "--demand-cache",
        default=None,
        help="Precomputed demand grids (build with scripts/build_demand_cache.py). "
        "Cuts worker memory from 1.33 GB to about 0.47 GB.",
    )
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    if args.workers <= 0:
        args.workers = max((os.cpu_count() or 2) - 1, 1)

    demand_cache = resolve_demand_cache(args.demand_cache, args.workers, args.allow_full_frame)
    resources = build_resources()
    bounds = scenario_bounds_list(resources)

    measurement = measure_machine(resources, n_runs=50 if args.measure else 20)
    if args.measure:
        for label, n_scen, reps in (
            ("training 32,768 x 10", 32_768, 10),
            ("T1 1,000 x 30", 1_000, 30),
            ("T2 2 x 1,000 x 30", 2_000, 30),
            ("OOD 300 x 30", 300, 30),
        ):
            projection = project(measurement, n_scen, reps, args.workers)
            logger.info(
                "%-22s %9d runs  %6.1f core-hours  -> %5.2f h wall "
                "(%.1f runs/s, %.1fx parallelism)",
                label,
                projection["total_runs"],
                projection["core_hours"],
                projection["projected_wall_hours"],
                projection["measured_runs_per_second"],
                projection["effective_parallelism"],
            )
        return 0

    if args.smoke:
        args.contexts, args.scenarios_per_context, args.n_reps = 1, 64, 2
        directory = ROLLOUT_DIR / "smoke"
    elif args.dev:
        args.contexts, args.scenarios_per_context = 32, 64
        directory = ROLLOUT_DIR / "dev"
    else:
        directory = Path(args.out) if args.out else ROLLOUT_DIR / "train"

    if args.smoke and args.test_splits:
        # A smoke run of the TEST SPLITS, so the whole path -- including each
        # worker's memory profile -- can be exercised in about a minute rather
        # than the 42 it would take to discover a problem the slow way.
        args.t1_contexts, args.t1_scenarios = 2, 2
        args.t2_scenarios, args.ood_scenarios, args.test_reps = 2, 2, 2

    if args.test_splits:
        settings = RunSettings(
            workers=args.workers, n_reps=args.test_reps, demand_cache=demand_cache
        )
        summary = generate_test_splits(resources, args, bounds, settings)
        logger.info("test splits: %s", json.dumps({k: v["blocks"] for k, v in summary.items()}))
        return 0

    if args.force_new_manifest:
        stale = directory / "manifest.json"
        if stale.exists():
            stale.unlink()
            logger.warning("removed the existing manifest at %s", stale)

    design = SobolDesign.build(args.contexts * args.scenarios_per_context, TRAIN_SCRAMBLE, bounds)
    manifest = build_manifest(resources, args)
    manifest["split"] = "train"
    settings = RunSettings(workers=args.workers, n_reps=args.n_reps, demand_cache=demand_cache)
    projection = project(
        measurement, args.contexts * args.scenarios_per_context, args.n_reps, args.workers
    )
    logger.info(
        "projected %.2f h wall on %d workers (%.1f core-hours)",
        projection["projected_wall_hours"],
        args.workers,
        projection["core_hours"],
    )
    generate(
        directory,
        design,
        args.contexts,
        args.scenarios_per_context,
        TRAIN_SEED_ROOT,
        manifest,
        settings,
    )
    return 0


def scenario_bounds_list(resources: Any) -> list[tuple[float, float]]:
    """Bounds in SCENARIO_DIMS order, from twin.yaml."""
    lookup = {d.name: (float(d.min), float(d.max)) for d in resources.config.scenario_space}
    return [lookup[name] for name in SCENARIO_DIMS]


if __name__ == "__main__":
    sys.exit(main())
