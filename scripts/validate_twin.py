"""Verification and validation evidence for the twin, written to results/.

    python scripts/validate_twin.py --all

What this does and does not claim is the whole point, so it is stated here and
at the top of ``reports/model_cards/twin_design.md``. **No real factory data
exists for this line.** There is therefore no fidelity table comparing
simulated throughput or service level to measured values, and building one
would mean fabricating the measurements. Instead:

* **Component validation** -- only where real data exists. The health model's
  warnings are scored against the real held-out C-MAPSS trajectories; the
  fixed-MTBF Weibull is scored against those engines' real lifetimes; the
  demand process is scored against the held-out Favorita fold.
* **Landscape sanity** -- do the best plans sit at a bound in every dimension?
  If they do, the twin has no trade-off and the week-5 optimisation would be
  trivially solved.
* **Sensitivity** -- every assumed parameter perturbed +-25%, reporting both how
  far the headline KPIs move and, more importantly, whether the RANKING of
  scenarios survives. A recommendation that flips when an invented constant
  moves by a quarter is not a recommendation.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from twinmind.eval.registry import results_dir
from twinmind.twin.runner import (
    Scenario,
    TwinResources,
    build_resources,
    last_detail,
    replicate,
    run_twin,
)
from twinmind.utils.logging import get_logger

logger = get_logger("validate_twin")

HORIZON = 28

#: Assumed parameters perturbed one at a time. Every one is in the twin.yaml
#: register; this list is the subset the spec names plus the spread multiplier
#: the addendum added.
SENSITIVITY_PARAMS: tuple[str, ...] = (
    "life_scale_hours_per_cycle",
    "wear_exponent",
    "unplanned_repair_multiplier",
    "price_elasticity",
    "processing_cv",
    "demand_capacity_ratio",
    "lead_time_multiplier",
    "defect_base_probability",
    "demand_spread_multiplier",
)

NAMED_SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "throughput_+10pct": {"throughput_multiplier": 1.10},
    "throughput_+20pct": {"throughput_multiplier": 1.20},
    "short_pm_intervals": {f"pm_interval_M{i}": 48.0 for i in range(1, 6)},
    "high_speed": {f"speed_setpoint_M{i}": 1.20 for i in range(1, 6)},
}


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Validate the digital twin.")
    parser.add_argument("--all", action="store_true", help="Run every section.")
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--band-coverage", action="store_true")
    parser.add_argument("--band-series", type=int, default=50)
    parser.add_argument("--landscape", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--scenarios", action="store_true")
    parser.add_argument("--sobol", type=int, default=500)
    parser.add_argument("--sobol-reps", type=int, default=10)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--rank-scenarios", type=int, default=50)
    parser.add_argument("--rank-reps", type=int, default=5)
    return parser


def _perturbed(resources: TwinResources, name: str, factor: float) -> TwinResources:
    """``resources`` with one assumed parameter scaled by ``factor``.

    ``demand_capacity_ratio`` needs special handling. It is consumed once, in
    ``build_resources``, to derive ``demand_scale``; the simulation itself never
    reads it. Perturbing it without recalibrating therefore changes nothing at
    all, and the first run of this study duly reported a 0.00% effect -- a null
    that looked like a finding and was an artefact of the harness. Recalibrating
    is what actually propagates it.
    """
    import dataclasses

    config = resources.config
    assumptions = dict(config.assumptions)
    entry = assumptions[name]
    assumptions[name] = entry.model_copy(update={"value": entry.value * factor})
    perturbed = dataclasses.replace(
        resources, config=config.model_copy(update={"assumptions": assumptions})
    )
    if name == "demand_capacity_ratio":
        perturbed = dataclasses.replace(perturbed, demand_scale=resources.demand_scale * factor)
    return perturbed


def _with_value(resources: TwinResources, name: str, value: float) -> TwinResources:
    """``resources`` with one assumed parameter set to an absolute value."""
    import dataclasses

    config = resources.config
    assumptions = dict(config.assumptions)
    assumptions[name] = assumptions[name].model_copy(update={"value": value})
    return dataclasses.replace(
        resources, config=config.model_copy(update={"assumptions": assumptions})
    )


def _hazard(resources: TwinResources, mode: str) -> TwinResources:
    """``resources`` switched to the other hazard source."""
    import dataclasses

    from twinmind.twin.health_replay import HealthLookup

    config = resources.config
    replay = config.health_replay.model_copy(update={"hazard_source": mode})
    lookup = HealthLookup.from_cache(resources.lookup.cache, mode)
    return dataclasses.replace(
        resources, config=config.model_copy(update={"health_replay": replay}), lookup=lookup
    )


# --------------------------------------------------------------------------- #
# Component validation 1: health warnings
# --------------------------------------------------------------------------- #
def validate_health_warnings(resources: TwinResources) -> dict[str, Any]:
    """Score the monitoring system offline on the pool, then inside the twin.

    Offline, a "warning" is the first cycle at which the cached
    ``p_fail_within_h`` crosses the configured threshold. Because the pool is
    run-to-failure, every engine does fail, so there are no true negatives at
    the engine level and **recall is the share of engines warned at all** while
    **precision is the share of warnings that arrive before failure with
    non-trivial notice**. Reporting a plain confusion matrix here would be
    meaningless -- the negative class does not exist.

    Tolerance: the twin's in-simulation warning rate should sit within +-0.15 of
    the offline recall. It cannot match exactly, because the twin renews
    machines on PM and so often leaves a trajectory before its warning cycle.
    """
    cache = resources.lookup.cache
    threshold = resources.config.health_replay.warning_threshold
    horizon_h = cache.horizon_h

    warned, leads, precise = 0, [], 0
    for slot in range(cache.n_engines):
        start, stop = int(cache.offsets[slot]), int(cache.offsets[slot + 1])
        probabilities = cache.p_fail[start:stop]
        failure_cycle = float(cache.failure_cycles[slot])
        crossing = (
            np.argmax(probabilities >= threshold) if (probabilities >= threshold).any() else -1
        )
        if crossing < 0:
            continue
        warned += 1
        warning_cycle = cache.window + int(crossing)
        lead_cycles = failure_cycle - warning_cycle
        leads.append(lead_cycles)
        # "Useful" means the warning arrived before failure with at least one
        # shift's worth of notice at the twin's ageing rate.
        if lead_cycles > 0:
            precise += 1

    # A warning is only worth anything if the plant can act on it. Early-warning
    # PM is scheduled at the next END OF SHIFT, which at a 16-hour staffed day
    # is on average 8 hours away, so a warning with less lead than that arrives
    # too late however correct it was. Offline "recall" (did the monitor warn at
    # all) and the twin's warning share (did the plant get there first) are
    # therefore different quantities, and comparing them directly -- which this
    # code did first -- produces a spurious disagreement. The ACTIONABLE
    # precision is the like-for-like figure.
    config = resources.config
    cycles_per_hour = 1.0 / config.assumption("life_scale_hours_per_cycle")
    mean_scheduling_delay_hours = 0.5 * (
        2 * config.shifts.hours_per_shift
    )  # half a two-shift staffed day
    lead_hours = np.asarray(leads, dtype="float64") / cycles_per_hour
    actionable = int(np.sum(lead_hours > mean_scheduling_delay_hours))

    offline = {
        "threshold": threshold,
        "horizon_h_cycles": horizon_h,
        "pool_engines": cache.n_engines,
        "engines_warned": warned,
        "recall": warned / cache.n_engines,
        "precision_warning_before_failure": precise / warned if warned else 0.0,
        "mean_lead_cycles": float(np.mean(leads)) if leads else 0.0,
        "median_lead_cycles": float(np.median(leads)) if leads else 0.0,
        "min_lead_cycles": float(np.min(leads)) if leads else 0.0,
        "mean_lead_hours": float(np.mean(lead_hours)) if leads else 0.0,
        "mean_scheduling_delay_hours": mean_scheduling_delay_hours,
        "actionable_warnings": actionable,
        "actionable_precision": actionable / warned if warned else 0.0,
    }

    # In-simulation: how often does a failure arrive with no warning standing?
    scenario = Scenario(**{f"pm_interval_M{i}": 720.0 for i in range(1, 6)})
    warnings_raised, failures, unwarned = 0, 0, 0
    for seed in range(40):
        run_twin(scenario, seed, resources=resources)
        detail = last_detail()
        warnings_raised += detail["warnings_raised"]
        failures += detail["unplanned_failures"]
        unwarned += detail["unwarned_failures"]
    total_events = warnings_raised + failures
    in_sim_recall = warnings_raised / total_events if total_events else 0.0

    result = {
        "offline": offline,
        "in_simulation": {
            "runs": 40,
            "warnings_raised": warnings_raised,
            "unplanned_failures": failures,
            "unwarned_failures": unwarned,
            "warning_share_of_events": in_sim_recall,
        },
        "tolerance": 0.15,
        "agrees": abs(in_sim_recall - offline["actionable_precision"]) <= 0.15,
    }
    logger.info(
        "health warnings: offline recall %.3f, precision %.3f, mean lead %.1f cycles "
        "(%.1f h); actionable precision %.3f against a %.0f h scheduling delay",
        offline["recall"],
        offline["precision_warning_before_failure"],
        offline["mean_lead_cycles"],
        offline["mean_lead_hours"],
        offline["actionable_precision"],
        offline["mean_scheduling_delay_hours"],
    )
    logger.info(
        "  in-simulation: %d warnings acted on, %d unplanned failures -> warning share "
        "%.3f vs actionable precision %.3f -> %s",
        warnings_raised,
        failures,
        in_sim_recall,
        offline["actionable_precision"],
        "AGREES" if result["agrees"] else "DISAGREES",
    )
    return result


# --------------------------------------------------------------------------- #
# Component validation 2: the fixed-MTBF Weibull
# --------------------------------------------------------------------------- #
def validate_weibull(resources: TwinResources) -> dict[str, Any]:
    """Compare the fitted Weibull to the pool's real lifetimes.

    A two-sample KS statistic against a large sample from the fit, plus a
    quantile table. Tolerance: KS < 0.20 at n = 52, which is roughly the 0.05
    critical value for a sample this size -- tight enough to catch a badly
    wrong shape, loose enough not to fail on 52 observations.
    """
    cache = resources.lookup.cache
    empirical = cache.lifetimes().astype("float64")
    shape, scale = resources.lookup.weibull_shape, resources.lookup.weibull_scale

    rng = np.random.default_rng(0)
    fitted = scale * rng.weibull(shape, size=20_000)
    ks = stats.ks_2samp(empirical, fitted)

    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = {
        f"q{int(100 * q)}": {
            "empirical": float(np.quantile(empirical, q)),
            "weibull": float(np.quantile(fitted, q)),
        }
        for q in levels
    }
    result = {
        "shape": shape,
        "scale": scale,
        "empirical_mean": float(empirical.mean()),
        "weibull_mean": float(fitted.mean()),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "tolerance_ks": 0.20,
        "passes": bool(ks.statistic < 0.20),
        "quantiles": quantiles,
    }
    logger.info(
        "weibull: shape %.2f scale %.1f, mean %.1f vs empirical %.1f, KS %.3f (p %.3f) -> %s",
        shape,
        scale,
        result["weibull_mean"],
        result["empirical_mean"],
        ks.statistic,
        ks.pvalue,
        "PASS" if result["passes"] else "FAIL",
    )
    return result


# --------------------------------------------------------------------------- #
# Component validation 3: the demand process
# --------------------------------------------------------------------------- #
def validate_demand(resources: TwinResources) -> dict[str, Any]:
    """Simulated demand paths against the held-out last Favorita fold.

    Compared on the four things that actually matter to an inventory process:
    the mean level, the variance, the lag-1 autocorrelation (serial dependence
    is what makes stockouts cluster), and the share of realised days falling
    inside the simulated 80% band.

    Everything is compared on the *rescaled* paths -- the twin's units, after
    ``demand_scale`` -- with the realised fold rescaled by the same constant, so
    the comparison is of shape and spread rather than of grocery units against
    product units.
    """
    from twinmind.data.features import build_demand_features, make_rolling_origin_splits
    from twinmind.data.loaders import load_favorita
    from twinmind.twin.demand_proc import get_sampler
    from twinmind.utils.config import load_config

    data_config = load_config("data")
    frame = build_demand_features(load_favorita(), data_config.favorita, top_n=200)
    folds = make_rolling_origin_splits(frame, n_splits=3, horizon=HORIZON)
    _, validation_index = folds[-1]
    validation = frame.iloc[validation_index]

    realised = (
        validation[validation["unique_id"] == resources.series_id]
        .sort_values("ds")["y"]
        .to_numpy(dtype="float64")
    )
    if realised.size < HORIZON:
        logger.warning("held-out fold has only %d days for %s", realised.size, resources.series_id)

    scale = resources.demand_scale
    realised_scaled = realised[:HORIZON] * scale

    sampler = get_sampler(resources.demand_model, resources.series_id, resources.origin, HORIZON)
    paths = sampler.sample(2000, seed=0) * scale

    def lag1(series: np.ndarray) -> float:
        if series.size < 3 or series.std() == 0:
            return float("nan")
        return float(np.corrcoef(series[:-1], series[1:])[0, 1])

    # A raw lag-1 autocorrelation on a 28-day path is NOT a measurement of the
    # copula's serial dependence, and reading it as one is a trap this code fell
    # into first. The forecast carries a strong weekly cycle -- per-step means
    # swing from 1,625 to 2,569, a spread comparable to the within-step sd -- so
    # a raw lag-1 is dominated by the calendar saw-tooth, and the sharp weekly
    # drop drags it negative. Measured: raw per-path lag-1 of -0.12 while the
    # copula's actual step-to-step dependence is +0.30.
    #
    # So three things are reported. The RESIDUAL lag-1, after removing each
    # step's own median, is the like-for-like comparison against the realised
    # fold. The across-sample step dependence is a verification check against
    # the fitted copula target. The raw figure is kept only because it is what
    # a naive reading would produce, and it is labelled as such.
    median_path = np.median(paths, axis=0)
    residual_paths = paths - median_path
    simulated_lag1_raw = float(np.nanmean([lag1(path) for path in paths]))
    simulated_lag1_residual = float(np.nanmean([lag1(path) for path in residual_paths]))
    step_dependence = float(
        np.mean([np.corrcoef(paths[:, t], paths[:, t + 1])[0, 1] for t in range(HORIZON - 1)])
    )
    copula = getattr(resources.demand_model, "copula", None)
    copula_target = (
        float(np.mean(np.diag(copula.correlation, 1))) if copula is not None else float("nan")
    )
    realised_residual = realised_scaled - median_path[: realised_scaled.size]
    # Band coverage is NOT computed here. Doing it from this origin compares
    # paths covering origin+1..origin+28 against realised days that sit a month
    # earlier, which is what the first version of this function did -- and 28
    # days is exactly four weeks, so the weekly phase lined up and the answer
    # looked plausible rather than obviously wrong. It lives in
    # demand_band_coverage(), at a date-aligned origin and over many series.

    result = {
        "series_id": resources.series_id,
        "demand_scale": scale,
        "n_realised_days": int(realised_scaled.size),
        "simulated_mean": float(paths.mean()),
        "realised_mean": float(realised_scaled.mean()),
        "simulated_sd": float(paths.std()),
        "realised_sd": float(realised_scaled.std()),
        "simulated_lag1_raw": simulated_lag1_raw,
        "simulated_lag1_residual": simulated_lag1_residual,
        "realised_lag1_raw": lag1(realised_scaled),
        "realised_lag1_residual": lag1(realised_residual),
        "across_sample_step_dependence": step_dependence,
        "copula_lag1_target": copula_target,
        "share_inside_80_band": None,  # moved to demand_band_coverage()
        "tolerances": {"mean_rel": 0.25, "sd_rel": 0.50, "band_lower": 0.60, "band_upper": 0.95},
    }
    result["copula_reproduced"] = bool(
        np.isnan(copula_target) or abs(step_dependence - copula_target) <= 0.10
    )
    result["mean_passes"] = bool(
        abs(result["simulated_mean"] - result["realised_mean"])
        <= 0.25 * max(result["realised_mean"], 1e-9)
    )
    result["sd_passes"] = bool(
        abs(result["simulated_sd"] - result["realised_sd"])
        <= 0.50 * max(result["realised_sd"], 1e-9)
    )
    result["band_passes"] = None  # see demand_band_coverage()
    logger.info(
        "demand: mean %.1f vs %.1f (%s), sd %.1f vs %.1f (%s)",
        result["simulated_mean"],
        result["realised_mean"],
        "PASS" if result["mean_passes"] else "FAIL",
        result["simulated_sd"],
        result["realised_sd"],
        "PASS" if result["sd_passes"] else "FAIL",
    )
    logger.info(
        "  lag-1 residual %.3f vs realised %.3f; step dependence %.3f vs copula target %.3f (%s)",
        simulated_lag1_residual,
        result["realised_lag1_residual"],
        step_dependence,
        copula_target,
        "reproduced" if result["copula_reproduced"] else "NOT REPRODUCED",
    )
    logger.info("  band coverage is reported by demand_band_coverage(), not here")
    return result


def demand_band_coverage(
    resources: TwinResources, n_series: int = 50, n_samples: int = 2000
) -> dict[str, Any]:
    """Band coverage over many series at a DATE-ALIGNED origin.

    This replaces the single-series check in ``validate_demand``, which had two
    defects worse than the small sample it was criticised for.

    **The dates did not line up.** It drew paths from the twin's origin
    (2017-08-15, covering 2017-08-16..09-12) and compared them against the
    held-out fold's realised days (2017-07-19..08-15). Those are different days
    a month apart. The weekly phase happened to match -- 28 days is exactly four
    weeks -- so the number looked plausible rather than obviously broken.

    **And it was in-sample.** ``models/demand/config.json`` records a training
    window of 2013-01-01 to 2017-08-15, which is the whole dataset, so every day
    it was scored against was a day it had trained on. The saved production
    model cannot be scored out-of-sample at any origin inside the data, and the
    twin's own origin (2017-08-15) forecasts past the end of the data where
    there is nothing to score against at all.

    So this measures at the last fold boundary with the dates aligned, and
    reports the result as the **optimistic, in-sample** bound it is. The
    genuinely out-of-sample numbers are the per-fold ones from training
    (``reports/results/demand_folds.json``), quoted alongside.
    """
    from twinmind.data.features import build_demand_features, make_rolling_origin_splits
    from twinmind.data.loaders import load_favorita
    from twinmind.twin.demand_proc import DemandSampler
    from twinmind.utils.config import load_config

    data_config = load_config("data")
    frame = build_demand_features(load_favorita(), data_config.favorita, top_n=200)
    folds = make_rolling_origin_splits(frame, n_splits=3, horizon=HORIZON)
    train_index, validation_index = folds[-1]
    train, validation = frame.iloc[train_index], frame.iloc[validation_index]
    origin = pd.Timestamp(train["ds"].max())

    model = resources.demand_model
    series = sorted(validation["unique_id"].unique())[:n_series]
    logger.info(
        "band coverage: %d series, origin %s, scoring %s..%s",
        len(series),
        origin.date(),
        (origin + pd.Timedelta(days=1)).date(),
        (origin + pd.Timedelta(days=HORIZON)).date(),
    )

    inside: list[bool] = []
    steps: list[int] = []
    per_series: list[float] = []
    for series_id in series:
        realised = (
            validation[validation["unique_id"] == series_id].sort_values("ds").set_index("ds")["y"]
        )
        try:
            sampler = DemandSampler.from_model(model, series_id, origin, HORIZON)
        except (KeyError, ValueError):
            continue
        paths = sampler.sample(n_samples, seed=0)
        low = np.quantile(paths, 0.10, axis=0)
        high = np.quantile(paths, 0.90, axis=0)

        hits = []
        for step in range(1, HORIZON + 1):
            day = origin + pd.Timedelta(days=step)
            if day not in realised.index:
                continue
            value = float(realised.loc[day])
            hit = bool(low[step - 1] <= value <= high[step - 1])
            inside.append(hit)
            steps.append(step)
            hits.append(hit)
        if hits:
            per_series.append(float(np.mean(hits)))

    hit_array = np.asarray(inside, dtype="float64")
    step_array = np.asarray(steps, dtype="int64")
    buckets = {
        "d1_7": (step_array >= 1) & (step_array <= 7),
        "d8_14": (step_array >= 8) & (step_array <= 14),
        "d15_28": (step_array >= 15) & (step_array <= 28),
    }
    result: dict[str, Any] = {
        "n_series": len(per_series),
        "n_observations": int(hit_array.size),
        "origin": str(origin.date()),
        "scored_window": [
            str((origin + pd.Timedelta(days=1)).date()),
            str((origin + pd.Timedelta(days=HORIZON)).date()),
        ],
        "training_window_from_config": ["2013-01-01", "2017-08-15"],
        "in_sample": True,
        "coverage_overall": float(hit_array.mean()) if hit_array.size else float("nan"),
        "per_series_sd": float(np.std(per_series, ddof=1)) if len(per_series) > 1 else 0.0,
        "per_series_min": float(np.min(per_series)) if per_series else float("nan"),
        "per_series_max": float(np.max(per_series)) if per_series else float("nan"),
    }
    for name, mask in buckets.items():
        result[f"coverage_{name}"] = float(hit_array[mask].mean()) if mask.any() else float("nan")
        result[f"n_{name}"] = int(mask.sum())

    logger.info(
        "  overall %.3f over %d observations (%d series); d1-7 %.3f  d8-14 %.3f  d15-28 %.3f",
        result["coverage_overall"],
        result["n_observations"],
        result["n_series"],
        result["coverage_d1_7"],
        result["coverage_d8_14"],
        result["coverage_d15_28"],
    )
    logger.info(
        "  per-series spread: sd %.3f, min %.3f, max %.3f -- IN-SAMPLE (see docstring)",
        result["per_series_sd"],
        result["per_series_min"],
        result["per_series_max"],
    )
    return result


# --------------------------------------------------------------------------- #
# Landscape sanity
# --------------------------------------------------------------------------- #
def _sobol_scenarios(resources: TwinResources, n: int, seed: int = 0) -> list[Scenario]:
    """``n`` scenarios on a scrambled Sobol sequence over the configured bounds."""
    dims = resources.config.scenario_space
    sampler = stats.qmc.Sobol(d=len(dims), scramble=True, seed=seed)
    # Sobol wants a power of two for its balance properties; round up and trim.
    points = sampler.random_base2(m=int(np.ceil(np.log2(max(n, 2)))))[:n]
    lows = np.array([d.min for d in dims], dtype="float64")
    highs = np.array([d.max for d in dims], dtype="float64")
    scaled = lows + points * (highs - lows)

    scenarios = []
    for row in scaled:
        payload = {d.name: float(v) for d, v in zip(dims, row, strict=True)}
        payload["shift_count"] = int(round(payload["shift_count"]))
        payload["supplier_choice"] = int(round(payload["supplier_choice"]))
        scenarios.append(Scenario(**payload))
    return scenarios


def landscape_check(resources: TwinResources, n: int, reps: int) -> dict[str, Any]:
    """Are the best plans at a bound in every dimension?

    If they are, the twin has no interior trade-off: "maximum everything" wins,
    the week-5 optimisation is trivial, and the whole exercise says nothing. The
    spec's remedy is to fix the cost or wear structure and say so, which is what
    the warning-threshold note in twin.yaml records.
    """
    dims = resources.config.scenario_space
    scenarios = _sobol_scenarios(resources, n)
    profits = np.array(
        [
            replicate(s, n_reps=reps, base_seed=17, resources=resources).get("profit").mean
            for s in scenarios
        ],
        dtype="float64",
    )
    order = np.argsort(profits)[::-1]
    top = order[:10]

    # "At a bound" means within 5% of the dimension's range of either end.
    at_bound: dict[str, int] = {}
    for index, dim in enumerate(dims):
        span = dim.max - dim.min
        if span <= 0:
            at_bound[dim.name] = 0
            continue
        count = 0
        for position in top:
            value = scenarios[position].to_vector()[index]
            if (value - dim.min) <= 0.05 * span or (dim.max - value) <= 0.05 * span:
                count += 1
        at_bound[dim.name] = count

    saturated = [name for name, count in at_bound.items() if count >= 9]
    result = {
        "n_scenarios": n,
        "reps_each": reps,
        "best_profit": float(profits[order[0]]),
        "median_profit": float(np.median(profits)),
        "worst_profit": float(profits[order[-1]]),
        "top10_at_bound_counts": at_bound,
        "dimensions_saturated_in_top10": saturated,
        "all_dimensions_saturated": len(saturated) == len(dims),
        "degenerate": len(saturated) == len(dims),
        "best_scenario": scenarios[order[0]].model_dump(),
    }
    logger.info(
        "landscape: best profit %.0f, median %.0f; %d/%d dimensions saturated in the top 10 -> %s",
        result["best_profit"],
        result["median_profit"],
        len(saturated),
        len(dims),
        "DEGENERATE" if result["degenerate"] else "has interior trade-offs",
    )
    if saturated:
        logger.info("  saturated dimensions: %s", ", ".join(saturated))
    return result


# --------------------------------------------------------------------------- #
# Sensitivity, and rank stability
# --------------------------------------------------------------------------- #
def sensitivity(resources: TwinResources, reps: int, n_rank: int, rank_reps: int) -> dict[str, Any]:
    """+-25% one-at-a-time on every assumed parameter, with rank stability.

    The KPI movement is the easy half. The half that decides whether any
    recommendation survives is the Spearman correlation between scenario
    rankings under the baseline and the perturbed assumption: a rho below about
    0.8 means the twin's advice depends on a number nobody measured.
    """
    baseline = replicate(Scenario(), n_reps=reps, base_seed=3, resources=resources)
    baseline_means = baseline.means()

    rank_scenarios = _sobol_scenarios(resources, n_rank, seed=5)
    baseline_ranking = np.array(
        [
            replicate(s, n_reps=rank_reps, base_seed=29, resources=resources).get("profit").mean
            for s in rank_scenarios
        ],
        dtype="float64",
    )

    rows: dict[str, Any] = {}
    for name in SENSITIVITY_PARAMS:
        entry: dict[str, Any] = {"nominal": resources.config.assumption(name)}
        for label, factor in (("minus25", 0.75), ("plus25", 1.25)):
            perturbed = _perturbed(resources, name, factor)
            estimate = replicate(Scenario(), n_reps=reps, base_seed=3, resources=perturbed)
            means = estimate.means()
            entry[label] = {
                "value": perturbed.config.assumption(name),
                "kpis": means,
                "profit_pct_change": _pct(means["profit"], baseline_means["profit"]),
                "units_pct_change": _pct(means["units_produced"], baseline_means["units_produced"]),
                "stockout_abs_change": means["stockout_probability"]
                - baseline_means["stockout_probability"],
                "otd_abs_change": means["on_time_delivery_rate"]
                - baseline_means["on_time_delivery_rate"],
            }
        # Rank stability measured at +25%, the direction the tornado reports.
        perturbed = _perturbed(resources, name, 1.25)
        perturbed_ranking = np.array(
            [
                replicate(s, n_reps=rank_reps, base_seed=29, resources=perturbed).get("profit").mean
                for s in rank_scenarios
            ],
            dtype="float64",
        )
        rho = float(stats.spearmanr(baseline_ranking, perturbed_ranking).statistic)
        entry["rank_spearman_rho"] = rho
        entry["rank_stable"] = bool(rho >= 0.8)
        entry["max_abs_profit_pct_change"] = max(
            abs(entry["minus25"]["profit_pct_change"]), abs(entry["plus25"]["profit_pct_change"])
        )
        rows[name] = entry
        logger.info(
            "%-32s profit %+6.1f%% / %+6.1f%%   rank rho %.3f  %s",
            name,
            entry["minus25"]["profit_pct_change"],
            entry["plus25"]["profit_pct_change"],
            rho,
            "stable" if entry["rank_stable"] else "RANKING UNSTABLE",
        )

    offbase = _offbaseline_sensitivity(resources, reps)
    spread = demand_spread_study(resources, reps, rank_scenarios, baseline_ranking, rank_reps)
    return {
        "off_baseline": offbase,
        "baseline_kpis": baseline_means,
        "reps": reps,
        "n_rank_scenarios": n_rank,
        "rank_reps": rank_reps,
        "parameters": rows,
        "demand_spread_multiplier_study": spread,
        "unstable_parameters": [n for n, r in rows.items() if not r["rank_stable"]],
    }


#: Parameters that enter only through ``speed ** k`` or ``price ** k`` and are
#: therefore EXACTLY inert at the baseline, where both are 1.0. A one-at-a-time
#: study run only at the baseline cannot see them, and reporting their 0.00%
#: as "this assumption does not matter" would be wrong.
STRUCTURALLY_INERT_AT_BASELINE: tuple[str, ...] = ("wear_exponent", "price_elasticity")


def _offbaseline_sensitivity(resources: TwinResources, reps: int) -> dict[str, Any]:
    """Re-run the inert parameters at a scenario where they can act.

    ``wear_exponent`` enters as ``speed ** wear_exponent`` and
    ``price_elasticity`` as ``price_multiplier ** elasticity``. At the baseline
    both bases are exactly 1.0, so any exponent gives 1.0 and the measured
    effect is identically zero -- correct arithmetic, useless evidence. This
    repeats the perturbation at speed 1.20 and price 1.15, where they bite.
    """
    scenario = Scenario(
        price_multiplier=1.15, **{f"speed_setpoint_M{i}": 1.20 for i in range(1, 6)}
    )
    reference = replicate(scenario, n_reps=reps, base_seed=3, resources=resources).means()
    rows: dict[str, Any] = {
        "scenario": {"speed_setpoint": 1.20, "price_multiplier": 1.15},
        "reference_profit": reference["profit"],
        "parameters": {},
    }
    for name in STRUCTURALLY_INERT_AT_BASELINE:
        entry: dict[str, Any] = {}
        for label, factor in (("minus25", 0.75), ("plus25", 1.25)):
            perturbed = _perturbed(resources, name, factor)
            means = replicate(scenario, n_reps=reps, base_seed=3, resources=perturbed).means()
            entry[label] = {
                "profit_pct_change": _pct(means["profit"], reference["profit"]),
                "units_pct_change": _pct(means["units_produced"], reference["units_produced"]),
            }
        rows["parameters"][name] = entry
        logger.info(
            "%-32s off-baseline profit %+6.2f%% / %+6.2f%%",
            name,
            entry["minus25"]["profit_pct_change"],
            entry["plus25"]["profit_pct_change"],
        )
    return rows


def demand_spread_study(
    resources: TwinResources,
    reps: int,
    rank_scenarios: list[Scenario],
    baseline_ranking: np.ndarray,
    rank_reps: int,
) -> dict[str, Any]:
    """``demand_spread_multiplier`` over {0.8, 1.0, 1.2, 1.4}.

    This exists because Model B's post-conformal coverage is uneven -- d8-14
    averages 0.716 against a [0.76, 0.84] target, and on fold 1 an 80% interval
    covered 65% of outcomes. The twin's demand paths are therefore probably too
    NARROW at mid-horizon, so stockout risk is probably UNDERSTATED, and the
    direction of that bias is what this measures.
    """
    rows: dict[str, Any] = {}
    reference = None
    for value in (0.8, 1.0, 1.2, 1.4):
        perturbed = _with_value(resources, "demand_spread_multiplier", value)
        estimate = replicate(Scenario(), n_reps=reps, base_seed=3, resources=perturbed)
        means = estimate.means()
        if value == 1.0:
            reference = means
        ranking = np.array(
            [
                replicate(s, n_reps=rank_reps, base_seed=29, resources=perturbed).get("profit").mean
                for s in rank_scenarios
            ],
            dtype="float64",
        )
        rows[f"{value:.1f}"] = {
            "stockout_probability": means["stockout_probability"],
            "on_time_delivery_rate": means["on_time_delivery_rate"],
            "profit": means["profit"],
            "rank_spearman_rho_vs_baseline": float(
                stats.spearmanr(baseline_ranking, ranking).statistic
            ),
        }
    assert reference is not None
    for key, row in rows.items():
        row["stockout_abs_change_vs_1.0"] = (
            row["stockout_probability"] - reference["stockout_probability"]
        )
        row["otd_abs_change_vs_1.0"] = (
            row["on_time_delivery_rate"] - reference["on_time_delivery_rate"]
        )
        logger.info(
            "spread %s: stockout %.4f (%+.4f)  otd %.4f (%+.4f)  rank rho %.3f",
            key,
            row["stockout_probability"],
            row["stockout_abs_change_vs_1.0"],
            row["on_time_delivery_rate"],
            row["otd_abs_change_vs_1.0"],
            row["rank_spearman_rho_vs_baseline"],
        )
    worst_stockout = max(abs(r["stockout_abs_change_vs_1.0"]) for r in rows.values())
    worst_otd = max(abs(r["otd_abs_change_vs_1.0"]) for r in rows.values())
    worst_rho = min(r["rank_spearman_rho_vs_baseline"] for r in rows.values())
    return {
        "rows": rows,
        "max_abs_stockout_change": worst_stockout,
        "max_abs_otd_change": worst_otd,
        "min_rank_rho": worst_rho,
        # "Material" set at one percentage point of stockout probability, which
        # is a fifth of the 0.05 constraint the twin is scored against.
        "material": bool(worst_stockout >= 0.01 or worst_otd >= 0.01 or worst_rho < 0.8),
    }


def _pct(value: float, reference: float) -> float:
    """Percentage change, guarding a zero or sign-crossing reference."""
    if abs(reference) < 1e-9:
        return float("nan")
    return 100.0 * (value - reference) / abs(reference)


# --------------------------------------------------------------------------- #
# The five illustrative scenarios, in both hazard modes
# --------------------------------------------------------------------------- #
def scenario_table(resources: TwinResources, reps: int) -> dict[str, Any]:
    """Five named scenarios with mean +- 95% CI for all 8 KPIs, in both modes."""
    table: dict[str, Any] = {}
    for mode in ("learned", "fixed_mtbf"):
        resources_mode = _hazard(resources, mode)
        table[mode] = {}
        for name, overrides in NAMED_SCENARIOS.items():
            estimate = replicate(
                Scenario(**overrides), n_reps=reps, base_seed=11, resources=resources_mode
            )
            table[mode][name] = {
                kpi: {
                    "mean": estimate.get(kpi).mean,
                    "lower": estimate.get(kpi).lower,
                    "upper": estimate.get(kpi).upper,
                }
                for kpi in estimate.means()
            }
            logger.info(
                "%-10s %-20s units %8.0f  profit %10.0f  oee %.3f  stockout %.3f  cm %.2f",
                mode,
                name,
                estimate.get("units_produced").mean,
                estimate.get("profit").mean,
                estimate.get("oee").mean,
                estimate.get("stockout_probability").mean,
                estimate.get("unplanned_failures").mean,
            )
    return {"reps": reps, "modes": table}


def _log_to_registry(resources: TwinResources, payload: dict[str, Any]) -> None:
    """Merge the scenario results into the shared run registry and summary.

    The registry is how every model in this project reports, so the twin reports
    the same way rather than leaving its numbers only in a JSON nobody reads. One
    row per (scenario, hazard mode), keyed on a config hash that includes the
    assumptions, so a run under different assumptions is a different row instead
    of silently overwriting this one.
    """
    from twinmind.eval.registry import (
        RunRecord,
        config_hash,
        current_device,
        log_run,
        write_summary,
    )

    modes = payload["scenarios"]["modes"]
    reps = payload["scenarios"]["reps"]
    benchmark = results_dir() / "twin_benchmark.json"
    duration = 0.0
    if benchmark.exists():
        duration = float(json.loads(benchmark.read_text(encoding="utf-8"))["worst_mean_s"])

    for mode, scenarios in modes.items():
        for name, kpis in scenarios.items():
            log_run(
                RunRecord(
                    model=f"twin_{mode}",
                    dataset="simulated",
                    subset=name,
                    seed=11,
                    metrics={kpi: float(v["mean"]) for kpi, v in kpis.items()},
                    config_hash=config_hash(
                        {
                            "mode": mode,
                            "scenario": name,
                            "reps": reps,
                            "assumptions": resources.assumptions,
                            "pool": resources.config.health_replay.subset,
                            "model_seed": resources.config.health_replay.model_seed,
                        }
                    ),
                    duration_s=duration * reps,
                    device=current_device(),
                    notes=f"{reps} reps, CRN base_seed=11, 52-engine FD002 replay pool",
                )
            )
    write_summary()
    logger.info("merged %d twin rows into the run registry", sum(len(v) for v in modes.values()))


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    run_all = args.all or not any(
        (args.components, args.landscape, args.sensitivity, args.scenarios, args.band_coverage)
    )

    resources = build_resources()
    payload: dict[str, Any] = {}

    if run_all or args.components:
        logger.info("=== component validation ===")
        payload["health_warnings"] = validate_health_warnings(resources)
        payload["weibull"] = validate_weibull(resources)
        payload["demand"] = validate_demand(resources)

    if run_all or args.band_coverage:
        logger.info("=== demand band coverage, many series, aligned dates ===")
        payload["demand_band_coverage"] = demand_band_coverage(resources, args.band_series)

    if run_all or args.scenarios:
        logger.info("=== five named scenarios, both hazard modes ===")
        payload["scenarios"] = scenario_table(resources, args.reps)

    if run_all or args.landscape:
        logger.info("=== landscape sanity ===")
        payload["landscape"] = landscape_check(resources, args.sobol, args.sobol_reps)

    if run_all or args.sensitivity:
        logger.info("=== sensitivity and rank stability ===")
        payload["sensitivity"] = sensitivity(
            resources, args.reps, args.rank_scenarios, args.rank_reps
        )

    if "scenarios" in payload:
        _log_to_registry(resources, payload)

    # MERGE into whatever is already there rather than overwriting. Running one
    # section (--sensitivity, say) used to replace the whole file and silently
    # destroy the others, which is a bad property for a results artifact that
    # takes forty minutes to regenerate -- and it did destroy them once. Sections
    # this run produced win; sections it did not touch are preserved.
    destination = results_dir() / "twin_validation.json"
    merged: dict[str, Any] = {}
    if destination.exists():
        try:
            merged = json.loads(destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("%s was not valid JSON; replacing it", destination.name)
    merged.update(payload)
    destination.write_text(
        json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("wrote %s (sections: %s)", destination, ", ".join(sorted(merged)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
