"""Noise-aware evaluation of the surrogate on T1, T2 and OOD.

    python scripts/evaluate_surrogate.py

Every headline number comes with its ceiling, because the test labels are
30-replication means of a stochastic simulator and therefore noisy. A perfect
surrogate cannot reach R² = 1 against them, and cannot reproduce their ranking
exactly. Reading 0.96 as "4% short of perfect" would be wrong.

* Regression ceilings are analytic:
  ``R²_max = 1 - mean(sigma_rep²/30) / Var(truth)``.
* The ranking ceiling is **measured**: T2 was run twice with independent seeds,
  so the agreement between those two estimates is what a perfect surrogate would
  score.

This is the only script that opens the test splits, and it says so explicitly
with ``allow_test=True``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twinmind.eval.registry import results_dir
from twinmind.eval.surrogate_metrics import (
    epistemic_tracks_error,
    flags_large_error,
    interval_coverage,
    ood_separation,
    rank_agreement,
    score_all,
)
from twinmind.models.surrogate import KPI_ORDER
from twinmind.models.surrogate_data import RolloutData, load_rollouts
from twinmind.models.surrogate_mlp import SurrogateEnsemble
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("evaluate_surrogate")

ROLLOUTS = PROJECT_ROOT / "data" / "processed" / "rollouts"
MODEL_DIR = PROJECT_ROOT / "models" / "surrogate"
#: A 30-rep twin evaluation at the measured 155 ms/run, which is what the
#: surrogate replaces.
TWIN_SECONDS_PER_EVALUATION = 30 * 0.155


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Evaluate the surrogate.")
    parser.add_argument("--model", default=str(MODEL_DIR))
    parser.add_argument("--rollouts", default=str(ROLLOUTS))
    parser.add_argument("--learning-curve", action="store_true")
    return parser


def encode_rows(ensemble: SurrogateEnsemble, data: RolloutData) -> NDArray[np.float64]:
    """Encode every row of a split with the ensemble's own training statistics.

    The statistics come from the saved model, never recomputed from the test
    split. Recomputing them here is the classic leak: the features would be
    standardised using the distribution of the data being scored.
    """
    from twinmind.models.surrogate import CATEGORICAL_DIMS, _categorical_levels, scenario_bounds
    from twinmind.twin.runner import SCENARIO_DIMS

    bounds = scenario_bounds()
    scenarios = data.scenario_matrix()
    columns: list[NDArray[np.float64]] = []
    for index, dim in enumerate(SCENARIO_DIMS):
        raw = scenarios[:, index]
        if dim in CATEGORICAL_DIMS:
            for level in _categorical_levels(dim):
                columns.append((np.round(raw).astype("int64") == level).astype("float64"))
            continue
        low, high = bounds[dim]
        span = high - low
        columns.append((raw - low) / span if span > 0 else np.zeros_like(raw))

    scenario_part = np.column_stack(columns)
    scale = np.where(np.abs(ensemble.state_scale) < 1e-12, 1.0, ensemble.state_scale)
    state_part = (data.state_matrix() - ensemble.state_mean) / scale
    return np.hstack([scenario_part, state_part])


def evaluate_t1(ensemble: SurrogateEnsemble, data: RolloutData) -> dict[str, Any]:
    """Per-KPI accuracy against the noise ceiling, plus calibration."""
    features = encode_rows(ensemble, data)
    output = ensemble.predict_features(features)
    truth = data.targets()
    rep_std = data.target_stds()
    n_reps = int(data.manifest["n_reps"])

    scores = score_all(output.mean, truth, rep_std, n_reps)
    logger.info(
        "%-24s %8s %8s %8s %10s %10s  %s",
        "KPI",
        "R2",
        "R2max",
        "ratio",
        "RMSE",
        "bias",
        "target",
    )
    for score in scores:
        logger.info(
            "%-24s %8.4f %8.4f %8.4f %10.4g %10.4g  %s",
            score.kpi,
            score.r2,
            score.r2_max,
            score.r2_ratio,
            score.rmse,
            score.bias,
            "HIT" if score.meets_target else "MISS",
        )

    label_noise = rep_std / np.sqrt(n_reps)
    coverage = interval_coverage(output.mean, truth, output.total_std, label_noise)
    correlation = epistemic_tracks_error(output.mean, truth, output.epistemic_std)
    flags = flags_large_error(output.mean, truth, output.epistemic_std)
    logger.info(
        "calibration: mean 95%% coverage %.3f; mean |error| vs epistemic rho %.3f; "
        "mean AUROC for flagging a large error %.3f",
        float(np.mean(list(coverage.values()))),
        float(np.nanmean(list(correlation.values()))),
        float(np.nanmean(list(flags.values()))),
    )

    # Generalisation by origin period, which is what "a context it has never
    # seen" really means here: every test origin is after every training one.
    by_origin: dict[str, dict[str, float]] = {}
    origins = data.frame["origin"].to_numpy()
    profit = KPI_ORDER.index("profit")
    for origin in sorted(set(origins.tolist())):
        mask = origins == origin
        if mask.sum() < 10:
            continue
        residual = output.mean[mask, profit] - truth[mask, profit]
        variance = float(np.var(truth[mask, profit], ddof=1))
        by_origin[str(origin)] = {
            "n": int(mask.sum()),
            "profit_r2": 1.0 - float(np.mean(residual**2)) / variance
            if variance > 0
            else float("nan"),
            "profit_rmse": float(np.sqrt(np.mean(residual**2))),
        }

    return {
        "n": data.n_rows,
        "n_reps": n_reps,
        "scores": [vars(s) for s in scores],
        "coverage_95": coverage,
        "epistemic_error_rho": correlation,
        "auroc_large_error": flags,
        "by_origin": by_origin,
        "all_targets_met": all(s.meets_target for s in scores),
    }


def evaluate_t2(
    ensemble: SurrogateEnsemble, first: RolloutData, second: RolloutData
) -> dict[str, Any]:
    """Ranking within one context, against the twin-vs-twin ceiling."""
    if not np.array_equal(first.scenario_indices, second.scenario_indices):
        raise ValueError("the two T2 estimates do not cover the same scenarios")

    features = encode_rows(ensemble, first)
    output = ensemble.predict_features(features)
    profit = KPI_ORDER.index("profit")

    truth = first.targets()[:, profit]
    truth_b = second.targets()[:, profit]
    score = rank_agreement(output.mean[:, profit], truth, truth_b)

    logger.info(
        "T2 ranking on %d scenarios: Spearman %.4f (ceiling %.4f), "
        "top-10 %d/10 (ceiling %d), top-50 %d/50 (ceiling %d)",
        score.n,
        score.spearman,
        score.spearman_ceiling,
        score.top10_overlap,
        score.top10_ceiling,
        score.top50_overlap,
        score.top50_ceiling,
    )
    logger.info(
        "  regret of the surrogate's argmax: %.0f of a %.0f range (%.2f%%); "
        "chosen %.0f against best %.0f",
        score.regret,
        score.best_truth - float(np.min(truth)),
        100 * score.regret_fraction,
        score.chosen_truth,
        score.best_truth,
    )
    logger.info(
        "  Spearman target: %s   top-10 target: %s",
        "HIT" if score.meets_spearman else "MISS",
        "HIT" if score.meets_top10 else "MISS",
    )
    return {**vars(score), "meets_spearman": score.meets_spearman, "meets_top10": score.meets_top10}


def evaluate_ood(ensemble: SurrogateEnsemble, t1: RolloutData, ood: RolloutData) -> dict[str, Any]:
    """Does the epistemic std rise where the training data never went?"""
    t1_output = ensemble.predict_features(encode_rows(ensemble, t1))
    ood_output = ensemble.predict_features(encode_rows(ensemble, ood))

    separation = ood_separation(t1_output.epistemic_std, ood_output.epistemic_std)
    scores = score_all(
        ood_output.mean, ood.targets(), ood.target_stds(), int(ood.manifest["n_reps"])
    )
    correlation = epistemic_tracks_error(ood_output.mean, ood.targets(), ood_output.epistemic_std)

    profit = KPI_ORDER.index("profit")
    logger.info(
        "OOD: epistemic std %.4g in-distribution vs %.4g out (%.2fx); "
        "AUROC separating them %.3f (profit)",
        float(np.mean(t1_output.epistemic_std[:, profit])),
        float(np.mean(ood_output.epistemic_std[:, profit])),
        float(np.mean(ood_output.epistemic_std[:, profit]))
        / max(float(np.mean(t1_output.epistemic_std[:, profit])), 1e-12),
        separation["profit"],
    )
    logger.info(
        "  OOD profit R2 %.4f (ceiling %.4f); mean separation AUROC %.3f",
        scores[profit].r2,
        scores[profit].r2_max,
        float(np.nanmean(list(separation.values()))),
    )
    return {
        "n": ood.n_rows,
        "separation_auroc": separation,
        "mean_separation_auroc": float(np.nanmean(list(separation.values()))),
        "scores": [vars(s) for s in scores],
        "epistemic_error_rho": correlation,
        "epistemic_mean_in": {
            kpi: float(np.mean(t1_output.epistemic_std[:, i])) for i, kpi in enumerate(KPI_ORDER)
        },
        "epistemic_mean_ood": {
            kpi: float(np.mean(ood_output.epistemic_std[:, i])) for i, kpi in enumerate(KPI_ORDER)
        },
    }


def measure_speed(ensemble: SurrogateEnsemble, data: RolloutData) -> dict[str, Any]:
    """Milliseconds per scenario, single-call and batched, against the twin."""
    features = encode_rows(ensemble, data)
    single = features[:1]

    ensemble.predict_features(single)
    started = time.perf_counter()
    for _ in range(50):
        ensemble.predict_features(single)
    single_ms = 1000 * (time.perf_counter() - started) / 50

    # A 10,000-row batch, tiled from the split so the shape is realistic.
    big = np.tile(features, (int(np.ceil(10_000 / len(features))), 1))[:10_000]
    ensemble.predict_features(big[:100])
    started = time.perf_counter()
    ensemble.predict_features(big)
    batch_total = time.perf_counter() - started
    batch_ms = 1000 * batch_total / len(big)

    speedup = TWIN_SECONDS_PER_EVALUATION / (batch_ms / 1000)
    logger.info(
        "speed: %.2f ms single call, %.4f ms/scenario in a 10,000 batch "
        "(%.1f s total); twin 30-rep evaluation is %.2f s -> %.0fx speed-up",
        single_ms,
        batch_ms,
        batch_total,
        TWIN_SECONDS_PER_EVALUATION,
        speedup,
    )
    return {
        "single_call_ms": single_ms,
        "batch_ms_per_scenario": batch_ms,
        "batch_size": len(big),
        "batch_total_s": batch_total,
        "twin_30rep_seconds": TWIN_SECONDS_PER_EVALUATION,
        "speedup_vs_twin": speedup,
        "meets_10000x": speedup >= 10_000,
    }


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    root = Path(args.rollouts)
    ensemble = SurrogateEnsemble.load(Path(args.model))
    logger.info("loaded %d ensemble members from %s", len(ensemble.members), args.model)

    t1 = load_rollouts(root / "test_t1", allow_test=True)
    t2a = load_rollouts(root / "test_t2_estimate1", allow_test=True)
    t2b = load_rollouts(root / "test_t2_estimate2", allow_test=True)
    ood = load_rollouts(root / "test_ood", allow_test=True)

    logger.info("=== T1: accuracy against the noise ceiling ===")
    payload: dict[str, Any] = {"T1": evaluate_t1(ensemble, t1)}
    logger.info("=== T2: ranking against the twin-vs-twin ceiling ===")
    payload["T2"] = evaluate_t2(ensemble, t2a, t2b)
    logger.info("=== OOD: does the epistemic std notice? ===")
    payload["OOD"] = evaluate_ood(ensemble, t1, ood)
    logger.info("=== speed ===")
    payload["speed"] = measure_speed(ensemble, t1)

    if args.learning_curve:
        logger.info("=== learning curve ===")
        payload["learning_curve"] = evaluate_learning_curve(root, t1, t2a, t2b)

    destination = results_dir() / "surrogate_evaluation.json"
    destination.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("wrote %s", destination)
    return 0


def evaluate_learning_curve(
    root: Path, t1: RolloutData, t2a: RolloutData, t2b: RolloutData
) -> list[dict[str, Any]]:
    """Accuracy and ranking against training size, from the saved prefix models."""
    profit = KPI_ORDER.index("profit")
    curve: list[dict[str, Any]] = []
    # Sorted NUMERICALLY, not by name: glob order puts 16384 before 2048, which
    # would plot the learning curve as a zigzag.
    directories = sorted(
        (MODEL_DIR.parent).glob("surrogate_lc_*"),
        key=lambda d: int(d.name.rsplit("_", 1)[1]),
    )
    for directory in directories:
        n_scenarios = int(directory.name.rsplit("_", 1)[1])
        ensemble = SurrogateEnsemble.load(directory)
        scores = score_all(
            ensemble.predict_features(encode_rows(ensemble, t1)).mean,
            t1.targets(),
            t1.target_stds(),
            int(t1.manifest["n_reps"]),
        )
        ranking = rank_agreement(
            ensemble.predict_features(encode_rows(ensemble, t2a)).mean[:, profit],
            t2a.targets()[:, profit],
            t2b.targets()[:, profit],
        )
        row = {
            "n_scenarios": n_scenarios,
            "profit_r2": scores[profit].r2,
            "profit_r2_ratio": scores[profit].r2_ratio,
            "mean_r2": float(np.nanmean([s.r2 for s in scores])),
            "spearman": ranking.spearman,
            "top10_overlap": ranking.top10_overlap,
        }
        curve.append(row)
        logger.info(
            "  %6d scenarios: profit R2 %.4f (%.3f of ceiling), mean R2 %.4f, "
            "Spearman %.4f, top-10 %d/10",
            n_scenarios,
            row["profit_r2"],
            row["profit_r2_ratio"],
            row["mean_r2"],
            row["spearman"],
            row["top10_overlap"],
        )
    return curve


if __name__ == "__main__":
    sys.exit(main())
