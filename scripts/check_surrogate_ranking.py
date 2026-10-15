"""A0: two cheap checks on the shipped surrogate, before building the optimiser.

    python scripts/check_surrogate_ranking.py

**(a) Does the T2 ranking result generalise?** T2 measured ranking at ONE
context and got Spearman 0.9917. The optimiser will rank at whatever context a
goal names, so the question is whether that was a lucky context. This scores all
40 T1 contexts (25 scenarios each) and reports the *distribution*, not a mean.

The noise ceiling here is a **split-half** one: two 15-replication halves of the
30-rep truth. Two things follow and both are stated in the output. It is a
**15-rep ceiling and therefore pessimistic** — the real truth is a 30-rep mean,
which is quieter than either half — so the surrogate is being read against a
harder bar than it actually faces. And it is computed **parametrically**, from
the recorded per-replication std, because the shards store means and stds rather
than individual replications; an empirical split-half would need a 30,000-run
re-run. A subset is re-run empirically to check the parametric estimate.

**(b) Is the novelty scorer a usable OOD detector?** Model C's epistemic std was
not (AUROC 0.590). The gate: wire novelty into the optimiser as an *active*
signal only if AUROC >= 0.75, otherwise ship it advisory-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from twinmind.eval.registry import results_dir
from twinmind.eval.surrogate_metrics import auroc
from twinmind.models.surrogate import KPI_ORDER
from twinmind.models.surrogate_data import RolloutData, load_rollouts
from twinmind.models.surrogate_mlp import SurrogateEnsemble
from twinmind.optim.novelty import DEFAULT_K, NoveltyScorer
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("check_surrogate_ranking")

ROLLOUTS = PROJECT_ROOT / "data" / "processed" / "rollouts"
PROFIT = KPI_ORDER.index("profit")
#: Gate from the spec. Below this, novelty is advisory only.
AUROC_GATE = 0.75


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="A0 checks on the shipped surrogate.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "surrogate"))
    parser.add_argument(
        "--empirical-contexts",
        type=int,
        default=0,
        help="Re-run this many contexts on the twin to check the parametric ceiling.",
    )
    parser.add_argument("--seed", type=int, default=20_260_101)
    return parser


def _encode(ensemble: SurrogateEnsemble, data: RolloutData) -> NDArray[np.float64]:
    """Encode a split with the ensemble's own training statistics."""
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "evaluate_surrogate.py"
    spec = importlib.util.spec_from_file_location("evaluate_surrogate_for_a0", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_rows(ensemble, data)  # type: ignore[no-any-return]


def within_context_ranking(
    ensemble: SurrogateEnsemble, t1: RolloutData, seed: int
) -> dict[str, Any]:
    """Spearman and regret per context, with a parametric split-half ceiling."""
    predicted = ensemble.predict_features(_encode(ensemble, t1)).mean[:, PROFIT]
    truth = t1.targets()[:, PROFIT]
    rep_std = t1.target_stds()[:, PROFIT]
    blocks = t1.context_blocks
    rng = np.random.default_rng(seed)

    rows: list[dict[str, float]] = []
    for block in sorted(set(blocks.tolist())):
        mask = blocks == block
        y, p, s = truth[mask], predicted[mask], rep_std[mask]
        if y.size < 5 or np.std(y) == 0:
            continue

        # Parametric split-half: two independent 15-rep means of the same
        # underlying expectation. Their agreement is what a PERFECT surrogate
        # would score against a 15-rep truth.
        half_se = s / np.sqrt(15.0)
        half_a = y + rng.normal(0.0, half_se)
        half_b = y + rng.normal(0.0, half_se)

        chosen = int(np.argmax(p))
        best = float(np.max(y))
        span = best - float(np.min(y))
        rows.append(
            {
                "context_block": int(block),
                "n": int(y.size),
                "spearman": float(stats.spearmanr(p, y).statistic),
                "spearman_ceiling_15rep": float(stats.spearmanr(half_a, half_b).statistic),
                "regret": best - float(y[chosen]),
                "regret_fraction": (best - float(y[chosen])) / span if span > 0 else 0.0,
                "top1_correct": float(chosen == int(np.argmax(y))),
            }
        )

    def summarise(key: str) -> dict[str, float]:
        values = np.array([r[key] for r in rows], dtype="float64")
        return {
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p10": float(np.percentile(values, 10)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary = {
        "n_contexts": len(rows),
        "spearman": summarise("spearman"),
        "spearman_ceiling_15rep": summarise("spearman_ceiling_15rep"),
        "regret_fraction": summarise("regret_fraction"),
        "top1_hit_rate": float(np.mean([r["top1_correct"] for r in rows])),
        "ceiling_is_parametric": True,
        "ceiling_is_15rep_so_pessimistic": True,
        "per_context": rows,
    }

    logger.info(
        "--- (a) within-context ranking over %d T1 contexts, 25 scenarios each ---", len(rows)
    )
    logger.info("%-26s %9s %9s %9s %9s", "quantity", "median", "p10", "min", "max")
    for key in ("spearman", "spearman_ceiling_15rep", "regret_fraction"):
        stat = summary[key]
        logger.info(
            "%-26s %9.4f %9.4f %9.4f %9.4f",
            key,
            stat["median"],
            stat["p10"],
            stat["min"],
            stat["max"],
        )
    logger.info(
        "surrogate picks the true best scenario in %.0f%% of contexts; "
        "median regret %.2f%% of the within-context profit range",
        100 * summary["top1_hit_rate"],
        100 * summary["regret_fraction"]["median"],
    )
    logger.info(
        "the ceiling is a 15-rep split-half and therefore PESSIMISTIC: the real truth "
        "is a 30-rep mean, which is quieter than either half"
    )
    return summary


def empirical_ceiling_check(t1: RolloutData, n_contexts: int, seed: int) -> dict[str, Any] | None:
    """Re-run a few contexts on the twin to check the parametric ceiling.

    The parametric split-half assumes the replication noise is Gaussian with the
    recorded std. That is an assumption, so it is checked rather than trusted --
    on a subset, because the full version is a 30,000-run job.
    """
    if n_contexts <= 0:
        return None

    from twinmind.twin.runner import Scenario, build_resources, run_twin
    from twinmind.twin.streams import derive_seeds

    resources = build_resources()
    blocks = sorted(set(t1.context_blocks.tolist()))[:n_contexts]
    logger.info("--- empirical check of the parametric ceiling on %d contexts ---", len(blocks))

    from twinmind.twin.context import sample_context

    cache = getattr(resources.lookup, "cache", None)
    agreements: list[float] = []
    for block in blocks:
        mask = t1.context_blocks == block
        frame = t1.frame[mask]
        origin = frame["origin"].iloc[0]
        # Context seeds for T1 were offset by 10_000 in the generator.
        context = sample_context(
            10_000 + int(block), cache=cache, series_id=resources.series_id, origin=origin
        )
        seeds = derive_seeds(seed + int(block), 30)
        first, second = [], []
        for _, row in frame.iterrows():
            scenario = Scenario(**{d: row[d] for d in Scenario.model_fields})
            values = [run_twin(scenario, s, context, resources).profit for s in seeds]
            first.append(float(np.mean(values[:15])))
            second.append(float(np.mean(values[15:])))
        agreements.append(float(stats.spearmanr(first, second).statistic))

    result = {
        "n_contexts": len(agreements),
        "empirical_split_half_spearman": {
            "median": float(np.median(agreements)),
            "min": float(np.min(agreements)),
            "max": float(np.max(agreements)),
        },
        "per_context": agreements,
    }
    logger.info(
        "empirical 15-rep split-half Spearman: median %.4f (min %.4f, max %.4f)",
        result["empirical_split_half_spearman"]["median"],
        result["empirical_split_half_spearman"]["min"],
        result["empirical_split_half_spearman"]["max"],
    )
    return result


def novelty_gate(train: RolloutData, t1: RolloutData, ood: RolloutData) -> dict[str, Any]:
    """AUROC of novelty separating OOD from T1, and the gate decision."""
    scorer = NoveltyScorer.from_scenarios(train.scenario_matrix(), k=DEFAULT_K)
    t1_score = scorer.score(t1.scenario_matrix())
    ood_score = scorer.score(ood.scenario_matrix())

    scores = np.concatenate([t1_score, ood_score])
    labels = np.concatenate(
        [np.zeros(t1_score.size, dtype=bool), np.ones(ood_score.size, dtype=bool)]
    )
    area = auroc(scores, labels)
    active = bool(area >= AUROC_GATE)

    logger.info("--- (b) novelty as an OOD detector (k=%d, fixed in advance) ---", DEFAULT_K)
    logger.info(
        "mean novelty: T1 %.4f, OOD %.4f (%.2fx); AUROC %.4f",
        float(t1_score.mean()),
        float(ood_score.mean()),
        float(ood_score.mean()) / max(float(t1_score.mean()), 1e-12),
        area,
    )
    logger.info(
        "gate (>= %.2f): %s -- novelty is %s",
        AUROC_GATE,
        "PASS" if active else "FAIL",
        "wired in as an ACTIVE signal" if active else "ADVISORY ONLY",
    )
    logger.info(
        "caveat: the OOD split is built from BOUND CORNERS (>=3 dimensions snapped to a "
        "bound), so part of this separation is by construction -- a corner is far from the "
        "interior almost by definition. It is not evidence of detecting arbitrary novelty."
    )
    return {
        "k": DEFAULT_K,
        "auroc": area,
        "gate": AUROC_GATE,
        "active": active,
        "mean_novelty_t1": float(t1_score.mean()),
        "mean_novelty_ood": float(ood_score.mean()),
        "ratio": float(ood_score.mean()) / max(float(t1_score.mean()), 1e-12),
        "surrogate_epistemic_auroc_for_comparison": 0.590,
        "caveat": (
            "the OOD split is built from bound corners, so part of the separation is by "
            "construction and this is not evidence of detecting arbitrary novelty"
        ),
    }


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    ensemble = SurrogateEnsemble.load(Path(args.model))
    train = load_rollouts(ROLLOUTS / "train")
    t1 = load_rollouts(ROLLOUTS / "test_t1", allow_test=True)
    ood = load_rollouts(ROLLOUTS / "test_ood", allow_test=True)

    payload: dict[str, Any] = {
        "within_context_ranking": within_context_ranking(ensemble, t1, args.seed),
        "novelty": novelty_gate(train, t1, ood),
    }
    empirical = empirical_ceiling_check(t1, args.empirical_contexts, args.seed)
    if empirical is not None:
        payload["empirical_ceiling_check"] = empirical

    destination = results_dir() / "optim_a0_checks.json"
    destination.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
