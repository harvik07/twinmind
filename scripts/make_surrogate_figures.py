"""The four surrogate figures.

    python scripts/make_surrogate_figures.py

Exactly four, per the spec:

1. ``c_surrogate_profit_t1`` -- surrogate against twin profit on T1, with the
   noise-ceiling band, because the truth is a 30-rep mean and the band is how
   much of the scatter is the yardstick rather than the model.
2. ``c_surrogate_rank_t2`` -- rank agreement on T2 with the twin-vs-twin
   ceiling marked, so the reader sees what a perfect surrogate would score.
3. ``c_surrogate_learning_curve`` -- accuracy and ranking against training size.
4. ``c_surrogate_epistemic`` -- binned epistemic std against ``|error|``, with
   the OOD split overlaid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from twinmind.eval.figures import BLUE, GREEN, ORANGE, RED, save  # noqa: E402
from twinmind.eval.registry import results_dir  # noqa: E402
from twinmind.eval.surrogate_metrics import rank_agreement, score_all  # noqa: E402
from twinmind.models.surrogate import KPI_ORDER  # noqa: E402
from twinmind.models.surrogate_data import load_rollouts  # noqa: E402
from twinmind.models.surrogate_mlp import SurrogateEnsemble  # noqa: E402
from twinmind.utils.logging import get_logger  # noqa: E402
from twinmind.utils.paths import PROJECT_ROOT  # noqa: E402

logger = get_logger("make_surrogate_figures")
ROLLOUTS = PROJECT_ROOT / "data" / "processed" / "rollouts"
PROFIT = KPI_ORDER.index("profit")


def _encode(ensemble, data):  # type: ignore[no-untyped-def]
    """Encode a split with the ensemble's own training statistics."""
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "evaluate_surrogate.py"
    spec = importlib.util.spec_from_file_location("evaluate_surrogate_for_figures", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_rows(ensemble, data)


def figure_profit(ensemble, t1) -> None:  # type: ignore[no-untyped-def]
    """Surrogate against twin profit, with the noise-ceiling band."""
    output = ensemble.predict_features(_encode(ensemble, t1))
    truth = t1.targets()[:, PROFIT]
    predicted = output.mean[:, PROFIT]
    label_se = t1.target_stds()[:, PROFIT] / np.sqrt(int(t1.manifest["n_reps"]))
    score = score_all(output.mean, t1.targets(), t1.target_stds(), int(t1.manifest["n_reps"]))[
        PROFIT
    ]

    fig, axis = plt.subplots(figsize=(8, 7.5))
    low, high = float(min(truth.min(), predicted.min())), float(max(truth.max(), predicted.max()))
    pad = 0.04 * (high - low)
    line = np.array([low - pad, high + pad])

    # The band is +-1.96 x the MEAN label standard error: inside it, a
    # disagreement is the yardstick's noise rather than the model's error.
    band = 1.96 * float(np.mean(label_se))
    axis.fill_between(
        line,
        line - band,
        line + band,
        color=GREEN,
        alpha=0.18,
        label=f"±1.96 × the truth's own noise (±£{band:,.0f})",
    )
    axis.plot(line, line, color="black", lw=1.2, ls="--", label="perfect agreement")
    axis.scatter(truth, predicted, s=10, alpha=0.4, color=BLUE, edgecolors="none")

    axis.set_xlabel("twin profit, 30-replication mean (£)")
    axis.set_ylabel("surrogate predicted profit (£)")
    axis.set_title(
        f"Profit on T1: R² = {score.r2:.4f} against a noise ceiling of {score.r2_max:.4f}\n"
        f"({100 * score.r2_ratio:.1f}% of what the 30-rep truth permits; "
        f"RMSE £{score.rmse:,.0f}, bias £{score.bias:,.0f})",
        fontsize=11,
    )
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(alpha=0.3)
    axis.set_xlim(line)
    axis.set_ylim(line)
    fig.text(
        0.5,
        -0.02,
        "The truth is itself a 30-replication mean, so a perfect surrogate would still "
        "scatter inside the green band.",
        ha="center",
        fontsize=8.5,
        style="italic",
    )
    save(fig, "c_surrogate_profit_t1")


def figure_ranking(ensemble, t2a, t2b) -> None:  # type: ignore[no-untyped-def]
    """Rank agreement on T2, with the twin-vs-twin ceiling."""
    predicted = ensemble.predict_features(_encode(ensemble, t2a)).mean[:, PROFIT]
    truth = t2a.targets()[:, PROFIT]
    truth_b = t2b.targets()[:, PROFIT]
    score = rank_agreement(predicted, truth, truth_b)

    rank_truth = np.argsort(np.argsort(truth))
    rank_pred = np.argsort(np.argsort(predicted))
    rank_b = np.argsort(np.argsort(truth_b))

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6))

    left.scatter(rank_truth, rank_pred, s=9, alpha=0.4, color=BLUE, edgecolors="none")
    left.plot([0, len(truth)], [0, len(truth)], color="black", lw=1, ls="--")
    left.set_xlabel("twin profit rank (estimate 1)")
    left.set_ylabel("surrogate profit rank")
    left.set_title(f"Surrogate vs twin\nSpearman {score.spearman:.4f}", fontsize=10)
    left.grid(alpha=0.3)

    right.scatter(rank_truth, rank_b, s=9, alpha=0.4, color=ORANGE, edgecolors="none")
    right.plot([0, len(truth)], [0, len(truth)], color="black", lw=1, ls="--")
    right.set_xlabel("twin profit rank (estimate 1)")
    right.set_ylabel("twin profit rank (estimate 2)")
    right.set_title(
        f"THE CEILING: twin vs itself\nSpearman {score.spearman_ceiling:.4f}", fontsize=10
    )
    right.grid(alpha=0.3)

    fig.suptitle(
        f"Ranking 1,000 scenarios at one context — surrogate {score.spearman:.4f} against a "
        f"ceiling of {score.spearman_ceiling:.4f}\n"
        f"top-10 {score.top10_overlap}/10 (ceiling {score.top10_ceiling}), "
        f"top-50 {score.top50_overlap}/50 (ceiling {score.top50_ceiling}), "
        f"regret of the argmax £{score.regret:,.0f}",
        fontsize=11,
    )
    fig.text(
        0.5,
        -0.02,
        "Both panels are scored against the same 30-rep estimate. The right panel is a second, "
        "independent 30-rep run of the identical scenarios: it is what a PERFECT surrogate "
        "would score, because the only difference is replication noise.",
        ha="center",
        fontsize=8.5,
        style="italic",
    )
    fig.tight_layout()
    save(fig, "c_surrogate_rank_t2")


def figure_learning_curve() -> None:
    """Accuracy and ranking against training size."""
    path = results_dir() / "surrogate_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    curve = payload.get("learning_curve")
    if not curve:
        logger.warning(
            "no learning_curve in %s; run evaluate_surrogate.py --learning-curve", path.name
        )
        return

    sizes = [row["n_scenarios"] for row in curve]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    left.semilogx(
        sizes, [r["profit_r2"] for r in curve], "o-", color=BLUE, lw=1.8, label="profit R²"
    )
    left.semilogx(
        sizes,
        [r["mean_r2"] for r in curve],
        "s--",
        color=GREEN,
        lw=1.6,
        label="mean R² over 8 KPIs",
    )
    if curve:
        ceiling = curve[-1]["profit_r2"] / max(curve[-1]["profit_r2_ratio"], 1e-9)
        left.axhline(
            ceiling, color="grey", ls=":", lw=1.5, label=f"profit noise ceiling {ceiling:.4f}"
        )
    left.set_xlabel("training scenarios")
    left.set_ylabel("R² on T1")
    left.set_title("Accuracy against training size", fontsize=10)
    left.grid(alpha=0.3, which="both")
    left.legend(fontsize=9)

    right.semilogx(
        sizes, [r["spearman"] for r in curve], "o-", color=ORANGE, lw=1.8, label="Spearman ρ on T2"
    )
    axis2 = right.twinx()
    axis2.semilogx(
        sizes, [r["top10_overlap"] for r in curve], "s--", color=RED, lw=1.6, label="top-10 overlap"
    )
    axis2.set_ylabel("top-10 overlap (of 10)")
    axis2.set_ylim(0, 10.5)
    right.set_xlabel("training scenarios")
    right.set_ylabel("Spearman ρ")
    right.set_title("Ranking against training size", fontsize=10)
    right.grid(alpha=0.3, which="both")
    handles = right.get_legend_handles_labels()[0] + axis2.get_legend_handles_labels()[0]
    labels = right.get_legend_handles_labels()[1] + axis2.get_legend_handles_labels()[1]
    right.legend(handles, labels, fontsize=9, loc="lower right")

    fig.suptitle(
        "Learning curve from nested Sobol prefixes — a prefix of complete blocks is itself a "
        "valid space-filling design",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "c_surrogate_learning_curve")


def figure_epistemic(ensemble, t1, ood) -> None:  # type: ignore[no-untyped-def]
    """Binned epistemic std against |error|, in-distribution and OOD."""
    t1_out = ensemble.predict_features(_encode(ensemble, t1))
    ood_out = ensemble.predict_features(_encode(ensemble, ood))
    payload = json.loads((results_dir() / "surrogate_evaluation.json").read_text(encoding="utf-8"))

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))

    for data, output, colour, label in (
        (t1, t1_out, BLUE, "T1 (in-distribution)"),
        (ood, ood_out, RED, "OOD (≥3 dims at a bound)"),
    ):
        spread = output.epistemic_std[:, PROFIT]
        error = np.abs(output.mean[:, PROFIT] - data.targets()[:, PROFIT])
        edges = np.quantile(spread, np.linspace(0, 1, 9))
        centres, medians = [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            mask = (spread >= lo) & (spread <= hi)
            if mask.sum() >= 5:
                centres.append(float(np.median(spread[mask])))
                medians.append(float(np.median(error[mask])))
        left.plot(centres, medians, "o-", color=colour, lw=1.8, label=label)

    left.set_xlabel("epistemic std (ensemble disagreement, £)")
    left.set_ylabel("median |error| in the bin (£)")
    left.set_title("Does disagreement predict error?", fontsize=10)
    left.grid(alpha=0.3)
    left.legend(fontsize=9)

    parts = right.violinplot(
        [t1_out.epistemic_std[:, PROFIT], ood_out.epistemic_std[:, PROFIT]],
        showmedians=True,
    )
    for body, colour in zip(parts["bodies"], (BLUE, RED), strict=False):
        body.set_facecolor(colour)
        body.set_alpha(0.55)
    right.set_xticks([1, 2])
    right.set_xticklabels(["T1", "OOD"])
    right.set_ylabel("epistemic std (£)")
    auroc = payload.get("OOD", {}).get("separation_auroc", {}).get("profit", float("nan"))
    right.set_title(
        f"Separation of OOD from in-distribution\nAUROC {auroc:.3f} "
        f"({'usable' if auroc >= 0.7 else 'barely above chance'})",
        fontsize=10,
    )
    right.grid(alpha=0.3, axis="y")

    rho = payload.get("T1", {}).get("epistemic_error_rho", {}).get("profit", float("nan"))
    fig.suptitle(
        f"Epistemic uncertainty: |error| correlation ρ = {rho:.3f} in-distribution, "
        f"OOD separation AUROC = {auroc:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "c_surrogate_epistemic")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Generate the four surrogate figures.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "surrogate"))
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    ensemble = SurrogateEnsemble.load(Path(args.model))
    t1 = load_rollouts(ROLLOUTS / "test_t1", allow_test=True)
    t2a = load_rollouts(ROLLOUTS / "test_t2_estimate1", allow_test=True)
    t2b = load_rollouts(ROLLOUTS / "test_t2_estimate2", allow_test=True)
    ood = load_rollouts(ROLLOUTS / "test_ood", allow_test=True)

    figure_profit(ensemble, t1)
    figure_ranking(ensemble, t2a, t2b)
    figure_learning_curve()
    figure_epistemic(ensemble, t1, ood)
    return 0


if __name__ == "__main__":
    sys.exit(main())
