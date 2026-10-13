"""Noise-aware evaluation for the surrogate.

THE POINT OF THIS MODULE, IN ONE PARAGRAPH.

The test labels are **30-replication means of a stochastic simulator**, so they
carry noise of their own. A perfect surrogate -- one that returns the twin's
exact expected value for every scenario -- still cannot reach R² = 1 against
them, and still cannot reproduce their ranking exactly. Reporting a raw R² of
0.96 as "4% short of perfect" would be wrong, and reporting a Spearman ρ of 0.91
as a failure against a 0.90 target would be wrong in the other direction.

So every headline number here comes with its **ceiling**:

* For regression, the noise ceiling is analytic.
  ``R²_max = 1 - mean(sigma_rep² / n_reps) / Var(truth)``: the share of the
  truth's variance that is not replication noise.
* For ranking, the ceiling is **measured**, not derived. T2 was evaluated twice
  with independent seeds, so the agreement between those two 30-rep estimates is
  exactly the agreement a perfect surrogate would show.

A KPI that cannot reach 0.95 because its own ceiling is lower is a finding about
the twin's noise, not a failure of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from twinmind.models.surrogate import KPI_ORDER


@dataclass
class KPIScore:
    """One KPI's accuracy against its own noise ceiling."""

    kpi: str
    r2: float
    r2_max: float
    r2_ratio: float
    rmse: float
    mae: float
    bias: float
    truth_sd: float
    label_noise_sd: float
    n: int

    @property
    def meets_target(self) -> bool:
        """R² >= 0.95, or >= 95% of what the noise permits."""
        return self.r2 >= 0.95 or self.r2_ratio >= 0.95


def noise_ceiling(truth: NDArray[np.float64], rep_std: NDArray[np.float64], n_reps: int) -> float:
    """``R²_max`` for a label that is a mean of ``n_reps`` replications.

    The label's own variance is ``sigma_rep² / n_reps``. A perfect predictor's
    residual is exactly that noise, so the best achievable R² is
    ``1 - E[label noise variance] / Var(truth)``.

    Returns ``nan`` when the truth has no variance, which would make R²
    undefined rather than zero.
    """
    truth_var = float(np.var(truth, ddof=1))
    if truth_var <= 0:
        return float("nan")
    label_var = float(np.mean(np.asarray(rep_std, dtype="float64") ** 2) / max(n_reps, 1))
    return 1.0 - label_var / truth_var


def score_kpi(
    kpi: str,
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    rep_std: NDArray[np.float64],
    n_reps: int,
) -> KPIScore:
    """Accuracy for one KPI, with its ceiling."""
    predicted = np.asarray(predicted, dtype="float64")
    truth = np.asarray(truth, dtype="float64")
    residual = predicted - truth
    truth_var = float(np.var(truth, ddof=1))
    r2 = 1.0 - float(np.mean(residual**2)) / truth_var if truth_var > 0 else float("nan")
    ceiling = noise_ceiling(truth, rep_std, n_reps)
    return KPIScore(
        kpi=kpi,
        r2=r2,
        r2_max=ceiling,
        r2_ratio=r2 / ceiling if ceiling and np.isfinite(ceiling) and ceiling > 0 else float("nan"),
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(np.abs(residual))),
        bias=float(np.mean(residual)),
        truth_sd=float(np.sqrt(truth_var)),
        label_noise_sd=float(np.sqrt(np.mean(rep_std**2) / max(n_reps, 1))),
        n=int(truth.size),
    )


def score_all(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    rep_std: NDArray[np.float64],
    n_reps: int,
) -> list[KPIScore]:
    """Scores for all eight KPIs, in canonical order."""
    return [
        score_kpi(kpi, predicted[:, i], truth[:, i], rep_std[:, i], n_reps)
        for i, kpi in enumerate(KPI_ORDER)
    ]


@dataclass
class RankingScore:
    """Ranking agreement on T2, against the twin-vs-twin ceiling."""

    spearman: float
    spearman_ceiling: float
    top10_overlap: int
    top10_ceiling: int
    top50_overlap: int
    top50_ceiling: int
    regret: float
    regret_fraction: float
    best_truth: float
    chosen_truth: float
    n: int

    @property
    def meets_spearman(self) -> bool:
        """ρ >= 0.90, or >= 90% of the ceiling."""
        if self.spearman >= 0.90:
            return True
        return bool(self.spearman_ceiling > 0 and self.spearman / self.spearman_ceiling >= 0.90)

    @property
    def meets_top10(self) -> bool:
        """>= 7 of 10, or >= 90% of the ceiling."""
        if self.top10_overlap >= 7:
            return True
        return bool(self.top10_ceiling > 0 and self.top10_overlap / self.top10_ceiling >= 0.90)


def rank_agreement(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    truth_b: NDArray[np.float64] | None = None,
) -> RankingScore:
    """Spearman, top-k overlap and regret, with the ceiling from a second estimate.

    ``truth_b`` is the *independent* second 30-rep estimate of the same
    scenarios. Comparing it to ``truth`` gives the agreement a **perfect**
    surrogate would show, because the only difference between the two is
    replication noise. Without it the targets are being read against an
    unattainable 1.0.

    Regret is the thing that actually matters to the optimiser: the profit lost
    by acting on the surrogate's argmax instead of the true best, measured in
    the twin's own units.
    """
    predicted = np.asarray(predicted, dtype="float64")
    truth = np.asarray(truth, dtype="float64")

    spearman = float(stats.spearmanr(predicted, truth).statistic)
    if truth_b is not None:
        ceiling = float(stats.spearmanr(np.asarray(truth_b, dtype="float64"), truth).statistic)
    else:
        ceiling = float("nan")

    def overlap(a: NDArray[np.float64], b: NDArray[np.float64], k: int) -> int:
        return len(set(np.argsort(a)[::-1][:k].tolist()) & set(np.argsort(b)[::-1][:k].tolist()))

    top10 = overlap(predicted, truth, 10)
    top50 = overlap(predicted, truth, 50)
    top10_ceiling = overlap(truth_b, truth, 10) if truth_b is not None else 0
    top50_ceiling = overlap(truth_b, truth, 50) if truth_b is not None else 0

    chosen = int(np.argmax(predicted))
    best_truth = float(np.max(truth))
    chosen_truth = float(truth[chosen])
    regret = best_truth - chosen_truth
    span = best_truth - float(np.min(truth))

    return RankingScore(
        spearman=spearman,
        spearman_ceiling=ceiling,
        top10_overlap=top10,
        top10_ceiling=top10_ceiling,
        top50_overlap=top50,
        top50_ceiling=top50_ceiling,
        regret=regret,
        regret_fraction=regret / span if span > 0 else float("nan"),
        best_truth=best_truth,
        chosen_truth=chosen_truth,
        n=int(truth.size),
    )


@dataclass
class CalibrationScore:
    """Interval coverage and whether the epistemic std tracks error."""

    coverage_95: dict[str, float] = field(default_factory=dict)
    epistemic_error_correlation: dict[str, float] = field(default_factory=dict)
    auroc_large_error: dict[str, float] = field(default_factory=dict)


def interval_coverage(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    total_std: NDArray[np.float64],
    label_noise_std: NDArray[np.float64],
) -> dict[str, float]:
    """Share of truths inside ``+-1.96 x`` the combined std, per KPI.

    The error budget includes **the truth's own noise**. The surrogate predicts
    the twin's expected value; the test label is a noisy estimate of it, so the
    gap between them carries both the model's error and the label's. Comparing
    against the model's std alone would score a well-calibrated model as
    over-confident purely because the yardstick is noisy.
    """
    combined = np.sqrt(np.asarray(total_std) ** 2 + np.asarray(label_noise_std) ** 2)
    inside = np.abs(predicted - truth) <= 1.96 * np.maximum(combined, 1e-12)
    return {kpi: float(inside[:, i].mean()) for i, kpi in enumerate(KPI_ORDER)}


def epistemic_tracks_error(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    epistemic_std: NDArray[np.float64],
) -> dict[str, float]:
    """Spearman correlation of ``|error|`` with the epistemic std, per KPI.

    A positive correlation is the claim that ensemble disagreement is a usable
    "I do not know this" signal. Near zero means it is decoration.
    """
    out: dict[str, float] = {}
    for index, kpi in enumerate(KPI_ORDER):
        error = np.abs(predicted[:, index] - truth[:, index])
        spread = epistemic_std[:, index]
        if np.std(spread) < 1e-12 or np.std(error) < 1e-12:
            out[kpi] = float("nan")
            continue
        out[kpi] = float(stats.spearmanr(spread, error).statistic)
    return out


def auroc(scores: NDArray[np.float64], labels: NDArray[np.bool_]) -> float:
    """AUROC via the rank-sum identity, with no sklearn dependency.

    ``(mean rank of positives - (n_pos + 1)/2) / n_neg``, which is the
    Mann-Whitney U statistic normalised -- identical to the ROC area and robust
    to ties through ``rankdata``'s average ranks.
    """
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(np.asarray(scores, dtype="float64"))
    return float((ranks[labels].mean() - (n_pos + 1) / 2) / n_neg)


def flags_large_error(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    epistemic_std: NDArray[np.float64],
    quantile: float = 0.9,
) -> dict[str, float]:
    """AUROC for "epistemic std flags a large error", per KPI.

    "Large" is the top decile of absolute error on this split. This is the
    operational question for a fallback-to-twin trigger: if the surrogate is
    about to be badly wrong, does the epistemic std say so?
    """
    out: dict[str, float] = {}
    for index, kpi in enumerate(KPI_ORDER):
        error = np.abs(predicted[:, index] - truth[:, index])
        threshold = float(np.quantile(error, quantile))
        out[kpi] = auroc(epistemic_std[:, index], error > threshold)
    return out


def ood_separation(
    in_distribution: NDArray[np.float64], out_of_distribution: NDArray[np.float64]
) -> dict[str, float]:
    """AUROC for "epistemic std separates OOD from in-distribution", per KPI.

    The strongest test of the epistemic signal: it should be larger in a region
    the training data never covered, with no access to the labels.
    """
    out: dict[str, float] = {}
    for index, kpi in enumerate(KPI_ORDER):
        scores = np.concatenate([in_distribution[:, index], out_of_distribution[:, index]])
        labels = np.concatenate(
            [
                np.zeros(in_distribution.shape[0], dtype=bool),
                np.ones(out_of_distribution.shape[0], dtype=bool),
            ]
        )
        out[kpi] = auroc(scores, labels)
    return out
