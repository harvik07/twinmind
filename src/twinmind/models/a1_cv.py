"""H1: GroupKFold cross-validation over engine IDs, and the fold ensemble.

The diagnosis this addresses. FD001 holds 100 training engines; an 80/20 split
leaves 20 in validation. Validation RMSE on 20 engines is noisy enough that
early stopping with patience 15 was selecting epoch 4-6 more or less at random,
and the selected epoch varied wildly between seeds -- one seed in three landed on
a checkpoint that tested at 17.0 against 14.6 for its siblings.

Two changes follow from that:

* **Select the epoch on a mean across folds.** Five folds average five noisy
  curves, so the epoch chosen is the one that is genuinely best rather than the
  one whose fold happened to dip.
* **Ensemble the folds.** Five models trained on 80% of engines each, averaged,
  is both a better estimator and a cheaper one than five seeds of the full model.

Every fold is split on *engine id*, never on rows: two windows from the same
engine are nearly identical, so a row-level split measures interpolation between
adjacent cycles instead of generalisation to an unseen engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from twinmind.data.cmapss_prep import CMAPSSData
from twinmind.eval.metrics import evaluate_cmapss_test
from twinmind.models.a1_cnn_bilstm import A1Net, predict_with_uncertainty
from twinmind.models.a1_train import TrainConfig, train_a1
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CVResult:
    """Outcome of one cross-validated run."""

    fold_metrics: list[dict[str, float]]
    ensemble_metrics: dict[str, float]
    mean_best_epoch: float
    duration_s: float
    n_parameters: int
    fold_val_rmse: list[float] = field(default_factory=list)

    @property
    def mean_fold_metrics(self) -> dict[str, float]:
        """Per-fold metrics averaged -- the single-model expectation."""
        if not self.fold_metrics:
            return {}
        return {
            key: float(np.mean([m[key] for m in self.fold_metrics])) for key in self.fold_metrics[0]
        }


def _fold_view(
    data: CMAPSSData, train_idx: NDArray[np.int64], val_idx: NDArray[np.int64]
) -> CMAPSSData:
    """A CMAPSSData whose train/val arrays are one fold of the original training set."""
    return replace(
        data,
        x_train=data.x_train[train_idx],
        y_train=data.y_train[train_idx],
        units_train=data.units_train[train_idx],
        x_val=data.x_train[val_idx],
        y_val=data.y_train[val_idx],
        units_val=data.units_train[val_idx],
    )


def train_a1_cv(
    data: CMAPSSData,
    n_splits: int = 5,
    seed: int = 42,
    horizon_h: int = 30,
    dropout: float = 0.3,
    lambda_fail: float | None = 0.5,
    train_config: TrainConfig | None = None,
    device: str = "cpu",
    arch_overrides: dict[str, Any] | None = None,
    quiet: bool = True,
) -> tuple[list[A1Net], CVResult]:
    """Train ``n_splits`` fold models and return them with per-fold and ensemble metrics.

    The training and validation arrays of ``data`` are pooled first, so the folds
    partition every available engine rather than only the 80% the original split
    left in training. The test arrays are untouched.
    """
    settings = train_config or TrainConfig()
    started = time.perf_counter()

    # Pool train + val, then refold. The original 80/20 split is an artefact of
    # prepare_cmapss; cross-validation should see all the engines there are.
    pooled = replace(
        data,
        x_train=np.concatenate([data.x_train, data.x_val]),
        y_train=np.concatenate([data.y_train, data.y_val]),
        units_train=np.concatenate([data.units_train, data.units_val]),
    )
    groups = pooled.units_train
    n_engines = len(np.unique(groups))
    if n_splits > n_engines:
        raise ValueError(f"{n_splits} folds requested but only {n_engines} engines available")

    splitter = GroupKFold(n_splits=n_splits)
    models: list[A1Net] = []
    fold_metrics: list[dict[str, float]] = []
    fold_val: list[float] = []
    best_epochs: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(pooled.x_train, pooled.y_train, groups=groups)
    ):
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        if overlap:  # pragma: no cover - GroupKFold guarantees this
            raise AssertionError(f"fold {fold} leaks engines across the split: {sorted(overlap)}")

        model, result = train_a1(
            _fold_view(pooled, train_idx, val_idx),
            seed=seed + fold,
            horizon_h=horizon_h,
            dropout=dropout,
            lambda_fail=lambda_fail,
            train_config=settings,
            device=device,
            checkpoint_dir=None,
            quiet=quiet,
            arch_overrides=arch_overrides,
        )
        models.append(model)
        fold_metrics.append(result.metrics)
        fold_val.append(result.best_val_rmse)
        best_epochs.append(result.best_epoch)
        logger.info(
            "  fold %d/%d: val %.3f  test %.3f  (best epoch %d, %d engines held out)",
            fold + 1,
            n_splits,
            result.best_val_rmse,
            result.metrics["rmse"],
            result.best_epoch,
            len(np.unique(groups[val_idx])),
        )

    ensemble = evaluate_cmapss_test(
        data.y_test, cv_ensemble_predict(models, data, device), data.units_test
    )
    return models, CVResult(
        fold_metrics=fold_metrics,
        ensemble_metrics=ensemble,
        mean_best_epoch=float(np.mean(best_epochs)),
        duration_s=time.perf_counter() - started,
        n_parameters=sum(p.numel() for p in models[0].parameters()),
        fold_val_rmse=fold_val,
    )


def cv_ensemble_predict(
    models: list[A1Net], data: CMAPSSData, device: str = "cpu"
) -> NDArray[np.float64]:
    """Mean of the fold models' deterministic predictions on the test windows."""
    x = torch.from_numpy(data.x_test.astype("float32")).to(torch.device(device))
    stacked = []
    for model in models:
        mean, _, _, _ = predict_with_uncertainty(model, x, n_passes=1, rul_cap=float(data.rul_cap))
        stacked.append(mean)
    return np.mean(np.stack(stacked), axis=0)
