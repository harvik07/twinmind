"""A2: failure classifier on AI4I 2020.

A separate model from A1, not a third head. AI4I is a single row of
milling-machine measurements; A1's input is a 30-cycle window of turbofan
sensors. There is no shared input space, so there is no trunk to share.

A2 exists for two reasons: the class-imbalance and calibration story (3.39%
positives, and the twin turns a probability into a hazard so it must be
calibrated, not merely ranked), and as an independent check that the health
signal is learnable from process physics rather than from C-MAPSS quirks.

Leakage: ``TWF/HDF/PWF/OSF/RNF`` OR together into the target for 99.73% of rows.
They never enter the feature matrix, and :func:`build_features` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twinmind.eval.metrics import (
    best_threshold_fbeta,
    brier_score,
    expected_calibration_error,
    pr_auc,
    recall_at_precision,
    roc_auc,
)
from twinmind.utils.config import AI4IConfig
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Engineered features, added because the raw columns do not express them.
ENGINEERED = ("power_w", "temperature_difference_k")


def build_features(
    features: pd.DataFrame,
    config: AI4IConfig,
    failure_modes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add the two physically motivated features and assert no mode leaked in.

    ``power_w = torque * rpm * 2*pi/60`` is mechanical power in watts. The AI4I
    generator defines its power-failure mode by a power threshold, so the product
    is the quantity that actually separates that class -- a model given only
    torque and speed has to learn the multiplication from data.

    ``temperature_difference_k = process - air`` is what drives heat dissipation
    failure; the two absolute temperatures carry it only implicitly, and their
    difference is small relative to their magnitudes (roughly 10 K against 300 K),
    so it is easily lost in scaling.
    """
    out = features.copy()

    required = {"torque_nm", "rotational_speed_rpm", "process_temperature_k", "air_temperature_k"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"AI4I features are missing {sorted(missing)}")

    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * (2 * np.pi / 60.0)
    out["temperature_difference_k"] = out["process_temperature_k"] - out["air_temperature_k"]

    leaked = set(out.columns) & set(config.failure_mode_columns)
    if leaked:
        raise AssertionError(f"failure-mode columns leaked into A2 features: {sorted(leaked)}")
    if config.target_column in out.columns:
        raise AssertionError(f"the target {config.target_column!r} is in the A2 feature matrix")
    if failure_modes is not None and set(out.columns) & set(failure_modes.columns):
        raise AssertionError("a failure-mode column reached the A2 feature matrix")

    return out


@dataclass
class FoldResult:
    """Out-of-fold predictions and per-fold metrics."""

    fold: int
    y_true: NDArray[np.int64]
    y_prob: NDArray[np.float64]
    metrics: dict[str, float] = field(default_factory=dict)


def score_predictions(
    y_true: NDArray[np.int64], y_prob: NDArray[np.float64], target_precision: float = 0.80
) -> dict[str, float]:
    """The A2 metric set: ranking, operating point and calibration."""
    return {
        "pr_auc": pr_auc(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
        "recall_at_p80": recall_at_precision(y_true, y_prob, target_precision),
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=10),
    }


def cross_validate(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str = "lightgbm",
    n_splits: int = 5,
    seed: int = 42,
    calibrate: bool | str = "auto",
) -> tuple[list[FoldResult], dict[str, float]]:
    """Stratified k-fold CV returning per-fold results and pooled out-of-fold metrics.

    Metrics are computed on the pooled out-of-fold predictions rather than averaged
    across folds. With 339 positives in total, a single fold holds about 68 of
    them, and PR-AUC on that few is noisy enough that the fold-to-fold spread
    would swamp the effect being measured.

    ``calibrate="auto"`` applies isotonic calibration only where it was measured
    to help, which is per-estimator rather than universal:

    ==========  ============  =============  ==========  ==========
    model       calibrated    PR-AUC         recall@P80  ECE
    ==========  ============  =============  ==========  ==========
    lightgbm    no            **0.8647**     **0.805**   **0.0083**
    lightgbm    yes           0.8525         0.785       0.0102
    mlp         no            **0.7400**     **0.546**   0.1981
    mlp         yes           0.6771         0.425       **0.0122**
    ==========  ============  =============  ==========  ==========

    LightGBM is already well calibrated despite ``scale_pos_weight``, so the
    wrapper only costs ranking by refitting on 2/3 subsets. The focal-loss MLP is
    badly miscalibrated -- focal deliberately down-weights confident examples,
    which distorts the probability scale -- and isotonic cuts its ECE from 0.198
    to 0.012. Since the twin consumes this probability as a hazard rather than
    merely ranking with it, the MLP is unusable uncalibrated.
    """
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_array = y.to_numpy().astype("int64")
    results: list[FoldResult] = []

    apply_calibration = model_name == "mlp" if calibrate == "auto" else bool(calibrate)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y_array)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        estimator = build_estimator(model_name, y_train, seed=seed, calibrate=apply_calibration)
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_test)[:, 1].astype("float64")

        results.append(
            FoldResult(
                fold=fold,
                y_true=y_test,
                y_prob=probabilities,
                metrics=score_predictions(y_test, probabilities),
            )
        )
        logger.info(
            "  fold %d: PR-AUC %.4f  recall@P80 %.3f  ECE %.4f",
            fold,
            results[-1].metrics["pr_auc"],
            results[-1].metrics["recall_at_p80"],
            results[-1].metrics["ece"],
        )

    pooled_true = np.concatenate([r.y_true for r in results])
    pooled_prob = np.concatenate([r.y_prob for r in results])
    pooled = score_predictions(pooled_true, pooled_prob)

    threshold, fbeta = best_threshold_fbeta(pooled_true, pooled_prob, beta=2.0)
    pooled["operating_threshold"] = threshold
    pooled["f2_at_threshold"] = fbeta
    return results, pooled


def build_estimator(
    model_name: str, y_train: NDArray[np.int64], seed: int = 42, calibrate: bool = False
) -> Any:
    """Construct one A2 estimator, optionally isotonic-calibrated."""
    from sklearn.calibration import CalibratedClassifierCV

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        positives = max(int(y_train.sum()), 1)
        negatives = int(len(y_train) - positives)
        base: Any = LGBMClassifier(
            # Selected by a 5-config sweep on pooled out-of-fold PR-AUC; the
            # metric is flat across a wide range (0.847-0.859), which is itself
            # evidence that the ceiling is the data, not the tuning.
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=10,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=negatives / positives,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_name == "mlp":
        base = TorchMLPClassifier(seed=seed)
    else:
        raise ValueError(f"unknown A2 model {model_name!r}; expected 'lightgbm' or 'mlp'")

    if not calibrate:
        return base
    # Isotonic rather than sigmoid: the distortion from scale_pos_weight is a
    # monotone rescaling, not a logistic shift, so a non-parametric map fits it.
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


class TorchMLPClassifier:
    """A small MLP with focal loss, exposing the scikit-learn estimator API.

    Two hidden layers with dropout, trained with focal loss rather than plain BCE
    so the abundant easy negatives stop dominating the gradient. Implements
    ``fit``/``predict_proba``/``get_params``/``set_params`` so it drops into
    ``CalibratedClassifierCV`` and the CV loop unchanged.
    """

    # scikit-learn infers estimator type from this; without it
    # CalibratedClassifierCV rejects the estimator as a regressor.
    _estimator_type = "classifier"

    def __init__(
        self,
        hidden: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        gamma: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.seed = seed
        self._model: Any = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self.classes_ = np.array([0, 1])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """scikit-learn estimator protocol."""
        return {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> TorchMLPClassifier:
        """scikit-learn estimator protocol."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(
        self, x: pd.DataFrame | NDArray[np.float64], y: NDArray[np.int64]
    ) -> TorchMLPClassifier:
        """Standardise, then train with focal loss."""
        import torch
        from torch import nn

        from twinmind.models.a1_cnn_bilstm import focal_bce_with_logits
        from twinmind.utils.seed import set_seed

        set_seed(self.seed)
        values = np.asarray(x, dtype="float64")
        self._mean = values.mean(axis=0)
        self._std = values.std(axis=0)
        self._std[self._std == 0] = 1.0
        scaled = (values - self._mean) / self._std

        targets = np.asarray(y, dtype="float32")
        alpha = float(np.clip(1.0 - targets.mean(), 0.05, 0.95))

        layers: list[nn.Module] = []
        width = scaled.shape[1]
        for size in self.hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self._model = nn.Sequential(*layers)

        optimiser = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)
        x_tensor = torch.from_numpy(scaled.astype("float32"))
        y_tensor = torch.from_numpy(targets)
        n = len(x_tensor)

        self._model.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                index = permutation[start : start + self.batch_size]
                optimiser.zero_grad()
                logits = self._model(x_tensor[index]).squeeze(-1)
                loss = focal_bce_with_logits(logits, y_tensor[index], self.gamma, alpha)
                loss.backward()
                optimiser.step()
        return self

    def predict_proba(self, x: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        """Two-column probability array, as scikit-learn expects."""
        import torch

        if self._model is None or self._mean is None or self._std is None:
            raise RuntimeError("TorchMLPClassifier is not fitted")
        scaled = (np.asarray(x, dtype="float64") - self._mean) / self._std
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.from_numpy(scaled.astype("float32"))).squeeze(-1)
            positive = torch.sigmoid(logits).numpy().astype("float64")
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.int64]:
        """Hard labels at 0.5. The operating threshold is chosen elsewhere."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype("int64")
