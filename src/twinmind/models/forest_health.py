"""ForestHealthModel: a random forest behind the HealthModel Protocol.

The stopping rule made this the production hazard model. At FD001 scale a tuned
tree ensemble on explicit degradation-trend features beats a 738k-parameter
sequence model, so the twin should consume the forest and A1 stays as the
deep-learning comparison. See ``reports/model_cards/a1_health.md``.

The twin's interface does not change. This satisfies exactly the same Protocol
A1 does and returns the same validated :class:`RULPrediction`, so swapping the
hazard source is a one-line change in the simulator and nothing downstream
notices.

**Uncertainty without MC-dropout.** A forest carries its own ensemble spread:
every tree is a separate estimate of the same quantity, so the standard deviation
across trees is a predictive spread that needs no extra machinery and no
stochastic forward passes. Two estimators are offered:

``spread="tree_std"``
    Standard deviation of the per-tree predictions. Cheap, and the natural
    analogue of the MC-dropout std it replaces.
``spread="quantile"``
    A quantile-regression reading of the same per-tree values: the interval is
    taken from empirical quantiles rather than assuming symmetry, and the std is
    derived from it. Better behaved near the 125 cap, where the distribution is
    hard against a boundary and a symmetric std overstates the upside.

Failure probability comes from a second forest trained as a classifier on the
same features, rather than from thresholding the RUL point estimate -- the twin
consumes it as a hazard, so it should be a probability the model actually fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from twinmind.data.cmapss_prep import CMAPSSData, failure_labels, window_features
from twinmind.models.health import RULPrediction
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

Spread = Literal["tree_std", "quantile"]


@dataclass
class ForestConfig:
    """Hyperparameters for both forests."""

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    max_features: float | str = 1.0
    n_jobs: int = -1
    horizon_h: int = 30
    rul_cap: int = 125
    spread: Spread = "tree_std"
    #: Interval used when spread="quantile"; 0.68 mirrors one standard deviation.
    interval: float = 0.68


class ForestHealthModel:
    """Random-forest RUL and failure probability, satisfying ``HealthModel``."""

    def __init__(self, config: ForestConfig | None = None) -> None:
        self.config = config or ForestConfig()
        self.regressor: RandomForestRegressor | None = None
        self.classifier: RandomForestClassifier | None = None
        self.n_features_: int = 0

    # ------------------------------------------------------------------ fit --
    def fit(self, data: CMAPSSData, seed: int = 42) -> ForestHealthModel:
        """Train both forests on the training split's window features."""
        x = window_features(data.x_train)
        y = np.asarray(data.y_train, dtype="float64")
        fail = failure_labels(data.y_train, self.config.horizon_h)

        self.regressor = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=seed,
            n_jobs=self.config.n_jobs,
        ).fit(x, y)

        # A degenerate split (every window inside or outside the horizon) leaves
        # nothing to classify; fall back to thresholding the regressor rather
        # than fitting a classifier that has seen one class.
        self.classifier = None
        if len(np.unique(fail)) > 1:
            self.classifier = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=self.config.n_jobs,
            ).fit(x, fail)
        else:  # pragma: no cover - only on a degenerate horizon
            logger.warning("failure labels are single-class; p_fail falls back to the RUL head")

        self.n_features_ = x.shape[1]
        logger.info(
            "forest fitted: %d trees, %d features, %d windows (%.1f%% within h=%d)",
            self.config.n_estimators,
            self.n_features_,
            len(x),
            100 * float(fail.mean()),
            self.config.horizon_h,
        )
        return self

    # -------------------------------------------------------------- predict --
    def _tree_predictions(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """``[n_trees, n_samples]`` of per-tree RUL estimates."""
        if self.regressor is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return np.stack([tree.predict(x) for tree in self.regressor.estimators_])

    def _spread(self, per_tree: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predictive standard deviation, by the configured estimator."""
        if self.config.spread == "tree_std":
            return np.asarray(per_tree.std(axis=0), dtype="float64")

        # Quantile reading: take the central interval empirically and convert its
        # width to a standard deviation. Near the cap the distribution is pressed
        # against a boundary, and a symmetric std reads wider than reality.
        tail = (1.0 - self.config.interval) / 2.0
        lo = np.quantile(per_tree, tail, axis=0)
        hi = np.quantile(per_tree, 1.0 - tail, axis=0)
        # For a normal distribution a 68% interval spans about 2 standard deviations.
        return np.asarray(np.maximum(hi - lo, 0.0) / 2.0, dtype="float64")

    def predict_arrays(
        self, windows: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(rul_mean, rul_std, p_fail)`` for ``[N, T, C]`` windows."""
        if self.regressor is None:
            raise RuntimeError("model is not fitted; call fit() first")
        x = window_features(np.asarray(windows, dtype="float32"))

        per_tree = self._tree_predictions(x)
        mean = np.clip(per_tree.mean(axis=0), 0.0, float(self.config.rul_cap))
        std = self._spread(per_tree)

        if self.classifier is not None:
            probability = self.classifier.predict_proba(x)[:, 1]
        else:  # pragma: no cover - degenerate-horizon fallback
            probability = (mean <= self.config.horizon_h).astype("float64")
        return mean, std, np.clip(probability, 0.0, 1.0)

    def predict(self, window: NDArray[np.floating]) -> RULPrediction:
        """Predict from a single window of shape ``[T, C]``."""
        array = np.asarray(window)
        if array.ndim != 2:
            raise ValueError(f"expected a single window [T, C], got {array.shape}")
        return self.predict_batch(array[None, ...])[0]

    def predict_batch(self, windows: NDArray[np.floating]) -> list[RULPrediction]:
        """Predict from a batch of shape ``[N, T, C]``."""
        array = np.asarray(windows)
        if array.ndim != 3:
            raise ValueError(f"expected a batch [N, T, C], got {array.shape}")

        mean, std, probability = self.predict_arrays(array)
        return [
            RULPrediction(
                rul_mean=float(m),
                rul_std=float(s),
                p_fail_within_h=float(p),
                horizon_h=self.config.horizon_h,
            )
            for m, s, p in zip(mean, std, probability, strict=True)
        ]

    # ----------------------------------------------------------- persistence --
    def save(self, path: Path) -> Path:
        """Persist both forests and the configuration."""
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config,
                "regressor": self.regressor,
                "classifier": self.classifier,
                "n_features": self.n_features_,
            },
            path,
            compress=3,
        )
        logger.info("saved forest to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @classmethod
    def load(cls, path: Path) -> ForestHealthModel:
        """Restore a model saved by :meth:`save`."""
        import joblib

        payload: dict[str, Any] = joblib.load(path)
        model = cls(payload["config"])
        model.regressor = payload["regressor"]
        model.classifier = payload["classifier"]
        model.n_features_ = int(payload["n_features"])
        return model
