"""LightGBM quantile demand model, with a Gaussian-copula path sampler.

Two parts, because the twin needs two different things.

**Marginals.** One booster per quantile level, trained with LightGBM's quantile
objective on ``log1p(sales)``. Quantiles are equivariant under a monotone
transform, so ``expm1`` of a predicted quantile *is* the quantile on the
original scale -- which is what makes the log transform safe here, and is why
the same transform is used for every model so the comparison stays fair.
Independent boosters can still cross, so the outputs are sorted per timestep.

**Joint structure.** Sampling each day independently from its own marginal
throws away serial correlation, and the twin's inventory dynamics would become
far too easy: a week of independently-drawn demand almost never produces the
run of high days that actually empties a buffer. A Gaussian copula fitted to
training-period residuals restores that dependence while leaving every marginal
exactly as the quantile model predicted it.

What this does *not* capture, stated here and in the model card: cross-series
dependence (each series is sampled on its own), and any effect of twin-side
decisions beyond what the covariates already encode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from twinmind.models.demand import DemandData, DemandQuantiles
from twinmind.models.demand_features import (
    FeatureSpec,
    build_inference_rows,
    feature_columns,
)
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Grid the sampler interpolates over. The headline q10/q50/q90 are a subset;
#: the extra levels exist only to give the copula's marginals enough resolution
#: in the tails, and are never reported as results.
SAMPLER_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass
class LGBMParams:
    """Booster settings. Not swept -- the time box says point at the twin instead."""

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: int | float = 0.8
    reg_lambda: float = 1.0
    verbose: int = -1


@dataclass
class CopulaSpec:
    """The fitted temporal dependence, and provenance for it."""

    correlation: NDArray[np.float64]
    horizon: int
    source: str = "training-fold residuals"
    shrinkage: float = 0.0
    n_residual_series: int = 0

    def to_npz(self, path: Path) -> Path:
        """Persist the correlation matrix with its provenance."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            correlation=self.correlation,
            horizon=np.array([self.horizon]),
            shrinkage=np.array([self.shrinkage]),
            n_residual_series=np.array([self.n_residual_series]),
            source=np.array([self.source]),
        )
        return path

    @classmethod
    def from_npz(cls, path: Path) -> CopulaSpec:
        """Restore a spec written by :meth:`to_npz`."""
        blob = np.load(path, allow_pickle=False)
        return cls(
            correlation=blob["correlation"],
            horizon=int(blob["horizon"][0]),
            source=str(blob["source"][0]),
            shrinkage=float(blob["shrinkage"][0]),
            n_residual_series=int(blob["n_residual_series"][0]),
        )


def nearest_psd(matrix: NDArray[np.float64], floor: float = 1e-8) -> NDArray[np.float64]:
    """Project a symmetric matrix onto the nearest positive semi-definite correlation matrix.

    A correlation matrix estimated from finite, unevenly-sized residual samples
    is routinely indefinite; sampling from it would fail outright. Clipping the
    eigenvalues and re-standardising the diagonal is the standard repair.
    """
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, floor, None)) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(repaired), floor, None))
    normalised = repaired / np.outer(scale, scale)
    np.fill_diagonal(normalised, 1.0)
    return np.asarray(normalised, dtype="float64")


def fit_copula(
    residuals: NDArray[np.float64],
    horizon: int,
    shrinkage: float = 0.3,
) -> CopulaSpec:
    """Estimate the ``horizon x horizon`` normal-score correlation of residuals.

    ``residuals`` is ``[n_paths, horizon]`` of **training-period** residuals.
    Each column is ranked to a probability-integral-transform value, mapped to a
    normal score, and correlated.

    The estimate is shrunk toward an AR(1)-shaped target rather than used raw.
    With a few hundred residual paths a free 28x28 matrix has 378 parameters and
    is badly under-determined in the far off-diagonal corners; the AR(1) target
    encodes the one thing that is certainly true, that dependence decays with
    separation.
    """
    if residuals.ndim != 2 or residuals.shape[1] != horizon:
        raise ValueError(f"expected [n, {horizon}] residuals, got {residuals.shape}")
    usable = residuals[~np.isnan(residuals).any(axis=1)]
    if len(usable) < 10:
        raise ValueError(f"only {len(usable)} complete residual paths; need at least 10")

    # Rank -> uniform -> normal score, per step.
    ranks = np.apply_along_axis(stats.rankdata, 0, usable)
    uniform = ranks / (len(usable) + 1.0)
    scores = stats.norm.ppf(uniform)
    empirical = np.corrcoef(scores, rowvar=False)

    # AR(1) target, with rho taken from the observed lag-1 correlation.
    lag1 = float(np.mean([empirical[i, i + 1] for i in range(horizon - 1)]))
    rho = float(np.clip(lag1, 0.0, 0.99))
    separation = np.abs(np.subtract.outer(np.arange(horizon), np.arange(horizon)))
    target = rho**separation

    blended = (1.0 - shrinkage) * empirical + shrinkage * target
    logger.info(
        "copula fitted on %d residual paths: lag-1 %.3f, shrinkage %.2f toward AR(1)",
        len(usable),
        lag1,
        shrinkage,
    )
    return CopulaSpec(
        correlation=nearest_psd(blended),
        horizon=horizon,
        shrinkage=shrinkage,
        n_residual_series=int(len(usable)),
    )


def interpolate_quantiles(
    levels: NDArray[np.float64], values: NDArray[np.float64], u: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Invert a discrete quantile function at probabilities ``u``.

    Linear between the fitted levels. Outside them the nearest fitted quantile is
    held flat rather than extrapolated -- extrapolating a tail from a 5% and a
    95% point invents structure the model never estimated, and for demand it
    readily produces absurd peaks.
    """
    return np.interp(u, levels, values)


class LightGBMQuantileModel:
    """Quantile demand forecasts and copula-coupled sample paths."""

    def __init__(
        self,
        boosters: dict[float, Any],
        spec: FeatureSpec,
        data: DemandData,
        copula: CopulaSpec | None = None,
        conformal: dict[str, NDArray[np.float64]] | None = None,
        horizon: int = 28,
    ) -> None:
        self.boosters = boosters
        self.spec = spec
        self.data = data
        self.copula = copula
        self.conformal = conformal or {}
        self.horizon = horizon

    # ------------------------------------------------------------- predict --
    def _predict_grid(
        self, series_id: str, origin: pd.Timestamp, horizon: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(levels, [horizon, n_levels])`` predictions in original units, sorted."""
        history = self.data.history(series_id, origin)
        if history.empty:
            raise KeyError(f"no history for series {series_id!r} at or before {origin}")
        future = self.data.future(series_id, origin, horizon)
        rows = build_inference_rows(history, future, origin, horizon, target=self.data.target)
        x = rows[self.spec.columns]

        levels = np.array(sorted(self.boosters), dtype="float64")
        raw = np.column_stack([self.boosters[level].predict(x) for level in levels])
        values = np.clip(np.expm1(raw), 0.0, None)

        if self.conformal:
            values = self._apply_conformal(values, levels, horizon)

        # Independent boosters can cross; sorting each row restores monotonicity.
        # Done here, once, so every consumer -- quantiles() and the sampler --
        # sees a valid quantile function.
        return levels, np.sort(values, axis=1)

    def _apply_conformal(
        self, values: NDArray[np.float64], levels: NDArray[np.float64], horizon: int
    ) -> NDArray[np.float64]:
        """Widen the interval by the stored split-conformal offsets."""
        lower = self.conformal.get("lower")
        upper = self.conformal.get("upper")
        if lower is None or upper is None:
            return values
        out = values.copy()
        for index, level in enumerate(levels):
            if level < 0.5:
                out[:, index] -= lower[:horizon]
            elif level > 0.5:
                out[:, index] += upper[:horizon]
        return np.clip(out, 0.0, None)

    def quantiles(self, series_id: str, origin: pd.Timestamp, horizon: int) -> DemandQuantiles:
        """Headline q10/q50/q90 for ``origin+1 .. origin+horizon``."""
        levels, values = self._predict_grid(series_id, origin, horizon)
        picked = {q: values[:, int(np.argmin(np.abs(levels - q)))] for q in (0.1, 0.5, 0.9)}
        return DemandQuantiles(
            horizon=horizon,
            q10=picked[0.1].tolist(),
            q50=picked[0.5].tolist(),
            q90=picked[0.9].tolist(),
        )

    def sample_paths(
        self,
        series_id: str,
        origin: pd.Timestamp,
        horizon: int,
        n_samples: int,
        seed: int,
    ) -> NDArray[np.float64]:
        """``[n_samples, horizon]`` trajectories with the fitted serial dependence."""
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        levels, values = self._predict_grid(series_id, origin, horizon)

        rng = np.random.default_rng(seed)
        correlation = self._correlation_for(horizon)
        normal = rng.multivariate_normal(np.zeros(horizon), correlation, size=n_samples)
        uniform = stats.norm.cdf(normal)

        paths = np.empty((n_samples, horizon), dtype="float64")
        for step in range(horizon):
            paths[:, step] = interpolate_quantiles(levels, values[step], uniform[:, step])
        return np.clip(paths, 0.0, None)

    def _correlation_for(self, horizon: int) -> NDArray[np.float64]:
        """The copula correlation, sliced or identity if none was fitted."""
        if self.copula is None:
            return np.eye(horizon)
        if self.copula.horizon < horizon:
            raise ValueError(f"copula covers {self.copula.horizon} steps, {horizon} requested")
        return nearest_psd(self.copula.correlation[:horizon, :horizon])

    # ----------------------------------------------------------- artifacts --
    def save(self, directory: Path, metadata: dict[str, Any] | None = None) -> Path:
        """Write boosters, feature spec, copula and conformal offsets.

        Boosters go out in LightGBM's native text format, not pickled and not
        renamed to ``.pt``: the format is portable across LightGBM versions and
        readable, and a ``.pt`` extension on a non-PyTorch artifact is a lie that
        costs somebody an afternoon later.
        """
        directory.mkdir(parents=True, exist_ok=True)
        for level, booster in self.boosters.items():
            name = f"lgbm_q{int(round(level * 100)):02d}.txt"
            booster.booster_.save_model(str(directory / name))

        config: dict[str, Any] = {
            "feature_spec": asdict(self.spec),
            "quantiles": sorted(self.boosters),
            "headline_quantiles": [0.1, 0.5, 0.9],
            "horizon": self.horizon,
            "transform": "log1p / expm1",
            **(metadata or {}),
        }
        (directory / "config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )
        if self.copula is not None:
            self.copula.to_npz(directory / "copula.npz")
        if self.conformal:
            np.savez(
                directory / "conformal.npz",
                **{k: np.asarray(v) for k, v in self.conformal.items()},
            )
        logger.info("saved demand model to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Path, data: DemandData) -> LightGBMQuantileModel:
        """Restore a model saved by :meth:`save` and bind it to ``data``."""
        import lightgbm as lgb

        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        spec = FeatureSpec(**config["feature_spec"])

        boosters: dict[float, Any] = {}
        for level in config["quantiles"]:
            path = directory / f"lgbm_q{int(round(float(level) * 100)):02d}.txt"
            wrapper = _BoosterWrapper(lgb.Booster(model_file=str(path)))
            boosters[float(level)] = wrapper

        copula = (
            CopulaSpec.from_npz(directory / "copula.npz")
            if (directory / "copula.npz").exists()
            else None
        )
        conformal = {}
        if (directory / "conformal.npz").exists():
            blob = np.load(directory / "conformal.npz")
            conformal = {k: blob[k] for k in blob.files}

        return cls(boosters, spec, data, copula, conformal, int(config.get("horizon", 28)))


@dataclass
class _BoosterWrapper:
    """Gives a raw ``lgb.Booster`` the ``.predict`` / ``.booster_`` shape used above."""

    booster_: Any = field(repr=False)

    def predict(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Predict, matching the sklearn wrapper's interface."""
        return np.asarray(self.booster_.predict(x), dtype="float64")


def fit_quantile_models(
    rows: pd.DataFrame,
    quantiles: tuple[float, ...] = SAMPLER_QUANTILES,
    params: LGBMParams | None = None,
    target: str = "y",
    seed: int = 42,
    objective: str = "quantile",
) -> tuple[dict[float, Any], FeatureSpec]:
    """Fit one booster per quantile level on ``log1p(target)``.

    ``objective="l2"`` fits a squared-error point model instead, which is what
    the point baseline uses.
    """
    import lightgbm as lgb

    settings = params or LGBMParams()
    columns = feature_columns(rows, target)
    x = rows[columns]
    y = np.log1p(np.clip(rows[target].to_numpy(dtype="float64"), 0.0, None))

    boosters: dict[float, Any] = {}
    for level in quantiles:
        if objective == "l2":
            # A genuine point forecast: squared error, estimating the conditional
            # MEAN. Fitting alpha=0.5 instead would be the quantile model's own
            # median booster under another name, so the "baseline" would differ
            # from the model it is meant to benchmark only by the monotonicity
            # sort -- a comparison of a model against itself.
            model = lgb.LGBMRegressor(objective="regression", random_state=seed, **asdict(settings))
        else:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=level, random_state=seed, **asdict(settings)
            )
        model.fit(x, y)
        boosters[level] = model
    logger.info(
        "fitted %d quantile boosters on %d rows, %d features",
        len(boosters),
        len(rows),
        len(columns),
    )
    return boosters, FeatureSpec(columns=columns)


def fit_conformal(
    y_true: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    steps: NDArray[np.int64],
    horizon: int,
    alpha: float = 0.2,
) -> dict[str, NDArray[np.float64]]:
    """Split-conformal (CQR) offsets, estimated per horizon step.

    The conformity score is the CQR one: how far outside its own interval the
    outcome fell, ``max(lower - y, y - upper)``, negative when the outcome was
    comfortably inside. Taking the ``1 - alpha`` quantile of those scores gives
    the amount the interval must be widened to reach nominal coverage on data
    distributed like the calibration set.

    **Per step, not pooled.** Intervals undercover far more at day 28 than at
    day 1, and a single pooled offset would over-widen the near horizon to fix
    the far one. A separate offset per step costs nothing and keeps both honest.

    The offsets are symmetric here (the same widening applied below and above),
    which is the standard CQR form. Asymmetric offsets are possible but need
    roughly twice the calibration data to estimate stably, and this calibration
    fold holds one 28-day window.

    ``alpha=0.2`` targets the 80% interval that q10/q90 nominally define.
    """
    lower_offsets = np.zeros(horizon, dtype="float64")
    upper_offsets = np.zeros(horizon, dtype="float64")

    for step in range(1, horizon + 1):
        mask = steps == step
        if not mask.any():
            continue
        scores = np.maximum(lower[mask] - y_true[mask], y_true[mask] - upper[mask])
        # The finite-sample correction: with n calibration points the valid level
        # is ceil((n+1)(1-alpha))/n, which matters when n is small.
        n = int(mask.sum())
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        offset = float(np.quantile(scores, level))
        # A negative offset would tighten an already over-wide interval. Allowed,
        # but floored at the point where the interval would invert.
        lower_offsets[step - 1] = max(offset, -0.0)
        upper_offsets[step - 1] = max(offset, -0.0)

    logger.info(
        "conformal offsets fitted on %d points: mean widening %.2f (day 1 %.2f, day %d %.2f)",
        len(y_true),
        float(np.mean(upper_offsets)),
        float(upper_offsets[0]),
        horizon,
        float(upper_offsets[-1]),
    )
    return {"lower": lower_offsets, "upper": upper_offsets}
