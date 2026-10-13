"""The twin's demand process: Model B's paths, rescaled and reshaped for the line.

THREE ADJUSTMENTS, ALL ASSUMPTIONS, ALL APPLIED OUTSIDE THE FROZEN PROTOCOL.

``DemandModel.sample_paths`` returns grocery units for one Favorita store-family
series. The twin makes a product on a five-station line. Something has to bridge
them, and that something is an assumption rather than a measurement, so it lives
here and is declared in ``configs/twin.yaml``:

* ``demand_scale`` -- multiplies the path so mean demand is about
  ``demand_capacity_ratio`` of nominal plant capacity. Without it the line is
  either permanently starved of orders or permanently saturated, and in both
  cases every throughput lever is worthless.
* ``price_elasticity`` -- demand multiplies by ``price_multiplier ** elasticity``.
  Favorita has no price column, so this cannot be estimated from the data.
* ``demand_spread_multiplier`` -- rescales each path's deviation from its own
  median. This one exists for a specific documented reason: Model B's
  post-conformal coverage is uneven (d8-14 averages 0.716 against a
  [0.76, 0.84] target; on fold 1 an 80% interval covered 65% of outcomes), so
  the sampled paths are probably too NARROW at mid-horizon, which means the
  twin probably UNDERSTATES stockout risk. The multiplier makes that bias
  testable instead of invisible.

None of the three is applied inside ``LightGBMQuantileModel``. The Protocol is
frozen and shared with the evaluation code, where these rescalings would be
wrong; they are twin-side modelling choices, so they belong to the twin.

WHY A SAMPLER OBJECT EXISTS.

``sample_paths`` runs LightGBM over 28 feature rows for each of 7 quantile
boosters, then builds a copula. Measured at about 25 ms, which against a 250 ms
budget for a whole replication is a tenth of it spent re-deriving a quantile
grid that cannot change -- the scenario does not affect the forecast. So the
grid and the copula factor are computed once per ``(series, origin, horizon)``
and cached in a :class:`DemandSampler`.

The important part is that this is not a second implementation.
``LightGBMQuantileModel.sample_paths`` *delegates* to the same
:meth:`DemandSampler.sample`, so there is one sampling code path and a test
asserts the two entry points return bit-identical arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from twinmind.models.demand import DemandModel
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Cached per process, keyed by (id(model), series_id, origin, horizon). Built
#: lazily and never mutated, so it is safe under multiprocessing "spawn": each
#: worker imports this module fresh and fills its own.
_SAMPLER_MEMO: dict[tuple[int, str, pd.Timestamp, int], DemandSampler] = {}


@dataclass(frozen=True)
class DemandSampler:
    """A reusable sampler for one series at one origin.

    Holds the calibrated quantile grid and the Cholesky factor of the copula
    correlation, which are the two expensive things ``sample_paths`` computes
    and the two that do not depend on the scenario or the seed.
    """

    levels: NDArray[np.float64]
    values: NDArray[np.float64]
    cholesky: NDArray[np.float64]
    horizon: int

    @classmethod
    def from_model(
        cls, model: DemandModel, series_id: str, origin: pd.Timestamp, horizon: int
    ) -> DemandSampler:
        """Build from any ``DemandModel``, using the model's own grid when it has one.

        A ``LightGBMQuantileModel`` exposes ``_predict_grid``, which is the
        calibrated 7-quantile grid the conformal step and the copula were both
        built on -- using it means the sampler inherits the calibration rather
        than re-deriving a narrower band from q10/q50/q90. A stub model that has
        only the Protocol falls back to those three headline quantiles, which is
        enough for the non-data tests.
        """
        grid = getattr(model, "_predict_grid", None)
        if callable(grid):
            levels, values = grid(series_id, origin, horizon)
            correlation = model._correlation_for(horizon)  # type: ignore[attr-defined]
        else:
            quantiles = model.quantiles(series_id, origin, horizon)
            levels = np.array([0.1, 0.5, 0.9], dtype="float64")
            values = np.column_stack(
                [
                    np.asarray(quantiles.q10, dtype="float64"),
                    np.asarray(quantiles.q50, dtype="float64"),
                    np.asarray(quantiles.q90, dtype="float64"),
                ]
            )
            correlation = np.eye(horizon)

        # Cholesky once here rather than per call: multivariate_normal refactors
        # the covariance on every invocation, which is the other half of the
        # 25 ms.
        jittered = np.asarray(correlation, dtype="float64") + 1e-10 * np.eye(horizon)
        return cls(
            levels=np.asarray(levels, dtype="float64"),
            values=np.asarray(values, dtype="float64"),
            cholesky=np.linalg.cholesky(jittered),
            horizon=horizon,
        )

    def sample(self, n_samples: int, seed: int) -> NDArray[np.float64]:
        """``[n_samples, horizon]`` paths, identical to ``sample_paths``' output.

        Reproduces ``rng.multivariate_normal(0, corr)`` exactly as
        ``standard_normal @ cholesky.T``, which is what numpy does internally
        for a Cholesky factorisation. The equality is asserted by a test rather
        than assumed.
        """
        from twinmind.models.demand_lgbm import interpolate_quantiles

        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        rng = np.random.default_rng(seed)
        normal = rng.standard_normal((n_samples, self.horizon)) @ self.cholesky.T
        uniform = stats.norm.cdf(normal)
        paths = np.empty((n_samples, self.horizon), dtype="float64")
        for step in range(self.horizon):
            paths[:, step] = interpolate_quantiles(self.levels, self.values[step], uniform[:, step])
        return np.clip(paths, 0.0, None)


def get_sampler(
    model: DemandModel, series_id: str, origin: pd.Timestamp, horizon: int
) -> DemandSampler:
    """A memoised :class:`DemandSampler`, built at most once per process."""
    key = (id(model), str(series_id), pd.Timestamp(origin), int(horizon))
    cached = _SAMPLER_MEMO.get(key)
    if cached is None:
        cached = DemandSampler.from_model(model, series_id, origin, horizon)
        _SAMPLER_MEMO[key] = cached
    return cached


def daily_demand(
    model: DemandModel,
    series_id: str,
    origin: pd.Timestamp,
    horizon: int,
    seed: int,
    *,
    demand_scale: float,
    price_multiplier: float,
    price_elasticity: float,
    spread_multiplier: float,
) -> NDArray[np.float64]:
    """One realised demand path of length ``horizon``, in product units.

    Order of operations matters and is deliberate:

    1. Draw the path from the (calibrated) forecast.
    2. Widen or narrow it about its own median by ``spread_multiplier``. Done
       before scaling so the multiplier means the same thing whatever the scale.
    3. Scale to plant units.
    4. Apply price elasticity.

    Elasticity is applied *last and multiplicatively*, which is what makes the
    common-random-numbers test in section 4 pass: two scenarios differing only
    in ``price_multiplier`` get the same underlying path and differ by a scalar,
    so the demand randomness has not been reshuffled.
    """
    sampler = get_sampler(model, series_id, origin, horizon)
    path = sampler.sample(1, seed)[0]

    if spread_multiplier != 1.0:
        median = float(np.median(path))
        path = median + spread_multiplier * (path - median)

    scaled = np.clip(path, 0.0, None) * demand_scale
    elasticity_factor = float(price_multiplier) ** float(price_elasticity)
    return np.clip(scaled * elasticity_factor, 0.0, None)


def calibrate_demand_scale(
    model: DemandModel,
    series_id: str,
    origin: pd.Timestamp,
    horizon: int,
    *,
    nominal_capacity_per_day: float,
    target_ratio: float,
    seed: int = 0,
    n_samples: int = 200,
) -> float:
    """The ``demand_scale`` making mean demand ``target_ratio`` of daily capacity.

    Computed from the sampler's own mean over ``n_samples`` paths rather than
    from the q50, because the mean of a right-skewed demand distribution sits
    above its median and it is the mean that determines whether the line is
    saturated.
    """
    sampler = get_sampler(model, series_id, origin, horizon)
    mean_demand = float(np.mean(sampler.sample(n_samples, seed)))
    if mean_demand <= 0.0:
        raise ValueError("demand sampler produced a non-positive mean; cannot calibrate")
    scale = target_ratio * nominal_capacity_per_day / mean_demand
    logger.info(
        "demand_scale=%.4f (mean path %.1f -> %.1f units/day, %.0f%% of %.1f capacity)",
        scale,
        mean_demand,
        mean_demand * scale,
        100 * target_ratio,
        nominal_capacity_per_day,
    )
    return scale


@dataclass(frozen=True)
class CachedDemandModel:
    """The demand forecast as precomputed grids, with no Favorita frame behind it.

    WHY THIS EXISTS: a measured memory wall, not a guess.

    Each rollout worker was holding 1.33 GB, of which **0.71 GB was the
    featurised Favorita frame** (336,800 x 47) and another 0.15 GB the LightGBM
    boosters. On a 7.3 GB machine that caps the pool at about 4 workers, and an
    11-worker pool died outright. But the twin never touches that frame at run
    time: it asks the sampler for one calibrated quantile grid per
    ``(series, origin)``, and the grid cannot change, because the forecast does
    not depend on the scenario or the seed.

    So the grids are computed once, offline, and stored. A worker then carries
    about 0.47 GB instead of 1.33 GB, which is what makes a larger pool
    possible. The sampling code is untouched -- this object exposes the same two
    private methods ``DemandSampler.from_model`` already uses, so there is still
    exactly one sampling implementation and the equivalence is tested.
    """

    levels: NDArray[np.float64]
    grids: dict[tuple[str, pd.Timestamp], NDArray[np.float64]]
    correlation: NDArray[np.float64]
    series_ids: tuple[str, ...]

    def _predict_grid(
        self, series_id: str, origin: pd.Timestamp, horizon: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The stored calibrated quantile grid, sliced to ``horizon``."""
        key = (str(series_id), pd.Timestamp(origin))
        grid = self.grids.get(key)
        if grid is None:
            raise KeyError(
                f"no cached demand grid for {key}. The cache is built for a fixed set "
                "of origins; rebuild it with scripts/build_demand_cache.py if the "
                "rollout design's origins changed."
            )
        if horizon > grid.shape[0]:
            raise ValueError(f"cached grid covers {grid.shape[0]} steps, {horizon} requested")
        return self.levels, grid[:horizon]

    def _correlation_for(self, horizon: int) -> NDArray[np.float64]:
        """The copula correlation, sliced to ``horizon``."""
        if horizon > self.correlation.shape[0]:
            raise ValueError(
                f"cached copula covers {self.correlation.shape[0]} steps, {horizon} requested"
            )
        return self.correlation[:horizon, :horizon]

    def quantiles(self, series_id: str, origin: pd.Timestamp, horizon: int) -> Any:
        """Headline q10/q50/q90, so this still satisfies ``DemandModel``."""
        from twinmind.models.demand import DemandQuantiles

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
        """Delegates to the one sampler, so there is no second implementation."""
        return get_sampler(self, series_id, origin, horizon).sample(n_samples, seed)

    def save(self, path: Path) -> Path:
        """Write the cache as a single compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.grids)
        np.savez_compressed(
            path,
            levels=self.levels,
            correlation=self.correlation,
            series=np.array([k[0] for k in keys]),
            origins=np.array([str(k[1].date()) for k in keys]),
            grids=np.stack([self.grids[k] for k in keys]),
        )
        logger.info("wrote demand grid cache %s (%d grids)", path, len(keys))
        return path

    @classmethod
    def load(cls, path: Path) -> CachedDemandModel:
        """Read a cache written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as payload:
            series = [str(s) for s in payload["series"]]
            origins = [pd.Timestamp(str(o)) for o in payload["origins"]]
            grids = payload["grids"].astype("float64")
            return cls(
                levels=payload["levels"].astype("float64"),
                grids={
                    (s, o): grids[i] for i, (s, o) in enumerate(zip(series, origins, strict=True))
                },
                correlation=payload["correlation"].astype("float64"),
                series_ids=tuple(sorted(set(series))),
            )

    @classmethod
    def build(
        cls,
        model: DemandModel,
        series_id: str,
        origins: Sequence[pd.Timestamp],
        horizon: int,
    ) -> CachedDemandModel:
        """Precompute the grid for every origin the rollout design will use."""
        grid_fn = getattr(model, "_predict_grid", None)
        if not callable(grid_fn):
            raise TypeError("model does not expose a quantile grid to cache")
        levels = None
        grids: dict[tuple[str, pd.Timestamp], NDArray[np.float64]] = {}
        for origin in origins:
            stamp = pd.Timestamp(origin)
            levels, values = grid_fn(series_id, stamp, horizon)
            grids[(str(series_id), stamp)] = np.asarray(values, dtype="float64")
        if levels is None:
            raise ValueError("no origins supplied")
        return cls(
            levels=np.asarray(levels, dtype="float64"),
            grids=grids,
            correlation=model._correlation_for(horizon),  # type: ignore[attr-defined]
            series_ids=(str(series_id),),
        )
