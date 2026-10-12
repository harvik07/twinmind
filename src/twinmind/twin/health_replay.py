"""Replayed C-MAPSS degradation, and the health model as the plant's monitor.

THE CIRCULARITY THIS AVOIDS.

The obvious way to couple a health model to a simulator is to ask it for a
failure probability and then sample a failure from it. That is circular. The
model's belief would *be* the ground truth, so a maintenance policy driven by
the model's warnings would be evaluated against a world the model defined, and
early warning would look perfect by construction.

So: **failures are REPLAYED from real run-to-failure trajectories.** Each
machine carries a C-MAPSS engine id and an age in cycles. It fails when its age
reaches that engine's actual failure cycle -- a number from the dataset, not
from the model. The health model plays the role the real article plays in a
plant: it sees the sensor window at the machine's current age and issues a
warning. It can be early, late or wrong, and the twin records which, because
ground truth is independent of it.

THE POOL MUST BE OUT-OF-SAMPLE, AND THAT IS HARDER HERE THAN IT LOOKS.

Warnings from engines the network trained on are in-sample, and in-sample
warnings are optimistically good. That would bias every scenario toward
predictive maintenance -- the twin would report that early warning pays, because
the warnings it saw were cheating.

The repo makes this awkward in a way worth recording. ``prepare_cmapss`` is
called as ``prepare_cmapss(subset, config, seed=seed)`` and passes that same
seed to ``split_by_unit``, so **the train/validation split is a function of the
training seed**. The three A1 seeds therefore held out three *different* engine
sets, and the intersection is nearly empty: on FD002 only 3 of 260 engines were
held out by all three seeds, on FD001 and FD003 none at all.

There is consequently no pool that is out-of-sample for the three-seed
ensemble. The twin therefore drives its monitoring from a **single A1 model**
(FD002, seed 42) and replays that model's own 52 held-out engines. A real plant
deploys one model, so this is not much of a concession, but it does mean the
twin's warnings come from a single network rather than the ensemble the model
card reports, and single-seed FD002 RMSE is 13.78 against the ensemble's 14.36
-- see ``reports/model_cards/twin_design.md``.

A further honest caveat, which the spec asked to be disclosed: these engines
were used for **early stopping**. No gradient step was taken on them, but the
checkpoint was *selected* by its loss on them, so they are not perfectly naive.
That is the cleanest held-out set that exists without retraining.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Cached per process, keyed by resolved path. Built lazily on first use and
#: never mutated afterwards, which is what makes it safe to touch from worker
#: processes started under the "spawn" method: each worker imports this module
#: fresh, finds an empty dict, and loads its own read-only copy.
_CACHE_MEMO: dict[str, HealthCache] = {}


@dataclass(frozen=True)
class PoolSpec:
    """Identifies a replay pool exactly enough to key a cache on it."""

    subset: str
    model_seed: int
    val_fraction: float
    window: int
    checkpoint: str

    def key(self) -> str:
        """A short stable hash of everything that changes the cached numbers.

        Includes the checkpoint's *contents*, not just its path, so retraining
        the model under the same filename produces a different key rather than
        silently reusing stale predictions.
        """
        path = Path(self.checkpoint)
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "subset": self.subset,
                    "model_seed": self.model_seed,
                    "val_fraction": self.val_fraction,
                    "window": self.window,
                },
                sort_keys=True,
            ).encode()
        )
        if path.exists():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()[:16]


def pool_engine_ids(
    subset: str, model_seed: int, val_fraction: float
) -> tuple[list[int], list[int]]:
    """``(train_ids, pool_ids)`` for one subset under one training seed.

    The pool is the validation split, reproduced by calling the same
    ``split_by_unit`` with the same seed the training run used. Reproducing the
    split rather than recording it means the two cannot drift apart.
    """
    from twinmind.data.loaders import load_cmapss
    from twinmind.data.windowing import split_by_unit

    # load_cmapss narrows the subset to a Literal; the config carries it as a
    # plain string, and an invalid value fails inside the loader with a clearer
    # message than a cast would give here.
    raw = load_cmapss(cast("Any", subset), "train")
    train, validation = split_by_unit(raw, val_fraction, seed=model_seed)
    train_ids = sorted(int(u) for u in train["unit"].unique())
    pool_ids = sorted(int(u) for u in validation["unit"].unique())
    overlap = set(train_ids) & set(pool_ids)
    if overlap:  # pragma: no cover - split_by_unit already asserts this
        raise AssertionError(f"replay pool overlaps the training engines: {sorted(overlap)}")
    return train_ids, pool_ids


@dataclass(frozen=True)
class HealthCache:
    """Precomputed ``RULPrediction`` arrays for every pool engine and cycle.

    Stored flat with an offset index rather than as a dict of ragged arrays:
    engines have different lifetimes, and a single contiguous float array with
    ``offsets[i]:offsets[i+1]`` slices is both smaller on disk and much faster
    to look up than a dict lookup plus a bounds check per simulated hour. The
    twin does this lookup tens of thousands of times per run, so it is on the
    hot path.
    """

    engine_ids: NDArray[np.int64]
    failure_cycles: NDArray[np.int64]
    offsets: NDArray[np.int64]
    rul_mean: NDArray[np.float64]
    rul_std: NDArray[np.float64]
    p_fail: NDArray[np.float64]
    horizon_h: int
    window: int
    key: str
    train_engine_ids: NDArray[np.int64]

    @property
    def n_engines(self) -> int:
        """Number of engines in the pool."""
        return int(self.engine_ids.size)

    def lifetimes(self) -> NDArray[np.int64]:
        """Each pool engine's true failure cycle."""
        return self.failure_cycles

    def index_of(self, engine_id: int) -> int:
        """Position of an engine id in the pool, raising if absent."""
        found = np.searchsorted(self.engine_ids, engine_id)
        if found >= self.engine_ids.size or self.engine_ids[found] != engine_id:
            raise KeyError(f"engine {engine_id} is not in the replay pool")
        return int(found)

    def predictions_at(self, index: int, age_cycles: int) -> tuple[float, float, float]:
        """``(rul_mean, rul_std, p_fail)`` for pool slot ``index`` at an age.

        ``age_cycles`` is clamped into the engine's observed range. Below the
        window length there is no sensor window to predict from, so the first
        available prediction is reused; past the failure cycle the last one is.
        Both ends are degenerate by construction rather than by accident, which
        is why they clamp rather than raise.
        """
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        n = stop - start
        if n <= 0:  # pragma: no cover - build_health_cache never emits empty engines
            raise ValueError(f"pool slot {index} has no cached predictions")
        slot = start + min(max(age_cycles - self.window, 0), n - 1)
        return (
            float(self.rul_mean[slot]),
            float(self.rul_std[slot]),
            float(self.p_fail[slot]),
        )

    def save(self, path: Path) -> Path:
        """Write the cache to ``path`` as a single compressed ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            engine_ids=self.engine_ids,
            failure_cycles=self.failure_cycles,
            offsets=self.offsets,
            rul_mean=self.rul_mean,
            rul_std=self.rul_std,
            p_fail=self.p_fail,
            horizon_h=np.int64(self.horizon_h),
            window=np.int64(self.window),
            key=np.array(self.key),
            train_engine_ids=self.train_engine_ids,
        )
        logger.info("wrote health cache %s (%d engines)", path, self.n_engines)
        return path

    @classmethod
    def load(cls, path: Path) -> HealthCache:
        """Read a cache written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(
                engine_ids=payload["engine_ids"].astype("int64"),
                failure_cycles=payload["failure_cycles"].astype("int64"),
                offsets=payload["offsets"].astype("int64"),
                rul_mean=payload["rul_mean"].astype("float64"),
                rul_std=payload["rul_std"].astype("float64"),
                p_fail=payload["p_fail"].astype("float64"),
                horizon_h=int(payload["horizon_h"]),
                window=int(payload["window"]),
                key=str(payload["key"]),
                train_engine_ids=payload["train_engine_ids"].astype("int64"),
            )


def cache_file(cache_dir: Path | str, spec: PoolSpec) -> Path:
    """Where the cache for one pool spec lives.

    The key is in the filename, so a changed checkpoint, split or window writes
    a new file instead of overwriting one that other results depend on. This is
    the same structural-keying lesson the A1 checkpoints learned the hard way.
    """
    return Path(cache_dir) / f"health_{spec.subset}_seed{spec.model_seed}_{spec.key()}.npz"


def load_cached(cache_dir: Path | str, spec: PoolSpec) -> HealthCache:
    """Load the cache for ``spec``, memoised once per process.

    The memo is a read-only module-level dict. Under multiprocessing "spawn"
    each worker re-imports this module and loads independently, which is
    correct and costs one file read per worker rather than a pickle of the whole
    array set through the process boundary.
    """
    path = cache_file(cache_dir, spec)
    resolved = str(path.resolve()) if path.exists() else str(path)
    cached = _CACHE_MEMO.get(resolved)
    if cached is not None:
        return cached
    if not path.exists():
        raise FileNotFoundError(
            f"no health cache at {path}. Build it with:\n"
            f"    python scripts/build_health_cache.py --subset {spec.subset} "
            f"--model-seed {spec.model_seed}"
        )
    loaded = HealthCache.load(path)
    _CACHE_MEMO[resolved] = loaded
    return loaded


# --------------------------------------------------------------------------- #
# fixed_mtbf mode
# --------------------------------------------------------------------------- #
def fit_weibull(lifetimes: NDArray[np.floating]) -> tuple[float, float]:
    """Fit a two-parameter Weibull to observed lifetimes by least squares.

    On the linearised CDF: ``log(-log(1 - F)) = k log(t) - k log(lambda)``, with
    ``F`` from median ranks. This is the standard Weibull-plot estimator. It is
    used instead of a maximum-likelihood fit because the pool is fully observed
    (every engine ran to failure, so there is no censoring for MLE to earn its
    keep) and because a closed-form fit keeps ``fixed_mtbf`` mode free of a
    scipy optimisation on the hot path.
    """
    values = np.sort(np.asarray(lifetimes, dtype="float64"))
    values = values[values > 0]
    if values.size < 2:
        raise ValueError("need at least two positive lifetimes to fit a Weibull")
    n = values.size

    # A pool whose engines all share one lifetime has no spread for the
    # regression to fit, and polyfit on a constant x raises rather than
    # returning something useless. That is not a pathological input -- a
    # synthetic pool built for a test looks exactly like it -- so it is handled
    # explicitly: a degenerate Weibull is one with a very large shape, which is
    # a near-deterministic lifetime at the observed value.
    if np.ptp(values) < 1e-9:
        return 50.0, float(values[0])

    # Median-rank (Bernard) plotting positions: less biased for small n than i/n.
    ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    x = np.log(values)
    y = np.log(-np.log1p(-ranks))
    shape, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(shape) or shape <= 0:
        return 50.0, float(np.mean(values))
    scale = float(np.exp(-intercept / shape))
    return float(shape), scale


def weibull_lifetimes(
    rng: np.random.Generator, shape: float, scale: float, size: int
) -> NDArray[np.float64]:
    """Draw ``size`` Weibull lifetimes in cycles."""
    return np.asarray(scale * rng.weibull(shape, size=size), dtype="float64")


@dataclass(frozen=True)
class HealthLookup:
    """What a :class:`~twinmind.twin.machines.Station` is allowed to ask.

    A thin façade over :class:`HealthCache` with exactly three methods, for two
    reasons. It keeps the station free of any knowledge of offsets and slot
    indices, and it is the seam the ``fixed_mtbf`` mode substitutes at: in that
    mode ``predictions`` is never called, because there is no monitoring system.

    ``weibull`` is fitted once here rather than per draw -- the fit is a
    least-squares solve and the stations draw a new lifetime on every renewal.
    """

    cache: HealthCache
    hazard_source: str
    weibull_shape: float
    weibull_scale: float

    @classmethod
    def from_cache(cls, cache: HealthCache, hazard_source: str) -> HealthLookup:
        """Build a lookup, fitting the Weibull that ``fixed_mtbf`` mode needs."""
        shape, scale = fit_weibull(cache.lifetimes().astype("float64"))
        return cls(
            cache=cache,
            hazard_source=hazard_source,
            weibull_shape=shape,
            weibull_scale=scale,
        )

    def draw_engine(self, rng: np.random.Generator) -> tuple[int, int, int]:
        """``(slot, engine_id, failure_cycle)`` for a fresh machine.

        Uniform over the pool. Weighting by lifetime would be the "length-biased
        sampling" correction appropriate if we were observing machines at a
        random moment in their lives, but a renewal installs a *new* unit, so
        uniform is right.
        """
        slot = int(rng.integers(0, self.cache.n_engines))
        return slot, int(self.cache.engine_ids[slot]), int(self.cache.failure_cycles[slot])

    def draw_lifetime(self, rng: np.random.Generator) -> float:
        """A lifetime in cycles from the fitted Weibull, for ``fixed_mtbf`` mode."""
        return float(weibull_lifetimes(rng, self.weibull_shape, self.weibull_scale, 1)[0])

    def predictions(self, slot: int, age_cycles: int) -> tuple[float, float, float]:
        """The monitoring system's current belief for one machine."""
        return self.cache.predictions_at(slot, age_cycles)
