"""C-MAPSS preprocessing: RUL targets, sensor selection, regime normalisation,
windowing and unit-wise splits.

Most of Model A's accuracy comes from this file rather than from the network, so
each step documents *why* it is done the way it is.

The leakage rule throughout: anything that learns a statistic (which sensors are
constant, the regime clusters, the per-regime means) is fitted on the training
split only and then applied to validation and test. The APIs are shaped so that
doing this correctly is the path of least resistance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_RUL_CAP = 125


# --------------------------------------------------------------------------- #
# (a) Targets
# --------------------------------------------------------------------------- #
def compute_rul(
    frame: pd.DataFrame,
    cap: int = DEFAULT_RUL_CAP,
    group_column: str = "unit",
    cycle_column: str = "cycle",
) -> pd.Series:
    """Piecewise-linear remaining useful life, clipped at ``cap``.

    ``RUL = min(max_cycle_of_unit - cycle, cap)``.

    Why the cap, and why piecewise-linear:
        A raw linear target claims an engine at cycle 5 of a 300-cycle life is
        "295 cycles from failure". That is not something the sensors can support:
        early in life a healthy engine looks identical whether it will run for
        200 more cycles or 300, because degradation has not started. Regressing
        on that unknowable difference forces the network to fit noise, and the
        error it makes there dominates the loss.

        The standard fix on C-MAPSS is a piecewise-linear target: hold RUL
        constant at ``R_early`` (125 cycles) until the engine is genuinely inside
        its degradation phase, then decrease linearly. The model is only asked to
        predict what the data can actually tell it. This is worth several RMSE
        points and is used by essentially every published C-MAPSS result, so
        comparisons against the literature are only meaningful with it applied.

        125 is the conventional value (Heimes 2008; Zheng et al. 2017) and is
        configurable via ``cmapss.rul_cap`` in configs/data.yaml.

    Returns an int64 Series aligned to ``frame``'s index.
    """
    if cap <= 0:
        raise ValueError(f"rul cap must be positive, got {cap}")
    max_cycle = frame.groupby(group_column)[cycle_column].transform("max")
    rul = (max_cycle - frame[cycle_column]).clip(upper=cap)
    return rul.astype("int64").rename("rul")


def compute_test_rul(
    frame: pd.DataFrame,
    cap: int = DEFAULT_RUL_CAP,
    group_column: str = "unit",
    cycle_column: str = "cycle",
    rul_column: str = "rul_at_last_cycle",
) -> pd.Series:
    """RUL for a *test* frame, where trajectories stop before failure.

    A test unit's last cycle has a known RUL from ``RUL_<subset>.txt`` rather than
    zero, so the remaining life at any earlier cycle is
    ``(max_cycle - cycle) + rul_at_last_cycle``, then capped.
    """
    if rul_column not in frame.columns:
        raise ValueError(
            f"{rul_column!r} is missing; load the test split with load_cmapss(split='test')"
        )
    max_cycle = frame.groupby(group_column)[cycle_column].transform("max")
    rul = (max_cycle - frame[cycle_column] + frame[rul_column]).clip(upper=cap)
    return rul.astype("int64").rename("rul")


# --------------------------------------------------------------------------- #
# (b) Sensor selection
# --------------------------------------------------------------------------- #
def drop_constant_sensors(
    frame: pd.DataFrame,
    columns: list[str],
    threshold: float = 1e-4,
) -> tuple[list[str], list[str]]:
    """Split ``columns`` into ``(kept, dropped)`` by training-split variance.

    Detected, not hard-coded: which sensors are flat differs by subset (FD002 and
    FD004 vary across six operating regimes, so sensors that look constant on
    FD001 carry signal there). Call this on the training split and reuse the
    returned lists for validation and test, so the two are preprocessed
    identically.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"columns not in frame: {missing}")

    variances = frame[columns].var(numeric_only=True)
    kept = [c for c in columns if float(variances.get(c, 0.0)) > threshold]
    dropped = [c for c in columns if c not in kept]
    if dropped:
        logger.info(
            "dropping %d near-constant sensors (var <= %g): %s",
            len(dropped),
            threshold,
            ", ".join(dropped),
        )
    return kept, dropped


# --------------------------------------------------------------------------- #
# (c) Operating-condition normalisation
# --------------------------------------------------------------------------- #
@dataclass
class OperatingConditionNormaliser:
    """Z-scores each sensor, optionally within its operating regime.

    FD002 and FD004 are flown across six discrete operating conditions. A sensor's
    raw level is dominated by which regime the engine is in, not by how degraded
    it is, so global z-scoring leaves the degradation signal buried. Clustering
    the three op-setting columns (k=6) and standardising *within* cluster removes
    the regime effect and exposes the trend.

    FD001 and FD003 are single-regime, so ``n_conditions=1`` falls back to a plain
    global z-score.

    Fit on training data only::

        norm = OperatingConditionNormaliser(n_conditions=6).fit(train, sensors, settings)
        train_z = norm.transform(train)
        test_z = norm.transform(test)          # reuses the training statistics

    Calling ``transform`` before ``fit`` raises, and ``fit`` is the only method
    that ever looks at the data's statistics -- there is no ``fit_transform`` on
    a test frame to reach for by accident.
    """

    n_conditions: int = 1
    seed: int = 42
    sensor_columns: list[str] = field(default_factory=list)
    setting_columns: list[str] = field(default_factory=list)
    kmeans_: KMeans | None = None
    means_: dict[int, pd.Series] = field(default_factory=dict)
    stds_: dict[int, pd.Series] = field(default_factory=dict)
    fitted_: bool = False

    def _assign(self, frame: pd.DataFrame) -> np.ndarray:
        """Regime label per row (all zeros when single-regime)."""
        if self.n_conditions <= 1 or self.kmeans_ is None:
            return np.zeros(len(frame), dtype=int)
        return np.asarray(self.kmeans_.predict(frame[self.setting_columns].to_numpy()), dtype=int)

    def fit(
        self,
        frame: pd.DataFrame,
        sensor_columns: list[str] | None = None,
        setting_columns: list[str] | None = None,
    ) -> OperatingConditionNormaliser:
        """Learn regime clusters and per-regime sensor statistics. Training data only."""
        if sensor_columns is not None:
            self.sensor_columns = list(sensor_columns)
        if setting_columns is not None:
            self.setting_columns = list(setting_columns)
        if not self.sensor_columns:
            raise ValueError("sensor_columns must be provided to fit()")

        if self.n_conditions > 1:
            if not self.setting_columns:
                raise ValueError("setting_columns are required when n_conditions > 1")
            self.kmeans_ = KMeans(
                n_clusters=self.n_conditions, random_state=self.seed, n_init=10
            ).fit(frame[self.setting_columns].to_numpy())

        labels = self._assign(frame)
        self.means_, self.stds_ = {}, {}
        for label in np.unique(labels):
            block = frame.loc[labels == label, self.sensor_columns]
            self.means_[int(label)] = block.mean()
            # A sensor that is constant *within* a regime would divide by zero.
            self.stds_[int(label)] = block.std().replace(0.0, 1.0).fillna(1.0)

        self.fitted_ = True
        logger.info(
            "normaliser fitted: %d regime(s), %d sensors, %d rows",
            max(len(self.means_), 1),
            len(self.sensor_columns),
            len(frame),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted statistics. Never reads statistics from ``frame``."""
        if not self.fitted_:
            raise RuntimeError("normaliser is not fitted; call fit() on the training split first")

        out = frame.copy()
        labels = self._assign(frame)
        for label in np.unique(labels):
            key = int(label)
            # A regime unseen in training falls back to the first fitted regime
            # rather than computing fresh statistics from this frame.
            mean = self.means_.get(key, next(iter(self.means_.values())))
            std = self.stds_.get(key, next(iter(self.stds_.values())))
            mask = labels == label
            out.loc[mask, self.sensor_columns] = (
                (frame.loc[mask, self.sensor_columns] - mean) / std
            ).to_numpy()
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialisable state, so the transform can be saved beside a checkpoint."""
        if not self.fitted_:
            raise RuntimeError("cannot serialise an unfitted normaliser")
        return {
            "n_conditions": self.n_conditions,
            "seed": self.seed,
            "sensor_columns": self.sensor_columns,
            "setting_columns": self.setting_columns,
            "cluster_centers": (
                self.kmeans_.cluster_centers_.tolist() if self.kmeans_ is not None else None
            ),
            "means": {k: v.to_dict() for k, v in self.means_.items()},
            "stds": {k: v.to_dict() for k, v in self.stds_.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> OperatingConditionNormaliser:
        """Rebuild a normaliser saved by :meth:`to_dict`."""
        norm = cls(
            n_conditions=int(state["n_conditions"]),
            seed=int(state["seed"]),
            sensor_columns=list(state["sensor_columns"]),
            setting_columns=list(state["setting_columns"]),
        )
        centers = state.get("cluster_centers")
        if centers is not None:
            kmeans = KMeans(n_clusters=norm.n_conditions, random_state=norm.seed, n_init=10)
            kmeans.cluster_centers_ = np.asarray(centers, dtype=float)
            kmeans._n_threads = 1  # noqa: SLF001 - required by sklearn's predict path
            norm.kmeans_ = kmeans
        norm.means_ = {int(k): pd.Series(v) for k, v in state["means"].items()}
        norm.stds_ = {int(k): pd.Series(v) for k, v in state["stds"].items()}
        norm.fitted_ = True
        return norm


def fit_operating_condition_normaliser(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    setting_columns: list[str],
    n_conditions: int = 1,
    seed: int = 42,
) -> OperatingConditionNormaliser:
    """Fit a normaliser on the *training* split.

    Pass ``n_conditions=6`` for FD002/FD004 and ``1`` for FD001/FD003.
    """
    return OperatingConditionNormaliser(n_conditions=n_conditions, seed=seed).fit(
        frame, sensor_columns, setting_columns
    )


# --------------------------------------------------------------------------- #
# (d) Windowing
# --------------------------------------------------------------------------- #
def make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window: int = 30,
    stride: int = 1,
    target_column: str = "rul",
    group_column: str = "unit",
    cycle_column: str = "cycle",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice each unit's trajectory into fixed-length windows.

    Returns ``(X, y, unit_ids)`` with shapes ``[N, window, C]``, ``[N]``, ``[N]``.
    Each window's label is the target at its *last* cycle, which is the value the
    model must predict from the preceding ``window`` cycles.

    Short units: a trajectory with fewer than ``window`` cycles is left-padded by
    repeating its first observation, so it yields exactly one window instead of
    being silently dropped. Dropping them would discard the fastest-failing
    engines -- precisely the ones a maintenance model exists to catch. Padding
    with the first observation (rather than zeros) keeps the padded region on the
    sensor manifold, so it reads as "the engine sat at its initial condition".
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"feature columns not in frame: {missing}")
    if target_column not in frame.columns:
        raise ValueError(f"target column {target_column!r} not in frame")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    units: list[Any] = []

    ordered = frame.sort_values([group_column, cycle_column])
    for unit, block in ordered.groupby(group_column, sort=True):
        values = block[feature_columns].to_numpy(dtype="float32")
        labels = block[target_column].to_numpy()
        n = len(block)

        if n < window:
            pad = np.repeat(values[:1], window - n, axis=0)
            windows.append(np.vstack([pad, values]))
            targets.append(labels[-1])
            units.append(unit)
            continue

        for end in range(window, n + 1, stride):
            windows.append(values[end - window : end])
            targets.append(labels[end - 1])
            units.append(unit)

    if not windows:
        empty = np.empty((0, window, len(feature_columns)), dtype="float32")
        return empty, np.empty(0), np.empty(0)

    x = np.stack(windows).astype("float32")
    y = np.asarray(targets)
    unit_ids = np.asarray(units)
    logger.info("built %d windows of shape (%d, %d)", len(x), window, len(feature_columns))
    return x, y, unit_ids


def make_last_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window: int = 30,
    target_column: str = "rul",
    group_column: str = "unit",
    cycle_column: str = "cycle",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One window per unit, ending at its final cycle.

    This is the C-MAPSS test protocol: the benchmark scores a single prediction
    per test engine, made at the point where its trajectory was truncated.
    """
    ordered = frame.sort_values([group_column, cycle_column])
    tails = ordered.groupby(group_column, sort=True).tail(window)
    return make_windows(
        tails,
        feature_columns,
        window=window,
        stride=max(window, 1),
        target_column=target_column,
        group_column=group_column,
        cycle_column=cycle_column,
    )


# --------------------------------------------------------------------------- #
# (e) Splitting
# --------------------------------------------------------------------------- #
def split_by_unit(
    frame: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
    group_column: str = "unit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on unit id, never within a trajectory.

    Splitting rows at random would put cycle 100 of an engine in train and cycle
    101 in validation. Those two rows are nearly identical, so validation error
    would measure interpolation between adjacent cycles rather than the thing
    that matters: generalisation to an engine never seen before.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    units = np.sort(frame[group_column].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(len(units) * val_fraction)))
    val_units = set(shuffled[:n_val].tolist())

    val_mask = frame[group_column].isin(val_units)
    train = frame.loc[~val_mask].reset_index(drop=True)
    val = frame.loc[val_mask].reset_index(drop=True)

    overlap = set(train[group_column]) & set(val[group_column])
    if overlap:  # pragma: no cover - guarded by construction and by a test
        raise AssertionError(f"units appear in both splits: {sorted(overlap)}")

    logger.info(
        "split %d units -> %d train / %d val (%d / %d rows)",
        len(units),
        len(units) - n_val,
        n_val,
        len(train),
        len(val),
    )
    return train, val
