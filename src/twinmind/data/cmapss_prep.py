"""One C-MAPSS preparation path, shared by the baselines and by A1.

Both must see identical inputs or the comparison is meaningless: the same
sensors dropped, the same normaliser fitted on the same training split, the same
windows, the same targets. Putting that in one function is the only way to be
sure -- two "equivalent" pipelines drift within a week.

Everything that learns a statistic is fitted on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twinmind.data.loaders import Subset, load_cmapss
from twinmind.data.windowing import (
    OperatingConditionNormaliser,
    compute_rul,
    compute_test_rul,
    drop_constant_sensors,
    fit_operating_condition_normaliser,
    make_last_windows,
    make_windows,
    split_by_unit,
)
from twinmind.utils.config import CMAPSSConfig
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CMAPSSData:
    """Windowed, normalised C-MAPSS ready for any model.

    ``x_*`` are ``[N, window, C]`` float32; ``y_*`` are RUL in cycles; ``units_*``
    identify the engine behind each window. ``x_test`` holds exactly one window
    per test engine, which is the benchmark protocol.
    """

    subset: str
    x_train: NDArray[np.float32]
    y_train: NDArray[np.float64]
    units_train: NDArray[np.int64]
    x_val: NDArray[np.float32]
    y_val: NDArray[np.float64]
    units_val: NDArray[np.int64]
    x_test: NDArray[np.float32]
    y_test: NDArray[np.float64]
    units_test: NDArray[np.int64]
    sensors: list[str]
    dropped_sensors: list[str]
    normaliser: OperatingConditionNormaliser
    rul_cap: int

    @property
    def n_features(self) -> int:
        """Channels per timestep."""
        return int(self.x_train.shape[2])

    @property
    def window(self) -> int:
        """Timesteps per window."""
        return int(self.x_train.shape[1])

    def describe(self) -> str:
        """A one-line shape summary for logs."""
        return (
            f"{self.subset}: train {self.x_train.shape} val {self.x_val.shape} "
            f"test {self.x_test.shape} ({len(self.sensors)} sensors, "
            f"{len(self.dropped_sensors)} dropped)"
        )


def prepare_cmapss(
    subset: Subset,
    config: CMAPSSConfig,
    seed: int = 42,
    val_fraction: float | None = None,
    scaling: Literal["zscore", "minmax"] = "zscore",
) -> CMAPSSData:
    """Load, target, split, select sensors, normalise and window one subset.

    Order matters and is not arbitrary:

    1. Split by **unit** first, so nothing learned later can see a validation
       engine. Splitting after fitting the normaliser would leak its statistics.
    2. Detect constant sensors on the training split, and drop the same columns
       everywhere.
    3. Fit the normaliser on training data only. FD002 and FD004 get
       within-regime standardisation over six clusters; FD001 and FD003 a global
       z-score.
    4. Window. Training and validation use every window at stride 1; the test set
       uses one window per engine, ending at its final observed cycle.

    Test targets come from ``compute_test_rul``, which adds the ground-truth
    remaining life at truncation -- test trajectories stop before failure, so the
    training-side formula would score every engine against a fictional zero.
    """
    val_fraction = config.val_fraction if val_fraction is None else val_fraction

    train_raw = load_cmapss(subset, "train")
    test_raw = load_cmapss(subset, "test")

    train_raw = train_raw.assign(rul=compute_rul(train_raw, cap=config.rul_cap))
    test_raw = test_raw.assign(rul=compute_test_rul(test_raw, cap=config.rul_cap))

    train_df, val_df = split_by_unit(train_raw, val_fraction, seed=seed)

    kept, dropped = drop_constant_sensors(
        train_df, config.sensor_columns, config.constant_variance_threshold
    )

    n_conditions = config.n_operating_conditions if subset in config.multi_regime_subsets else 1
    normaliser = fit_operating_condition_normaliser(
        train_df, kept, config.setting_columns, n_conditions=n_conditions, seed=seed
    )

    train_z = normaliser.transform(train_df)
    val_z = normaliser.transform(val_df)
    test_z = normaliser.transform(test_raw)

    window, stride = config.window_length, config.window_stride
    x_train, y_train, units_train = make_windows(train_z, kept, window=window, stride=stride)
    x_val, y_val, units_val = make_windows(val_z, kept, window=window, stride=stride)
    # One window per test engine: the benchmark scores a single prediction each.
    x_test, y_test, units_test = make_last_windows(test_z, kept, window=window)

    if scaling == "minmax":
        # H6: rescale each sensor to [-1, 1], the most common C-MAPSS convention.
        # Fitted on the training windows ONLY, then applied unchanged to
        # validation and test -- fitting on all three would leak test statistics.
        lo = x_train.min(axis=(0, 1), keepdims=True)
        hi = x_train.max(axis=(0, 1), keepdims=True)
        span = np.maximum(hi - lo, 1e-8)

        def rescale(a: NDArray[np.float32]) -> NDArray[np.float32]:
            return np.clip(2.0 * (a - lo) / span - 1.0, -3.0, 3.0).astype("float32")

        x_train, x_val, x_test = rescale(x_train), rescale(x_val), rescale(x_test)

    data = CMAPSSData(
        subset=subset,
        x_train=x_train,
        y_train=y_train.astype("float64"),
        units_train=units_train.astype("int64"),
        x_val=x_val,
        y_val=y_val.astype("float64"),
        units_val=units_val.astype("int64"),
        x_test=x_test,
        y_test=y_test.astype("float64"),
        units_test=units_test.astype("int64"),
        sensors=kept,
        dropped_sensors=dropped,
        normaliser=normaliser,
        rul_cap=config.rul_cap,
    )
    logger.info("%s", data.describe())
    return data


# --------------------------------------------------------------------------- #
# Hand-crafted window features, for the classical baselines
# --------------------------------------------------------------------------- #
FEATURE_STATS = ("last", "mean", "std", "slope", "min", "max")


def window_features(x: NDArray[np.float32]) -> NDArray[np.float64]:
    """Flatten ``[N, T, C]`` windows into ``[N, C * 6]`` summary features.

    Per sensor: last value, mean, standard deviation, the slope of a linear fit
    over the window, min and max. The slope is what carries most of the signal --
    degradation shows up as drift, and a Ridge or forest on the raw flattened
    window cannot see a trend without it.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [N, T, C], got {x.shape}")
    n, timesteps, channels = x.shape
    values = x.astype("float64")

    t = np.arange(timesteps, dtype="float64")
    t_centred = t - t.mean()
    denominator = float((t_centred**2).sum()) or 1.0

    last = values[:, -1, :]
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    slope = np.einsum("ntc,t->nc", values - mean[:, None, :], t_centred) / denominator
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)

    stacked = np.stack([last, mean, std, slope, minimum, maximum], axis=2)
    return stacked.reshape(n, channels * len(FEATURE_STATS))


def feature_names(sensors: list[str]) -> list[str]:
    """Column names matching :func:`window_features` output order."""
    return [f"{sensor}_{stat}" for sensor in sensors for stat in FEATURE_STATS]


def failure_labels(y_rul: NDArray[np.float64], horizon: int) -> NDArray[np.int64]:
    """Binary "will fail within ``horizon`` cycles", derived from the RUL target.

    This is why A1's two heads can share a trunk: the classification label is a
    function of the regression target on the same inputs, not a separate dataset.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return (np.asarray(y_rul) <= horizon).astype("int64")


def as_frame(x: NDArray[np.float32], sensors: list[str]) -> pd.DataFrame:
    """Window features as a named DataFrame, for inspection in a notebook."""
    return pd.DataFrame(window_features(x), columns=feature_names(sensors))
