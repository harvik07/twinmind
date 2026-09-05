"""Tidy loaders, one per dataset.

Every loader reads from ``data/raw/<dataset>/``, normalises names and dtypes, and
caches the tidy result to ``data/interim/<name>.parquet``. Subsequent calls read
the parquet. Pass ``use_cache=False`` to force a rebuild from raw.

Column contracts are documented per loader; downstream code should rely on these
names, not on whatever the upstream source happened to ship.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from twinmind.utils.config import AI4IConfig, CMAPSSConfig, DataConfig, SKABConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import interim_dir, raw_dir

logger = get_logger(__name__)

Subset = Literal["FD001", "FD002", "FD003", "FD004"]
Split = Literal["train", "test"]


class DataNotDownloadedError(FileNotFoundError):
    """Raised when a loader is called before the raw files exist."""

    def __init__(self, dataset: str, path: Path) -> None:
        super().__init__(
            f"{dataset} is not in data/raw ({path} is missing). "
            f"Run: python scripts/download_data.py --dataset {dataset}"
        )


def _data_config() -> DataConfig:
    return cast(DataConfig, load_config("data"))


def _cache_path(name: str) -> Path:
    return interim_dir() / f"{name}.parquet"


def _read_cache(name: str, use_cache: bool) -> pd.DataFrame | None:
    path = _cache_path(name)
    if use_cache and path.exists():
        logger.debug("loading %s from cache %s", name, path)
        return pd.read_parquet(path)
    return None


def _write_cache(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    path = _cache_path(name)
    frame.to_parquet(path, index=False)
    logger.debug("cached %s -> %s (%d rows)", name, path, len(frame))
    return frame


def normalise_column(name: str) -> str:
    """Lower-case snake_case, with bracketed units folded into the name.

    ``"Air temperature [K]"`` and ``"Air temperature"`` both become
    ``air_temperature_k``, so the AI4I loader is indifferent to whether the data
    came from the UCI CSV or from ``ucimlrepo``.
    """
    text = name.strip()
    text = re.sub(r"\[([^\]]*)\]", r" \1", text)  # "[K]" -> " K"
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


# --------------------------------------------------------------------------- #
# C-MAPSS
# --------------------------------------------------------------------------- #
def cmapss_columns(config: CMAPSSConfig) -> list[str]:
    """The 26 column names of a raw C-MAPSS file, in file order."""
    return ["unit", "cycle", *config.setting_columns, *config.sensor_columns]


def load_cmapss(
    subset: Subset = "FD001",
    split: Split = "train",
    config: CMAPSSConfig | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load one C-MAPSS subset/split as a tidy frame.

    The raw files are space-delimited, headerless, and carry two trailing empty
    columns from a trailing separator on every line.

    Returns columns:
        unit                int64    engine id, 1-based
        cycle               int64    operating cycle, 1-based within a unit
        op_setting_1..3     float64  operating condition descriptors
        sensor_1..21        float64  sensor measurements

    For ``split="test"`` two extra columns are attached from ``RUL_<subset>.txt``:
        rul_at_last_cycle   int64    ground-truth RUL at the unit's final cycle
        is_last_cycle       bool     marks the row the ground truth refers to

    The test trajectories are truncated some time before failure -- that is the
    point of the benchmark -- so RUL is only known for the final cycle of each
    unit, not for every row.
    """
    config = config or _data_config().cmapss
    name = f"cmapss_{subset}_{split}"
    cached = _read_cache(name, use_cache)
    if cached is not None:
        return cached

    directory = raw_dir() / "cmapss"
    path = directory / f"{split}_{subset}.txt"
    if not path.exists():
        raise DataNotDownloadedError("cmapss", path)

    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    expected = len(cmapss_columns(config))
    # Trailing whitespace on each line yields phantom all-NaN columns; drop them.
    if frame.shape[1] > expected:
        extra = frame.iloc[:, expected:]
        if not extra.isna().all().all():
            raise ValueError(f"{path} has {frame.shape[1]} columns with real data in the tail")
        frame = frame.iloc[:, :expected]
    if frame.shape[1] != expected:
        raise ValueError(f"{path}: expected {expected} columns, found {frame.shape[1]}")

    frame.columns = pd.Index(cmapss_columns(config))
    frame["unit"] = frame["unit"].astype("int64")
    frame["cycle"] = frame["cycle"].astype("int64")
    for column in frame.columns[2:]:
        frame[column] = frame[column].astype("float64")

    if split == "test":
        rul_path = directory / f"RUL_{subset}.txt"
        if not rul_path.exists():
            raise DataNotDownloadedError("cmapss", rul_path)
        rul = pd.read_csv(rul_path, header=None, names=["rul_at_last_cycle"])
        rul["unit"] = np.arange(1, len(rul) + 1, dtype="int64")
        rul["rul_at_last_cycle"] = rul["rul_at_last_cycle"].astype("int64")

        units_in_frame = frame["unit"].nunique()
        if len(rul) != units_in_frame:
            raise ValueError(
                f"{rul_path} has {len(rul)} rows but {path} has {units_in_frame} units"
            )
        frame = frame.merge(rul, on="unit", how="left", validate="many_to_one")
        last_cycle = frame.groupby("unit")["cycle"].transform("max")
        frame["is_last_cycle"] = frame["cycle"].eq(last_cycle)

    frame = frame.sort_values(["unit", "cycle"], ignore_index=True)
    return _write_cache(name, frame)


# --------------------------------------------------------------------------- #
# AI4I 2020
# --------------------------------------------------------------------------- #
#: The two upstream spellings differ: the UCI CSV carries unit suffixes
#: ("Air temperature [K]") while ucimlrepo strips them ("Air temperature").
#: normalise_column() folds a present unit into the name but cannot invent a
#: missing one, so map the unit-less form onto the canonical name explicitly.
AI4I_ALIASES = {
    "air_temperature": "air_temperature_k",
    "process_temperature": "process_temperature_k",
    "rotational_speed": "rotational_speed_rpm",
    "torque": "torque_nm",
    "tool_wear": "tool_wear_min",
}


def load_ai4i(
    config: AI4IConfig | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load AI4I 2020 as ``(features, target, failure_modes)``.

    LEAKAGE WARNING -- read before using this dataset.
        The columns TWF, HDF, PWF, OSF and RNF are the individual failure modes.
        Their logical OR reproduces the ``machine_failure`` target almost exactly
        (RNF, the random failure mode, is the only discrepancy). A classifier
        handed those five columns scores ~1.0 AUC and has learned nothing about
        the process physics -- it has read the answer.

        This function therefore returns three objects, and the mode columns are
        deliberately NOT in the feature frame. Use ``failure_modes`` only for
        analysis and plots, never as model input. ``test_ai4i_no_leakage`` in the
        test suite asserts the separation holds.

    Returns:
        features       DataFrame  the 5 process measurements plus one-hot 'type'
        target         Series     int64 0/1, named ``machine_failure``
        failure_modes  DataFrame  the 5 mode columns, for analysis only
    """
    config = config or _data_config().ai4i
    cached = _read_cache("ai4i", use_cache)
    if cached is None:
        path = raw_dir() / "ai4i" / "ai4i2020.csv"
        if not path.exists():
            raise DataNotDownloadedError("ai4i", path)
        frame = pd.read_csv(path)
        columns = [normalise_column(c) for c in frame.columns]
        frame.columns = pd.Index([AI4I_ALIASES.get(c, c) for c in columns])
        cached = _write_cache("ai4i", frame)

    frame = cached
    missing = [c for c in [*config.feature_columns, config.target_column] if c not in frame.columns]
    if missing:
        raise ValueError(f"AI4I data is missing expected columns {missing}; found {list(frame)}")

    features = frame[list(config.feature_columns)].astype("float64").copy()
    if "type" in frame.columns:
        # Product quality variant L/M/H -- ordinal, and genuinely predictive.
        dummies = pd.get_dummies(frame["type"], prefix="type", dtype="int64")
        features = pd.concat([features, dummies], axis=1)

    target = frame[config.target_column].astype("int64").rename(config.target_column)
    failure_modes = frame[list(config.failure_mode_columns)].astype("int64").copy()

    leaked = set(features.columns) & set(config.failure_mode_columns)
    if leaked:  # pragma: no cover - guarded by construction and by a test
        raise AssertionError(f"failure modes leaked into the feature frame: {sorted(leaked)}")

    return features, target, failure_modes


# --------------------------------------------------------------------------- #
# Favorita
# --------------------------------------------------------------------------- #
FAVORITA_FILES = ("train", "test", "stores", "oil", "holidays_events", "transactions")


def load_favorita(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Load the Favorita competition tables as a dict of tidy frames.

    Keys: ``train``, ``test``, ``stores``, ``oil``, ``holidays``, ``transactions``.
    Date columns are parsed to datetime64[ns]; ``family`` and the store metadata
    columns are left as strings for the feature builder to encode.
    """
    directory = raw_dir() / "favorita"
    frames: dict[str, pd.DataFrame] = {}

    for stem in FAVORITA_FILES:
        key = "holidays" if stem == "holidays_events" else stem
        cached = _read_cache(f"favorita_{key}", use_cache)
        if cached is not None:
            frames[key] = cached
            continue

        path = directory / f"{stem}.csv"
        if not path.exists():
            raise DataNotDownloadedError("favorita", path)
        frame = pd.read_csv(path)
        frame.columns = pd.Index([normalise_column(c) for c in frame.columns])
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        frames[key] = _write_cache(f"favorita_{key}", frame)

    return frames


# --------------------------------------------------------------------------- #
# SKAB (optional)
# --------------------------------------------------------------------------- #
def load_skab(
    config: SKABConfig | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load SKAB as one long frame with an ``anomaly`` label column.

    The raw CSVs are semicolon-delimited. Each file is one experiment; ``group``
    and ``experiment`` columns are added so files stay distinguishable after the
    concatenation.
    """
    config = config or _data_config().skab
    cached = _read_cache("skab", use_cache)
    if cached is not None:
        return cached

    directory = raw_dir() / "skab"
    paths = sorted(directory.rglob("*.csv"))
    if not paths:
        raise DataNotDownloadedError("skab", directory)

    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, sep=";")
        frame.columns = pd.Index([normalise_column(c) for c in frame.columns])
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame.insert(0, "experiment", path.stem)
        frame.insert(0, "group", path.parent.name)
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    if config.label_column not in combined.columns:
        raise ValueError(f"SKAB is missing its label column {config.label_column!r}")
    combined[config.label_column] = combined[config.label_column].astype("int64")
    return _write_cache("skab", combined)
