"""Loading rollout shards for the surrogate, with the leak guards built in.

Two things here exist to stop a specific, easy, invisible mistake.

**Training code must not be able to read a test split.** The test splits are the
only evidence the surrogate works, and they are a directory of parquet files
indistinguishable from the training ones. So every split's manifest records what
it is, and :func:`load_rollouts` refuses to hand a test split to a caller that
has not explicitly asked for one. A guard you have to opt out of is worth far
more than a naming convention you have to remember.

**The validation split holds out whole context blocks.** Scenarios within one
block share a context, so splitting rows at random would put 63 of a block's 64
scenarios in training and the 64th in validation — measuring interpolation
between neighbouring scenarios at a state the model has already seen, which is
not what the surrogate is for. The agent ranks scenarios at states the surrogate
has never seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twinmind.models.surrogate import KPI_ORDER
from twinmind.twin.context import STATE_FIELDS
from twinmind.twin.runner import SCENARIO_DIMS
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Column prefix for the 8 KPI targets. Deliberately explicit: three STATE
#: fields end in ``_mean`` too (``demand_q10_mean`` and friends), so selecting
#: targets by suffix alone would quietly pull context features in as labels.
KPI_PREFIX = "kpi_"


@dataclass(frozen=True)
class RolloutData:
    """One split, loaded and ready to encode."""

    frame: pd.DataFrame
    manifest: dict[str, Any]
    directory: Path

    @property
    def split(self) -> str:
        """``train``, ``test`` or ``unknown``."""
        return str(self.manifest.get("split", "unknown"))

    @property
    def n_rows(self) -> int:
        """Number of scenarios in the split."""
        return int(len(self.frame))

    @property
    def context_blocks(self) -> NDArray[np.int64]:
        """Each row's context block id."""
        return self.frame["context_block"].to_numpy(dtype="int64")

    @property
    def scenario_indices(self) -> NDArray[np.int64]:
        """Each row's global Sobol index, the stable scenario id."""
        return self.frame["scenario_index"].to_numpy(dtype="int64")

    def seeds(self) -> list[list[int]]:
        """The replication seeds behind every row, for the disjointness test."""
        return [json.loads(s) for s in self.frame["seeds"]]

    def scenario_matrix(self) -> NDArray[np.float64]:
        """``[N, 18]`` raw scenario values in canonical order."""
        return self.frame[list(SCENARIO_DIMS)].to_numpy(dtype="float64")

    def state_matrix(self) -> NDArray[np.float64]:
        """``[N, 15]`` raw context state vectors."""
        return self.frame[[f"state_{f}" for f in STATE_FIELDS]].to_numpy(dtype="float64")

    def targets(self) -> NDArray[np.float64]:
        """``[N, 8]`` KPI means, the labels."""
        return self.frame[[f"{KPI_PREFIX}{k}_mean" for k in KPI_ORDER]].to_numpy(dtype="float64")

    def target_stds(self) -> NDArray[np.float64]:
        """``[N, 8]`` per-KPI sample std ACROSS replications.

        Note this is the std of a single replication, not of the label. The
        label is a mean of ``n_reps``, so its own std is this divided by
        ``sqrt(n_reps)`` -- which is what the noise ceiling and the aleatoric
        target both need, and which is the single easiest factor to drop.
        """
        return self.frame[[f"{KPI_PREFIX}{k}_std" for k in KPI_ORDER]].to_numpy(dtype="float64")

    def label_stds(self) -> NDArray[np.float64]:
        """``[N, 8]`` std OF THE LABEL: replication std / sqrt(n_reps)."""
        reps = self.frame["n_reps"].to_numpy(dtype="float64")[:, None]
        return self.target_stds() / np.sqrt(np.maximum(reps, 1.0))


def load_rollouts(
    directory: Path | str,
    allow_test: bool = False,
    max_blocks: int | None = None,
) -> RolloutData:
    """Load every valid shard in ``directory``.

    ``allow_test`` must be set explicitly to read a split whose manifest is not
    ``train``. Training code never passes it, so a test split cannot reach a
    training loop through a mistyped path.

    ``max_blocks`` keeps only the first N context blocks, which is what makes
    the learning curve work: because the Sobol sequence is balanced in prefixes
    of 2^m, a prefix of complete blocks is itself a valid space-filling design
    rather than an arbitrary subsample.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path}; generate the split with "
            "scripts/generate_rollouts.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(manifest.get("split", "unknown"))
    if split != "train" and not allow_test:
        raise PermissionError(
            f"{directory} is a {split!r} split and this caller did not pass "
            "allow_test=True. Training must never read the test splits; if you are "
            "writing evaluation code, say so explicitly."
        )

    shards = sorted(directory.glob("block_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shards in {directory}")

    frames = []
    for shard in shards:
        frame = pd.read_parquet(shard)
        if max_blocks is not None and int(frame["context_block"].iloc[0]) >= max_blocks:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"no shards left in {directory} after max_blocks={max_blocks}")

    combined = pd.concat(frames, ignore_index=True).sort_values("scenario_index")
    combined = combined.reset_index(drop=True)
    logger.info(
        "loaded %d rows from %d shards in %s (split=%s)",
        len(combined),
        len(frames),
        directory.name,
        split,
    )
    return RolloutData(frame=combined, manifest=manifest, directory=directory)


def split_by_context_block(
    data: RolloutData, val_fraction: float = 0.2, seed: int = 42
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Row indices for ``(train, validation)``, holding out whole context blocks.

    Splitting rows at random would leave 63 of a block's 64 scenarios in
    training and one in validation, so validation error would measure
    interpolation between adjacent scenarios at an already-seen state. The
    surrogate's job is to generalise to states it has not seen.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    blocks = np.unique(data.context_blocks)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(blocks)
    n_val = max(1, int(round(len(blocks) * val_fraction)))
    val_blocks = set(shuffled[:n_val].tolist())

    mask = np.array([b in val_blocks for b in data.context_blocks])
    train_index = np.flatnonzero(~mask).astype("int64")
    val_index = np.flatnonzero(mask).astype("int64")
    overlap = set(data.context_blocks[train_index]) & set(data.context_blocks[val_index])
    if overlap:  # pragma: no cover - guarded by construction
        raise AssertionError(f"context blocks in both splits: {sorted(overlap)}")
    logger.info(
        "split %d blocks -> %d train rows, %d validation rows (%d held-out blocks)",
        len(blocks),
        train_index.size,
        val_index.size,
        len(val_blocks),
    )
    return train_index, val_index


def state_statistics(
    data: RolloutData, rows: NDArray[np.int64] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(mean, std)`` of the state features, from the TRAINING rows only.

    Passed explicitly into ``encode`` rather than computed wherever it is
    needed, because computing them over the whole file -- the convenient thing
    to do -- standardises the training features using statistics that saw the
    validation and test states.
    """
    matrix = data.state_matrix()
    if rows is not None:
        matrix = matrix[rows]
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    return mean, np.where(std < 1e-12, 1.0, std)
