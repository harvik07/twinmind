"""The rollout design: Sobol scenarios, seeds, shards and the manifest.

Library code rather than script code, because the tests need it and so does any
later tool that reads a rollout set. ``scripts/generate_rollouts.py`` is the
orchestration on top: process pool, progress, CLI.

The three ideas worth knowing before reading:

* **Sobol prefixes are balanced.** Block ``k`` consumes points
  ``[per_block * k, per_block * (k + 1))`` of one scrambled sequence, so any
  prefix of complete blocks is itself a valid space-filling design. That is what
  makes a multi-hour job stoppable and the learning curve trainable from nested
  prefixes rather than five separate samples.
* **Replication seeds are hashed pairs, not consecutive integers.**
  ``SeedSequence((root, scenario_index))`` gives neighbouring scenarios
  unrelated streams; ``root + index`` gives them correlated ones.
* **The manifest is what makes a resume safe.** It records the twin's config
  hash, the artifact hashes and every declared assumption, so shards generated
  under a different ``demand_capacity_ratio`` -- a number worth 45% of profit --
  cannot be silently concatenated with these.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from twinmind.models.surrogate import KPI_ORDER
from twinmind.twin.context import STATE_FIELDS
from twinmind.twin.runner import SCENARIO_DIMS, Scenario
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:  # pragma: no cover
    import argparse

logger = get_logger(__name__)

#: Where shards live. Gitignored: tens of thousands of rows of derived data that
#: regenerates from the manifest.
ROLLOUT_DIR = PROJECT_ROOT / "data" / "processed" / "rollouts"

#: Training contexts take origins strictly before TRAIN_ORIGIN_END; test
#: contexts take later ones. 2017-06-20 is fold 1's boundary in the demand
#: splits, which leaves training a full history and test two clear months.
TRAIN_ORIGIN_END = pd.Timestamp("2017-06-20")
TEST_ORIGIN_START = pd.Timestamp("2017-06-21")
LAST_ORIGIN = pd.Timestamp("2017-08-15")
FIRST_TRAIN_ORIGIN = pd.Timestamp("2016-01-01")

#: Entropy roots. Train and test differ in the FIRST element, so every derived
#: seed differs; a test asserts the realised seed sets are disjoint rather than
#: trusting that.
TRAIN_SEED_ROOT = 20_250_101
TEST_SEED_ROOT = 77_770_001

#: Sobol scramble seeds. The test splits use a different scramble so their
#: points are not a subset of the training sequence.
TRAIN_SCRAMBLE = 12_345
TEST_SCRAMBLE = 98_765


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def _git_sha() -> str:
    """Current commit, or ``unknown`` outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _file_digest(path: Path) -> str:
    """Short sha256 of a file's contents, or ``missing``."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_manifest(resources: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Everything that, if changed, invalidates existing shards.

    The assumption values are in here in full. A rollout set generated under
    ``demand_capacity_ratio = 0.85`` describes a different twin from one
    generated at 0.64 — the sensitivity study puts 45% of profit on that single
    number — and silently mixing the two would produce a surrogate trained on
    two different simulators.
    """
    config = resources.config
    replay = config.health_replay
    return {
        "git_sha": _git_sha(),
        "created": pd.Timestamp.utcnow().isoformat(),
        "twin_config_hash": hashlib.sha256(
            json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "health_checkpoint_hash": _file_digest(PROJECT_ROOT / replay.checkpoint),
        "health_cache_hash": _file_digest(
            next((PROJECT_ROOT / replay.cache_path).glob("*.npz"), Path("missing"))
        ),
        "demand_config_hash": _file_digest(PROJECT_ROOT / "models" / "demand" / "config.json"),
        "demand_scale": float(resources.demand_scale),
        "series_id": resources.series_id,
        "assumptions": {k: float(v.value) for k, v in config.assumptions.items()},
        "hazard_source": replay.hazard_source,
        "warning_threshold": replay.warning_threshold,
        "lot_size": config.flow.lot_size,
        "scenario_dims": list(SCENARIO_DIMS),
        "kpi_order": list(KPI_ORDER),
        "state_fields": list(STATE_FIELDS),
        "n_reps": args.n_reps,
        "scenarios_per_context": args.scenarios_per_context,
        "scramble_seed": TRAIN_SCRAMBLE,
        "seed_root": TRAIN_SEED_ROOT,
        "train_origin_window": [str(FIRST_TRAIN_ORIGIN.date()), str(TRAIN_ORIGIN_END.date())],
    }


#: Manifest keys that must match for a resume to be safe. ``created`` and
#: ``git_sha`` deliberately are not: re-running after an unrelated commit is
#: fine, re-running after the twin's assumptions moved is not.
INVALIDATING_KEYS: tuple[str, ...] = (
    "twin_config_hash",
    "health_checkpoint_hash",
    "health_cache_hash",
    "demand_config_hash",
    "demand_scale",
    "assumptions",
    "hazard_source",
    "warning_threshold",
    "lot_size",
    "scenario_dims",
    "kpi_order",
    "n_reps",
    "scenarios_per_context",
    "scramble_seed",
    "seed_root",
)


def manifest_conflicts(existing: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Keys on which a stored manifest disagrees with the current one."""
    return [k for k in INVALIDATING_KEYS if existing.get(k) != current.get(k)]


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SobolDesign:
    """A scrambled Sobol sequence over the 18 scenario dimensions."""

    points: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    scramble_seed: int
    integer_dims: tuple[str, ...] = field(default=("shift_count", "supplier_choice"))

    @classmethod
    def build(
        cls, n_points: int, scramble_seed: int, bounds: list[tuple[float, float]]
    ) -> SobolDesign:
        """``n_points`` points, drawn as a power-of-two block then trimmed.

        ``random_base2`` rather than ``random``: Sobol's balance guarantees hold
        for prefixes of 2^m, and scipy warns when that is broken. Trimming after
        the fact keeps the prefix property that makes the job resumable.
        """
        sampler = stats.qmc.Sobol(d=len(bounds), scramble=True, seed=scramble_seed)
        m = max(int(np.ceil(np.log2(max(n_points, 2)))), 1)
        points = sampler.random_base2(m=m)[:n_points]
        return cls(
            points=points,
            lows=np.array([b[0] for b in bounds], dtype="float64"),
            highs=np.array([b[1] for b in bounds], dtype="float64"),
            scramble_seed=scramble_seed,
        )

    def scenario(self, index: int) -> Scenario:
        """The scenario at a global Sobol index."""
        scaled = self.lows + self.points[index] * (self.highs - self.lows)
        payload: dict[str, Any] = dict(zip(SCENARIO_DIMS, scaled.tolist(), strict=True))
        for dim in self.integer_dims:
            payload[dim] = int(round(payload[dim]))
        return Scenario(**payload)

    def __len__(self) -> int:
        """Number of points available."""
        return int(self.points.shape[0])


def ood_design(
    n_scenarios: int, scramble_seed: int, bounds: list[tuple[float, float]]
) -> SobolDesign:
    """A design where at least 3 dimensions sit at a bound in every scenario.

    Sobol points with a random subset of at least three coordinates snapped to 0
    or 1 before scaling. Extrapolation is the regime where a surrogate's
    confidence is least trustworthy, so the OOD split has to be genuinely
    outside the training density rather than merely unlikely under it -- an
    "unlikely" interior point still sits inside the convex hull of the training
    data and tells you nothing about extrapolation.
    """
    design = SobolDesign.build(n_scenarios, scramble_seed, bounds)
    rng = np.random.default_rng(scramble_seed)
    points = design.points.copy()
    n_dims = points.shape[1]
    for row in range(points.shape[0]):
        k = int(rng.integers(3, min(6, n_dims) + 1))
        chosen = rng.choice(n_dims, size=k, replace=False)
        points[row, chosen] = rng.integers(0, 2, size=k).astype("float64")
    return SobolDesign(
        points=points,
        lows=design.lows,
        highs=design.highs,
        scramble_seed=scramble_seed,
    )


def origin_for_block(block: int, n_blocks: int, test: bool = False) -> pd.Timestamp:
    """A forecast origin for one context block, spread over the allowed window.

    Spread deterministically rather than sampled, so the set of origins is
    reproducible and evenly covers the window. The chronological split is the
    whole point: a training context can never share an origin with a test one.
    """
    start, end = (
        (TEST_ORIGIN_START, LAST_ORIGIN) if test else (FIRST_TRAIN_ORIGIN, TRAIN_ORIGIN_END)
    )
    span = (end - start).days
    offset = 0 if n_blocks <= 1 else int(round(block * span / (n_blocks - 1)))
    return start + pd.Timedelta(days=offset)


def rep_seeds(seed_root: int, scenario_index: int, n_reps: int) -> list[int]:
    """Independent replication seeds for one scenario.

    ``SeedSequence((root, index))`` hashes the pair, so neighbouring scenarios
    get unrelated streams -- unlike ``root + index``, which produces correlated
    starting states and would make adjacent scenarios share luck.
    """
    children = np.random.SeedSequence((seed_root, scenario_index)).spawn(n_reps)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_shard(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write one shard atomically.

    Temp file then ``os.replace``, which is atomic on Windows and POSIX alike.
    A multi-hour job interrupted mid-write would otherwise leave a truncated
    parquet that the resume logic counts as done and silently trains on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)
    return path


def shard_path(directory: Path, block: int) -> Path:
    """Deterministic shard filename for a context block."""
    return directory / f"block_{block:05d}.parquet"


def valid_shard(path: Path, expected_rows: int) -> bool:
    """True when a shard exists, reads back and has the rows it should.

    Checking the row count rather than existence alone is what makes resume
    safe against a shard that was being written when the job died.
    """
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["scenario_index"])
    except Exception:  # noqa: BLE001 - a corrupt shard is simply not valid
        return False
    return len(frame) == expected_rows
