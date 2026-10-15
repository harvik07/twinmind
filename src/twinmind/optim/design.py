"""The benchmark's run design: which held-out (goal, method, seed) cells count.

Amendment 2 reduced the benchmark to 8 of the 12 held-out goals, one
``best_known`` seed, and seeds 1 and 2 for every other method. The selection
lives in ``configs/benchmark_design.yaml``, NOT in the goal specs, so the goal
library and its hash are untouched and every existing cell stays valid.

Two things read the design, and both go through this module so they cannot
disagree:

* the runner, which only ever STARTS design cells on the held-out split; and
* the analysis, which only ever READS design cells (:func:`load_design_cells`).

A cell on disk that is outside the design -- a seed-3 cell from before the
amendment, or anything from a reserve goal -- is skipped by both, and the
loader reports how many it skipped so the exclusion is visible rather than
silent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from twinmind.optim.contracts import SpecError
from twinmind.utils.config import BenchmarkDesignConfig, load_raw
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

DESIGN_PATH = PROJECT_ROOT / "configs" / "benchmark_design.yaml"


@dataclass(frozen=True)
class BenchmarkDesign:
    """The core goals, the reserve, and the seed rule per method."""

    core_goals: tuple[str, ...]
    reserve_goals: tuple[str, ...]
    seeds: dict[str, tuple[int, ...]]
    amendment: int
    dated: str
    raw: dict[str, Any] = field(repr=False, compare=False, default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> BenchmarkDesign:
        """Parse and validate a design file."""
        path = Path(path) if path is not None else DESIGN_PATH
        # Schema first (unknown or missing keys, wrong types), through the same
        # pydantic model every other config file has.
        config = BenchmarkDesignConfig.model_validate(load_raw(path))
        seeds = {str(k): tuple(v) for k, v in config.seeds.items()}
        if "default" not in seeds:
            raise SpecError(f"{path}: seeds needs a 'default' entry")
        design = cls(
            core_goals=tuple(config.core_goals),
            reserve_goals=tuple(config.reserve_goals),
            seeds=seeds,
            amendment=config.amendment,
            dated=config.dated,
            raw=config.model_dump(),
        )
        design._check()
        return design

    def _check(self) -> None:
        overlap = set(self.core_goals) & set(self.reserve_goals)
        if overlap:
            raise SpecError(f"goals both core and reserve: {sorted(overlap)}")
        for name, values in (("core", self.core_goals), ("reserve", self.reserve_goals)):
            if len(set(values)) != len(values):
                raise SpecError(f"duplicate {name} goal")
        for method, seeds in self.seeds.items():
            if not seeds or len(set(seeds)) != len(seeds):
                raise SpecError(f"seeds for {method!r} must be non-empty and distinct")

    def check_against(self, held_out_goals: Iterable[str], methods: Sequence[str]) -> None:
        """Core + reserve must be exactly the held-out library; seed keys real methods.

        A goal in neither list would be silently unrun and unreported, which is
        the cherry-picking the reserve is meant to rule out.
        """
        library = set(held_out_goals)
        listed = set(self.core_goals) | set(self.reserve_goals)
        if listed != library:
            raise SpecError(
                "design must partition the held-out goals exactly: "
                f"unlisted {sorted(library - listed)}, not in library {sorted(listed - library)}"
            )
        unknown = set(self.seeds) - set(methods) - {"default"}
        if unknown:
            raise SpecError(f"seed rule for unknown method(s) {sorted(unknown)}")

    def seeds_for(self, method: str) -> tuple[int, ...]:
        """The optimiser seeds this method runs under the design."""
        return self.seeds.get(method, self.seeds["default"])

    def includes(self, goal: str, method: str, seed: int) -> bool:
        """Is this cell part of the design?"""
        return goal in self.core_goals and seed in self.seeds_for(method)

    def cells(self, methods: Sequence[str]) -> list[tuple[str, str, int]]:
        """Every design cell, goal-major."""
        return [
            (goal, method, seed)
            for goal in self.core_goals
            for method in methods
            for seed in self.seeds_for(method)
        ]

    def hash(self) -> str:
        """A stable hash of the selection itself (comments and key order ignored)."""
        payload = json.dumps(
            {
                "core_goals": list(self.core_goals),
                "reserve_goals": list(self.reserve_goals),
                "seeds": {k: list(v) for k, v in sorted(self.seeds.items())},
                "amendment": self.amendment,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """For the manifest, so the record says what the hash stands for."""
        return {
            "amendment": self.amendment,
            "dated": self.dated,
            "core_goals": list(self.core_goals),
            "reserve_goals": list(self.reserve_goals),
            "seeds": {k: list(v) for k, v in sorted(self.seeds.items())},
        }


@dataclass
class DesignCells:
    """What :func:`load_design_cells` found."""

    cells: list[dict[str, Any]]
    missing: list[tuple[str, str, int]]
    excluded: list[str]
    stale: list[str]


def load_design_cells(
    results: Path,
    design: BenchmarkDesign,
    methods: Sequence[str],
    budgets_hash: str,
) -> DesignCells:
    """Read ONLY the design's cells from a results tree. The analysis entry point.

    * ``cells`` -- design cells present and produced under ``budgets_hash``.
    * ``missing`` -- design cells with no valid file yet.
    * ``excluded`` -- files on disk outside the design (seed 3, reserve goals,
      unknown names). Listed by filename only; they are never opened, so no
      objective value outside the design is read even by accident.
    * ``stale`` -- design cells whose file is unparseable or from other budgets.
    """
    wanted = {f"{g}__{m}__seed{s}.json": (g, m, s) for g, m, s in design.cells(methods)}
    excluded = sorted(
        p.name
        for p in Path(results).glob("*__*.json")
        if p.name not in wanted and p.parent.name != "failures"
    )
    cells: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []
    stale: list[str] = []
    for name, key in wanted.items():
        path = Path(results) / name
        if not path.exists():
            missing.append(key)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stale.append(name)
            continue
        if payload.get("budgets_hash") != budgets_hash:
            stale.append(name)
            continue
        cells.append(payload)
    logger.info(
        "design cells: %d present, %d missing, %d stale; %d file(s) outside the design ignored",
        len(cells),
        len(missing),
        len(stale),
        len(excluded),
    )
    return DesignCells(cells=cells, missing=missing, excluded=excluded, stale=stale)
