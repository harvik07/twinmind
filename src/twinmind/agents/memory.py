"""Episodic memory of past agent runs, backed by DuckDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twinmind.twin.runner import Scenario, TwinResult


class EpisodeMemory:
    """Stores (query, scenario, KPIs) triples so later runs can warm-start."""

    def __init__(self, path: Path, max_episodes: int = 500) -> None:
        self.path = path
        self.max_episodes = max_episodes

    def initialise(self) -> None:
        """Create the schema if it does not exist."""
        raise NotImplementedError("EpisodeMemory.initialise is not implemented yet")

    def record(self, query: str, result: TwinResult, metadata: dict[str, Any] | None = None
               ) -> None:
        """Persist one evaluated scenario."""
        raise NotImplementedError("EpisodeMemory.record is not implemented yet")

    def similar(self, query: str, limit: int = 5) -> list[tuple[Scenario, TwinResult]]:
        """Retrieve past episodes relevant to ``query``."""
        raise NotImplementedError("EpisodeMemory.similar is not implemented yet")
