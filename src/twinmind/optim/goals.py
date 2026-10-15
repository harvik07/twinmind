"""The goal library: 20 YAML specifications, split dev / held-out.

THE SPLIT IS THE POINT. Eight **dev** goals are the only ones any hyperparameter
may be tuned on — the surrogate safety margin, the shortlist size, the minimum
diversity distance. Twelve **held-out** goals are touched exactly once, by the
final benchmark, after the pre-registration is committed. They use disjoint T1
contexts (dev 0–19, held-out 20–39) so a goal cannot be "nearly the same
problem" across the split.

BASELINE FOR NORMALISATION. ``Scenario`` has defaults, and those defaults are
the current operating policy — they are what the twin runs when nobody has
optimised anything. So the baseline is the **default scenario evaluated on the
goal's own context**, not the mean of random plans. That matters for reading the
results: a normalised gap is measured against "what the plant does today",
which is the comparison a plant manager would ask for.

Two goals are **infeasible by design**. They exist to prove the infeasible path
returns ``status="infeasible_unverified"`` rather than quietly relaxing a
constraint, and they are marked in the YAML so the benchmark can score them
separately instead of counting them as failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from twinmind.optim.contracts import ObjectiveSpec, SpecError
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import CONFIG_DIR

logger = get_logger(__name__)

GOALS_DIR = CONFIG_DIR / "goals"

#: T1 context blocks reserved for each split. Disjoint by construction.
DEV_CONTEXTS = range(0, 20)
HELD_OUT_CONTEXTS = range(20, 40)


@dataclass(frozen=True)
class Goal:
    """One goal: its spec, which split it belongs to, and where it came from."""

    spec: ObjectiveSpec
    split: str
    path: Path

    @property
    def name(self) -> str:
        """The goal's name."""
        return self.spec.name


def load_goal(path: Path) -> Goal:
    """Parse one goal YAML, raising :class:`SpecError` with the file named."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpecError(f"{path}: a goal file must contain a mapping.")
    split = payload.pop("split", None)
    if split not in {"dev", "held_out"}:
        raise SpecError(
            f"{path}: 'split' must be 'dev' or 'held_out', got {split!r}. "
            "Dev goals may be tuned on; held-out goals may not."
        )
    try:
        spec = ObjectiveSpec.from_dict(payload)
    except SpecError as error:
        raise SpecError(f"{path}: {error}") from error

    allowed = DEV_CONTEXTS if split == "dev" else HELD_OUT_CONTEXTS
    if spec.context_id not in allowed:
        raise SpecError(
            f"{path}: context_id {spec.context_id} is not in the {split} range "
            f"[{allowed.start}, {allowed.stop - 1}]. Dev and held-out goals must use "
            "disjoint contexts so a goal cannot appear in both halves."
        )
    return Goal(spec=spec, split=split, path=Path(path))


def load_goals(directory: Path | None = None, split: str | None = None) -> list[Goal]:
    """Every goal in ``directory``, optionally filtered to one split."""
    root = Path(directory) if directory is not None else GOALS_DIR
    goals = [load_goal(p) for p in sorted(root.glob("*.yaml"))]
    if split is not None:
        goals = [g for g in goals if g.split == split]
    logger.info(
        "loaded %d goals from %s%s", len(goals), root.name, f" (split={split})" if split else ""
    )
    return goals


def baseline_scenario(spec: ObjectiveSpec) -> Any:
    """The current-policy baseline: Scenario defaults with the goal's fixed values.

    Scenario's defaults *are* the current operating policy, so this is "what the
    plant does today under this what-if", which is the right thing to normalise
    a gap against.
    """
    return spec.base_scenario()
