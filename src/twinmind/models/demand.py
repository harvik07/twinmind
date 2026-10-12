"""Model B: the demand contract the digital twin consumes.

Frozen before any training code, for the same reason ``HealthModel`` was: the
twin's order-book process depends on this shape, and renegotiating it once four
modules import it is expensive.

Two things this contract insists on, both learned from the health model:

**The series and the forecast origin are explicit arguments.** A forecaster that
remembers "where it got to" is impossible to test and impossible to reuse across
the hundreds of scenario replications the twin runs. ``quantiles(series_id,
origin, horizon)`` is a pure function of its arguments plus an immutable
snapshot, so the same call always gives the same answer.

**``origin`` is the last *observed* day.** The forecast covers
``origin + 1 ... origin + horizon``. Stating that here rather than in a comment
removes the off-by-one that otherwise appears at every call site.

The twin needs whole trajectories, not just marginals, which is why
``sample_paths`` is part of the contract rather than an afterthought: sampling
each day independently from its own quantiles destroys the serial correlation
that makes inventory dynamics interesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

#: Headline quantile levels. The sampler may fit a denser grid (see
#: ``demand_lgbm``); these three are what the contract exposes and what the
#: model card reports.
HEADLINE_QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)


class DemandQuantiles(BaseModel):
    """A quantile forecast for one series over one horizon.

    Validated rather than trusted: a crossed quantile (q10 above q50) or a
    negative demand is meaningless to the twin's inventory process, and silently
    passing one through produces a stockout probability that looks plausible and
    is wrong.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: int = Field(gt=0, description="Number of days forecast.")
    q10: list[float]
    q50: list[float]
    q90: list[float]

    @model_validator(mode="after")
    def _check(self) -> DemandQuantiles:
        for name, values in (("q10", self.q10), ("q50", self.q50), ("q90", self.q90)):
            if len(values) != self.horizon:
                raise ValueError(
                    f"{name} has {len(values)} values, expected horizon={self.horizon}"
                )
            if any(v < 0 for v in values):
                raise ValueError(f"{name} contains a negative demand")
        crossed = [
            i
            for i, (lo, mid, hi) in enumerate(zip(self.q10, self.q50, self.q90, strict=True))
            if not lo <= mid <= hi
        ]
        if crossed:
            raise ValueError(
                f"quantiles cross at step(s) {crossed[:5]}: q10 <= q50 <= q90 must hold"
            )
        return self

    def as_frame(self, origin: pd.Timestamp) -> pd.DataFrame:
        """The forecast as a frame indexed by the dates it actually covers."""
        dates = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=self.horizon)
        return pd.DataFrame({"ds": dates, "q10": self.q10, "q50": self.q50, "q90": self.q90})


@dataclass(frozen=True)
class DemandData:
    """Immutable snapshot of the featurised frame a forecaster reads from.

    Frozen, and the frame is defensively copied on construction, so a model
    cannot accumulate state between calls and a caller cannot mutate history out
    from under a fitted model. That is what makes the truncation-invariance test
    meaningful: ``truncate(origin)`` returns a *new* snapshot rather than
    editing this one.
    """

    frame: pd.DataFrame
    target: str = "y"
    date_column: str = "ds"
    id_column: str = "unique_id"

    def __post_init__(self) -> None:
        missing = [
            c for c in (self.id_column, self.date_column, self.target) if c not in self.frame
        ]
        if missing:
            raise ValueError(f"snapshot frame is missing {missing}")
        ordered = (
            self.frame.copy()
            .assign(**{self.date_column: lambda f: pd.to_datetime(f[self.date_column])})
            .sort_values([self.id_column, self.date_column])
            .reset_index(drop=True)
        )
        object.__setattr__(self, "frame", ordered)

    @property
    def series_ids(self) -> list[str]:
        """Every series in the snapshot, sorted."""
        return sorted(self.frame[self.id_column].unique().tolist())

    def history(self, series_id: str, origin: pd.Timestamp) -> pd.DataFrame:
        """Rows for ``series_id`` up to and including ``origin``."""
        frame = self.frame
        mask = (frame[self.id_column] == series_id) & (
            frame[self.date_column] <= pd.Timestamp(origin)
        )
        return frame.loc[mask]

    def future(self, series_id: str, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Rows for the forecast window ``origin+1 .. origin+horizon``.

        Only the *known-future* columns of these rows may be read by a model;
        :mod:`twinmind.models.demand_features` enforces that split.
        """
        start = pd.Timestamp(origin) + pd.Timedelta(days=1)
        end = pd.Timestamp(origin) + pd.Timedelta(days=horizon)
        frame = self.frame
        mask = (
            (frame[self.id_column] == series_id)
            & (frame[self.date_column] >= start)
            & (frame[self.date_column] <= end)
        )
        return frame.loc[mask]

    def truncate(self, origin: pd.Timestamp) -> DemandData:
        """A new snapshot with every row after ``origin`` removed.

        The contract test uses this: a forecast made at ``origin`` must be
        identical whether or not the future exists in the snapshot. Any
        difference is leakage.
        """
        kept = self.frame[self.frame[self.date_column] <= pd.Timestamp(origin)]
        return DemandData(kept, self.target, self.date_column, self.id_column)


@runtime_checkable
class DemandModel(Protocol):
    """What the twin's order-book process depends on, and nothing more."""

    def quantiles(self, series_id: str, origin: pd.Timestamp, horizon: int) -> DemandQuantiles:
        """Quantile forecast for ``origin+1 .. origin+horizon``."""
        ...

    def sample_paths(
        self,
        series_id: str,
        origin: pd.Timestamp,
        horizon: int,
        n_samples: int,
        seed: int,
    ) -> NDArray[np.float64]:
        """``[n_samples, horizon]`` demand trajectories, non-negative, seed-deterministic."""
        ...


def load_demand_model(path: Path, data: DemandData) -> DemandModel:
    """Load a saved demand model and bind it to a snapshot.

    The model and the data it reads are separate on disk: boosters are fitted
    artifacts, the snapshot is whatever history the caller currently has. Binding
    them at load time is what lets the twin re-point a trained model at a
    different history without refitting.
    """
    from twinmind.models.demand_lgbm import LightGBMQuantileModel

    return LightGBMQuantileModel.load(path, data)
