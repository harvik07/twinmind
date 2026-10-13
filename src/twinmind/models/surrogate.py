"""Model C: the surrogate contract, frozen before any training code.

WHAT THIS EMULATES, STATED FIRST BECAUSE IT IS THE EASIEST THING TO FORGET.

The surrogate emulates **the twin**, not reality. It inherits every assumption
in ``configs/twin.yaml`` — including ``demand_capacity_ratio``, whose 25%
perturbation moves twin profit by 45%, and the 0.95 warning threshold that was
chosen to keep the landscape non-degenerate rather than measured from data. A
surrogate that reproduces the twin perfectly is still only as good as the twin.
``reports/model_cards/c_surrogate.md`` says this at the top too.

THREE KINDS OF UNCERTAINTY, AND WHY THEY ARE SEPARATE FIELDS.

This is the distinction the whole design turns on, and conflating the first two
is the standard way a surrogate ends up confidently wrong:

* **aleatoric_std** — the replication noise the model expects at this scenario.
  The twin is stochastic, so even a perfect surrogate cannot predict a single
  replication. This is what the Gaussian NLL head learns, and because the labels
  are means of ``n_reps`` replications, it is the std *of the label*, which is
  the per-replication std divided by ``sqrt(n_reps)``.
* **epistemic_std** — ensemble disagreement, which is model ignorance. "I have
  not seen this region." This is the only one of the two that should ever
  trigger a fallback to the real twin, because a scenario can be perfectly
  predictable and still unseen.
* The **truth's own noise** — the test labels are 30-rep means and carry noise
  too, so a perfect surrogate cannot reach R2 = 1. That is not a field here; it
  is the noise ceiling in the evaluation (see the model card).

Reporting one number for all three would make a confident prediction in an
unexplored region indistinguishable from a confident prediction in a
well-sampled one.

BATCH-FIRST. The agent screens hundreds of scenarios at once, so ``predict``
takes a sequence and returns ``[N, 8]`` arrays. A per-item API that the caller
loops over would spend its whole life in Python overhead and throw away the only
reason the surrogate exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from twinmind.twin.runner import KPI_NAMES, SCENARIO_DIMS, Scenario

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from twinmind.twin.context import TwinContext

#: The 8 KPIs in the order every surrogate array uses. Identical to
#: ``KPI_NAMES`` by construction rather than by copying -- a second hand-written
#: tuple is a second thing to get out of step, and a column permutation between
#: training and inference is invisible until the numbers are quietly wrong.
KPI_ORDER: tuple[str, ...] = KPI_NAMES

#: KPIs the twin defines on [0, 1]. Enforced on surrogate output.
BOUNDED_KPIS: tuple[str, ...] = ("oee", "stockout_probability", "on_time_delivery_rate")

#: KPIs that cannot be negative. ``profit`` is deliberately absent: a loss-making
#: scenario is a real and important outcome, and clipping it at zero would hide
#: exactly the plans the optimiser must avoid.
NONNEGATIVE_KPIS: tuple[str, ...] = (
    "units_produced",
    "unit_cost",
    "downtime_hours",
    "unplanned_failures",
    *BOUNDED_KPIS,
)

#: ``supplier_choice`` is one-hot rather than scaled. The three suppliers differ
#: in price, lead time and reliability with no natural ordering -- S1 is not
#: "more supplier" than S0 -- so a single scaled column would invite the model to
#: interpolate between the incumbent and the offshore option, which means
#: nothing. ``shift_count`` stays scalar because 1, 2, 3 shifts genuinely is
#: ordered and monotone in capacity.
CATEGORICAL_DIMS: tuple[str, ...] = ("supplier_choice",)


class SurrogateOutput(BaseModel):
    """One batched prediction: means with both kinds of uncertainty.

    Validity is enforced here rather than hoped for downstream. The optimiser
    will rank thousands of these, and a negative ``units_produced`` or an OEE of
    1.3 is not a small error -- it is a scenario that looks artificially good
    and gets recommended.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    mean: NDArray[np.float64]
    aleatoric_std: NDArray[np.float64]
    epistemic_std: NDArray[np.float64]

    @model_validator(mode="after")
    def _check(self) -> SurrogateOutput:
        shapes = {
            "mean": self.mean.shape,
            "aleatoric_std": self.aleatoric_std.shape,
            "epistemic_std": self.epistemic_std.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(f"all arrays must share a shape, got {shapes}")
        if self.mean.ndim != 2 or self.mean.shape[1] != len(KPI_ORDER):
            raise ValueError(
                f"expected arrays of shape [N, {len(KPI_ORDER)}], got {self.mean.shape}"
            )
        for name, array in (
            ("mean", self.mean),
            ("aleatoric_std", self.aleatoric_std),
            ("epistemic_std", self.epistemic_std),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.aleatoric_std < 0) or np.any(self.epistemic_std < 0):
            raise ValueError("standard deviations must be non-negative")

        for index, kpi in enumerate(KPI_ORDER):
            column = self.mean[:, index]
            if kpi in NONNEGATIVE_KPIS and np.any(column < -1e-9):
                raise ValueError(f"{kpi} must be non-negative, got min {column.min()}")
            if kpi in BOUNDED_KPIS and np.any(column > 1.0 + 1e-9):
                raise ValueError(f"{kpi} must be <= 1, got max {column.max()}")
        return self

    @property
    def total_std(self) -> NDArray[np.float64]:
        """Aleatoric and epistemic combined in quadrature.

        The right interval for "how far might the twin's answer be from this
        prediction": the two are independent sources, one irreducible noise and
        one reducible ignorance.
        """
        return np.sqrt(self.aleatoric_std**2 + self.epistemic_std**2)

    def column(self, kpi: str) -> NDArray[np.float64]:
        """One KPI's predicted means, by name."""
        if kpi not in KPI_ORDER:
            raise KeyError(f"unknown KPI {kpi!r}; expected one of {KPI_ORDER}")
        return self.mean[:, KPI_ORDER.index(kpi)]

    def __len__(self) -> int:
        """Number of scenarios in the batch."""
        return int(self.mean.shape[0])


def enforce_validity(mean: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip a raw mean array into the KPI domain.

    Applied at inference rather than trusted from the network. The transforms in
    the heads make violations rare, but "rare" over a 10,000-scenario batch that
    an optimiser is about to take the argmax of is not good enough.
    """
    out = np.array(mean, dtype="float64", copy=True)
    for index, kpi in enumerate(KPI_ORDER):
        if kpi in NONNEGATIVE_KPIS:
            out[:, index] = np.clip(out[:, index], 0.0, None)
        if kpi in BOUNDED_KPIS:
            out[:, index] = np.clip(out[:, index], 0.0, 1.0)
    return out


@runtime_checkable
class SurrogateModel(Protocol):
    """What the optimisers and the agent depend on, and nothing more."""

    def predict(self, scenarios: Sequence[Scenario], context: TwinContext) -> SurrogateOutput:
        """Predict KPIs for a batch of scenarios at one context."""
        ...


# --------------------------------------------------------------------------- #
# Feature encoding
# --------------------------------------------------------------------------- #
def feature_names(bounds: dict[str, tuple[float, float]] | None = None) -> list[str]:
    """Canonical feature order. Frozen: the trained weights depend on it."""
    from twinmind.twin.context import STATE_FIELDS

    del bounds
    names: list[str] = []
    for dim in SCENARIO_DIMS:
        if dim in CATEGORICAL_DIMS:
            names.extend(f"{dim}_is_{level}" for level in _categorical_levels(dim))
        else:
            names.append(dim)
    names.extend(f"state_{field}" for field in STATE_FIELDS)
    return names


@lru_cache(maxsize=1)
def _bounds_from_config() -> tuple[tuple[str, float, float], ...]:
    """The scenario bounds, parsed from twin.yaml **once per process**.

    Cached because it is on a very hot path and was not cheap. ``load_config``
    re-reads and re-validates the YAML on every call, measured at **17.4 ms**,
    and ``build_scenario`` called this plus ``_categorical_levels`` for every
    candidate — 18 ms per candidate. At ``best_known``'s 500,000-point budget
    that is 2.5 hours of YAML parsing per cell, which would have made the full
    benchmark take days rather than hours.

    A tuple rather than a dict is returned so a caller cannot mutate the cached
    value; the public helpers build a fresh container from it each call, which
    costs about a microsecond.
    """
    from twinmind.utils.config import TwinConfig, load_config

    config = cast(TwinConfig, load_config("twin"))
    return tuple((d.name, float(d.min), float(d.max)) for d in config.scenario_space)


def clear_bounds_cache() -> None:
    """Drop the cached bounds. For tests that rewrite ``configs/twin.yaml``."""
    _bounds_from_config.cache_clear()


def _categorical_levels(dim: str) -> list[int]:
    """The integer levels a categorical dimension can take, from twin.yaml."""
    for name, low, high in _bounds_from_config():
        if name == dim:
            return list(range(int(round(low)), int(round(high)) + 1))
    raise KeyError(f"{dim!r} is not a scenario dimension")


def scenario_bounds() -> dict[str, tuple[float, float]]:
    """``{dim: (min, max)}`` from twin.yaml, in canonical order."""
    return {name: (low, high) for name, low, high in _bounds_from_config()}


def encode(
    scenario: Scenario,
    context: TwinContext,
    bounds: dict[str, tuple[float, float]] | None = None,
    state_mean: NDArray[np.float64] | None = None,
    state_scale: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Canonical, deterministic features for one ``(scenario, context)`` pair.

    Scenario dimensions are normalised to [0, 1] by the configured bounds, with
    ``supplier_choice`` one-hot (see :data:`CATEGORICAL_DIMS`). The context's
    state vector is appended, standardised by statistics that must come from the
    **training split only** -- passing test statistics here is a leak, which is
    why they are explicit arguments rather than computed from whatever array
    happens to be to hand.

    A zero entry in ``state_scale`` means a constant feature; it is replaced by
    1.0 so the feature becomes a constant 0 rather than a NaN that silently
    poisons the whole row.
    """
    return encode_batch(
        [scenario], context, bounds=bounds, state_mean=state_mean, state_scale=state_scale
    )[0]


def encode_batch(
    scenarios: Sequence[Scenario],
    context: TwinContext,
    bounds: dict[str, tuple[float, float]] | None = None,
    state_mean: NDArray[np.float64] | None = None,
    state_scale: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """``[N, F]`` features for many scenarios at one context.

    The context encodes once and is broadcast, which is the whole point of the
    batch form: the agent screens hundreds of scenarios against a single plant
    state, and re-standardising the same 15 numbers per scenario is pure waste.
    """
    resolved_bounds = bounds if bounds is not None else scenario_bounds()
    rows = np.empty((len(scenarios), _n_scenario_features()), dtype="float64")
    for row, scenario in enumerate(scenarios):
        rows[row] = _encode_scenario(scenario, resolved_bounds)

    state = np.asarray(context.state_vector(), dtype="float64")
    if state_mean is not None and state_scale is not None:
        scale = np.where(np.abs(state_scale) < 1e-12, 1.0, state_scale)
        state = (state - state_mean) / scale
    return np.hstack([rows, np.tile(state, (len(scenarios), 1))])


def _n_scenario_features() -> int:
    """Width of the scenario half of the feature vector."""
    width = 0
    for dim in SCENARIO_DIMS:
        width += len(_categorical_levels(dim)) if dim in CATEGORICAL_DIMS else 1
    return width


def _encode_scenario(
    scenario: Scenario, bounds: dict[str, tuple[float, float]]
) -> NDArray[np.float64]:
    """The scenario half of one feature row."""
    values: list[float] = []
    for dim in SCENARIO_DIMS:
        raw = float(getattr(scenario, dim))
        if dim in CATEGORICAL_DIMS:
            levels = _categorical_levels(dim)
            chosen = int(round(raw))
            values.extend(1.0 if level == chosen else 0.0 for level in levels)
            continue
        low, high = bounds[dim]
        span = high - low
        values.append((raw - low) / span if span > 0 else 0.0)
    return np.asarray(values, dtype="float64")


def decode(
    features: NDArray[np.float64], bounds: dict[str, tuple[float, float]] | None = None
) -> Scenario:
    """Invert :func:`encode`'s scenario half back to a ``Scenario``.

    Exists for the round-trip test rather than for production use. A feature
    encoding that cannot be inverted is one where a dimension has been dropped,
    duplicated or mis-scaled, and that is otherwise invisible until a trained
    surrogate quietly ignores a decision variable.
    """
    resolved = bounds if bounds is not None else scenario_bounds()
    array = np.asarray(features, dtype="float64").ravel()
    payload: dict[str, Any] = {}
    cursor = 0
    for dim in SCENARIO_DIMS:
        if dim in CATEGORICAL_DIMS:
            levels = _categorical_levels(dim)
            block = array[cursor : cursor + len(levels)]
            payload[dim] = int(levels[int(np.argmax(block))])
            cursor += len(levels)
            continue
        low, high = resolved[dim]
        payload[dim] = low + float(array[cursor]) * (high - low)
        cursor += 1
    payload["shift_count"] = int(round(payload["shift_count"]))
    return Scenario(**payload)


def load_surrogate(path: Path) -> SurrogateModel:
    """Load a trained ensemble and return it behind the Protocol."""
    from twinmind.models.surrogate_mlp import SurrogateEnsemble

    return SurrogateEnsemble.load(path)
