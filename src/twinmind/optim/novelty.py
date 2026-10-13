"""How far is a scenario from anything the surrogate was trained on?

WHY THIS IS A SEPARATE CLASS. The ``SurrogateModel`` Protocol is frozen and the
agent layer depends on it, so novelty is not bolted onto ``predict``. It is also
a different kind of quantity: ``epistemic_std`` is what the *model* thinks it
does not know, while this is a plain geometric fact about the training set that
needs no model at all.

That distinction earns its keep here. Model C's ensemble disagreement turned out
**not** to be a usable out-of-distribution detector (AUROC 0.590 on the OOD
split, barely above chance — see ``reports/model_cards/c_surrogate.md`` §4), so
a second, independent signal is worth having.

THE SUPPLIER CATEGORICAL. ``supplier_choice`` is one of three unordered options
differing in price, lead time and reliability. Treating it as a scaled
coordinate would make "incumbent vs premium-local" a *smaller* distance than
"incumbent vs offshore", which is meaningless. So it is one-hot and the three
columns are scaled by ``1/sqrt(2)``, which makes the distance between any two
distinct suppliers exactly 1.0 — the same as the full span of a normalised
continuous dimension, so one supplier switch counts as one dimension of
difference. ``shift_count`` stays scalar because 1, 2, 3 shifts genuinely is
ordered and monotone in capacity.

``k`` IS FIXED AT 10 IN ADVANCE and is not tuned against the OOD split. Tuning
it there would be choosing the detector on the test set, and the resulting AUROC
would mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twinmind.models.surrogate import CATEGORICAL_DIMS, _categorical_levels, scenario_bounds
from twinmind.twin.runner import SCENARIO_DIMS, Scenario
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)

#: Fixed in advance. Not a tuned hyperparameter; see the module docstring.
DEFAULT_K = 10

#: One-hot supplier columns are scaled so that two distinct suppliers sit
#: exactly 1.0 apart, matching the full range of a normalised continuous
#: dimension. ``sqrt(2)`` because two one-hot columns differ by 1 each.
CATEGORICAL_SCALE = 1.0 / np.sqrt(2.0)


def normalise_scenarios(scenarios: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map raw ``[N, 18]`` scenario values into the distance space.

    Continuous and ordinal dimensions are scaled to [0, 1] by the configured
    bounds; the supplier categorical becomes three scaled one-hot columns.
    """
    bounds = scenario_bounds()
    raw = np.asarray(scenarios, dtype="float64")
    columns: list[NDArray[np.float64]] = []
    for index, dim in enumerate(SCENARIO_DIMS):
        values = raw[:, index]
        if dim in CATEGORICAL_DIMS:
            for level in _categorical_levels(dim):
                columns.append(
                    (np.round(values).astype("int64") == level).astype("float64")
                    * CATEGORICAL_SCALE
                )
            continue
        low, high = bounds[dim]
        span = high - low
        columns.append((values - low) / span if span > 0 else np.zeros_like(values))
    return np.column_stack(columns)


@dataclass(frozen=True)
class NoveltyScorer:
    """Mean distance to the ``k`` nearest training scenarios.

    Higher means further from anything the surrogate saw. The score is in
    normalised-dimension units, so 0.25 means "a quarter of one dimension's full
    range away, averaged over the 10 nearest neighbours".
    """

    training: NDArray[np.float64]
    k: int = DEFAULT_K

    @classmethod
    def from_scenarios(cls, scenarios: NDArray[np.float64], k: int = DEFAULT_K) -> NoveltyScorer:
        """Build from a raw ``[N, 18]`` matrix of training scenarios."""
        normalised = normalise_scenarios(scenarios)
        logger.info(
            "novelty scorer over %d training scenarios in %d normalised dimensions (k=%d)",
            normalised.shape[0],
            normalised.shape[1],
            k,
        )
        return cls(training=normalised, k=k)

    def score(self, scenarios: NDArray[np.float64]) -> NDArray[np.float64]:
        """Novelty for a raw ``[N, 18]`` batch."""
        return self.score_normalised(normalise_scenarios(scenarios))

    def score_normalised(self, query: NDArray[np.float64]) -> NDArray[np.float64]:
        """Novelty for already-normalised points.

        Chunked so a large query does not allocate an ``[N, 32768]`` distance
        matrix all at once: the benchmark scores tens of thousands of candidates
        and the full matrix would be gigabytes.
        """
        query = np.atleast_2d(np.asarray(query, dtype="float64"))
        k = min(self.k, self.training.shape[0])
        out = np.empty(query.shape[0], dtype="float64")
        chunk = max(1, int(4e7 // max(self.training.shape[0], 1)))
        for start in range(0, query.shape[0], chunk):
            block = query[start : start + chunk]
            distances = np.linalg.norm(block[:, None, :] - self.training[None, :, :], axis=2)
            nearest = np.partition(distances, k - 1, axis=1)[:, :k]
            out[start : start + block.shape[0]] = nearest.mean(axis=1)
        return out

    def score_scenario(self, scenario: Scenario) -> float:
        """Novelty for one :class:`Scenario`."""
        return float(self.score(np.asarray([scenario.to_vector()], dtype="float64"))[0])
