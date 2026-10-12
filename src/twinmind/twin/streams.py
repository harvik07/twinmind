"""Common random numbers: one named, independent stream per source of noise.

WHY THIS MODULE EXISTS.

The optimiser and the surrogate spend their whole lives *comparing* scenarios.
If scenario A and scenario B are evaluated with differently-shuffled randomness,
the difference between their KPIs contains both the effect of the decision and
the difference in luck, and the luck term is large: a single unplanned failure
moves profit by more than most parameter changes do.

Common random numbers fix this by making the noise a function of the *seed*
alone, never of the scenario. Two scenarios run at the same seed then see the
same machine lifetimes, the same repair durations and the same demand path, so
their difference is attributable to the decision. It is the cheapest variance
reduction available and it is why the spec forbids ``base_seed + i``.

HOW.

``numpy.random.SeedSequence(seed).spawn(k)`` derives k statistically independent
child sequences by hashing, not by stepping a single stream. Each named source
gets its own child, so drawing one more repair time cannot shift the demand
path. Adding a new stream name appends to the end of ``STREAM_NAMES``, which
leaves every existing stream's numbers unchanged -- important, because a result
table regenerated after a code change should not move for a reason that has
nothing to do with the change.

The ordering guarantee is the whole contract, so ``STREAM_NAMES`` is append-only
and there is a test asserting its exact value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Append-only. Inserting a name in the middle renumbers every later stream and
#: silently invalidates every stored result. There is a test pinning this tuple.
STREAM_NAMES: tuple[str, ...] = (
    "machine_life",
    "repair",
    "processing",
    "defects",
    "supplier",
    "demand",
)


@dataclass(frozen=True)
class StreamBundle:
    """One independent :class:`numpy.random.Generator` per named source.

    Frozen, and the generators are created once in ``from_seed``. The twin pulls
    every random number through this object, so there is exactly one place that
    decides what is correlated with what.
    """

    generators: dict[str, np.random.Generator]
    seed: int

    @classmethod
    def from_seed(cls, seed: int) -> StreamBundle:
        """Derive all streams from one integer seed via ``SeedSequence.spawn``."""
        children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
        return cls(
            generators={
                name: np.random.default_rng(child)
                for name, child in zip(STREAM_NAMES, children, strict=True)
            },
            seed=int(seed),
        )

    def __getitem__(self, name: str) -> np.random.Generator:
        """The generator for one named stream, raising on an unknown name."""
        if name not in self.generators:
            raise KeyError(f"unknown stream {name!r}; expected one of {STREAM_NAMES}")
        return self.generators[name]


def derive_seeds(base_seed: int, n: int) -> list[int]:
    """``n`` independent replication seeds from one base seed.

    Via ``SeedSequence.spawn``, not ``base_seed + i``. Consecutive integer seeds
    are a real hazard rather than a stylistic one: they produce *correlated*
    starting states in several generators, and because the twin spawns further
    children from each replication seed, neighbouring replications can end up
    sharing substreams. That inflates the apparent precision of a mean -- the
    confidence interval shrinks without the information to justify it.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    children = np.random.SeedSequence(base_seed).spawn(n)
    # generate_state gives the raw 32-bit words; one is plenty for a seed and
    # keeps these reproducible across numpy versions.
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
