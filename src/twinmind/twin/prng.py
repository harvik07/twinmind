"""Chunked pre-drawn random numbers for the simulation's hot path.

Profiling the twin at a 10-unit lot size showed 0.559 s per replication against
a 0.25 s budget. Two of the addressable costs were per-draw calls into numpy:
``Generator.lognormal`` and ``Generator.binomial`` were together about 0.11 s of
that, because each call pays Python-level dispatch and argument validation to
produce a single scalar.

Drawing in chunks of a few thousand and handing them out one at a time amortises
that away. This is the spec's second sanctioned optimisation ("pre-drawn numpy
random arrays per stream") and it is **not** a fidelity change: the numbers come
from the same generator in the same order, so a chunked draw and a per-call draw
produce the identical sequence. There is a test asserting exactly that.

The chunk is a buffer, not a cache: once consumed it is refilled from the same
generator, so a long replication is not restricted to the first N draws.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ChunkedNormal:
    """Standard normals from one generator, drawn in blocks."""

    __slots__ = ("_rng", "_chunk", "_buffer", "_index")

    def __init__(self, rng: np.random.Generator, chunk: int = 4096) -> None:
        self._rng = rng
        self._chunk = chunk
        self._buffer: NDArray[np.float64] = rng.standard_normal(chunk)
        self._index = 0

    def next(self) -> float:
        """One standard normal."""
        if self._index >= self._buffer.size:
            self._buffer = self._rng.standard_normal(self._chunk)
            self._index = 0
        value = self._buffer[self._index]
        self._index += 1
        return float(value)


class ChunkedBinomial:
    """Binomial draws at fixed ``(n, p)``, drawn in blocks.

    Fixed parameters are what makes the chunking safe: the lot size and the
    defect probability are constant for a station over a replication, so a block
    drawn in advance is distributed exactly as the per-call draws would be.
    """

    __slots__ = ("_rng", "_chunk", "_n", "_p", "_buffer", "_index")

    def __init__(self, rng: np.random.Generator, n: int, p: float, chunk: int = 4096) -> None:
        self._rng = rng
        self._chunk = chunk
        self._n = int(n)
        self._p = float(p)
        self._buffer: NDArray[np.int64] = self._draw()
        self._index = 0

    def _draw(self) -> NDArray[np.int64]:
        if self._p <= 0.0:
            return np.zeros(self._chunk, dtype="int64")
        return np.asarray(self._rng.binomial(self._n, self._p, size=self._chunk), dtype="int64")

    def next(self) -> int:
        """One binomial draw."""
        if self._index >= self._buffer.size:
            self._buffer = self._draw()
            self._index = 0
        value = self._buffer[self._index]
        self._index += 1
        return int(value)
