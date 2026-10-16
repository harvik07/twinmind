"""Peak memory of the current process over a block of work, sampled on a thread.

Two figures, because on Windows they diverge and the second is the one that
fails: ``rss`` (working set, what is resident) and ``private`` (committed
memory; USS elsewhere). A ``MemoryError`` on Windows means the commit limit was
hit, not RAM. Sampled every 0.25 s, so a shorter spike can be missed.
"""

from __future__ import annotations

import contextlib
import os
import threading


class PeakMemory:
    """Start, do work, stop: returns peak rss and peak committed memory in GB."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.peak_rss = 0.0
        self.peak_private = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        info = process.memory_info()
        self.peak_rss = max(self.peak_rss, float(info.rss))
        private = getattr(info, "private", None)
        if private is None:
            private = process.memory_full_info().uss
        self.peak_private = max(self.peak_private, float(private))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:  # noqa: BLE001 - diagnostics must never break a run
                return
            self._stop.wait(self.interval)

    def start(self) -> PeakMemory:
        """Begin sampling."""
        self._thread.start()
        return self

    def stop(self) -> dict[str, float]:
        """Stop, take a last sample, and return the peaks in GB."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        with contextlib.suppress(Exception):
            self._sample()
        return {"peak_rss_gb": self.peak_rss / 1e9, "peak_private_gb": self.peak_private / 1e9}
