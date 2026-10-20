"""One background run at a time, for the Plan page. No Streamlit import.

A run takes minutes, and Streamlit reruns the page on every click, so the
work runs in ONE worker thread in the same process. The page only reads a
lock-protected status snapshot; the thread never touches Streamlit.

Guarantees, each tested with fakes:

- **One run at a time.** :meth:`JobRunner.start` refuses while a run is
  active, so a rerun or a double click can never start a second thread.
- **Every run ends with a status**: finished, failed, cancelled or timed out.
  Exceptions are caught and kept in plain words (and redacted detail).
- **Cancel and a hard time limit** take effect between agent steps: the
  progress hook raises, and the graph stops at the next node boundary. The
  current step (a twin search can take a minute) finishes first, and the
  status says so.
- ``on_done`` (which writes Past results) runs exactly once per run, and its
  own failure cannot leave the run without a status.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

#: Hard limit for one run. The agent card measured 106-205 s per goal on a
#: quiet machine; ten minutes leaves room for a busy laptop.
TIME_LIMIT_SECONDS = 600.0

ACTIVE = ("running", "cancelling", "stopping")
FINAL = ("finished", "failed", "cancelled", "timed_out")

#: Agent node -> the plain step shown while it runs.
STEPS = (
    ("planner", "Understanding your goal"),
    ("explorer", "Searching for plans"),
    ("evaluator", "Checking the best candidates in the detailed simulation (30 runs)"),
    ("critic", "Reviewing for risks"),
    ("explainer", "Writing the summary"),
)


class JobCancelledError(Exception):
    """The user pressed Cancel."""


class JobTimedOutError(Exception):
    """The run passed the hard time limit."""


Progress = Callable[[str, int, int], None]
Work = Callable[[Progress], Any]


@dataclass(frozen=True)
class JobStatus:
    """A snapshot the page can read without holding the lock."""

    id: str = ""
    goal_text: str = ""
    state: str = "idle"  # idle | running | cancelling | stopping | finished | failed | ...
    nodes: tuple[str, ...] = ()
    llm_used: int = 0
    llm_cap: int = 0
    started: float = 0.0
    ended: float | None = None
    result: Any = None
    error_kind: str = ""  # plain message key in ui.plain.MESSAGES
    error_detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    def elapsed(self, now: float) -> float:
        if not self.id:
            return 0.0
        end = self.ended if self.ended is not None else now
        return end - self.started

    @property
    def steps(self) -> list[tuple[str, str]]:
        """Each plain step with "done", "now" or "todo"."""
        done = set(self.nodes)
        out, current_given = [], False
        for node, text in STEPS:
            if node in done:
                out.append((text, "done"))
            elif self.active and not current_given and node != "planner":
                out.append((text, "now"))
                current_given = True
            else:
                out.append((text, "todo"))
        return out


def classify_error(error: BaseException) -> str:
    """Which plain message fits an exception from a run."""
    name = type(error).__name__.lower()
    text = str(error).lower()
    if "ratelimit" in name or "rate limit" in text or "429" in text:
        return "ai_busy"
    if any(k in name for k in ("connection", "llmerror", "apierror", "timeout")) or (
        "llm unavailable" in text or "api_key" in text or "groq" in text
    ):
        return "ai_unreachable"
    return "failed"


class JobRunner:
    """Owns the single worker thread and its status."""

    def __init__(
        self,
        time_limit_seconds: float = TIME_LIMIT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        start_thread: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._status = JobStatus()
        self._cancel = False
        self._limit = time_limit_seconds
        self._clock = clock
        self._start_thread = start_thread or self._thread

    @staticmethod
    def _thread(target: Callable[[], None]) -> None:
        threading.Thread(target=target, name="twinmind-plan", daemon=True).start()

    # ------------------------------------------------------------- reading --
    def status(self) -> JobStatus:
        with self._lock:
            status = self._status
        if status.active and status.elapsed(self._clock()) > self._limit:
            return replace(status, state="stopping")
        return status

    @property
    def busy(self) -> bool:
        return self.status().active

    # ------------------------------------------------------------- control --
    def start(
        self,
        goal_text: str,
        work: Work,
        on_done: Callable[[JobStatus], None] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> JobStatus | None:
        """Start a run, or return None if one is already active."""
        with self._lock:
            if self._status.active:
                return None
            self._cancel = False
            self._status = JobStatus(
                id=uuid.uuid4().hex[:12],
                goal_text=goal_text,
                state="running",
                started=self._clock(),
                meta=dict(meta or {}),
            )
            snapshot = self._status
        self._start_thread(lambda: self._run(snapshot.id, work, on_done))
        return snapshot

    def cancel(self) -> None:
        with self._lock:
            if self._status.active:
                self._cancel = True
                self._status = replace(self._status, state="cancelling")

    # -------------------------------------------------------------- worker --
    def _progress(self, job_id: str) -> Progress:
        def progress(node: str, used: int, cap: int) -> None:
            with self._lock:
                if self._status.id != job_id:
                    raise JobCancelledError("superseded")
                self._status = replace(
                    self._status,
                    nodes=(*self._status.nodes, node),
                    llm_used=used,
                    llm_cap=cap,
                )
                cancel = self._cancel
                late = self._clock() - self._status.started > self._limit
            if cancel:
                raise JobCancelledError("cancelled by the user")
            if late:
                raise JobTimedOutError(f"over the {self._limit:.0f} s limit")

        return progress

    def _run(self, job_id: str, work: Work, on_done: Callable[[JobStatus], None] | None) -> None:
        from twinmind.ui.services import safe_error

        state, result, kind, detail = "failed", None, "failed", ""
        try:
            result = work(self._progress(job_id))
            state, kind = "finished", ""
        except JobCancelledError:
            state, kind = "cancelled", "cancelled"
        except JobTimedOutError:
            state, kind = "timed_out", "timed_out"
        except BaseException as error:  # noqa: BLE001 -- a run must always end with a status
            state, kind, detail = "failed", classify_error(error), safe_error(error)
        with self._lock:
            if self._status.id != job_id:
                return
            if state == "finished" and self._cancel:
                state, kind = "cancelled", "cancelled"
            self._status = replace(
                self._status,
                state=state,
                ended=self._clock(),
                result=result if state == "finished" else None,
                error_kind=kind,
                error_detail=detail,
            )
            final = self._status
        if on_done is not None:
            # Logging must never undo the status.
            with contextlib.suppress(Exception):
                on_done(final)
