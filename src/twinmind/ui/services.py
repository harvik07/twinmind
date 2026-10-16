"""Dashboard services: no Streamlit, so all of it is unit-tested.

Mode detection, the replayed demo cache and its closest-goal fallback, the
dashboard's history log, the one-run-at-a-time guard, and twin verification
under a replication and wall-clock cap. Nothing here reads, prints or stores
the Groq key: :func:`detect_mode` only learns whether one is set.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twinmind.agents.contracts import AgentResult
from twinmind.utils.paths import PROJECT_ROOT

DEMO_CACHE = PROJECT_ROOT / "app" / "demo_cache"
REPLAY_LABEL = "replayed benchmark result, not a live run"
HISTORY_LABELS = {
    "live": "live run",
    "replayed": REPLAY_LABEL,
    "aborted": "aborted, no result",
}

#: What-if twin verification: at most this many replications, and the runs
#: stop at the wall-clock cap (always after at least two, for an interval).
WHATIF_MAX_REPS = 10
WHATIF_MIN_REPS = 2
WHATIF_TIME_CAP_SECONDS = 60.0
#: One twin run's budget, measured on a quiet, plugged-in machine
#: (tests/test_twin.py). A busy or hot laptop is slower; the UI says so.
TWIN_SECONDS_PER_RUN_QUIET = 0.25


# --------------------------------------------------------------------------- #
# Mode
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mode:
    """Live needs a Groq key and the local model/data files; else demo."""

    live: bool
    twin_available: bool
    reasons: tuple[str, ...]

    @property
    def label(self) -> str:
        return "Live mode" if self.live else "Demo mode (replayed benchmark results)"


#: Files the live agent and the What-if page load (all gitignored; CI has none).
TWIN_FILES = (
    Path("models") / "surrogate",
    Path("data") / "processed" / "rollouts" / "demand_grids.npz",
    Path("data") / "processed" / "rollouts" / "train",
    Path("data") / "processed" / "rollouts" / "test_t1",
)


def _key_is_set(env: Mapping[str, str], root: Path) -> bool:
    """Whether GROQ_API_KEY is set. The value is never returned or kept."""
    if env.get("GROQ_API_KEY"):
        return True
    env_file = root / ".env"
    if not env_file.exists():
        return False
    from dotenv import dotenv_values

    return bool(dotenv_values(env_file).get("GROQ_API_KEY"))


def detect_mode(env: Mapping[str, str] | None = None, root: Path = PROJECT_ROOT) -> Mode:
    """Live only when nothing forces demo mode, a key is set and the files exist."""
    env = os.environ if env is None else env
    missing = [p.as_posix() for p in TWIN_FILES if not (root / p).exists()]
    reasons = []
    if env.get("TWINMIND_DEMO") == "1":
        reasons.append("TWINMIND_DEMO=1 is set")
    if missing:
        reasons.append("local model and data files are missing: " + ", ".join(missing))
    if not _key_is_set(env, root):
        reasons.append("no GROQ_API_KEY is set")
    forced = env.get("TWINMIND_DEMO") == "1"
    return Mode(live=not reasons, twin_available=not missing and not forced, reasons=tuple(reasons))


# --------------------------------------------------------------------------- #
# Demo cache
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoEntry:
    """One replayed benchmark result."""

    id: str
    goal_id: str
    seed: int
    category: str
    role: str
    context_id: int
    goal_text: str
    provenance: str
    result: AgentResult


def load_demo_cache(cache_dir: Path = DEMO_CACHE) -> list[DemoEntry]:
    """The six replayed results, in the index's order."""
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))["entries"]
    out = []
    for item in index:
        raw = json.loads((cache_dir / item["file"]).read_text(encoding="utf-8"))
        out.append(
            DemoEntry(
                id=raw["id"],
                goal_id=raw["goal_id"],
                seed=int(raw["seed"]),
                category=raw["category"],
                role=raw["role"],
                context_id=int(raw["context_id"]),
                goal_text=raw["result"]["goal_text"],
                provenance=raw["provenance"],
                result=AgentResult.from_dict(raw["result"]),
            )
        )
    return out


_WORD = re.compile(r"[a-z]+")
_STOP = frozenset(
    {
        "a", "an", "and", "the", "of", "to", "with", "by", "for", "at", "in", "on", "or",
        "no", "not", "be", "is", "are", "as", "it", "its", "this", "that", "more", "than",
        "over", "while", "keeping", "keep", "only",
    }
)  # fmt: skip


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def closest_demo(goal_text: str, entries: Sequence[DemoEntry]) -> DemoEntry:
    """The cached goal whose wording overlaps most (Jaccard on content words).

    Used only when a live run cannot reach the language model; the UI then
    labels the result as replayed, never as an answer to the typed goal.
    """
    if not entries:
        raise ValueError("the demo cache is empty")
    asked = _words(goal_text)

    def overlap(entry: DemoEntry) -> float:
        cached = _words(entry.goal_text)
        union = asked | cached
        return len(asked & cached) / len(union) if union else 0.0

    return max(entries, key=overlap)


def is_llm_failure(result: AgentResult) -> bool:
    """The planner could not reach the language model (Groq down, no network)."""
    return result.status == "spec_invalid" and (result.refusal or "").startswith("LLM unavailable")


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def history_path(env: Mapping[str, str] | None = None) -> Path:
    """The dashboard's own JSONL log (gitignored); overridable for tests."""
    env = os.environ if env is None else env
    override = env.get("TWINMIND_DASHBOARD_LOG")
    return Path(override) if override else PROJECT_ROOT / "reports" / "dashboard_runs.jsonl"


_SECRET = re.compile(r"gsk_[A-Za-z0-9]+|(?i:api[_-]?key)\S*")


def safe_error(error: BaseException) -> str:
    """An error for the UI and the log: type and message, key-like text removed."""
    message = _SECRET.sub("[redacted]", str(error)).strip()
    return f"{type(error).__name__}: {message}"[:300]


@dataclass(frozen=True)
class HistoryItem:
    """One dashboard event."""

    kind: str  # "live" | "replayed" | "aborted"
    when: str
    goal_text: str
    status: str
    detail: str = ""
    result: AgentResult | None = None

    @property
    def label(self) -> str:
        return HISTORY_LABELS[self.kind]


def append_history(
    path: Path,
    kind: str,
    goal_text: str,
    status: str,
    detail: str = "",
    result: AgentResult | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> HistoryItem:
    """Append one event. Every run that starts ends with exactly one event."""
    if kind not in HISTORY_LABELS:
        raise ValueError(f"unknown history kind {kind!r}")
    item = HistoryItem(
        kind=kind,
        when=now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        goal_text=goal_text,
        status=status,
        detail=detail,
        result=result,
    )
    record = {
        "kind": item.kind,
        "when": item.when,
        "goal_text": item.goal_text,
        "status": item.status,
        "detail": item.detail,
        "result": result.to_dict() if result is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return item


def read_history(path: Path) -> list[HistoryItem]:
    """Newest first. A malformed line is skipped, not fatal."""
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            raw = json.loads(line)
            items.append(
                HistoryItem(
                    kind=raw["kind"],
                    when=raw["when"],
                    goal_text=raw["goal_text"],
                    status=raw["status"],
                    detail=raw.get("detail", ""),
                    result=AgentResult.from_dict(raw["result"]) if raw.get("result") else None,
                )
            )
        except (ValueError, KeyError, TypeError):
            continue
    return items[::-1]


def read_run_log(path: Path) -> list[AgentResult]:
    """The agent's own JSONL run log (every live run, CLI included), newest first."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(AgentResult.from_dict(json.loads(line)))
        except (ValueError, KeyError, TypeError):
            continue
    return out[::-1]


# --------------------------------------------------------------------------- #
# One live run at a time
# --------------------------------------------------------------------------- #
class RunGuard:
    """A process-wide lock: the laptop runs one agent goal at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        if self._lock.locked():
            self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()


# --------------------------------------------------------------------------- #
# What-if
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SliderSpec:
    """One controllable lever's slider, from configs/twin.yaml."""

    name: str
    kind: str
    min: float
    max: float
    default: float
    description: str

    @property
    def step(self) -> float:
        if self.kind == "integer":
            return 1.0
        span = self.max - self.min
        return float(10 ** math.floor(math.log10(span / 100))) if span > 0 else 1.0


def slider_specs() -> list[SliderSpec]:
    """The 16 levers the optimiser may move, with their searchable bounds."""
    from twinmind.optim.contracts import CONTROLLABLE_DIMS
    from twinmind.utils.config import load_config

    twin: Any = load_config("twin")
    space = {d.name: d for d in twin.scenario_space}
    return [
        SliderSpec(
            name=name,
            kind=space[name].kind,
            min=float(space[name].min),
            max=float(space[name].max),
            default=float(space[name].default),
            description=space[name].description,
        )
        for name in CONTROLLABLE_DIMS
    ]


@dataclass(frozen=True)
class VerifyRun:
    """Twin replications of one what-if plan, under the caps."""

    n_reps: int
    requested: int
    seconds: float
    stopped_by_time_cap: bool
    means: dict[str, float] = field(default_factory=dict)
    lows: dict[str, float] = field(default_factory=dict)
    highs: dict[str, float] = field(default_factory=dict)

    @property
    def seconds_per_run(self) -> float:
        return self.seconds / self.n_reps if self.n_reps else float("nan")

    @property
    def slower_than_quiet(self) -> bool:
        return self.seconds_per_run > TWIN_SECONDS_PER_RUN_QUIET


def verify_on_twin(
    run_one: Callable[[int], Sequence[float]],
    seeds: Sequence[int],
    kpi_names: Sequence[str],
    requested: int,
    time_cap_seconds: float = WHATIF_TIME_CAP_SECONDS,
    clock: Callable[[], float] = time.perf_counter,
) -> VerifyRun:
    """Run up to ``requested`` (<= 10) replications, stopping at the time cap.

    Never fails for being slow: it stops early, after at least two runs, and
    reports how many it managed and how long they took. Intervals are 95%
    t-intervals over the runs completed.
    """
    from scipy import stats

    requested = max(WHATIF_MIN_REPS, min(int(requested), WHATIF_MAX_REPS, len(seeds)))
    rows: list[Sequence[float]] = []
    started = clock()
    stopped = False
    seconds = 0.0
    for seed in seeds[:requested]:
        rows.append(run_one(seed))
        seconds = clock() - started
        if len(rows) >= WHATIF_MIN_REPS and seconds > time_cap_seconds:
            stopped = len(rows) < requested
            break
    n = len(rows)
    means, lows, highs = {}, {}, {}
    critical = float(stats.t.ppf(0.975, df=n - 1))
    for index, kpi in enumerate(kpi_names):
        values = [float(r[index]) for r in rows]
        mean = sum(values) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = critical * sd / math.sqrt(n)
        means[kpi], lows[kpi], highs[kpi] = mean, mean - half, mean + half
    return VerifyRun(
        n_reps=n,
        requested=requested,
        seconds=seconds,
        stopped_by_time_cap=stopped,
        means=means,
        lows=lows,
        highs=highs,
    )


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
CARDS_DIR = PROJECT_ROOT / "reports" / "model_cards"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

#: The note B5 must always carry, wherever it is shown.
B5_NOTE = (
    "B5 was measured with explanation checker v1 and the original facts format. "
    "The live agent now uses checker v2 and plain-decimal facts, which have NOT been "
    "benchmarked. B5 stands as registered: FAIL, 35/40."
)


@dataclass(frozen=True)
class BenchRow:
    """One pre-registered hypothesis and its result, PASS or FAIL alike."""

    id: str
    claim: str
    verdict: str
    deciding: str


def benchmark_rows(analysis: Mapping[str, Any]) -> list[BenchRow]:
    """B1-B5 from reports/results/agent_bench_analysis.json, all five, in order."""
    b1, b2, b3, b4, b5 = (analysis[k] for k in ("B1", "B2", "B3", "B4", "B5"))
    inf_ok, room_ok = b4["infeasible_ok"], b4["no_room_ok"]
    explainer = b5["explainer"]
    return [
        BenchRow(
            "B1",
            "The agent is non-inferior to twin-only on normalised gap (bound +0.10)",
            b1["verdict"],
            f"mean(agent − twin-only) = {b1['mean']:.3f}, 95% CI {b1['ci95'][0]:.3f} to "
            f"{b1['ci95'][1]:.3f}, {b1['n']} goals",
        ),
        BenchRow(
            "B2",
            "Recommended plans are robustly feasible at least as often as twin-only's",
            b2["verdict"],
            f"agent {b2['agent']['robust']}/{b2['agent']['recommended']} = "
            f"{100 * b2['agent']['rate']:.1f}% against twin-only "
            f"{b2['twin_only']['robust']}/{b2['twin_only']['recommended']} = "
            f"{100 * b2['twin_only']['rate']:.1f}%",
        ),
        BenchRow(
            "B3",
            "The critic leaves strictly fewer bad recommendations than no critic",
            b3["verdict"] if "verdict" in b3 else _b3_verdict(b3),
            f"agent {b3['agent']['count']} against no-critic {b3['agent_nocritic']['count']}",
        ),
        BenchRow(
            "B4",
            "All 4 infeasible goals end infeasible with options; at least 2 of 3 no-room "
            "goals are flagged",
            b4["verdict"],
            f"{sum(inf_ok.values())}/{len(inf_ok)} infeasible goals declared infeasible with "
            f"options on both seeds; {sum(room_ok.values())}/{len(room_ok)} no-room goals "
            "flagged and not recommended on both seeds",
        ),
        BenchRow(
            "B5",
            "Planner spec-match at least 85% AND 100% of explanations pass the checks",
            b5["verdict"],
            f"planner {b5['planner_spec_match']['hits']}/{b5['planner_spec_match']['n']}; "
            f"explanations {explainer['llm_text_pass_within_two_attempts']}/{explainer['n']}",
        ),
    ]


def _b3_verdict(b3: Mapping[str, Any]) -> str:
    return "PASS" if b3["agent"]["count"] < b3["agent_nocritic"]["count"] else "FAIL"


@dataclass(frozen=True)
class CardInfo:
    stem: str
    title: str
    path: Path


def model_cards(cards_dir: Path = CARDS_DIR) -> list[CardInfo]:
    """Every model card, A to F then the twin, with its first heading as title."""
    out = []
    for path in sorted(cards_dir.glob("*.md")):
        first = path.read_text(encoding="utf-8").splitlines()[0]
        out.append(CardInfo(path.stem, first.lstrip("# ").strip(), path))
    return out


_SECTION = re.compile(r"^## ", re.M)


def underperformed_sections(cards_dir: Path = CARDS_DIR) -> list[tuple[str, str]]:
    """``(card title, section text)`` for every "Underperformed ..." section."""
    out = []
    for card in model_cards(cards_dir):
        text = card.path.read_text(encoding="utf-8")
        for block in _SECTION.split(text)[1:]:
            heading, _, body = block.partition("\n")
            if "underperformed" in heading.lower():
                out.append((card.title, body.strip().rstrip("-").strip()))
    return out


_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def split_markdown(text: str, base_dir: Path) -> list[tuple[str, str]]:
    """Markdown split around local images, which st.markdown cannot load.

    Returns ``("md", text)`` and ``("img", resolved path)`` parts; an image
    whose file is missing becomes its alt text.
    """
    parts: list[tuple[str, str]] = []
    last = 0
    for match in _IMAGE.finditer(text):
        parts.append(("md", text[last : match.start()]))
        path = (base_dir / match.group(2)).resolve()
        parts.append(("img", str(path)) if path.exists() else ("md", f"*{match.group(1)}*"))
        last = match.end()
    parts.append(("md", text[last:]))
    return [(kind, body) for kind, body in parts if body.strip()]
