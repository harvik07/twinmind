"""Append-only run registry: one JSON object per training run.

Deliberately not MLflow or W&B. A JSONL file is lighter, diffable in review,
survives a Kaggle session ending, needs no server, and pastes straight into the
report. Everything the dissertation's results section needs is one file:

    reports/results/runs.jsonl

Append-only is the point. A run is a fact about something that happened, so
nothing here edits or deletes a record; a superseded result is superseded by a
later record with a newer timestamp, and the history stays visible.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twinmind.eval.metrics import summarise
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT, reports_dir

logger = get_logger(__name__)

RUNS_FILENAME = "runs.jsonl"
SUMMARY_FILENAME = "summary.md"


def results_dir() -> Path:
    """``reports/results/``, created on first use."""
    path = reports_dir() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runs_path() -> Path:
    """Path to the append-only run log."""
    return results_dir() / RUNS_FILENAME


def git_sha(short: bool = True) -> str:
    """Current commit, or ``"unknown"`` outside a repository.

    Recorded so a number in the report can be traced to the exact code that
    produced it.
    """
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        result = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable 12-character hash of a configuration mapping.

    Sorted keys and a compact separator, so the same configuration always hashes
    the same regardless of dict ordering.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class RunRecord:
    """One training run. Written once, never edited."""

    model: str
    dataset: str
    subset: str
    seed: int
    metrics: dict[str, float]
    config_hash: str = ""
    duration_s: float = 0.0
    device: str = "cpu"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    git_sha: str = field(default_factory=git_sha)
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        """The record as a plain JSON-serialisable dict."""
        payload = asdict(self)
        payload["metrics"] = {k: float(v) for k, v in self.metrics.items()}
        return payload


def log_run(record: RunRecord, path: Path | None = None) -> RunRecord:
    """Append one record. Returns it, so a caller can log its run_id."""
    target = path or runs_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="" stops Python translating to CRLF on Windows. The file is
    # committed, so a mixed-line-ending hook would otherwise rewrite it under any
    # training run that is still appending.
    with target.open("a", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(record.to_json(), default=str) + "\n")
    logger.info(
        "logged run %s: %s/%s seed=%d %s",
        record.run_id,
        record.model,
        record.subset,
        record.seed,
        {k: round(v, 4) for k, v in record.metrics.items()},
    )
    return record


def read_runs(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every record. A malformed line is skipped with a warning, not fatal.

    A truncated final line is the normal consequence of a Kaggle session being
    killed mid-write, and it must not make the whole history unreadable.
    """
    target = path or runs_path()
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping malformed line %d in %s", number, target)
    return records


def current_device() -> str:
    """A short device label for the record."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001 - device detection must never break a run
        pass
    return f"cpu:{platform.processor() or platform.machine()}"[:60]


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
#: Metrics worth a column, in report order. Lower is better for all but the last group.
_PREFERRED_ORDER = [
    "rmse",
    "mae",
    "nasa_score",
    "pr_auc",
    "roc_auc",
    "recall_at_p80",
    "brier",
    "ece",
]


def _metric_order(names: Iterable[str]) -> list[str]:
    seen = list(dict.fromkeys(names))
    ranked = [m for m in _PREFERRED_ORDER if m in seen]
    return ranked + sorted(n for n in seen if n not in ranked)


def latest_config_only(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep, per (model, dataset, subset), only runs sharing its newest config hash.

    ``runs.jsonl`` is append-only on purpose -- it is the evidence trail, and
    nothing in it is ever rewritten. But that makes averaging naive: after a
    hyperparameter change the file holds runs from two different models under one
    name, and a mean across them describes neither.

    This happened concretely here. Three A1 FD001 runs were recorded at lr 1e-3
    reporting the MC-dropout mean; the point estimate was then corrected and the
    learning rate lowered. Averaging all six would have reported a model that was
    never trained.

    So the summary shows the current configuration while the history keeps
    everything. The newest run's ``config_hash`` selects its cohort.
    """
    newest: dict[tuple[str, str, str], tuple[str, str]] = {}
    for record in records:
        key = (
            str(record.get("model", "?")),
            str(record.get("dataset", "?")),
            str(record.get("subset", "?")),
        )
        stamp = str(record.get("timestamp", ""))
        if key not in newest or stamp >= newest[key][0]:
            newest[key] = (stamp, str(record.get("config_hash", "")))

    kept = []
    for record in records:
        key = (
            str(record.get("model", "?")),
            str(record.get("dataset", "?")),
            str(record.get("subset", "?")),
        )
        if str(record.get("config_hash", "")) == newest[key][1]:
            kept.append(record)
    return kept


def newest_per_seed(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated runs of the same seed to the most recent one.

    The registry records *attempts*; the summary reports *results*, and those are
    not the same thing. Re-running a seed -- because a probe run overwrote its
    checkpoint, or a machine fell over mid-sweep -- appends a second record, and
    averaging both would weight that seed twice in a mean that is supposed to be
    one number per independent training run.

    Seeding is deterministic, so a re-run of the same seed under the same config
    reproduces the same score; the duplicate carries no extra information, only
    extra weight.
    """
    newest: dict[tuple[str, str, str, Any], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("model", "?")),
            str(record.get("dataset", "?")),
            str(record.get("subset", "?")),
            record.get("seed"),
        )
        current = newest.get(key)
        if current is None or str(record.get("timestamp", "")) >= str(current.get("timestamp", "")):
            newest[key] = record
    return list(newest.values())


def build_summary(records: list[dict[str, Any]]) -> str:
    """Render runs as Markdown, grouped by dataset/subset then model.

    Every cell is mean +/- std across seeds, with the seed count shown, because a
    single-seed number is not a result. Superseded configurations are filtered out
    by :func:`latest_config_only` before anything is averaged.
    """
    if not records:
        return "# Results\n\nNo runs recorded yet. Run a training script first.\n"

    total_recorded = len(records)
    records = newest_per_seed(latest_config_only(records))

    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        key = (str(record.get("dataset", "?")), str(record.get("subset", "?")))
        grouped.setdefault(key, {}).setdefault(str(record.get("model", "?")), []).append(record)

    generated = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Results",
        "",
        f"Generated {generated} from `reports/results/{RUNS_FILENAME}`: "
        f"{len(records)} of {total_recorded} runs, keeping only the newest "
        f"configuration per model so superseded runs are never averaged in.",
        "",
        "Every cell is **mean +/- std across seeds**; `n` is the number of "
        "independent training runs behind it. C-MAPSS rows use the "
        "last-window-per-unit protocol (one prediction per test engine).",
        "",
    ]

    for (dataset, subset), by_model in sorted(grouped.items()):
        lines.append(f"## {dataset} / {subset}")
        lines.append("")

        metric_names = _metric_order(
            name for runs in by_model.values() for run in runs for name in run.get("metrics", {})
        )
        header = ["model", "n", "config", *metric_names]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")

        for model, runs in sorted(by_model.items()):
            # Show which configuration the row describes. Without it a reader
            # cannot tell that two subsets were trained under different presets,
            # which is exactly what happened when a stray baseline FD002 run
            # overlapped the production one and, being newer, won the filter.
            configs = sorted({str(run.get("config_hash", ""))[:8] for run in runs})
            row = [model, str(len(runs)), "/".join(configs) or "--"]
            for name in metric_names:
                values = [
                    float(run["metrics"][name]) for run in runs if name in run.get("metrics", {})
                ]
                if not values:
                    row.append("--")
                    continue
                stats = summarise(values)
                places = 4 if abs(stats["mean"]) < 10 else 2
                row.append(f"{stats['mean']:.{places}f} +/- {stats['std']:.{places}f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_summary(path: Path | None = None, runs: Path | None = None) -> Path:
    """Render the summary table to ``reports/results/summary.md``."""
    target = path or results_dir() / SUMMARY_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    # rstrip + a single "\n" guarantees exactly one trailing newline. Without it
    # build_summary's trailing blank line makes the end-of-file-fixer pre-commit
    # hook rewrite this file on every run and dirty an otherwise clean tree.
    body = build_summary(read_runs(runs)).rstrip("\n") + "\n"
    target.write_text(body, encoding="utf-8", newline="\n")
    logger.info("wrote %s", target)
    return target


def main() -> int:
    """Entry point for ``make results-table``."""
    path = write_summary()
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
