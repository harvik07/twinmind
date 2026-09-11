"""The append-only run registry and the summary table it renders.

The registry is the evidence trail behind every number in the report, so the
properties that matter are that it never rewrites history, that identical
configurations hash identically, and that superseded runs are not silently
averaged into current ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from twinmind.eval.registry import (
    RunRecord,
    build_summary,
    config_hash,
    latest_config_only,
    log_run,
    newest_per_seed,
    read_runs,
)


def _record(model: str, seed: int, cfg: str, stamp: str, rmse: float) -> dict[str, Any]:
    return {
        "model": model,
        "dataset": "cmapss",
        "subset": "FD001",
        "seed": seed,
        "config_hash": cfg,
        "timestamp": stamp,
        "metrics": {"rmse": rmse},
    }


# --------------------------------------------------------------------------- #
# Append-only behaviour
# --------------------------------------------------------------------------- #
def test_log_run_appends_and_never_rewrites(tmp_path: Path) -> None:
    target = tmp_path / "runs.jsonl"
    for seed in (42, 43):
        log_run(
            RunRecord(
                model="a1",
                dataset="cmapss",
                subset="FD001",
                seed=seed,
                metrics={"rmse": 14.0 + seed},
                config_hash="abc",
                duration_s=1.0,
                device="cpu",
            ),
            path=target,
        )

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, "each run must add exactly one line"
    assert [json.loads(line)["seed"] for line in lines] == [42, 43]


def test_every_record_carries_the_provenance_fields(tmp_path: Path) -> None:
    target = tmp_path / "runs.jsonl"
    log_run(
        RunRecord(
            model="a1",
            dataset="cmapss",
            subset="FD001",
            seed=42,
            metrics={"rmse": 14.0},
            config_hash="abc",
            duration_s=12.5,
            device="cpu",
        ),
        path=target,
    )
    record = read_runs(target)[0]

    for field in (
        "run_id",
        "timestamp",
        "git_sha",
        "model",
        "dataset",
        "subset",
        "seed",
        "config_hash",
        "metrics",
        "duration_s",
        "device",
    ):
        assert field in record, f"{field} missing from the run record"
    assert record["metrics"]["rmse"] == 14.0


def test_read_runs_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_runs(tmp_path / "nope.jsonl") == []


def test_read_runs_skips_a_corrupt_line(tmp_path: Path) -> None:
    """One bad line must not destroy the whole evidence trail."""
    target = tmp_path / "runs.jsonl"
    target.write_text(
        json.dumps({"model": "a1", "metrics": {}})
        + "\n{ not json\n"
        + json.dumps({"model": "ridge", "metrics": {}})
        + "\n",
        encoding="utf-8",
    )
    records = read_runs(target)
    assert [r["model"] for r in records] == ["a1", "ridge"]


# --------------------------------------------------------------------------- #
# Config hashing
# --------------------------------------------------------------------------- #
def test_config_hash_is_stable_and_order_independent() -> None:
    assert config_hash({"lr": 1e-3, "batch": 256}) == config_hash({"batch": 256, "lr": 1e-3})
    assert config_hash({"lr": 1e-3}) != config_hash({"lr": 3e-4})


# --------------------------------------------------------------------------- #
# Superseded configurations
# --------------------------------------------------------------------------- #
def test_latest_config_only_drops_superseded_runs() -> None:
    """A hyperparameter change must not be averaged together with what preceded it."""
    records = [
        _record("a1", 42, "old", "2026-01-01T00:00:00", 20.0),
        _record("a1", 43, "old", "2026-01-01T01:00:00", 21.0),
        _record("a1", 42, "new", "2026-02-01T00:00:00", 14.0),
        _record("a1", 43, "new", "2026-02-01T01:00:00", 15.0),
    ]
    kept = latest_config_only(records)

    assert len(kept) == 2
    assert {r["config_hash"] for r in kept} == {"new"}
    assert [r["metrics"]["rmse"] for r in kept] == [14.0, 15.0]


def test_latest_config_only_is_per_model_and_subset() -> None:
    """One model changing configuration must not evict another's runs."""
    records = [
        _record("a1", 42, "a1old", "2026-01-01T00:00:00", 20.0),
        _record("a1", 42, "a1new", "2026-02-01T00:00:00", 14.0),
        _record("ridge", 42, "ridge1", "2026-01-01T00:00:00", 15.0),
        _record("ridge", 43, "ridge1", "2026-01-01T01:00:00", 15.5),
    ]
    kept = latest_config_only(records)

    assert sum(r["model"] == "ridge" for r in kept) == 2, "ridge runs were wrongly dropped"
    assert sum(r["model"] == "a1" for r in kept) == 1


def test_latest_config_only_keeps_everything_when_nothing_changed() -> None:
    records = [
        _record("a1", 42, "same", "2026-01-01T00:00:00", 14.0),
        _record("a1", 43, "same", "2026-01-01T01:00:00", 15.0),
        _record("a1", 44, "same", "2026-01-01T02:00:00", 16.0),
    ]
    assert len(latest_config_only(records)) == 3


# --------------------------------------------------------------------------- #
# Summary rendering
# --------------------------------------------------------------------------- #
def test_summary_reports_how_many_runs_it_kept() -> None:
    records = [
        _record("a1", 42, "old", "2026-01-01T00:00:00", 20.0),
        _record("a1", 42, "new", "2026-02-01T00:00:00", 14.0),
    ]
    text = build_summary(records)

    assert "1 of 2 runs" in text
    assert "20.0" not in text, "a superseded run leaked into the rendered table"


def test_summary_shows_mean_and_std_with_the_seed_count() -> None:
    records = [
        _record("a1", 42, "c", "2026-01-01T00:00:00", 14.0),
        _record("a1", 43, "c", "2026-01-01T01:00:00", 16.0),
    ]
    text = build_summary(records)

    assert "| a1 |" in text
    assert "15" in text, "the mean of 14 and 16 should appear"
    assert "+/-" in text, "spread must be shown, not just a point estimate"
    assert "cmapss / FD001" in text


def test_summary_of_no_runs_is_not_an_error() -> None:
    assert "No runs recorded yet" in build_summary([])


def test_newest_per_seed_collapses_a_rerun_of_the_same_seed() -> None:
    """A re-run must not weight its seed twice in the mean."""
    records = [
        _record("a1", 42, "c", "2026-01-01T00:00:00", 20.0),
        _record("a1", 42, "c", "2026-02-01T00:00:00", 14.0),
        _record("a1", 43, "c", "2026-01-01T00:00:00", 15.0),
    ]
    kept = newest_per_seed(records)

    assert len(kept) == 2
    by_seed = {r["seed"]: r["metrics"]["rmse"] for r in kept}
    assert by_seed == {42: 14.0, 43: 15.0}, "the newer run of seed 42 should win"


def test_newest_per_seed_keeps_distinct_seeds() -> None:
    records = [
        _record("a1", s, "c", f"2026-01-0{i + 1}T00:00:00", 14.0)
        for i, s in enumerate((42, 43, 44))
    ]
    assert len(newest_per_seed(records)) == 3


def test_summary_does_not_double_count_a_rerun_seed() -> None:
    records = [
        _record("a1", 42, "c", "2026-01-01T00:00:00", 20.0),
        _record("a1", 42, "c", "2026-02-01T00:00:00", 14.0),
        _record("a1", 43, "c", "2026-01-01T00:00:00", 14.0),
    ]
    text = build_summary(records)
    assert "| a1 | 2 |" in text, "two seeds, not three records"
