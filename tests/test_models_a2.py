"""A2 feature construction, leakage guards and the run registry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twinmind.eval.registry import (
    RunRecord,
    build_summary,
    config_hash,
    log_run,
    read_runs,
    write_summary,
)
from twinmind.models.ai4i import ENGINEERED, build_features, score_predictions
from twinmind.utils.config import AI4IConfig


def _features(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "air_temperature_k": rng.normal(298, 2, n),
            "process_temperature_k": rng.normal(308, 1, n),
            "rotational_speed_rpm": rng.normal(1500, 100, n),
            "torque_nm": rng.normal(40, 10, n),
            "tool_wear_min": rng.uniform(0, 250, n),
            "type_L": rng.integers(0, 2, n),
            "type_M": rng.integers(0, 2, n),
            "type_H": rng.integers(0, 2, n),
        }
    )


def _modes(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({c: rng.integers(0, 2, n) for c in ("twf", "hdf", "pwf", "osf", "rnf")})


# --------------------------------------------------------------------------- #
# Engineered features
# --------------------------------------------------------------------------- #
def test_engineered_features_are_added(ai4i_config: AI4IConfig) -> None:
    out = build_features(_features(), ai4i_config)
    for name in ENGINEERED:
        assert name in out.columns


def test_power_is_the_physical_formula(ai4i_config: AI4IConfig) -> None:
    """power = torque * rpm * 2*pi/60, in watts."""
    frame = _features(3)
    out = build_features(frame, ai4i_config)
    expected = frame["torque_nm"] * frame["rotational_speed_rpm"] * (2 * np.pi / 60.0)
    np.testing.assert_allclose(out["power_w"], expected)


def test_temperature_difference_is_process_minus_air(ai4i_config: AI4IConfig) -> None:
    frame = _features(3)
    out = build_features(frame, ai4i_config)
    np.testing.assert_allclose(
        out["temperature_difference_k"],
        frame["process_temperature_k"] - frame["air_temperature_k"],
    )


def test_missing_source_columns_are_reported(ai4i_config: AI4IConfig) -> None:
    with pytest.raises(ValueError, match="missing"):
        build_features(_features().drop(columns=["torque_nm"]), ai4i_config)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_failure_modes_never_reach_the_feature_matrix(ai4i_config: AI4IConfig) -> None:
    out = build_features(_features(), ai4i_config, _modes())
    for mode in ai4i_config.failure_mode_columns:
        assert mode not in out.columns


def test_a_mode_column_smuggled_in_is_caught(ai4i_config: AI4IConfig) -> None:
    """The guard must fire even if a caller merges the modes in beforehand."""
    contaminated = _features().assign(twf=0)
    with pytest.raises(AssertionError, match="leaked into A2 features"):
        build_features(contaminated, ai4i_config)


def test_the_target_column_is_caught(ai4i_config: AI4IConfig) -> None:
    contaminated = _features().assign(machine_failure=0)
    with pytest.raises(AssertionError, match="target"):
        build_features(contaminated, ai4i_config)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_score_predictions_returns_the_full_metric_set() -> None:
    rng = np.random.default_rng(2)
    y = np.array([1] * 20 + [0] * 180)
    p = np.clip(np.concatenate([rng.normal(0.7, 0.2, 20), rng.normal(0.2, 0.15, 180)]), 0, 1)

    metrics = score_predictions(y, p)
    assert set(metrics) == {"pr_auc", "roc_auc", "recall_at_p80", "brier", "ece"}
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["pr_auc"] > 0.1, "must beat the base rate"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_config_hash_is_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert len(config_hash({"a": 1})) == 12


def test_log_run_appends_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    for seed in (1, 2):
        log_run(
            RunRecord(model="m", dataset="d", subset="s", seed=seed, metrics={"rmse": float(seed)}),
            path=path,
        )

    records = read_runs(path)
    assert len(records) == 2
    assert [r["seed"] for r in records] == [1, 2]
    assert records[0]["run_id"] != records[1]["run_id"]
    assert "timestamp" in records[0] and "git_sha" in records[0]


def test_registry_is_append_only(tmp_path: Path) -> None:
    """A second write must not disturb the first record."""
    path = tmp_path / "runs.jsonl"
    log_run(RunRecord(model="a", dataset="d", subset="s", seed=1, metrics={"rmse": 1.0}), path)
    first = read_runs(path)[0]
    log_run(RunRecord(model="b", dataset="d", subset="s", seed=2, metrics={"rmse": 2.0}), path)
    assert read_runs(path)[0] == first


def test_a_truncated_final_line_does_not_break_reading(tmp_path: Path) -> None:
    """A killed Kaggle session leaves a partial line; the history must survive it."""
    path = tmp_path / "runs.jsonl"
    log_run(RunRecord(model="a", dataset="d", subset="s", seed=1, metrics={"rmse": 1.0}), path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"model": "b", "seed": 2, "metr')

    records = read_runs(path)
    assert len(records) == 1
    assert records[0]["model"] == "a"


def test_missing_registry_reads_as_empty(tmp_path: Path) -> None:
    assert read_runs(tmp_path / "absent.jsonl") == []


def test_summary_groups_by_subset_and_model_with_mean_std() -> None:
    records = [
        {"model": "a1", "dataset": "cmapss", "subset": "FD001", "seed": s, "metrics": {"rmse": v}}
        for s, v in [(1, 12.0), (2, 14.0)]
    ] + [
        {
            "model": "ridge",
            "dataset": "cmapss",
            "subset": "FD001",
            "seed": 1,
            "metrics": {"rmse": 15.0},
        }
    ]

    summary = build_summary(records)
    assert "cmapss / FD001" in summary
    assert "a1" in summary and "ridge" in summary
    assert "13.00 +/- 1.00" in summary, "two seeds should render as mean +/- std"
    assert "| 2 |" in summary, "the seed count must be visible"


def test_summary_of_no_runs_is_still_valid_markdown() -> None:
    assert "No runs recorded yet" in build_summary([])


def test_write_summary_creates_the_file(tmp_path: Path) -> None:
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        json.dumps(
            {"model": "m", "dataset": "d", "subset": "s", "seed": 1, "metrics": {"rmse": 1.0}}
        )
        + "\n",
        encoding="utf-8",
    )
    out = write_summary(path=tmp_path / "summary.md", runs=runs)
    assert out.exists()
    assert "d / s" in out.read_text(encoding="utf-8")
