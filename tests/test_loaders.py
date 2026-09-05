"""Loaders, the AI4I leakage separation, and the download manifest logic.

The AI4I leakage tests are the ones that matter: that dataset is easy to score
1.0 on and learn nothing from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from twinmind.data.download import (
    build_manifest,
    manifest_is_satisfied,
    read_manifest,
    sha256_file,
    write_manifest,
)
from twinmind.data.loaders import AI4I_ALIASES, cmapss_columns, normalise_column
from twinmind.utils.config import AI4IConfig, CMAPSSConfig


# --------------------------------------------------------------------------- #
# Column normalisation
# --------------------------------------------------------------------------- #
def test_normalise_column_folds_units_into_the_name() -> None:
    assert normalise_column("Air temperature [K]") == "air_temperature_k"
    assert normalise_column("Rotational speed [rpm]") == "rotational_speed_rpm"
    assert normalise_column("Machine failure") == "machine_failure"
    assert normalise_column("TWF") == "twf"
    assert normalise_column("  Tool wear [min] ") == "tool_wear_min"


def test_both_upstream_spellings_reach_the_same_canonical_name() -> None:
    """ucimlrepo strips the unit suffixes that the UCI CSV carries."""
    for bare, canonical in AI4I_ALIASES.items():
        assert normalise_column(bare) == bare
        assert AI4I_ALIASES[normalise_column(bare)] == canonical
    assert normalise_column("Air temperature [K]") == AI4I_ALIASES["air_temperature"]


def test_cmapss_columns_are_26_wide(cmapss_config: CMAPSSConfig) -> None:
    columns = cmapss_columns(cmapss_config)
    assert len(columns) == 2 + 3 + 21
    assert columns[:2] == ["unit", "cycle"]
    assert columns[-1] == "sensor_21"


# --------------------------------------------------------------------------- #
# AI4I leakage -- config level (runs without data)
# --------------------------------------------------------------------------- #
def test_config_keeps_failure_modes_out_of_the_features(ai4i_config: AI4IConfig) -> None:
    assert not set(ai4i_config.feature_columns) & set(ai4i_config.failure_mode_columns)
    assert ai4i_config.target_column not in ai4i_config.feature_columns


def test_config_rejects_a_leaked_feature_column(ai4i_config: AI4IConfig) -> None:
    """The guard is a validator, so a future edit to data.yaml cannot reintroduce this."""
    with pytest.raises(ValueError, match="leaked into features"):
        ai4i_config.model_copy(
            update={"feature_columns": [*ai4i_config.feature_columns, "twf"]}
        ).model_validate(
            {
                **ai4i_config.model_dump(),
                "feature_columns": [*ai4i_config.feature_columns, "twf"],
            }
        )


def test_config_rejects_the_target_as_a_feature(ai4i_config: AI4IConfig) -> None:
    with pytest.raises(ValueError, match="declared as a feature"):
        AI4IConfig.model_validate(
            {
                **ai4i_config.model_dump(),
                "feature_columns": [*ai4i_config.feature_columns, ai4i_config.target_column],
            }
        )


# --------------------------------------------------------------------------- #
# AI4I leakage -- against the real file
# --------------------------------------------------------------------------- #
@pytest.mark.data
def test_ai4i_feature_frame_contains_no_failure_modes(ai4i_config: AI4IConfig) -> None:
    """THE leakage test. TWF/HDF/PWF/OSF/RNF OR together into the target.

    A model handed those five columns reports ~1.0 AUC and has learned to read
    the answer key rather than the process physics.
    """
    from twinmind.data.loaders import load_ai4i

    features, target, modes = load_ai4i()

    for mode in ai4i_config.failure_mode_columns:
        assert mode not in features.columns, f"{mode} leaked into the feature frame"
    assert ai4i_config.target_column not in features.columns
    assert set(modes.columns) == set(ai4i_config.failure_mode_columns)
    assert len(features) == len(target) == len(modes)


@pytest.mark.data
def test_ai4i_shape_and_class_balance() -> None:
    from twinmind.data.loaders import load_ai4i

    features, target, modes = load_ai4i()

    assert len(features) == 10_000, "AI4I 2020 is a 10,000-row dataset"
    assert set(target.unique()) == {0, 1}
    assert 0.02 < target.mean() < 0.05, "roughly 3.4% of rows are failures"
    assert (modes.sum(axis=1) >= 0).all()


@pytest.mark.data
def test_ai4i_modes_really_do_reconstruct_the_target() -> None:
    """Demonstrates why the separation exists, rather than just asserting it."""
    from twinmind.data.loaders import load_ai4i

    _, target, modes = load_ai4i()
    agreement = (modes.max(axis=1) == target).mean()
    assert agreement > 0.99, (
        f"expected the OR of the failure modes to reproduce the target; got {agreement:.4f}"
    )


@pytest.mark.data
def test_ai4i_features_are_numeric_and_complete() -> None:
    from twinmind.data.loaders import load_ai4i

    features, _, _ = load_ai4i()
    assert features.notna().all().all()
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns)
    assert any(c.startswith("type_") for c in features.columns), "product type should be encoded"


# --------------------------------------------------------------------------- #
# C-MAPSS loader -- against the real files
# --------------------------------------------------------------------------- #
@pytest.mark.data
@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
def test_cmapss_train_shape_and_dtypes(subset: str, cmapss_config: CMAPSSConfig) -> None:
    from twinmind.data.loaders import load_cmapss

    frame = load_cmapss(subset, "train")  # type: ignore[arg-type]

    assert list(frame.columns) == cmapss_columns(cmapss_config)
    assert frame["unit"].dtype == "int64"
    assert frame["cycle"].dtype == "int64"
    assert frame[cmapss_config.sensor_columns].dtypes.eq("float64").all()
    assert frame.notna().all().all(), "the trailing empty columns should have been dropped"


@pytest.mark.data
@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
def test_cmapss_cycles_are_contiguous_from_one(subset: str) -> None:
    from twinmind.data.loaders import load_cmapss

    frame = load_cmapss(subset, "train")  # type: ignore[arg-type]
    for _, block in frame.groupby("unit"):
        cycles = block["cycle"].to_numpy()
        assert cycles[0] == 1
        assert (cycles == range(1, len(cycles) + 1)).all()


@pytest.mark.data
@pytest.mark.parametrize("subset", ["FD001", "FD002", "FD003", "FD004"])
def test_cmapss_test_split_carries_ground_truth_rul(subset: str) -> None:
    from twinmind.data.loaders import load_cmapss

    frame = load_cmapss(subset, "test")  # type: ignore[arg-type]

    assert "rul_at_last_cycle" in frame.columns
    assert "is_last_cycle" in frame.columns
    assert int(frame["is_last_cycle"].sum()) == frame["unit"].nunique()
    assert frame["rul_at_last_cycle"].ge(0).all()


@pytest.mark.data
def test_cmapss_cache_round_trips() -> None:
    from twinmind.data.loaders import load_cmapss

    fresh = load_cmapss("FD001", "train", use_cache=False)
    cached = load_cmapss("FD001", "train", use_cache=True)
    pd.testing.assert_frame_equal(fresh, cached)


@pytest.mark.data
def test_loader_raises_a_useful_error_when_data_is_missing(tmp_path: Path) -> None:
    from twinmind.data.loaders import DataNotDownloadedError

    error = DataNotDownloadedError("cmapss", tmp_path / "train_FD001.txt")
    assert "download_data.py" in str(error)
    assert "cmapss" in str(error)


# --------------------------------------------------------------------------- #
# Manifests / idempotency (no network)
# --------------------------------------------------------------------------- #
def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "f.bin"
    payload = b"twinmind" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_manifest_records_names_sizes_and_checksums(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")

    manifest = write_manifest(tmp_path, [tmp_path / "a.txt", tmp_path / "b.txt"])

    assert manifest["n_files"] == 2
    assert {e["name"] for e in manifest["files"]} == {"a.txt", "b.txt"}
    assert all(len(e["sha256"]) == 64 for e in manifest["files"])
    assert "fetched_at" in manifest

    on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_manifest_is_satisfied_when_nothing_changed(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("alpha", encoding="utf-8")
    write_manifest(tmp_path, [path])
    assert manifest_is_satisfied(tmp_path) is True


def test_manifest_detects_a_modified_file(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("alpha", encoding="utf-8")
    write_manifest(tmp_path, [path])

    path.write_text("tampered", encoding="utf-8")
    assert manifest_is_satisfied(tmp_path) is False


def test_manifest_detects_a_deleted_file(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("alpha", encoding="utf-8")
    write_manifest(tmp_path, [path])

    path.unlink()
    assert manifest_is_satisfied(tmp_path) is False


def test_missing_manifest_is_not_satisfied(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) is None
    assert manifest_is_satisfied(tmp_path) is False


def test_corrupt_manifest_is_treated_as_absent(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert read_manifest(tmp_path) is None
    assert manifest_is_satisfied(tmp_path) is False


def test_build_manifest_uses_posix_relative_names(tmp_path: Path) -> None:
    nested = tmp_path / "group" / "x.csv"
    nested.parent.mkdir()
    nested.write_text("1,2", encoding="utf-8")

    manifest = build_manifest(tmp_path, [nested])
    assert manifest["files"][0]["name"] == "group/x.csv"


@pytest.mark.data
def test_downloaded_datasets_have_valid_manifests() -> None:
    """Whatever is actually in data/raw must match its manifest."""
    from twinmind.utils.paths import raw_dir

    directories = [d for d in raw_dir().iterdir() if d.is_dir()]
    checked = [d for d in directories if (d / "manifest.json").exists()]
    assert checked, "no downloaded datasets found; run scripts/download_data.py"

    for directory in checked:
        assert manifest_is_satisfied(directory), f"{directory.name} does not match its manifest"
