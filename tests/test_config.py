"""Every shipped config must validate; a bad config must fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from twinmind.twin.runner import SCENARIO_DIMS
from twinmind.utils.config import CONFIG_MODELS, TwinConfig, load_config


@pytest.mark.parametrize("name", sorted(CONFIG_MODELS))
def test_every_shipped_config_loads(name: str, config_dir: Path) -> None:
    config = load_config(name, config_dir)
    assert isinstance(config, CONFIG_MODELS[name])


def test_every_yaml_file_has_a_model(config_dir: Path) -> None:
    """No config file may sit in configs/ without a schema behind it."""
    on_disk = {p.stem for p in config_dir.glob("*.yaml")}
    assert on_disk == set(CONFIG_MODELS)


def test_scenario_space_is_eighteen_dimensional(twin_config: TwinConfig) -> None:
    assert twin_config.n_dims == 18
    assert tuple(twin_config.dim_names) == SCENARIO_DIMS


def test_corrective_maintenance_is_five_times_preventive(twin_config: TwinConfig) -> None:
    costs = twin_config.costs
    assert costs.corrective_maintenance_per_event == 5 * costs.preventive_maintenance_per_event


def test_unknown_key_is_rejected(tmp_path: Path, config_dir: Path) -> None:
    """extra='forbid' means a typo fails at load time, not at 3am mid-training."""
    raw = yaml.safe_load((config_dir / "twin.yaml").read_text(encoding="utf-8"))
    raw["simulaton"] = {"horizon_days": 30}  # deliberate typo
    (tmp_path / "twin.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config("twin", tmp_path)


def test_out_of_range_value_is_rejected(tmp_path: Path, config_dir: Path) -> None:
    raw = yaml.safe_load((config_dir / "twin.yaml").read_text(encoding="utf-8"))
    raw["constraints"]["min_oee"] = 1.5
    (tmp_path / "twin.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config("twin", tmp_path)


def test_broken_cost_ratio_is_rejected(tmp_path: Path, config_dir: Path) -> None:
    raw = yaml.safe_load((config_dir / "twin.yaml").read_text(encoding="utf-8"))
    raw["costs"]["corrective_maintenance_per_event"] = 600.0
    (tmp_path / "twin.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config("twin", tmp_path)


def test_env_interpolation(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWINMIND_TEST_MEMORY_PATH", "reports/from_env.duckdb")
    raw = yaml.safe_load((config_dir / "agent.yaml").read_text(encoding="utf-8"))
    raw["memory"]["path"] = "${TWINMIND_TEST_MEMORY_PATH}"
    raw["llm"]["model"] = "${TWINMIND_TEST_MISSING:-fallback-model}"
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config("agent", tmp_path)
    assert config.memory.path == "reports/from_env.duckdb"
    assert config.llm.model == "fallback-model"


def test_unknown_config_name_raises() -> None:
    with pytest.raises(KeyError):
        load_config("does_not_exist")
