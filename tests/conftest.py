"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from twinmind.utils.config import (
    AgentConfig,
    AI4IConfig,
    CMAPSSConfig,
    DataConfig,
    FavoritaConfig,
    LLMConfig,
    TwinConfig,
    load_config,
)
from twinmind.utils.paths import PROJECT_ROOT


@pytest.fixture(scope="session")
def config_dir() -> Path:
    """The repository's real configs directory."""
    return PROJECT_ROOT / "configs"


@pytest.fixture(scope="session")
def twin_config() -> TwinConfig:
    """The validated twin configuration."""
    return cast(TwinConfig, load_config("twin"))


@pytest.fixture(scope="session")
def agent_config() -> AgentConfig:
    """The validated agent configuration."""
    return cast(AgentConfig, load_config("agent"))


@pytest.fixture()
def llm_config(agent_config: AgentConfig) -> LLMConfig:
    """The LLM section, forced to the offline fake backend."""
    return agent_config.llm.model_copy(update={"provider": "fake"})


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    """The validated data configuration."""
    return cast(DataConfig, load_config("data"))


@pytest.fixture(scope="session")
def cmapss_config(data_config: DataConfig) -> CMAPSSConfig:
    """The C-MAPSS section of data.yaml."""
    return data_config.cmapss


@pytest.fixture(scope="session")
def ai4i_config(data_config: DataConfig) -> AI4IConfig:
    """The AI4I section of data.yaml."""
    return data_config.ai4i


@pytest.fixture(scope="session")
def favorita_config(data_config: DataConfig) -> FavoritaConfig:
    """The Favorita section of data.yaml."""
    return data_config.favorita
