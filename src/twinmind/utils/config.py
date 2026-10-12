"""Typed configuration loading.

Each file in ``configs/`` maps to exactly one Pydantic model. Models use
``extra="forbid"`` so that a mistyped key fails at load time rather than being
silently ignored and surfacing as a wrong number halfway through training.

``${VAR}`` and ``${VAR:-default}`` in any string value are interpolated from the
process environment (``.env`` is loaded first when python-dotenv is available).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from twinmind.utils.paths import CONFIG_DIR

# TWINMIND_SKIP_DOTENV lets a test run the real CLI with a deliberately empty
# credential environment; without it .env would be loaded and the test could not
# observe the missing-credentials path.
if not os.getenv("TWINMIND_SKIP_DOTENV"):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv is a hard dependency in practice
        pass


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #
class DownloadConfig(_Base):
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=1)
    backoff_factor: float = Field(gt=1.0)
    chunk_bytes: int = Field(gt=0)


class CMAPSSConfig(_Base):
    url: str
    subsets: list[str]
    primary_subset: str
    setting_columns: list[str]
    n_sensors: int = Field(gt=0)
    rul_cap: int = Field(gt=0)
    window_length: int = Field(gt=0)
    window_stride: int = Field(gt=0)
    val_fraction: float = Field(ge=0.0, lt=1.0)
    constant_variance_threshold: float = Field(gt=0)
    n_operating_conditions: int = Field(gt=0)
    multi_regime_subsets: list[str]

    @property
    def sensor_columns(self) -> list[str]:
        """Canonical sensor column names, sensor_1 .. sensor_N."""
        return [f"sensor_{i}" for i in range(1, self.n_sensors + 1)]

    @model_validator(mode="after")
    def _check_subsets(self) -> CMAPSSConfig:
        if self.primary_subset not in self.subsets:
            raise ValueError(f"primary_subset {self.primary_subset!r} not in subsets")
        unknown = set(self.multi_regime_subsets) - set(self.subsets)
        if unknown:
            raise ValueError(f"multi_regime_subsets not in subsets: {sorted(unknown)}")
        return self


class AI4IConfig(_Base):
    uci_id: int = Field(gt=0)
    csv_url: str
    target_column: str
    failure_mode_columns: list[str]
    feature_columns: list[str]
    categorical_columns: list[str]
    test_fraction: float = Field(ge=0.0, lt=1.0)
    stratify: bool

    @model_validator(mode="after")
    def _no_leakage(self) -> AI4IConfig:
        """The failure-mode columns must never be declared as features."""
        overlap = set(self.feature_columns) & set(self.failure_mode_columns)
        if overlap:
            raise ValueError(f"failure-mode columns leaked into features: {sorted(overlap)}")
        if self.target_column in self.feature_columns:
            raise ValueError(f"target {self.target_column!r} declared as a feature")
        return self


class FavoritaConfig(_Base):
    kaggle_competition: str
    target_column: str
    id_columns: list[str]
    date_column: str
    frequency: str
    horizon: int = Field(gt=0)
    input_size: int = Field(gt=0)
    quantiles: list[float]
    val_days: int = Field(ge=0)
    test_days: int = Field(ge=0)
    top_series: int = Field(gt=0)
    lags: list[int]
    rolling_windows: list[int]
    fourier_weekly_k: int = Field(ge=0)
    fourier_yearly_k: int = Field(ge=0)
    n_splits: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_lags(self) -> FavoritaConfig:
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("lags must all be positive; a lag of 0 would leak the target")
        if any(w <= 0 for w in self.rolling_windows):
            raise ValueError("rolling_windows must all be positive")
        return self


class SKABConfig(_Base):
    optional: bool
    base_url: str
    groups: list[str]
    label_column: str


class SplitsConfig(_Base):
    seed: int
    shuffle: bool


class DataConfig(_Base):
    raw_dir: str
    interim_dir: str
    processed_dir: str
    download: DownloadConfig
    cmapss: CMAPSSConfig
    ai4i: AI4IConfig
    favorita: FavoritaConfig
    skab: SKABConfig
    splits: SplitsConfig


# --------------------------------------------------------------------------- #
# twin.yaml
# --------------------------------------------------------------------------- #
class SimulationConfig(_Base):
    horizon_days: int = Field(gt=0)
    warmup_days: int = Field(ge=0)
    time_step_minutes: int = Field(gt=0)
    default_replications: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)


class MachineConfig(_Base):
    id: str
    name: str
    nominal_rate_units_per_hour: float = Field(gt=0)
    buffer_capacity: int = Field(ge=0)
    repair_hours_mean: float = Field(gt=0)
    repair_hours_sd: float = Field(ge=0)
    pm_hours: float = Field(ge=0)


class ShiftConfig(_Base):
    hours_per_shift: float = Field(gt=0)
    working_days_per_week: int = Field(gt=0, le=7)
    max_overtime_hours_per_day: float = Field(ge=0)


class SupplierConfig(_Base):
    id: str
    name: str
    lead_time_days_mean: float = Field(gt=0)
    lead_time_days_sd: float = Field(ge=0)
    unit_price: float = Field(gt=0)
    reliability: float = Field(gt=0.0, le=1.0)


class ScenarioDim(_Base):
    name: str
    kind: Literal["continuous", "integer", "categorical"]
    min: float
    max: float
    default: float
    description: str

    @model_validator(mode="after")
    def _check_bounds(self) -> ScenarioDim:
        if self.min > self.max:
            raise ValueError(f"dim {self.name!r}: min {self.min} > max {self.max}")
        if not self.min <= self.default <= self.max:
            raise ValueError(f"dim {self.name!r}: default {self.default} outside bounds")
        return self


class CostConfig(_Base):
    currency: str
    preventive_maintenance_per_event: float = Field(ge=0)
    corrective_maintenance_per_event: float = Field(ge=0)
    downtime_per_hour: float = Field(ge=0)
    labour_per_operator_hour: float = Field(ge=0)
    overtime_multiplier: float = Field(ge=1.0)
    holding_cost_per_unit_per_day: float = Field(ge=0)
    stockout_penalty_per_unit: float = Field(ge=0)
    late_delivery_penalty_per_order: float = Field(ge=0)
    raw_material_per_unit: float = Field(ge=0)
    base_selling_price_per_unit: float = Field(gt=0)
    energy_per_machine_hour: float = Field(ge=0)
    fixed_overhead_per_day: float = Field(ge=0)


class AssumptionEntry(_Base):
    """One declared assumption: its value, that it *is* assumed, and why.

    The ``assumed`` flag is required and validated to be ``True``. A parameter
    that is data-backed does not belong in this register, and one that is
    assumed must not be able to hide by omitting the flag -- that is the whole
    point of keeping them in one place the sensitivity study can enumerate.
    """

    value: float
    assumed: bool
    reason: str = Field(min_length=20)

    @model_validator(mode="after")
    def _must_be_assumed(self) -> AssumptionEntry:
        if not self.assumed:
            raise ValueError(
                "entries in the assumptions register must have assumed: true; "
                "move data-backed parameters out of the register"
            )
        return self


class FlowConfig(_Base):
    """Lot-based material flow."""

    lot_size: int = Field(gt=0)
    initial_raw_inventory: float = Field(ge=0)
    initial_finished_inventory: float = Field(ge=0)


class LabourConfig(_Base):
    """Staffing. Capacity-hours only; individual workers are not modelled."""

    operators_per_shift: int = Field(gt=0)


class HealthReplayConfig(_Base):
    """Which trajectories are replayed, and how warnings are produced."""

    subset: str
    model_seed: int
    checkpoint: str
    cache_path: str
    hazard_source: Literal["learned", "fixed_mtbf"]
    warning_threshold: float = Field(gt=0.0, lt=1.0)
    min_pool_engines: int = Field(gt=0)


class TwinConstraints(_Base):
    min_on_time_delivery_rate: float = Field(ge=0.0, le=1.0)
    max_stockout_probability: float = Field(ge=0.0, le=1.0)
    max_unit_cost: float = Field(gt=0)
    min_oee: float = Field(ge=0.0, le=1.0)


class TwinConfig(_Base):
    name: str
    seed: int
    simulation: SimulationConfig
    machines: list[MachineConfig] = Field(min_length=1)
    shifts: ShiftConfig
    suppliers: list[SupplierConfig] = Field(min_length=1)
    scenario_space: list[ScenarioDim] = Field(min_length=1)
    costs: CostConfig
    constraints: TwinConstraints
    assumptions: dict[str, AssumptionEntry] = Field(default_factory=dict)
    flow: FlowConfig
    labour: LabourConfig
    health_replay: HealthReplayConfig

    def assumption(self, name: str) -> float:
        """One assumed parameter's value, by register key.

        Deliberately the *only* way the simulator reads these. Going through a
        lookup that raises on an unknown key means a typo fails loudly, and it
        guarantees every number the twin treats as assumed is one the
        sensitivity study can find by enumerating the register.
        """
        if name not in self.assumptions:
            raise KeyError(
                f"unknown assumption {name!r}; the register holds {sorted(self.assumptions)}"
            )
        return self.assumptions[name].value

    @model_validator(mode="after")
    def _check_consistency(self) -> TwinConfig:
        names = [d.name for d in self.scenario_space]
        if len(set(names)) != len(names):
            raise ValueError("scenario_space contains duplicate dimension names")
        if self.simulation.warmup_days >= self.simulation.horizon_days:
            raise ValueError("warmup_days must be shorter than horizon_days")
        expected = 5 * self.costs.preventive_maintenance_per_event
        if abs(self.costs.corrective_maintenance_per_event - expected) > 1e-6:
            raise ValueError(
                "corrective maintenance cost must be exactly 5x preventive "
                f"(expected {expected}, got {self.costs.corrective_maintenance_per_event})"
            )
        return self

    @property
    def dim_names(self) -> list[str]:
        """Canonical scenario-vector order."""
        return [d.name for d in self.scenario_space]

    @property
    def n_dims(self) -> int:
        """Dimensionality of the scenario space."""
        return len(self.scenario_space)


# --------------------------------------------------------------------------- #
# models.yaml
# --------------------------------------------------------------------------- #
class HealthModelConfig(_Base):
    architecture: str
    input_window: int = Field(gt=0)
    conv_channels: list[int]
    kernel_size: int = Field(gt=0)
    lstm_hidden: int = Field(gt=0)
    lstm_layers: int = Field(gt=0)
    bidirectional: bool
    dropout: float = Field(ge=0.0, lt=1.0)
    head_hidden: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    early_stopping_patience: int = Field(ge=0)
    rul_loss: str
    failure_loss: str
    failure_loss_weight: float = Field(ge=0)
    mc_dropout_samples: int = Field(gt=0)
    checkpoint_name: str


class DemandModelConfig(_Base):
    models: list[str]
    horizon: int = Field(gt=0)
    input_size: int = Field(gt=0)
    quantiles: list[float]
    loss: str
    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    windows_batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    early_stop_patience_steps: int = Field(ge=0)
    val_check_steps: int = Field(gt=0)
    scaler_type: str
    checkpoint_dir: str


class SurrogateModelConfig(_Base):
    architecture: str
    input_dim: int = Field(gt=0)
    output_dim: int = Field(gt=0)
    hidden_dims: list[int]
    residual_blocks: int = Field(gt=0)
    activation: str
    dropout: float = Field(ge=0.0, lt=1.0)
    predict_log_variance: bool
    min_log_variance: float
    max_log_variance: float
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    lr_scheduler: str
    early_stopping_patience: int = Field(ge=0)
    n_rollouts: int = Field(gt=0)
    val_fraction: float = Field(ge=0.0, lt=1.0)
    test_fraction: float = Field(ge=0.0, lt=1.0)
    target_scaler: str
    checkpoint_name: str


class ModelsConfig(_Base):
    seed: int
    device: Literal["auto", "cpu", "cuda"]
    health: HealthModelConfig
    demand: DemandModelConfig
    surrogate: SurrogateModelConfig


# --------------------------------------------------------------------------- #
# agent.yaml
# --------------------------------------------------------------------------- #
class RateLimitConfig(_Base):
    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)


class LLMConfig(_Base):
    provider: Literal["groq", "echo"]
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    api_key_env: str
    json_mode: bool
    max_parse_retries: int = Field(ge=0)
    rate_limit: RateLimitConfig
    call_log: str


class GraphConfig(_Base):
    nodes: list[str]
    max_iterations: int = Field(gt=0)
    early_stop_on_no_improvement: bool


class PlannerConfig(_Base):
    max_objectives: int = Field(gt=0)
    default_objective: str


class ExplorerConfig(_Base):
    proposals_per_iteration: int = Field(gt=0)
    surrogate_screen_top_k: int = Field(gt=0)
    perturbation_scale: float = Field(gt=0)


class EvaluatorConfig(_Base):
    twin_replications: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    objective: str
    tie_break: str


class CriticConfig(_Base):
    enforce_constraints: bool
    min_effect_size: float = Field(ge=0)
    require_non_overlapping_ci: bool


class ExplainerConfig(_Base):
    max_words: int = Field(gt=0)
    cite_kpis: bool


class MemoryConfig(_Base):
    backend: Literal["duckdb", "memory"]
    path: str
    max_episodes: int = Field(gt=0)


class AgentConfig(_Base):
    llm: LLMConfig
    graph: GraphConfig
    planner: PlannerConfig
    explorer: ExplorerConfig
    evaluator: EvaluatorConfig
    critic: CriticConfig
    explainer: ExplainerConfig
    memory: MemoryConfig


CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "data": DataConfig,
    "twin": TwinConfig,
    "models": ModelsConfig,
    "agent": AgentConfig,
}


def _interpolate(value: Any) -> Any:
    """Recursively expand env-var references in string leaves."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            env = os.getenv(var)
            if env is not None:
                return env
            if default is not None:
                return default
            raise KeyError(f"environment variable {var!r} referenced in config is not set")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_raw(path: Path) -> dict[str, Any]:
    """Read a YAML file and interpolate environment variables, without validating."""
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping at the top level")
    return dict(_interpolate(loaded))


def load_config(name: str, config_dir: Path | None = None) -> BaseModel:
    """Load ``configs/<name>.yaml`` into its Pydantic model.

    Raises ``KeyError`` for an unknown config name, ``FileNotFoundError`` if the
    file is missing, and ``pydantic.ValidationError`` if the contents are wrong.
    """
    if name not in CONFIG_MODELS:
        raise KeyError(f"unknown config {name!r}; expected one of {sorted(CONFIG_MODELS)}")
    directory = config_dir or CONFIG_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return CONFIG_MODELS[name].model_validate(load_raw(path))


def load_all(config_dir: Path | None = None) -> dict[str, BaseModel]:
    """Load and validate every known config. Useful as a startup smoke check."""
    return {name: load_config(name, config_dir) for name in CONFIG_MODELS}
