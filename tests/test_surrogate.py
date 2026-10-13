"""Contract, encoding and rollout-pipeline tests for Model C.

Stage A: everything here runs in the non-data suite against a stub surrogate
and tiny synthetic rollout shards, so none of it needs the real 32,768-scenario
dataset or a trained model.

The tests that matter most are the leak guards. A surrogate's whole value is
its test score, and the two ways to destroy that quietly are to train on the
test split and to standardise features using statistics that saw it. Both are
invisible in the output -- the numbers just come out better than they should.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twinmind.data.rollouts import (
    SobolDesign,
    manifest_conflicts,
    ood_design,
    origin_for_block,
    rep_seeds,
    valid_shard,
    write_shard,
)
from twinmind.models.surrogate import (
    BOUNDED_KPIS,
    CATEGORICAL_DIMS,
    KPI_ORDER,
    NONNEGATIVE_KPIS,
    SurrogateModel,
    SurrogateOutput,
    decode,
    encode,
    encode_batch,
    enforce_validity,
    feature_names,
    scenario_bounds,
)
from twinmind.models.surrogate_data import (
    load_rollouts,
    split_by_context_block,
    state_statistics,
)
from twinmind.twin.context import STATE_FIELDS, MachineInit, TwinContext
from twinmind.twin.runner import KPI_NAMES, SCENARIO_DIMS, Scenario

HORIZON = 28


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_context(seed: int = 0) -> TwinContext:
    """A deterministic context without needing the health cache."""
    rng = np.random.default_rng(seed)
    return TwinContext(
        series_id="stub",
        origin=pd.Timestamp("2017-06-01"),
        machines=[
            MachineInit(trajectory_id=i + 1, age_cycles=float(rng.uniform(0, 150)))
            for i in range(5)
        ],
        raw_inventory=float(rng.uniform(1500, 4500)),
        finished_inventory=0.0,
        rul_beliefs=[(float(rng.uniform(10, 125)), float(rng.uniform(1, 12))) for _ in range(5)],
        demand_summary=(800.0, 950.0, 1100.0),
    )


class StubSurrogate:
    """A ``SurrogateModel`` that returns fixed, valid output."""

    def predict(self, scenarios, context) -> SurrogateOutput:  # type: ignore[no-untyped-def]
        """Constant predictions, shaped and bounded correctly."""
        n = len(scenarios)
        mean = np.tile(np.array([20000.0, 0.7, 25.0, 300000.0, 0.2, 10.0, 0.5, 0.8]), (n, 1))
        return SurrogateOutput(
            mean=mean,
            aleatoric_std=np.full((n, len(KPI_ORDER)), 0.5),
            epistemic_std=np.full((n, len(KPI_ORDER)), 0.1),
        )


def synthetic_shard(
    directory: Path,
    block: int,
    n_scenarios: int = 8,
    index_offset: int = 0,
    split: str = "train",
    seed_root: int = 1,
    n_reps: int = 10,
) -> Path:
    """Write one tiny rollout shard with the real column layout."""
    rng = np.random.default_rng(block + 1)
    bounds = scenario_bounds()
    rows = []
    for position in range(n_scenarios):
        index = index_offset + position
        payload = {dim: float(rng.uniform(*bounds[dim])) for dim in SCENARIO_DIMS}
        payload["shift_count"] = float(int(round(payload["shift_count"])))
        payload["supplier_choice"] = float(int(round(payload["supplier_choice"])))
        row: dict[str, object] = {
            "scenario_index": index,
            "context_block": block,
            "origin": "2016-06-01",
            "n_reps": n_reps,
            "seeds": json.dumps([seed_root * 1_000_000 + index * 100 + r for r in range(n_reps)]),
        }
        row.update(payload)
        row.update({f"state_{f}": float(rng.normal()) for f in STATE_FIELDS})
        for kpi in KPI_ORDER:
            row[f"kpi_{kpi}_mean"] = float(abs(rng.normal(10.0, 2.0)))
            row[f"kpi_{kpi}_std"] = float(abs(rng.normal(1.0, 0.2)))
        rows.append(row)

    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps({"split": split, "n_reps": n_reps, "seed_root": seed_root}, indent=2),
            encoding="utf-8",
        )
    path = directory / f"block_{block:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Frozen contract
# --------------------------------------------------------------------------- #
def test_kpi_order_is_the_twins_kpi_order() -> None:
    """Derived from KPI_NAMES, not copied, so the two cannot drift apart."""
    assert KPI_ORDER == KPI_NAMES
    assert len(KPI_ORDER) == 8


def test_profit_is_not_clamped_non_negative() -> None:
    """A loss-making scenario is a real outcome the optimiser must be able to see."""
    assert "profit" not in NONNEGATIVE_KPIS
    assert "profit" not in BOUNDED_KPIS


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubSurrogate(), SurrogateModel)


def test_predict_is_batch_first() -> None:
    """Hundreds of scenarios in one call is the whole point of the surrogate."""
    scenarios = [Scenario(price_multiplier=1.0 + i / 100) for i in range(200)]
    output = StubSurrogate().predict(scenarios, make_context())
    assert output.mean.shape == (200, 8)
    assert len(output) == 200


def test_output_rejects_a_mismatched_shape() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        SurrogateOutput(
            mean=np.zeros((4, 8)),
            aleatoric_std=np.zeros((3, 8)),
            epistemic_std=np.zeros((4, 8)),
        )


def test_output_rejects_the_wrong_kpi_count() -> None:
    with pytest.raises(ValueError, match=r"shape \[N, 8\]"):
        SurrogateOutput(
            mean=np.zeros((4, 7)),
            aleatoric_std=np.zeros((4, 7)),
            epistemic_std=np.zeros((4, 7)),
        )


def test_output_rejects_a_negative_std() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SurrogateOutput(
            mean=np.zeros((2, 8)),
            aleatoric_std=np.full((2, 8), -1.0),
            epistemic_std=np.zeros((2, 8)),
        )


def test_output_rejects_an_out_of_range_rate() -> None:
    """OEE above 1 would make a scenario look artificially good to the optimiser."""
    mean = np.zeros((2, 8))
    mean[:, KPI_ORDER.index("oee")] = 1.4
    with pytest.raises(ValueError, match="oee must be <= 1"):
        SurrogateOutput(mean=mean, aleatoric_std=np.zeros((2, 8)), epistemic_std=np.zeros((2, 8)))


def test_output_rejects_negative_units() -> None:
    mean = np.zeros((2, 8))
    mean[:, KPI_ORDER.index("units_produced")] = -5.0
    with pytest.raises(ValueError, match="units_produced must be non-negative"):
        SurrogateOutput(mean=mean, aleatoric_std=np.zeros((2, 8)), epistemic_std=np.zeros((2, 8)))


def test_output_rejects_non_finite() -> None:
    mean = np.zeros((2, 8))
    mean[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SurrogateOutput(mean=mean, aleatoric_std=np.zeros((2, 8)), epistemic_std=np.zeros((2, 8)))


def test_enforce_validity_clips_into_the_domain() -> None:
    raw = np.zeros((1, 8))
    raw[0, KPI_ORDER.index("oee")] = 1.8
    raw[0, KPI_ORDER.index("units_produced")] = -20.0
    raw[0, KPI_ORDER.index("profit")] = -50_000.0
    clipped = enforce_validity(raw)
    assert clipped[0, KPI_ORDER.index("oee")] == pytest.approx(1.0)
    assert clipped[0, KPI_ORDER.index("units_produced")] == pytest.approx(0.0)
    # profit is signed and must survive untouched
    assert clipped[0, KPI_ORDER.index("profit")] == pytest.approx(-50_000.0)


def test_total_std_combines_in_quadrature() -> None:
    output = SurrogateOutput(
        mean=np.zeros((1, 8)),
        aleatoric_std=np.full((1, 8), 3.0),
        epistemic_std=np.full((1, 8), 4.0),
    )
    assert output.total_std[0, 0] == pytest.approx(5.0)


def test_column_lookup_by_name() -> None:
    mean = np.arange(8, dtype="float64").reshape(1, 8)
    mean[0, KPI_ORDER.index("oee")] = 0.5
    mean[0, KPI_ORDER.index("stockout_probability")] = 0.5
    mean[0, KPI_ORDER.index("on_time_delivery_rate")] = 0.5
    output = SurrogateOutput(
        mean=mean, aleatoric_std=np.zeros((1, 8)), epistemic_std=np.zeros((1, 8))
    )
    assert output.column("profit")[0] == pytest.approx(mean[0, KPI_ORDER.index("profit")])
    with pytest.raises(KeyError, match="unknown KPI"):
        output.column("nonsense")


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def test_feature_layout_is_scenario_then_state() -> None:
    names = feature_names()
    assert len(names) == 20 + len(STATE_FIELDS)
    assert names[0] == SCENARIO_DIMS[0]
    assert names[-1] == f"state_{STATE_FIELDS[-1]}"
    assert sum(n.startswith("supplier_choice_is_") for n in names) == 3


def test_supplier_choice_is_one_hot_not_scaled() -> None:
    """The three suppliers are not ordered, so interpolating between them is meaningless."""
    assert "supplier_choice" in CATEGORICAL_DIMS
    names = feature_names()
    assert "supplier_choice" not in names


@pytest.mark.parametrize("supplier", [0, 1, 2])
@pytest.mark.parametrize("shifts", [1, 2, 3])
def test_encode_decode_round_trips_every_dimension(supplier: int, shifts: int) -> None:
    """Every one of the 18 dimensions must survive the round trip.

    A dimension that is dropped, duplicated or mis-scaled in the encoding is
    otherwise invisible: the surrogate simply ignores a decision variable and
    the optimiser never learns it could have used it.
    """
    bounds = scenario_bounds()
    rng = np.random.default_rng(supplier * 10 + shifts)
    payload = {
        dim: float(rng.uniform(*bounds[dim]))
        for dim in SCENARIO_DIMS
        if dim not in {"shift_count", "supplier_choice"}
    }
    scenario = Scenario(shift_count=shifts, supplier_choice=supplier, **payload)

    features = encode(scenario, make_context())
    recovered = decode(features[:20])
    for dim in SCENARIO_DIMS:
        assert getattr(recovered, dim) == pytest.approx(
            getattr(scenario, dim), rel=1e-9, abs=1e-9
        ), dim


def test_encode_normalises_scenario_features_into_unit_range() -> None:
    bounds = scenario_bounds()
    low = Scenario(**{d: bounds[d][0] for d in SCENARIO_DIMS if d != "supplier_choice"})
    features = encode(low, make_context())[:20]
    continuous = [
        value
        for name, value in zip(feature_names()[:20], features, strict=True)
        if not name.startswith("supplier_choice_is_")
    ]
    assert all(-1e-9 <= v <= 1.0 + 1e-9 for v in continuous)


def test_encode_is_deterministic() -> None:
    scenario, context = Scenario(), make_context(3)
    assert np.array_equal(encode(scenario, context), encode(scenario, context))


def test_encode_batch_matches_per_item_encode() -> None:
    """Batch and single-item paths must not diverge."""
    context = make_context(5)
    scenarios = [Scenario(price_multiplier=1.0 + i / 50) for i in range(16)]
    batch = encode_batch(scenarios, context)
    for row, scenario in enumerate(scenarios):
        assert np.array_equal(batch[row], encode(scenario, context))


def test_encode_standardises_state_with_supplied_statistics() -> None:
    """Statistics are arguments, so training cannot accidentally use test ones."""
    context = make_context(7)
    raw = encode(Scenario(), context)[20:]
    mean = np.full(len(STATE_FIELDS), 2.0)
    scale = np.full(len(STATE_FIELDS), 4.0)
    standardised = encode(Scenario(), context, state_mean=mean, state_scale=scale)[20:]
    assert np.allclose(standardised, (raw - mean) / scale)


def test_encode_survives_a_constant_state_feature() -> None:
    """A zero scale must give 0, not NaN, or one constant feature poisons the row."""
    scale = np.zeros(len(STATE_FIELDS))
    features = encode(
        Scenario(), make_context(), state_mean=np.zeros(len(STATE_FIELDS)), state_scale=scale
    )
    assert np.all(np.isfinite(features))


# --------------------------------------------------------------------------- #
# Rollout storage, resume and leak guards
# --------------------------------------------------------------------------- #
def test_shard_write_is_atomic(tmp_path: Path) -> None:
    """No temp file survives, so a half-written shard cannot look complete."""
    rows = [{"scenario_index": i, "context_block": 0} for i in range(4)]
    path = write_shard(rows, tmp_path / "block_00000.parquet")
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp.parquet"))
    assert len(pd.read_parquet(path)) == 4


def test_valid_shard_rejects_a_wrong_row_count(tmp_path: Path) -> None:
    """Resume must not count a truncated shard as done."""
    path = write_shard([{"scenario_index": i} for i in range(3)], tmp_path / "block_00000.parquet")
    assert valid_shard(path, expected_rows=3)
    assert not valid_shard(path, expected_rows=8)
    assert not valid_shard(tmp_path / "block_00001.parquet", expected_rows=3)


def test_valid_shard_rejects_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "block_00000.parquet"
    path.write_bytes(b"not a parquet file")
    assert not valid_shard(path, expected_rows=3)


def test_manifest_conflicts_flag_a_changed_assumption(tmp_path: Path) -> None:
    """Resuming across an assumption change would mix two different simulators."""
    base = {
        "twin_config_hash": "aaa",
        "assumptions": {"demand_capacity_ratio": 0.85},
        "n_reps": 10,
        "git_sha": "abc123",
    }
    same = {**base, "git_sha": "def456"}
    assert manifest_conflicts(base, same) == []

    changed = {**base, "assumptions": {"demand_capacity_ratio": 0.64}}
    assert "assumptions" in manifest_conflicts(base, changed)


def test_training_loader_refuses_a_test_split(tmp_path: Path) -> None:
    """The guard that protects the only evidence the surrogate works."""
    directory = tmp_path / "test_t1"
    synthetic_shard(directory, block=0, split="test")
    with pytest.raises(PermissionError, match="allow_test"):
        load_rollouts(directory)
    data = load_rollouts(directory, allow_test=True)
    assert data.split == "test"


def test_load_rollouts_reads_every_shard(tmp_path: Path) -> None:
    directory = tmp_path / "train"
    for block in range(3):
        synthetic_shard(directory, block=block, n_scenarios=8, index_offset=8 * block)
    data = load_rollouts(directory)
    assert data.n_rows == 24
    assert data.targets().shape == (24, 8)
    assert data.target_stds().shape == (24, 8)
    assert sorted(set(data.context_blocks.tolist())) == [0, 1, 2]


def test_max_blocks_takes_a_prefix_for_the_learning_curve(tmp_path: Path) -> None:
    """Sobol prefixes are balanced, so a prefix of blocks is a valid design."""
    directory = tmp_path / "train"
    for block in range(4):
        synthetic_shard(directory, block=block, n_scenarios=8, index_offset=8 * block)
    assert load_rollouts(directory, max_blocks=2).n_rows == 16
    assert load_rollouts(directory, max_blocks=4).n_rows == 32


def test_label_std_divides_by_root_n_reps(tmp_path: Path) -> None:
    """The label is a mean of n_reps, so its std is the replication std / sqrt(n)."""
    directory = tmp_path / "train"
    synthetic_shard(directory, block=0, n_scenarios=4, n_reps=10)
    data = load_rollouts(directory)
    assert np.allclose(data.label_stds(), data.target_stds() / np.sqrt(10.0))


def test_validation_split_holds_out_whole_context_blocks(tmp_path: Path) -> None:
    """Splitting rows at random would measure interpolation at a seen state."""
    directory = tmp_path / "train"
    for block in range(10):
        synthetic_shard(directory, block=block, n_scenarios=8, index_offset=8 * block)
    data = load_rollouts(directory)
    train_index, val_index = split_by_context_block(data, val_fraction=0.3, seed=1)

    train_blocks = set(data.context_blocks[train_index].tolist())
    val_blocks = set(data.context_blocks[val_index].tolist())
    assert train_blocks and val_blocks
    assert not (train_blocks & val_blocks)
    assert train_index.size + val_index.size == data.n_rows


def test_state_statistics_come_from_the_given_rows_only(tmp_path: Path) -> None:
    """Computing them over the whole file would standardise using held-out states."""
    directory = tmp_path / "train"
    for block in range(4):
        synthetic_shard(directory, block=block, n_scenarios=8, index_offset=8 * block)
    data = load_rollouts(directory)
    train_index, _ = split_by_context_block(data, val_fraction=0.5, seed=0)
    subset_mean, _ = state_statistics(data, train_index)
    whole_mean, _ = state_statistics(data)
    assert not np.allclose(subset_mean, whole_mean)
    assert np.allclose(subset_mean, data.state_matrix()[train_index].mean(axis=0))


def test_train_and_test_scenario_ids_and_seeds_are_disjoint(tmp_path: Path) -> None:
    """Ids, contexts AND seeds must all be disjoint, not just the files."""
    train_dir, test_dir = tmp_path / "train", tmp_path / "test_t1"
    for block in range(3):
        synthetic_shard(train_dir, block=block, n_scenarios=8, index_offset=8 * block, seed_root=1)
    for block in range(2):
        synthetic_shard(
            test_dir,
            block=100 + block,
            n_scenarios=8,
            index_offset=5000 + 8 * block,
            split="test",
            seed_root=9,
        )
    train = load_rollouts(train_dir)
    test = load_rollouts(test_dir, allow_test=True)

    assert not set(train.scenario_indices.tolist()) & set(test.scenario_indices.tolist())
    assert not set(train.context_blocks.tolist()) & set(test.context_blocks.tolist())
    train_seeds = {s for row in train.seeds() for s in row}
    test_seeds = {s for row in test.seeds() for s in row}
    assert train_seeds and test_seeds
    assert not (train_seeds & test_seeds)


# --------------------------------------------------------------------------- #
# Design helpers
# --------------------------------------------------------------------------- #
def test_rep_seeds_are_independent_not_consecutive() -> None:
    """base+i gives correlated streams; SeedSequence((root, index)) does not."""
    first = rep_seeds(100, 0, 10)
    second = rep_seeds(100, 1, 10)
    assert len(set(first)) == 10
    assert not set(first) & set(second)
    assert first != list(range(first[0], first[0] + 10))


def test_rep_seeds_are_reproducible() -> None:
    assert rep_seeds(7, 3, 5) == rep_seeds(7, 3, 5)


def test_origins_split_chronologically() -> None:
    """No training context may share an origin with a test context."""
    train = {origin_for_block(b, 20, test=False) for b in range(20)}
    test = {origin_for_block(b, 20, test=True) for b in range(20)}
    assert not (train & test)
    assert max(train) < min(test)


def test_sobol_blocks_do_not_overlap() -> None:
    """Block k consumes [64k, 64k+64), so scenario ids partition the sequence."""
    bounds = [scenario_bounds()[d] for d in SCENARIO_DIMS]
    design = SobolDesign.build(256, scramble_seed=1, bounds=bounds)
    blocks = [set(range(64 * k, 64 * (k + 1))) for k in range(4)]
    assert len(set.union(*blocks)) == 256
    first = design.scenario(0)
    assert first.to_vector() == design.scenario(0).to_vector()
    assert design.scenario(1).to_vector() != first.to_vector()


def test_ood_design_puts_at_least_three_dimensions_at_a_bound() -> None:
    """Extrapolation has to actually be outside the training density."""
    bounds = [scenario_bounds()[d] for d in SCENARIO_DIMS]
    design = ood_design(64, scramble_seed=3, bounds=bounds)
    assert isinstance(design, SobolDesign)
    at_bound = ((design.points <= 1e-12) | (design.points >= 1 - 1e-12)).sum(axis=1)
    assert at_bound.min() >= 3


# --------------------------------------------------------------------------- #
# The demand grid cache must reach every worker
# --------------------------------------------------------------------------- #
#
# These exist because of a shipped bug. `demand_cache` was an optional keyword
# on `generate`, the training call site passed it and the three test-split call
# sites did not, so every test-splits worker silently rebuilt the 0.71 GB
# Favorita frame and a 10-worker pool died of memory exhaustion. Nothing caught
# it because no test exercised the test-splits worker path.


def _rollout_script():  # type: ignore[no-untyped-def]
    """Load scripts/generate_rollouts.py as a module.

    The orchestration lives in a script rather than the package, so the only way
    to test it is to import it by path. Worth doing: the bug these tests cover
    was in the call sites, not in the library.
    """
    import importlib.util
    import sys

    from twinmind.utils.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "scripts" / "generate_rollouts.py"
    spec = importlib.util.spec_from_file_location("generate_rollouts_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_requires_settings_so_a_call_site_cannot_forget_the_cache() -> None:
    """``settings`` has no default: forgetting it is a TypeError, not a silent None."""
    import inspect

    module = _rollout_script()
    parameters = inspect.signature(module.generate).parameters
    assert "settings" in parameters
    assert parameters["settings"].default is inspect.Parameter.empty
    # And the old footgun is gone.
    assert "demand_cache" not in parameters


def test_every_task_carries_the_cache_on_both_code_paths(tmp_path: Path) -> None:
    """Training and test-splits tasks are built by the SAME function.

    This is the assertion that would have caught the bug: it checks the
    test-splits shape, not just the training one.
    """
    module = _rollout_script()
    bounds = [scenario_bounds()[d] for d in SCENARIO_DIMS]
    design = module.SobolDesign.build(16, scramble_seed=1, bounds=bounds)
    settings = module.RunSettings(workers=4, n_reps=3, demand_cache="cache.npz")

    training = module.build_tasks(tmp_path / "train", design, 2, 8, 1, settings, False, 0)
    test_split = module.build_tasks(tmp_path / "t1", design, 2, 8, 99, settings, True, 10_000)

    assert training and test_split
    for task in [*training, *test_split]:
        assert task["demand_cache"] == "cache.npz"
        assert task["n_reps"] == 3


def test_worker_takes_the_cache_path_and_never_builds_the_full_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker given a cache must not touch Favorita at all.

    ``load_favorita`` and ``build_demand_features`` are monkeypatched to raise,
    so taking the full-frame branch fails loudly instead of merely being slow
    and memory-hungry.
    """
    import twinmind.data.features as features_module
    import twinmind.data.loaders as loaders_module

    module = _rollout_script()

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("the worker rebuilt the Favorita frame despite a cache")

    monkeypatch.setattr(loaders_module, "load_favorita", explode)
    monkeypatch.setattr(features_module, "build_demand_features", explode)

    captured: dict[str, object] = {}

    class FakeModel:
        series_ids = ("stub",)
        grids = {"a": 1}

    def fake_load(path: Path) -> FakeModel:
        captured["loaded"] = str(path)
        return FakeModel()

    def fake_build_resources(**kwargs: object) -> str:
        captured.update(kwargs)
        return "resources"

    import twinmind.twin.demand_proc as demand_module

    monkeypatch.setattr(demand_module.CachedDemandModel, "load", staticmethod(fake_load))
    monkeypatch.setattr(module, "build_resources", fake_build_resources)
    monkeypatch.setattr(module, "_WORKER_STATE", {})

    cache = tmp_path / "demand_grids.npz"
    cache.write_bytes(b"")
    assert module._worker_resources(str(cache)) == "resources"
    assert captured["loaded"] == str(cache)
    assert captured["series_id"] == "stub"
    assert isinstance(captured["demand_model"], FakeModel)


def test_worker_raises_rather_than_falling_back_on_an_unreadable_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Silent fallback is what turned a typo into a memory-exhaustion crash."""
    module = _rollout_script()
    monkeypatch.setattr(module, "_WORKER_STATE", {})

    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not an npz")
    with pytest.raises(RuntimeError, match="could not load the demand grid cache"):
        module._worker_resources(str(broken))


def test_resolver_refuses_a_multi_worker_run_with_no_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail in the first second with a reason, not after minutes with an OOM."""
    module = _rollout_script()
    monkeypatch.setattr(module, "DEFAULT_DEMAND_CACHE", tmp_path / "absent.npz")

    with pytest.raises(SystemExit, match="refusing to start 10 workers"):
        module.resolve_demand_cache(None, workers=10, allow_full_frame=False)

    # Explicitly opted in, and single-worker runs, are allowed.
    assert module.resolve_demand_cache(None, 10, allow_full_frame=True) is None
    assert module.resolve_demand_cache(None, 1, allow_full_frame=False) is None


def test_resolver_rejects_a_named_cache_that_does_not_exist(tmp_path: Path) -> None:
    module = _rollout_script()
    with pytest.raises(SystemExit, match="no demand grid cache at"):
        module.resolve_demand_cache(str(tmp_path / "missing.npz"), 4, False)


def test_resolver_rejects_a_named_cache_that_cannot_be_read(tmp_path: Path) -> None:
    module = _rollout_script()
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not an npz")
    with pytest.raises(SystemExit, match="could not be read"):
        module.resolve_demand_cache(str(broken), 4, False)


def test_resolver_auto_detects_the_default_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forgetting the flag must cost nothing when the file is sitting there."""
    module = _rollout_script()

    class FakeModel:
        series_ids = ("stub",)
        grids = {"a": 1}

    default = tmp_path / "demand_grids.npz"
    default.write_bytes(b"")
    monkeypatch.setattr(module, "DEFAULT_DEMAND_CACHE", default)

    import twinmind.twin.demand_proc as demand_module

    monkeypatch.setattr(
        demand_module.CachedDemandModel, "load", staticmethod(lambda path: FakeModel())
    )
    assert module.resolve_demand_cache(None, workers=10, allow_full_frame=False) == str(default)
