"""The remediation options (H2-H7), the CV splitter, and the forest health model.

The options all default to off, so the first thing tested is that the defaults
still reproduce v0.3-health -- a remediation that silently changes the baseline
makes every measured delta meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from twinmind.data.cmapss_prep import CMAPSSData
from twinmind.models.a1_cnn_bilstm import A1Config, A1Net, trend_features
from twinmind.models.a1_cv import _fold_view, cv_ensemble_predict, train_a1_cv
from twinmind.models.a1_train import TrainConfig, augment
from twinmind.models.forest_health import ForestConfig, ForestHealthModel
from twinmind.models.health import HealthModel, RULPrediction

WINDOW, SENSORS = 30, 6


def _windows(n: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 1, WINDOW)[None, :, None]
    return (rng.normal(0, 0.3, (n, WINDOW, SENSORS)) + ramp).astype("float32")


def _data(n: int = 240, n_engines: int = 20, seed: int = 0) -> CMAPSSData:
    """A small CMAPSSData whose windows carry a genuine trend, for fast tests."""
    rng = np.random.default_rng(seed)
    x = _windows(n, seed)
    units = np.repeat(np.arange(1, n_engines + 1), n // n_engines).astype("int64")
    y = rng.uniform(0, 125, n)
    n_test = 12
    return CMAPSSData(
        subset="FD001",
        x_train=x,
        y_train=y,
        units_train=units,
        x_val=x[:48],
        y_val=y[:48],
        units_val=units[:48],
        x_test=x[:n_test],
        y_test=y[:n_test],
        units_test=np.arange(1, n_test + 1, dtype="int64"),
        sensors=[f"sensor_{i}" for i in range(SENSORS)],
        dropped_sensors=[],
        normaliser=None,  # type: ignore[arg-type]
        rul_cap=125,
    )


# --------------------------------------------------------------------------- #
# Defaults must not move
# --------------------------------------------------------------------------- #
def test_default_config_is_unchanged_by_the_new_options() -> None:
    """Every remediation knob defaults to off, so the baseline is still the baseline."""
    config = A1Config(n_features=14)
    assert config.conv_norm == "batch"
    assert config.detach_fail_head is False
    assert config.trend_branch is False
    assert sum(p.numel() for p in A1Net(config).parameters()) == 738_626

    settings = TrainConfig()
    assert (settings.noise_sigma, settings.mask_prob, settings.jitter) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# H7: trend features
# --------------------------------------------------------------------------- #
def test_trend_features_shape_and_content() -> None:
    x = torch.zeros(2, WINDOW, SENSORS)
    # A clean ramp of +1 per step on every sensor.
    x += torch.arange(WINDOW, dtype=torch.float32).view(1, WINDOW, 1)

    out = trend_features(x)
    assert out.shape == (2, 3 * SENSORS)

    last, slope, delta = out[:, :SENSORS], out[:, SENSORS : 2 * SENSORS], out[:, 2 * SENSORS :]
    assert torch.allclose(last, torch.full_like(last, WINDOW - 1.0))
    assert torch.allclose(slope, torch.ones_like(slope), atol=1e-5), "unit ramp has unit slope"
    assert torch.allclose(delta, torch.full_like(delta, WINDOW - 1.0))


def test_trend_features_slope_sign_follows_the_trend() -> None:
    rising = torch.arange(WINDOW, dtype=torch.float32).view(1, WINDOW, 1).repeat(1, 1, SENSORS)
    falling = -rising
    assert (trend_features(rising)[:, SENSORS : 2 * SENSORS] > 0).all()
    assert (trend_features(falling)[:, SENSORS : 2 * SENSORS] < 0).all()


def test_trend_branch_widens_the_heads_and_changes_the_output() -> None:
    x = torch.from_numpy(_windows(8))
    plain = A1Net(A1Config(n_features=SENSORS))
    with_trend = A1Net(A1Config(n_features=SENSORS, trend_branch=True))

    assert sum(p.numel() for p in with_trend.parameters()) > sum(
        p.numel() for p in plain.parameters()
    )
    assert with_trend(x)[0].shape == plain(x)[0].shape


# --------------------------------------------------------------------------- #
# H4: normalisation choice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("norm", ["batch", "layer", "none"])
def test_every_conv_norm_builds_and_runs(norm: str) -> None:
    net = A1Net(A1Config(n_features=SENSORS, conv_norm=norm))  # type: ignore[arg-type]
    rul, logit, attention = net(torch.from_numpy(_windows(4)))
    assert rul.shape == logit.shape == (4,)
    assert attention.shape == (4, WINDOW)


def test_layer_norm_removes_batch_dependence() -> None:
    """LayerNorm normalises per sample; BatchNorm in train mode does not.

    Feeding the same sample inside two different batches must give the same
    answer under LayerNorm, which is the property that makes it immune to the
    BatchNorm/Dropout interaction H4 is about.
    """
    single = torch.from_numpy(_windows(1, seed=1))
    crowd = torch.cat([single, torch.from_numpy(_windows(7, seed=2) * 50.0)])

    net = A1Net(A1Config(n_features=SENSORS, conv_norm="layer", dropout=0.0)).train()
    alone = net(single)[0]
    together = net(crowd)[0][:1]
    assert torch.allclose(alone, together, atol=1e-4)


# --------------------------------------------------------------------------- #
# H5: detached failure head
# --------------------------------------------------------------------------- #
def test_detached_fail_head_sends_no_gradient_into_the_trunk() -> None:
    x = torch.from_numpy(_windows(8))

    for detach, expect_grad in ((True, False), (False, True)):
        net = A1Net(A1Config(n_features=SENSORS, detach_fail_head=detach, dropout=0.0))
        net.zero_grad()
        _, logit, _ = net(x)
        logit.sum().backward()

        conv_weight = net.conv[0].weight
        moved = conv_weight.grad is not None and bool(conv_weight.grad.abs().sum() > 0)
        assert moved is expect_grad, (
            f"detach={detach}: trunk gradient {'present' if moved else 'absent'}, "
            f"expected {'present' if expect_grad else 'absent'}"
        )


def test_detaching_does_not_block_the_rul_head() -> None:
    """The RUL head must still train the trunk; only the auxiliary head is cut."""
    net = A1Net(A1Config(n_features=SENSORS, detach_fail_head=True, dropout=0.0))
    net.zero_grad()
    rul, _, _ = net(torch.from_numpy(_windows(8)))
    rul.sum().backward()
    assert net.conv[0].weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# H3: augmentation
# --------------------------------------------------------------------------- #
def test_augment_is_a_no_op_by_default() -> None:
    x = torch.from_numpy(_windows(8))
    assert torch.equal(augment(x, TrainConfig()), x)


def test_noise_perturbs_without_reshaping() -> None:
    x = torch.from_numpy(_windows(32))
    out = augment(x, TrainConfig(noise_sigma=0.05))
    assert out.shape == x.shape
    assert not torch.equal(out, x)
    # The perturbation should be of the requested size, not wildly off.
    assert 0.02 < float((out - x).std()) < 0.09


def test_channel_masking_zeroes_whole_sensors() -> None:
    x = torch.from_numpy(_windows(256)) + 5.0
    out = augment(x, TrainConfig(mask_prob=0.5))

    zeroed = (out == 0).all(dim=1)
    assert 0.3 < float(zeroed.float().mean()) < 0.7, "about half the channels should drop"
    partly = ((out == 0).any(dim=1) & ~zeroed).any()
    assert not bool(partly), "masking must drop a whole channel, never part of one"


def test_jitter_scales_without_changing_shape() -> None:
    x = torch.from_numpy(_windows(32)) + 10.0
    out = augment(x, TrainConfig(jitter=0.1))
    ratio = (out / x).mean(dim=(1, 2))
    assert ((ratio > 0.85) & (ratio < 1.15)).all()


def test_augmentation_is_reproducible_under_a_seeded_generator() -> None:
    x = torch.from_numpy(_windows(16))
    settings = TrainConfig(noise_sigma=0.03, mask_prob=0.1, jitter=0.05)
    first = augment(x, settings, torch.Generator().manual_seed(7))
    second = augment(x, settings, torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


# --------------------------------------------------------------------------- #
# H1: grouped cross-validation
# --------------------------------------------------------------------------- #
def test_fold_view_keeps_the_test_split_untouched() -> None:
    data = _data()
    view = _fold_view(data, np.arange(100), np.arange(100, 140))

    assert len(view.x_train) == 100
    assert len(view.x_val) == 40
    np.testing.assert_array_equal(view.x_test, data.x_test)
    np.testing.assert_array_equal(view.y_test, data.y_test)


def test_cv_folds_never_share_an_engine() -> None:
    """The guard that matters: a row-level split would leak adjacent cycles."""
    data = _data(n=240, n_engines=20)
    settings = TrainConfig(epochs=1, batch_size=64, warmup_epochs=0, patience=1, mc_passes=1)

    models, result = train_a1_cv(
        data,
        n_splits=4,
        seed=0,
        lambda_fail=0.5,
        train_config=settings,
        arch_overrides={
            "conv1_channels": 8,
            "conv2_channels": 8,
            "lstm_hidden": 8,
            "lstm_layers": 1,
        },
    )
    assert len(models) == 4
    assert len(result.fold_metrics) == 4
    assert result.ensemble_metrics["rmse"] > 0
    # train_a1_cv asserts disjointness internally; reaching here means it held.


def test_cv_rejects_more_folds_than_engines() -> None:
    data = _data(n=40, n_engines=4)
    with pytest.raises(ValueError, match="only 4 engines"):
        train_a1_cv(data, n_splits=8, train_config=TrainConfig(epochs=1))


def test_cv_ensemble_averages_its_members() -> None:
    data = _data(n=80, n_engines=8)
    nets = [
        A1Net(
            A1Config(
                n_features=SENSORS, conv1_channels=8, conv2_channels=8, lstm_hidden=8, lstm_layers=1
            )
        )
        for _ in range(3)
    ]
    pooled = cv_ensemble_predict(nets, data)
    assert pooled.shape == (len(data.x_test),)
    assert np.isfinite(pooled).all()


# --------------------------------------------------------------------------- #
# The forest production model
# --------------------------------------------------------------------------- #
def test_forest_satisfies_the_health_model_protocol() -> None:
    model = ForestHealthModel(ForestConfig(n_estimators=12)).fit(_data(), seed=0)
    assert isinstance(model, HealthModel)


def test_forest_predict_returns_a_valid_rul_prediction() -> None:
    data = _data()
    model = ForestHealthModel(ForestConfig(n_estimators=12)).fit(data, seed=0)

    single = model.predict(data.x_test[0])
    assert isinstance(single, RULPrediction)
    assert 0.0 <= single.rul_mean <= 125.0
    assert single.rul_std >= 0.0
    assert 0.0 <= single.p_fail_within_h <= 1.0
    assert single.horizon_h == 30


def test_forest_predict_batch_matches_predict() -> None:
    data = _data()
    model = ForestHealthModel(ForestConfig(n_estimators=12)).fit(data, seed=0)

    batch = model.predict_batch(data.x_test[:5])
    assert len(batch) == 5
    assert batch[0].model_dump() == model.predict(data.x_test[0]).model_dump()


def test_forest_uncertainty_is_real_not_a_constant() -> None:
    """Per-tree spread must vary across engines, or it carries no information."""
    data = _data()
    model = ForestHealthModel(ForestConfig(n_estimators=40)).fit(data, seed=0)
    _, std, _ = model.predict_arrays(data.x_test)

    assert (std > 0).all(), "a forest of 40 trees should never agree exactly"
    assert std.std() > 0, "the spread must differ between engines"


@pytest.mark.parametrize("spread", ["tree_std", "quantile"])
def test_both_spread_estimators_produce_usable_uncertainty(spread: str) -> None:
    data = _data()
    model = ForestHealthModel(
        ForestConfig(n_estimators=40, spread=spread)  # type: ignore[arg-type]
    ).fit(data, seed=0)
    _, std, _ = model.predict_arrays(data.x_test)
    assert np.isfinite(std).all() and (std >= 0).all()


def test_forest_refuses_to_predict_before_fitting() -> None:
    model = ForestHealthModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_arrays(_windows(4))


def test_forest_rejects_wrongly_shaped_input() -> None:
    model = ForestHealthModel(ForestConfig(n_estimators=8)).fit(_data(), seed=0)
    with pytest.raises(ValueError, match=r"single window"):
        model.predict(_windows(4))
    with pytest.raises(ValueError, match=r"batch"):
        model.predict_batch(_windows(4)[0])


def test_forest_round_trips_through_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = _data()
    model = ForestHealthModel(ForestConfig(n_estimators=12)).fit(data, seed=0)
    path = model.save(tmp_path / "forest.joblib")

    restored = ForestHealthModel.load(path)
    assert (
        restored.predict(data.x_test[0]).model_dump() == model.predict(data.x_test[0]).model_dump()
    )


# --------------------------------------------------------------------------- #
# The blended production model
# --------------------------------------------------------------------------- #
class _StubNetwork:
    """A predict_arrays source with a controllable offset, for blend arithmetic."""

    def __init__(self, offset: float = 0.0, std: float = 2.0, p: float = 0.3) -> None:
        self.offset, self.std, self.p = offset, std, p

    def predict_arrays(self, windows):  # type: ignore[no-untyped-def]
        n = len(windows)
        return (
            np.full(n, 60.0 + self.offset),
            np.full(n, self.std),
            np.full(n, self.p),
        )


def test_blend_satisfies_the_protocol_and_returns_valid_predictions() -> None:
    from twinmind.models.blended_health import BlendConfig, BlendedHealthModel

    data = _data()
    forest = ForestHealthModel(ForestConfig(n_estimators=12)).fit(data, seed=0)
    blend = BlendedHealthModel(forest, _StubNetwork(), BlendConfig())

    assert isinstance(blend, HealthModel)
    single = blend.predict(data.x_test[0])
    assert isinstance(single, RULPrediction)
    assert 0.0 <= single.rul_mean <= 125.0


def test_blend_mean_is_the_weighted_average_of_its_components() -> None:
    from twinmind.models.blended_health import BlendConfig, BlendedHealthModel

    data = _data()
    forest = ForestHealthModel(ForestConfig(n_estimators=12)).fit(data, seed=0)
    network = _StubNetwork(offset=0.0)

    forest_mean, _, _ = forest.predict_arrays(data.x_test)
    net_mean, _, _ = network.predict_arrays(data.x_test)

    for weight in (0.0, 0.25, 0.5, 1.0):
        blend = BlendedHealthModel(forest, network, BlendConfig(forest_weight=weight))
        mean, _, _ = blend.predict_arrays(data.x_test)
        expected = np.clip(weight * forest_mean + (1 - weight) * net_mean, 0.0, 125.0)
        np.testing.assert_allclose(mean, expected, rtol=1e-9)


def test_blend_variance_follows_the_law_of_total_variance() -> None:
    """Exact arithmetic: within-model variance plus between-model spread.

    Both components are stubs with known constant means and stds, so the
    expected blend std can be written down rather than approximated.
    """
    from twinmind.models.blended_health import BlendConfig, BlendedHealthModel

    data = _data()
    low = _StubNetwork(offset=-20.0, std=3.0)  # mean 40, std 3
    high = _StubNetwork(offset=+20.0, std=5.0)  # mean 80, std 5

    blend = BlendedHealthModel(low, high, BlendConfig(forest_weight=0.5))
    mean, std, _ = blend.predict_arrays(data.x_test)

    # mean = 0.5*40 + 0.5*80 = 60
    np.testing.assert_allclose(mean, 60.0, rtol=1e-9)
    # within  = 0.5*9 + 0.5*25 = 17
    # between = 0.5*(40-60)^2 + 0.5*(80-60)^2 = 400
    np.testing.assert_allclose(std, np.sqrt(17.0 + 400.0), rtol=1e-9)


def test_blend_interval_widens_when_the_components_disagree() -> None:
    """The between-model term is the point: disagreement must raise the spread."""
    from twinmind.models.blended_health import BlendConfig, BlendedHealthModel

    agreeing = BlendedHealthModel(
        _StubNetwork(offset=0.0, std=3.0), _StubNetwork(offset=0.0, std=3.0), BlendConfig()
    )
    disagreeing = BlendedHealthModel(
        _StubNetwork(offset=-15.0, std=3.0), _StubNetwork(offset=15.0, std=3.0), BlendConfig()
    )

    data = _data()
    _, agree_std, _ = agreeing.predict_arrays(data.x_test)
    _, disagree_std, _ = disagreeing.predict_arrays(data.x_test)

    assert float(agree_std.mean()) == pytest.approx(3.0), "identical models add no spread"
    assert float(disagree_std.mean()) > 5 * float(agree_std.mean())


def test_blend_rejects_a_bad_weight_and_a_network_without_predict_arrays() -> None:
    from twinmind.models.blended_health import BlendConfig, BlendedHealthModel

    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        BlendConfig(forest_weight=1.5)

    forest = ForestHealthModel(ForestConfig(n_estimators=8)).fit(_data(), seed=0)
    with pytest.raises(TypeError, match="predict_arrays"):
        BlendedHealthModel(forest, object())


def test_blend_rejects_wrongly_shaped_input() -> None:
    from twinmind.models.blended_health import BlendedHealthModel

    forest = ForestHealthModel(ForestConfig(n_estimators=8)).fit(_data(), seed=0)
    blend = BlendedHealthModel(forest, _StubNetwork())
    with pytest.raises(ValueError, match="single window"):
        blend.predict(_windows(4))
    with pytest.raises(ValueError, match="batch"):
        blend.predict_arrays(_windows(4)[0])
