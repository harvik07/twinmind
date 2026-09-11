"""A1 architecture, loss, MC-dropout and the C-MAPSS preparation path.

The MC-dropout test is the important one: enabling dropout by calling
``model.train()`` also re-enables BatchNorm's batch statistics, which silently
corrupts the model with test data and gets worse the more passes you take.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from twinmind.data.cmapss_prep import (
    FEATURE_STATS,
    failure_labels,
    feature_names,
    window_features,
)
from twinmind.models.a1_cnn_bilstm import (
    A1Config,
    A1HealthModel,
    A1Net,
    a1_loss,
    batchnorm_layers_are_in_eval,
    enable_mc_dropout,
    focal_bce_with_logits,
    predict_with_uncertainty,
)
from twinmind.models.health import HealthModel, RULPrediction

WINDOW, CHANNELS = 30, 14


@pytest.fixture()
def net() -> A1Net:
    torch.manual_seed(0)
    return A1Net(A1Config(n_features=CHANNELS, window=WINDOW, dropout=0.3))


@pytest.fixture()
def batch() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(8, WINDOW, CHANNELS)


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def test_forward_shapes(net: A1Net, batch: torch.Tensor) -> None:
    rul, logit, attention = net(batch)
    assert rul.shape == (8,)
    assert logit.shape == (8,)
    assert attention.shape == (8, WINDOW), "attention is one weight per cycle"


def test_attention_weights_are_a_distribution(net: A1Net, batch: torch.Tensor) -> None:
    _, _, attention = net(batch)
    assert torch.allclose(attention.sum(dim=1), torch.ones(8), atol=1e-5)
    assert (attention >= 0).all()


def test_wrong_input_rank_is_rejected(net: A1Net) -> None:
    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        net(torch.randn(WINDOW, CHANNELS))


def test_both_heads_read_the_same_trunk(net: A1Net) -> None:
    """Two heads, one trunk -- the shared representation is the design claim."""
    trunk = {"conv", "lstm", "attention", "head_dropout"}
    assert trunk <= {name for name, _ in net.named_children()}
    assert net.rul_head is not net.fail_head
    rul_ids = {id(p) for p in net.rul_head.parameters()}
    fail_ids = {id(p) for p in net.fail_head.parameters()}
    assert not (rul_ids & fail_ids), "the heads must not share parameters"


def test_dropout_sits_before_both_heads(net: A1Net) -> None:
    assert isinstance(net.head_dropout, nn.Dropout)
    assert net.head_dropout.p == pytest.approx(0.3)


def test_window_length_is_flexible(batch: torch.Tensor) -> None:
    """Padding='same' keeps the conv output length equal to the input."""
    model = A1Net(A1Config(n_features=CHANNELS, window=50))
    rul, _, attention = model(torch.randn(4, 50, CHANNELS))
    assert rul.shape == (4,)
    assert attention.shape == (4, 50)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def test_focal_loss_downweights_easy_examples() -> None:
    """A confident correct prediction should contribute far less than an unsure one."""
    confident = focal_bce_with_logits(torch.tensor([6.0]), torch.tensor([1.0]), gamma=2.0)
    unsure = focal_bce_with_logits(torch.tensor([0.1]), torch.tensor([1.0]), gamma=2.0)
    assert confident < unsure / 50


def test_focal_reduces_to_bce_at_gamma_zero() -> None:
    logits, targets = torch.tensor([0.4, -1.2]), torch.tensor([1.0, 0.0])
    focal = focal_bce_with_logits(logits, targets, gamma=0.0)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert focal == pytest.approx(float(bce), abs=1e-6)


def test_alpha_reweights_the_positive_class() -> None:
    logits, positive = torch.tensor([0.0]), torch.tensor([1.0])
    high = focal_bce_with_logits(logits, positive, alpha=0.9)
    low = focal_bce_with_logits(logits, positive, alpha=0.1)
    assert high > low


def test_a1_loss_combines_both_terms() -> None:
    rul_pred, fail_logit = torch.tensor([0.5]), torch.tensor([0.0])
    rul_true, fail_true = torch.tensor([0.5]), torch.tensor([1.0])

    total, mse, focal = a1_loss(rul_pred, fail_logit, rul_true, fail_true, lambda_fail=0.5)
    assert mse == pytest.approx(0.0, abs=1e-7), "an exact RUL prediction has no MSE"
    assert focal > 0
    assert total == pytest.approx(float(mse) + 0.5 * float(focal), abs=1e-6)


def test_lambda_scales_only_the_failure_term() -> None:
    args = (torch.tensor([0.2]), torch.tensor([0.3]), torch.tensor([0.5]), torch.tensor([1.0]))
    low, _, focal = a1_loss(*args, lambda_fail=0.1)
    high, _, _ = a1_loss(*args, lambda_fail=1.0)
    assert high - low == pytest.approx(0.9 * float(focal), abs=1e-6)


# --------------------------------------------------------------------------- #
# MC dropout -- BatchNorm must stay in eval
# --------------------------------------------------------------------------- #
def test_enable_mc_dropout_activates_only_dropout(net: A1Net) -> None:
    net.eval()
    switched = enable_mc_dropout(net)

    assert switched > 0, "there should be dropout layers to switch on"
    assert all(m.training for m in net.modules() if isinstance(m, nn.Dropout))
    assert batchnorm_layers_are_in_eval(net), "BatchNorm leaked into train mode"


def test_batchnorm_running_stats_are_untouched_by_mc_sampling(
    net: A1Net, batch: torch.Tensor
) -> None:
    """The actual failure mode, measured: BN buffers must not move during sampling.

    If sampling ran under model.train(), the running mean and variance would be
    updated with test batches on every pass.
    """
    net.eval()
    before = {
        name: buffer.detach().clone() for name, buffer in net.named_buffers() if "running_" in name
    }
    assert before, "expected BatchNorm running buffers to exist"

    predict_with_uncertainty(net, batch, n_passes=10)

    for name, original in before.items():
        current = dict(net.named_buffers())[name]
        assert torch.equal(original, current), f"MC-dropout modified BatchNorm buffer {name}"


def test_naive_model_train_would_have_corrupted_batchnorm(net: A1Net, batch: torch.Tensor) -> None:
    """Demonstrates the bug the implementation avoids, so the guard has a reason."""
    net.train()
    before = {
        name: buffer.detach().clone()
        for name, buffer in net.named_buffers()
        if "running_mean" in name
    }
    with torch.no_grad():
        net(batch)
    changed = [
        name
        for name, original in before.items()
        if not torch.equal(original, dict(net.named_buffers())[name])
    ]
    assert changed, "a forward pass in train mode should move the BN running mean"


def test_mc_dropout_produces_genuine_variance(net: A1Net, batch: torch.Tensor) -> None:
    mean, std, probability, attention = predict_with_uncertainty(net, batch, n_passes=25)

    assert mean.shape == std.shape == probability.shape == (8,)
    assert attention.shape == (8, WINDOW)
    assert (std > 0).any(), "dropout should make repeated passes differ"
    assert (mean >= 0).all() and (mean <= 125).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_model_returns_to_eval_after_sampling(net: A1Net, batch: torch.Tensor) -> None:
    predict_with_uncertainty(net, batch, n_passes=3)
    assert not net.training
    assert all(not m.training for m in net.modules() if isinstance(m, nn.Dropout))


def test_zero_dropout_gives_zero_variance(batch: torch.Tensor) -> None:
    deterministic = A1Net(A1Config(n_features=CHANNELS, dropout=0.0))
    _, std, _, _ = predict_with_uncertainty(deterministic, batch, n_passes=5)
    assert np.allclose(std, 0.0, atol=1e-6)


def test_mc_passes_must_be_positive(net: A1Net, batch: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="n_passes"):
        predict_with_uncertainty(net, batch, n_passes=0)


# --------------------------------------------------------------------------- #
# The HealthModel adapter
# --------------------------------------------------------------------------- #
def test_adapter_satisfies_the_protocol(net: A1Net) -> None:
    assert isinstance(A1HealthModel(net), HealthModel)


def test_adapter_returns_validated_predictions(net: A1Net) -> None:
    model = A1HealthModel(net, n_passes=5, horizon_h=30)
    prediction = model.predict(np.zeros((WINDOW, CHANNELS), dtype="float32"))

    assert isinstance(prediction, RULPrediction)
    assert prediction.horizon_h == 30
    assert 0.0 <= prediction.p_fail_within_h <= 1.0
    assert prediction.rul_mean >= 0.0


def test_adapter_batch_matches_input_length(net: A1Net) -> None:
    model = A1HealthModel(net, n_passes=3)
    assert len(model.predict_batch(np.zeros((5, WINDOW, CHANNELS), dtype="float32"))) == 5


def test_adapter_rejects_wrong_rank(net: A1Net) -> None:
    model = A1HealthModel(net, n_passes=2)
    with pytest.raises(ValueError, match=r"\[T, C\]"):
        model.predict(np.zeros((4, WINDOW, CHANNELS)))


# --------------------------------------------------------------------------- #
# Window features and derived labels
# --------------------------------------------------------------------------- #
def test_window_features_shape_and_names() -> None:
    x = np.random.default_rng(0).normal(size=(12, WINDOW, 5)).astype("float32")
    features = window_features(x)

    assert features.shape == (12, 5 * len(FEATURE_STATS))
    assert len(feature_names([f"s{i}" for i in range(5)])) == features.shape[1]


def test_window_features_are_hand_checkable() -> None:
    """A single ramping sensor: last, mean, min, max and slope are all known."""
    ramp = np.arange(WINDOW, dtype="float32").reshape(1, WINDOW, 1)
    features = window_features(ramp)[0]
    last, mean, std, slope, minimum, maximum = features

    assert last == pytest.approx(WINDOW - 1)
    assert mean == pytest.approx((WINDOW - 1) / 2)
    assert minimum == pytest.approx(0.0)
    assert maximum == pytest.approx(WINDOW - 1)
    assert slope == pytest.approx(1.0), "a unit ramp has unit slope"
    assert std > 0


def test_constant_window_has_zero_slope_and_std() -> None:
    flat = np.full((1, WINDOW, 1), 7.0, dtype="float32")
    _, mean, std, slope, _, _ = window_features(flat)[0]
    assert mean == pytest.approx(7.0)
    assert std == pytest.approx(0.0)
    assert slope == pytest.approx(0.0)


def test_window_features_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\[N, T, C\]"):
        window_features(np.zeros((10, 5)))


def test_failure_labels_threshold_the_rul_target() -> None:
    """The classification label is derived from the regression target, not separate."""
    rul = np.array([0.0, 15.0, 30.0, 31.0, 125.0])
    np.testing.assert_array_equal(failure_labels(rul, 30), [1, 1, 1, 0, 0])


def test_failure_horizon_must_be_positive() -> None:
    with pytest.raises(ValueError, match="horizon"):
        failure_labels(np.array([1.0]), 0)
