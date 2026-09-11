"""A1: CNN-BiLSTM with attention pooling and two heads, for C-MAPSS.

    [B, 30, C]
      -> Conv1d(C, 64, k=5)  + BN + GELU + Dropout
      -> Conv1d(64, 128, k=3) + BN + GELU + Dropout
      -> BiLSTM(128, 2 layers)                     -> [B, 30, 256]
      -> additive temporal attention pooling       -> [B, 256]
      -> Dropout
      -> rul_head:  Linear(256, 64) -> GELU -> Linear(64, 1)
      -> fail_head: Linear(256, 64) -> GELU -> Linear(64, 1)   (logit)

Why two heads on one trunk here, when A1 and A2 are separate models: both heads
read the same 30-cycle sensor window, and the failure label is derived from the
RUL target by thresholding at ``horizon_h``. They are two views of one quantity,
so sharing representation is principled. AI4I has no shared input space with
C-MAPSS, which is why A2 is a separate model rather than a third head.

Two details that are easy to get wrong and are tested:

* Dropout sits **before both heads** and stays active at inference for
  MC-dropout, while **BatchNorm must not**. Putting the whole model in
  ``train()`` for MC sampling corrupts the BN running statistics with test
  batches -- a silent, systematic error. :func:`enable_mc_dropout` toggles only
  ``nn.Dropout``.
* RUL is regressed on the target divided by the cap, so the MSE term and the
  focal-BCE term are on comparable scales. Predictions are un-normalised before
  any metric is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from twinmind.models.health import RULPrediction
from twinmind.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class A1Config:
    """Architecture and loss hyperparameters for A1."""

    n_features: int
    window: int = 30
    conv1_channels: int = 64
    conv2_channels: int = 128
    kernel1: int = 5
    kernel2: int = 3
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    head_hidden: int = 64
    horizon_h: int = 30
    rul_cap: int = 125
    lambda_fail: float = 0.5
    focal_gamma: float = 2.0

    @property
    def trunk_width(self) -> int:
        """BiLSTM output width: hidden size doubled for the two directions."""
        return self.lstm_hidden * 2


class AdditiveAttentionPooling(nn.Module):
    """Bahdanau-style additive attention over timesteps.

    Mean-pooling weights every cycle equally and last-step pooling throws away
    the trajectory. Attention learns where in the window the degradation signal
    lives, and returning the weights makes that inspectable -- the
    attention-over-cycles figure is produced from exactly these.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ``[B, T, D]`` to ``[B, D]``; also return ``[B, T]`` weights."""
        energy = self.score(torch.tanh(self.project(sequence))).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        pooled = torch.einsum("btd,bt->bd", sequence, weights)
        return pooled, weights


class A1Net(nn.Module):
    """The A1 network. Outputs ``(rul_normalised, fail_logit, attention)``."""

    def __init__(self, config: A1Config):
        super().__init__()
        self.config = config
        p = config.dropout

        self.conv = nn.Sequential(
            nn.Conv1d(config.n_features, config.conv1_channels, config.kernel1, padding="same"),
            nn.BatchNorm1d(config.conv1_channels),
            nn.GELU(),
            nn.Dropout(p),
            nn.Conv1d(config.conv1_channels, config.conv2_channels, config.kernel2, padding="same"),
            nn.BatchNorm1d(config.conv2_channels),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.lstm = nn.LSTM(
            config.conv2_channels,
            config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=p if config.lstm_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttentionPooling(config.trunk_width, config.head_hidden)
        self.head_dropout = nn.Dropout(p)

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(config.trunk_width, config.head_hidden),
                nn.GELU(),
                nn.Linear(config.head_hidden, 1),
            )

        self.rul_head = head()
        self.fail_head = head()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` is ``[B, T, C]``. Returns normalised RUL, failure logit, attention."""
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, C], got {tuple(x.shape)}")
        # Conv1d wants [B, C, T]; the LSTM wants [B, T, C].
        features = self.conv(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.lstm(features)
        pooled, weights = self.attention(sequence)
        pooled = self.head_dropout(pooled)
        return (
            self.rul_head(pooled).squeeze(-1),
            self.fail_head(pooled).squeeze(-1),
            weights,
        )

    def n_parameters(self) -> int:
        """Trainable parameter count, for the model card."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    """Focal binary cross-entropy.

    Down-weights examples the model already classifies confidently, so training
    signal concentrates on the hard, near-threshold cases rather than on the
    abundant obviously-healthy windows. ``alpha`` additionally reweights the
    positive class; passing the balanced value (negatives / total) is the usual
    choice and is what the trainer computes per split.
    """
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = ((1 - p_t) ** gamma) * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def a1_loss(
    rul_pred: torch.Tensor,
    fail_logit: torch.Tensor,
    rul_target: torch.Tensor,
    fail_target: torch.Tensor,
    lambda_fail: float = 0.5,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mse(rul) + lambda * focal_bce(fail)``.

    Both targets are on comparable scales -- RUL is divided by the cap upstream --
    so ``lambda_fail`` is a genuine trade-off knob rather than a units conversion.
    Returns ``(total, mse_part, focal_part)`` so the trainer can log both.
    """
    mse = nn.functional.mse_loss(rul_pred, rul_target)
    focal = focal_bce_with_logits(fail_logit, fail_target, gamma=gamma, alpha=alpha)
    return mse + lambda_fail * focal, mse, focal


# --------------------------------------------------------------------------- #
# MC dropout
# --------------------------------------------------------------------------- #
def enable_mc_dropout(model: nn.Module) -> int:
    """Put **only** ``nn.Dropout`` modules into train mode. Returns how many.

    The bug this exists to prevent: calling ``model.train()`` to activate dropout
    for MC sampling also re-enables BatchNorm's batch statistics, so every
    forward pass updates the running mean and variance with test data. The model
    is quietly corrupted, and the corruption grows with the number of MC passes.

    ``model.eval()`` first, then flip the dropout layers back on.
    """
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            count += 1
    return count


def batchnorm_layers_are_in_eval(model: nn.Module) -> bool:
    """True when every BatchNorm layer is in eval mode. Asserted after MC setup."""
    return all(
        not module.training
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )


@torch.no_grad()
def predict_with_uncertainty(
    model: A1Net,
    x: torch.Tensor,
    n_passes: int = 30,
    rul_cap: float = 125.0,
    point_estimate: Literal["deterministic", "mc_mean"] = "deterministic",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MC-dropout prediction.

    Returns ``(rul_mean, rul_std, p_fail, attention_mean)`` in cycles, with the
    RUL un-normalised by ``rul_cap``.

    ``point_estimate`` controls where the *mean* comes from, and the default is
    deliberate. Averaging stochastic forward passes is not the same as one
    deterministic pass: dropout perturbs the trunk and the heads are non-linear,
    so ``E[f(x)] != f(E[x])`` and the MC mean carries a bias.

    Measured on FD001 seed 42, the MC mean sat **+3.58 cycles above** the
    deterministic prediction, costing 0.63 RMSE (14.94 -> 15.57) and 139 NASA
    points. The bias is upward, which is the dangerous direction: it
    overestimates remaining life and schedules maintenance late, which is
    exactly what the NASA score punishes hardest.

    So the point estimate comes from a single deterministic pass with dropout
    off, and MC sampling is used only for the spread. ``"mc_mean"`` remains
    available to reproduce the comparison.

    Only dropout is ever stochastic; BatchNorm stays in eval throughout, which
    :func:`batchnorm_layers_are_in_eval` asserts before sampling begins.
    """
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")

    # The deterministic pass first, with every stochastic layer off.
    model.eval()
    deterministic_rul, deterministic_logit, deterministic_attention = model(x)
    deterministic = np.clip(deterministic_rul.cpu().numpy() * rul_cap, 0.0, rul_cap).astype(
        "float64"
    )

    enable_mc_dropout(model)
    if not batchnorm_layers_are_in_eval(model):  # pragma: no cover - guarded by a test
        raise RuntimeError("BatchNorm is in train mode during MC-dropout sampling")

    ruls, probabilities, attentions = [], [], []
    for _ in range(n_passes):
        rul, logit, attention = model(x)
        ruls.append(rul.cpu().numpy())
        probabilities.append(torch.sigmoid(logit).cpu().numpy())
        attentions.append(attention.cpu().numpy())

    stacked = np.stack(ruls) * rul_cap
    model.eval()

    mean = (
        deterministic
        if point_estimate == "deterministic"
        else np.clip(stacked.mean(axis=0), 0.0, rul_cap).astype("float64")
    )
    probability = (
        torch.sigmoid(deterministic_logit).cpu().numpy().astype("float64")
        if point_estimate == "deterministic"
        else np.stack(probabilities).mean(axis=0).astype("float64")
    )
    attention = (
        deterministic_attention.cpu().numpy().astype("float64")
        if point_estimate == "deterministic"
        else np.stack(attentions).mean(axis=0).astype("float64")
    )
    return mean, stacked.std(axis=0).astype("float64"), probability, attention


# --------------------------------------------------------------------------- #
# The HealthModel adapter the twin consumes
# --------------------------------------------------------------------------- #
class A1HealthModel:
    """Wraps a trained :class:`A1Net` as a :class:`~twinmind.models.health.HealthModel`.

    This is the only class the digital twin imports from A1. It hides the
    network, the device and the normalisation, so the twin never touches torch.
    """

    def __init__(
        self,
        net: A1Net,
        n_passes: int = 30,
        rul_cap: float = 125.0,
        horizon_h: int = 30,
        device: str = "cpu",
    ) -> None:
        self.net = net.to(torch.device(device))
        self.n_passes = n_passes
        self.rul_cap = rul_cap
        self.horizon_h = horizon_h
        self.device = torch.device(device)

    def predict(self, window: NDArray[np.floating]) -> RULPrediction:
        """Predict from one window of shape ``[T, C]``."""
        array = np.asarray(window)
        if array.ndim != 2:
            raise ValueError(f"expected a window of shape [T, C], got {array.shape}")
        return self.predict_batch(array[None, ...])[0]

    def predict_batch(self, windows: NDArray[np.floating]) -> list[RULPrediction]:
        """Predict from a batch of shape ``[B, T, C]``."""
        array = np.asarray(windows)
        if array.ndim != 3:
            raise ValueError(f"expected a batch of shape [B, T, C], got {array.shape}")

        tensor = torch.from_numpy(array.astype("float32")).to(self.device)
        mean, std, probability, _ = predict_with_uncertainty(
            self.net, tensor, n_passes=self.n_passes, rul_cap=self.rul_cap
        )
        return [
            RULPrediction(
                rul_mean=float(m),
                rul_std=float(s),
                p_fail_within_h=float(np.clip(p, 0.0, 1.0)),
                horizon_h=self.horizon_h,
            )
            for m, s, p in zip(mean, std, probability, strict=True)
        ]
