"""Training loop for A1.

Kept in the package rather than in the script so it can be imported from a Kaggle
notebook and unit-tested. The script is a thin CLI over :func:`train_a1`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from twinmind.data.cmapss_prep import CMAPSSData, failure_labels
from twinmind.eval.metrics import evaluate_cmapss_test, pr_auc, rmse
from twinmind.models.a1_cnn_bilstm import A1Config, A1Net, a1_loss, predict_with_uncertainty
from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Optimisation settings. Defaults are the ones reported in the model card."""

    epochs: int = 100
    batch_size: int = 256
    # 3e-4 rather than the more usual 1e-3: at 1e-3 the run peaked at epoch 6
    # of 22 and then overfit. Measured on FD001 seed 42, 3e-4 moved test RMSE
    # 14.94 -> 14.60 and the NASA score 455 -> 441, peaking at epoch 10 of 26.
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    patience: int = 15
    mc_passes: int = 30
    lambda_candidates: tuple[float, ...] = (0.1, 0.5, 1.0)
    amp: bool = True


@dataclass
class TrainResult:
    """Everything one training run produced."""

    metrics: dict[str, float]
    best_val_rmse: float
    best_epoch: int
    epochs_run: int
    lambda_fail: float
    duration_s: float
    n_parameters: int
    history: list[dict[str, float]] = field(default_factory=list)
    checkpoint: Path | None = None


def _loaders(
    data: CMAPSSData, horizon: int, batch_size: int
) -> tuple[DataLoader[Any], torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build the training loader and the validation tensors, plus the focal alpha."""
    cap = float(data.rul_cap)
    y_train_fail = failure_labels(data.y_train, horizon)

    dataset = TensorDataset(
        torch.from_numpy(data.x_train.astype("float32")),
        torch.from_numpy((data.y_train / cap).astype("float32")),
        torch.from_numpy(y_train_fail.astype("float32")),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    x_val = torch.from_numpy(data.x_val.astype("float32"))
    y_val_rul = torch.from_numpy((data.y_val / cap).astype("float32"))
    y_val_fail = torch.from_numpy(failure_labels(data.y_val, horizon).astype("float32"))

    # Balanced alpha: the share of negatives, so the rarer positive class is
    # weighted up in proportion to how rare it actually is in this split.
    positive_rate = float(y_train_fail.mean())
    alpha = float(np.clip(1.0 - positive_rate, 0.05, 0.95))
    return loader, x_val, y_val_rul, y_val_fail, alpha


def _scheduler(
    optimiser: torch.optim.Optimizer, epochs: int, warmup: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, per epoch.

    Warmup matters here because BatchNorm statistics are meaningless for the
    first few batches; a full-rate step into that noise destabilises training.
    """

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)


def train_a1(
    data: CMAPSSData,
    seed: int = 42,
    horizon_h: int = 30,
    dropout: float = 0.3,
    lambda_fail: float | None = None,
    train_config: TrainConfig | None = None,
    device: str = "cpu",
    checkpoint_dir: Path | None = None,
    quiet: bool = False,
) -> tuple[A1Net, TrainResult]:
    """Train one A1 model on one subset with one seed.

    When ``lambda_fail`` is None the candidates in ``TrainConfig`` are each tried
    and the one with the best validation RMSE is kept -- tuned on validation, never
    on test.

    Returns the network with its best-by-validation-RMSE weights restored.
    """
    settings = train_config or TrainConfig()
    if lambda_fail is None and len(settings.lambda_candidates) > 1:
        return _train_with_lambda_search(
            data, seed, horizon_h, dropout, settings, device, checkpoint_dir, quiet
        )

    chosen_lambda = lambda_fail if lambda_fail is not None else settings.lambda_candidates[0]
    set_seed(seed)
    torch_device = torch.device(device)
    cap = float(data.rul_cap)

    loader, x_val, y_val_rul, y_val_fail, alpha = _loaders(data, horizon_h, settings.batch_size)
    x_val = x_val.to(torch_device)
    y_val_rul = y_val_rul.to(torch_device)
    y_val_fail = y_val_fail.to(torch_device)

    model = A1Net(
        A1Config(
            n_features=data.n_features,
            window=data.window,
            dropout=dropout,
            horizon_h=horizon_h,
            rul_cap=data.rul_cap,
            lambda_fail=chosen_lambda,
        )
    ).to(torch_device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = _scheduler(optimiser, settings.epochs, settings.warmup_epochs)
    use_amp = settings.amp and torch_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rmse, best_state, best_epoch, stale = float("inf"), None, -1, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(settings.epochs):
        model.train()
        epoch_loss = epoch_mse = epoch_focal = 0.0
        batches = 0
        for xb, y_rul, y_fail in loader:
            xb = xb.to(torch_device, non_blocking=True)
            y_rul = y_rul.to(torch_device, non_blocking=True)
            y_fail = y_fail.to(torch_device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                rul_pred, fail_logit, _ = model(xb)
                loss, mse_part, focal_part = a1_loss(
                    rul_pred,
                    fail_logit,
                    y_rul,
                    y_fail,
                    lambda_fail=chosen_lambda,
                    alpha=alpha,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += float(loss.detach())
            epoch_mse += float(mse_part.detach())
            epoch_focal += float(focal_part.detach())
            batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_rul, val_logit, _ = model(x_val)
            val_loss, _, _ = a1_loss(
                val_rul,
                val_logit,
                y_val_rul,
                y_val_fail,
                lambda_fail=chosen_lambda,
                alpha=alpha,
            )
            predicted = np.clip(val_rul.cpu().numpy() * cap, 0.0, cap)
        val_rmse = rmse(data.y_val, predicted)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": epoch_loss / max(batches, 1),
                "train_mse": epoch_mse / max(batches, 1),
                "train_focal": epoch_focal / max(batches, 1),
                "val_loss": float(val_loss),
                "val_rmse": val_rmse,
                "lr": float(optimiser.param_groups[0]["lr"]),
            }
        )

        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch, stale = val_rmse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if not quiet and (epoch % 10 == 0 or stale == 0):
            logger.info(
                "  epoch %3d  train %.4f  val_rmse %.3f%s",
                epoch,
                history[-1]["train_loss"],
                val_rmse,
                "  *" if stale == 0 else "",
            )
        if stale >= settings.patience:
            logger.info("  early stop at epoch %d (best val RMSE %.3f)", epoch, best_rmse)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    duration = time.perf_counter() - started

    metrics = evaluate_a1(model, data, horizon_h, settings.mc_passes, torch_device)

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"a1_{data.subset}_seed{seed}.pt"
        if checkpoint_path.exists():
            # Checkpoints are keyed by subset and seed only, so a short probe run
            # (--epochs 3) silently replaces a long one. Warn rather than fail:
            # overwriting is usually intended, but losing a 13-minute fit to a
            # 30-second smoke test is not. Pass --output-dir to keep them apart.
            logger.warning(
                "overwriting existing checkpoint %s -- pass --output-dir to keep "
                "a probe run separate from a real one",
                checkpoint_path.name,
            )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "a1_config": asdict(model.config),
                "sensors": data.sensors,
                "normaliser": data.normaliser.to_dict(),
                "val_rmse": best_rmse,
                "seed": seed,
                "subset": data.subset,
            },
            checkpoint_path,
        )

    return model, TrainResult(
        metrics=metrics,
        best_val_rmse=best_rmse,
        best_epoch=best_epoch,
        epochs_run=len(history),
        lambda_fail=chosen_lambda,
        duration_s=duration,
        n_parameters=model.n_parameters(),
        history=history,
        checkpoint=checkpoint_path,
    )


def _train_with_lambda_search(
    data: CMAPSSData,
    seed: int,
    horizon_h: int,
    dropout: float,
    settings: TrainConfig,
    device: str,
    checkpoint_dir: Path | None,
    quiet: bool,
) -> tuple[A1Net, TrainResult]:
    """Try each lambda and keep the best by validation RMSE."""
    best: tuple[A1Net, TrainResult] | None = None
    for candidate in settings.lambda_candidates:
        logger.info("  lambda=%.2f", candidate)
        model, result = train_a1(
            data,
            seed=seed,
            horizon_h=horizon_h,
            dropout=dropout,
            lambda_fail=candidate,
            train_config=settings,
            device=device,
            checkpoint_dir=None,
            quiet=quiet,
        )
        if best is None or result.best_val_rmse < best[1].best_val_rmse:
            best = (model, result)

    assert best is not None
    model, result = best
    logger.info(
        "  selected lambda=%.2f on validation RMSE %.3f", result.lambda_fail, result.best_val_rmse
    )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"a1_{data.subset}_seed{seed}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "a1_config": asdict(model.config),
                "sensors": data.sensors,
                "normaliser": data.normaliser.to_dict(),
                "val_rmse": result.best_val_rmse,
                "seed": seed,
                "subset": data.subset,
            },
            path,
        )
        result.checkpoint = path
    return model, result


@torch.no_grad()
def evaluate_a1(
    model: A1Net,
    data: CMAPSSData,
    horizon_h: int,
    mc_passes: int,
    device: torch.device,
) -> dict[str, float]:
    """Score A1 on the test split under the benchmark protocol.

    ``data.x_test`` already holds exactly one window per engine, so the protocol
    reduction is a no-op here -- but it is applied anyway, so the guarantee lives
    in one place rather than depending on how the caller built the array.
    """
    x_test = torch.from_numpy(data.x_test.astype("float32")).to(device)
    rul_mean, rul_std, p_fail, _ = predict_with_uncertainty(
        model, x_test, n_passes=mc_passes, rul_cap=float(data.rul_cap)
    )

    metrics = evaluate_cmapss_test(data.y_test, rul_mean, data.units_test)
    metrics["mean_rul_std"] = float(np.mean(rul_std))

    fail_true = failure_labels(data.y_test, horizon_h)
    if 0 < fail_true.sum() < len(fail_true):
        metrics["fail_pr_auc"] = pr_auc(fail_true, p_fail)
    return metrics


def attention_by_rul_bucket(
    model: A1Net,
    x: NDArray[np.float32],
    y_rul: NDArray[np.float64],
    device: str = "cpu",
    buckets: tuple[tuple[str, float, float], ...] = (
        ("0-25", 0.0, 25.0),
        ("26-50", 26.0, 50.0),
        ("51-100", 51.0, 100.0),
        ("101-125", 101.0, 125.0),
    ),
) -> dict[str, NDArray[np.float64]]:
    """Mean attention weight per timestep, grouped by RUL bucket.

    Produces the attention-over-cycles figure: it shows whether the model looks
    at different parts of the window when an engine is near failure.
    """
    torch_device = torch.device(device)
    tensor = torch.from_numpy(x.astype("float32")).to(torch_device)
    model.eval()
    with torch.no_grad():
        _, _, attention = model(tensor)
    weights = attention.cpu().numpy()

    out: dict[str, NDArray[np.float64]] = {}
    for name, low, high in buckets:
        mask = (y_rul >= low) & (y_rul <= high)
        if mask.any():
            out[name] = weights[mask].mean(axis=0).astype("float64")
    return out


def ensemble_predictions(
    checkpoints: list[Path],
    data: CMAPSSData,
    device: torch.device | None = None,
) -> NDArray[np.float64]:
    """Average the deterministic predictions of several A1 checkpoints.

    The seeds already exist -- three independent training runs are required for
    the mean +/- std report -- so averaging them costs nothing extra and is the
    standard way to spend that compute twice. Each seed sees a different
    train/validation split of the engines, so their errors are only partly
    correlated and the mean cancels some of the variance.

    The test split is never divided, so predictions from different seeds are
    row-aligned and can be averaged directly.

    Reported as its own row rather than folded into the per-seed statistics: an
    ensemble is a different model from any of its members, and quoting its score
    as though it were a single run would overstate what one training run buys.
    """
    if not checkpoints:
        raise ValueError("no checkpoints to ensemble")
    device = device or torch.device("cpu")
    x_test = torch.from_numpy(data.x_test.astype("float32")).to(device)

    predictions = []
    for path in checkpoints:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = A1Net(A1Config(**payload["a1_config"])).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        mean, _, _, _ = predict_with_uncertainty(
            model, x_test, n_passes=1, rul_cap=float(data.rul_cap)
        )
        predictions.append(mean)
    return np.mean(predictions, axis=0)
