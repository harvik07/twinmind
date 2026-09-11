"""Classical and simple-neural baselines for C-MAPSS RUL.

Run before A1, and A1 must beat all of these on the same protocol or it is not
earning its complexity. Three baselines, deliberately spanning the space:

* **Ridge** on hand-crafted window features -- a linear model. If it is close to
  A1, the problem did not need a sequence model.
* **RandomForest** on the same features -- non-linear, no temporal structure. If
  it is close to A1, the problem did not need *sequence* modelling specifically.
* **LSTM(64) x2** with a single RUL head -- sequence modelling without the
  convolutional front end, attention pooling or second head. Isolates what those
  additions actually buy.

Every baseline sees exactly the inputs A1 sees, via
:func:`twinmind.data.cmapss_prep.prepare_cmapss`, and is scored with the
last-window-per-unit protocol.

    python scripts/train_baselines.py --subset FD001
    python scripts/train_baselines.py --subset all --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from twinmind.data.cmapss_prep import CMAPSSData, prepare_cmapss, window_features
from twinmind.eval.metrics import evaluate_cmapss_test
from twinmind.eval.registry import (
    RunRecord,
    config_hash,
    current_device,
    log_run,
    write_summary,
)
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.seed import set_seed

logger = get_logger("train_baselines")

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
BASELINES = ["ridge", "random_forest", "lstm"]


def _score(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], units: NDArray[np.int64], cap: int
) -> dict[str, float]:
    """Clip to the valid RUL range, then score under the benchmark protocol."""
    clipped = np.clip(y_pred, 0.0, float(cap))
    return evaluate_cmapss_test(y_true, clipped, units)


# --------------------------------------------------------------------------- #
# Classical baselines
# --------------------------------------------------------------------------- #
def run_ridge(data: CMAPSSData, seed: int) -> dict[str, float]:
    """Ridge regression on hand-crafted window features."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_train = window_features(data.x_train)
    x_test = window_features(data.x_test)

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(x_train, data.y_train)
    return _score(data.y_test, model.predict(x_test), data.units_test, data.rul_cap)


def run_random_forest(data: CMAPSSData, seed: int) -> dict[str, float]:
    """Random forest on the same hand-crafted features."""
    from sklearn.ensemble import RandomForestRegressor

    x_train = window_features(data.x_train)
    x_test = window_features(data.x_test)

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, data.y_train)
    return _score(data.y_test, model.predict(x_test), data.units_test, data.rul_cap)


# --------------------------------------------------------------------------- #
# Plain LSTM baseline
# --------------------------------------------------------------------------- #
class PlainLSTM(nn.Module):
    """Two stacked LSTM layers and one RUL head. No conv front end, no attention."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a RUL estimate per window, normalised to [0, 1]."""
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def run_lstm(
    data: CMAPSSData,
    seed: int,
    epochs: int = 40,
    batch_size: int = 256,
    device: str = "cpu",
    patience: int = 8,
) -> dict[str, float]:
    """Train the plain LSTM baseline, early-stopping on validation RMSE."""
    set_seed(seed)
    torch_device = torch.device(device)
    cap = float(data.rul_cap)

    def loader(x: NDArray[np.float32], y: NDArray[np.float64], shuffle: bool) -> DataLoader[Any]:
        tensors = TensorDataset(
            torch.from_numpy(x.astype("float32")),
            torch.from_numpy((y / cap).astype("float32")),
        )
        return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(data.x_train, data.y_train, True)
    model = PlainLSTM(data.n_features).to(torch_device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    x_val = torch.from_numpy(data.x_val.astype("float32")).to(torch_device)
    y_val = data.y_val
    best_rmse, best_state, stale = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(torch_device), yb.to(torch_device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            predicted = model(x_val).cpu().numpy() * cap
        val_rmse = float(np.sqrt(np.mean((np.clip(predicted, 0, cap) - y_val) ** 2)))

        if val_rmse < best_rmse - 1e-4:
            best_rmse, stale = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                logger.info("  lstm: early stop at epoch %d (best val RMSE %.3f)", epoch, best_rmse)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        x_test = torch.from_numpy(data.x_test.astype("float32")).to(torch_device)
        y_pred = model(x_test).cpu().numpy() * cap

    return _score(data.y_test, y_pred, data.units_test, data.rul_cap)


RUNNERS = {"ridge": run_ridge, "random_forest": run_random_forest, "lstm": run_lstm}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_one(
    name: str, subset: str, seed: int, config: DataConfig, device: str, epochs: int
) -> dict[str, float]:
    """Prepare data and run one baseline for one seed."""
    set_seed(seed)
    data = prepare_cmapss(subset, config.cmapss, seed=seed)  # type: ignore[arg-type]

    started = time.perf_counter()
    if name == "lstm":
        metrics = run_lstm(data, seed, epochs=epochs, device=device)
    else:
        metrics = RUNNERS[name](data, seed)
    duration = time.perf_counter() - started

    log_run(
        RunRecord(
            model=name,
            dataset="cmapss",
            subset=subset,
            seed=seed,
            metrics=metrics,
            config_hash=config_hash(
                {
                    "model": name,
                    "window": config.cmapss.window_length,
                    "cap": config.cmapss.rul_cap,
                    "epochs": epochs,
                }
            ),
            duration_s=round(duration, 2),
            device=current_device() if name == "lstm" else "cpu",
            notes="baseline; last-window-per-unit protocol",
        )
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Train C-MAPSS RUL baselines.")
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="FD001")
    parser.add_argument("--baseline", choices=[*BASELINES, "all"], default="all")
    parser.add_argument("--seeds", type=int, default=3, help="Number of independent seeds.")
    parser.add_argument("--seed", type=int, default=42, help="First seed; later ones increment.")
    parser.add_argument("--epochs", type=int, default=40, help="Max epochs for the LSTM baseline.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def resolve_device(requested: str) -> str:
    """Map ``auto`` onto whatever is actually available."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    config = cast(DataConfig, load_config("data"))
    device = resolve_device(args.device)

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    baselines = BASELINES if args.baseline == "all" else [args.baseline]
    seeds = [args.seed + i for i in range(args.seeds)]

    logger.info("baselines %s on %s, seeds %s, device %s", baselines, subsets, seeds, device)

    for subset in subsets:
        for name in baselines:
            for index, seed in enumerate(seeds, start=1):
                logger.info(
                    "=== %s / %s / seed %d (run %d of %d: an independent fit) ===",
                    subset,
                    name,
                    seed,
                    index,
                    len(seeds),
                )
                metrics = run_one(name, subset, seed, config, device, args.epochs)
                logger.info(
                    "  RMSE %.3f  NASA %.1f  (%d units)",
                    metrics["rmse"],
                    metrics["nasa_score"],
                    int(metrics["n_units"]),
                )

    write_summary()
    logger.info("done; see reports/results/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
