"""Generate every Model A figure from saved artefacts.

Reads checkpoints, per-epoch history and out-of-fold predictions from disk, so
the figures can be regenerated from what is committed rather than from a live
training session.

    python scripts/make_figures.py
    python scripts/make_figures.py --subset FD001 --only a1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch

from twinmind.data.cmapss_prep import prepare_cmapss
from twinmind.data.loaders import load_cmapss
from twinmind.data.windowing import OperatingConditionNormaliser, compute_test_rul, make_windows
from twinmind.eval.figures import (
    load_histories,
    load_oof,
    plot_attention_by_bucket,
    plot_pr_and_reliability,
    plot_pred_vs_true,
    plot_rul_trajectories,
    plot_training_curves,
)
from twinmind.eval.registry import results_dir
from twinmind.models.a1_cnn_bilstm import A1Config, A1Net, predict_with_uncertainty
from twinmind.models.a1_train import attention_by_rul_bucket
from twinmind.utils.config import DataConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import models_dir

logger = get_logger("make_figures")


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[A1Net, dict]:
    """Rebuild an A1 network from a saved checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=False)
    config = A1Config(**payload["a1_config"])
    model = A1Net(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def engine_trajectories(
    model: A1Net,
    subset: str,
    config: DataConfig,
    payload: dict,
    n_engines: int = 4,
    mc_passes: int = 30,
) -> list[dict]:
    """Build per-cycle RUL predictions for a few whole test engines.

    The scored protocol uses one window per engine; this walks the whole
    trajectory instead, which is what makes a trajectory plot meaningful.
    """
    cmapss = config.cmapss
    frame = load_cmapss(subset, "test")  # type: ignore[arg-type]
    frame = frame.assign(rul=compute_test_rul(frame, cap=cmapss.rul_cap))

    normaliser = OperatingConditionNormaliser.from_dict(payload["normaliser"])
    sensors = payload["sensors"]
    normalised = normaliser.transform(frame)

    lengths = frame.groupby("unit").size().sort_values(ascending=False)
    chosen = [int(u) for u in lengths.index[:n_engines]]

    trajectories = []
    for unit in chosen:
        block = normalised[normalised["unit"] == unit]
        x, y, _ = make_windows(block, sensors, window=cmapss.window_length, stride=1)
        if len(x) < 2:
            continue
        tensor = torch.from_numpy(x.astype("float32"))
        mean, std, _, _ = predict_with_uncertainty(
            model, tensor, n_passes=mc_passes, rul_cap=float(cmapss.rul_cap)
        )
        cycles = np.arange(cmapss.window_length, cmapss.window_length + len(x))
        trajectories.append({"unit": unit, "cycles": cycles, "true": y, "mean": mean, "std": std})
    return trajectories


def figures_for_a1(subset: str, config: DataConfig, seed: int, mc_passes: int) -> None:
    """Predicted-vs-true, trajectories, attention and training curves."""
    checkpoint = models_dir() / "a1" / f"a1_{subset}_seed{seed}.pt"
    if not checkpoint.exists():
        logger.warning("no checkpoint at %s; skipping A1 figures for %s", checkpoint, subset)
        return

    model, payload = load_checkpoint(checkpoint)
    data = prepare_cmapss(subset, config.cmapss, seed=seed)  # type: ignore[arg-type]

    x_test = torch.from_numpy(data.x_test.astype("float32"))
    mean, _, _, _ = predict_with_uncertainty(
        model, x_test, n_passes=mc_passes, rul_cap=float(data.rul_cap)
    )
    plot_pred_vs_true(data.y_test, mean, subset, rul_cap=data.rul_cap)

    trajectories = engine_trajectories(model, subset, config, payload, mc_passes=mc_passes)
    if trajectories:
        plot_rul_trajectories(trajectories, subset, rul_cap=data.rul_cap)

    attention = attention_by_rul_bucket(model, data.x_train, data.y_train)
    if attention:
        plot_attention_by_bucket(attention, subset)

    histories = load_histories(subset, results_dir())
    if histories:
        plot_training_curves(histories, subset)


def figures_for_a2(model_name: str, seed: int) -> None:
    """PR curve and reliability diagram from saved out-of-fold predictions."""
    path = results_dir() / f"oof_{model_name}_seed{seed}.json"
    if not path.exists():
        logger.warning("no out-of-fold file at %s; skipping A2 figures", path)
        return
    y_true, y_prob = load_oof(model_name, seed, results_dir())
    plot_pr_and_reliability(y_true, y_prob, model_name)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Generate the Model A figures.")
    parser.add_argument(
        "--subset", default="all", choices=["FD001", "FD002", "FD003", "FD004", "all"]
    )
    parser.add_argument("--seed", type=int, default=42, help="Which seed's checkpoint to plot.")
    parser.add_argument("--only", choices=["a1", "a2", "both"], default="both")
    parser.add_argument("--mc-passes", type=int, default=30)
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    config = cast(DataConfig, load_config("data"))

    subsets = ["FD001", "FD002", "FD003", "FD004"] if args.subset == "all" else [args.subset]

    if args.only in ("a1", "both"):
        for subset in subsets:
            logger.info("A1 figures for %s", subset)
            figures_for_a1(subset, config, args.seed, args.mc_passes)

    if args.only in ("a2", "both"):
        for model_name in ("a2_lightgbm", "a2_mlp"):
            logger.info("A2 figures for %s", model_name)
            figures_for_a2(model_name, args.seed)

    logger.info("done; see reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
