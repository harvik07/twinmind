"""Figures for the Model A results section.

Every figure is generated from saved artefacts -- checkpoints, the run registry,
per-epoch history JSON and out-of-fold predictions -- so a figure can always be
regenerated from what is in the repository rather than from a live session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from twinmind.eval.metrics import reliability_curve  # noqa: E402
from twinmind.utils.logging import get_logger  # noqa: E402
from twinmind.utils.paths import figures_dir  # noqa: E402

logger = get_logger(__name__)

DPI = 150
BLUE, RED, GREEN, ORANGE = "#4C72B0", "#C44E52", "#55A868", "#DD8452"


def save(fig: plt.Figure, name: str) -> Path:
    """Write a figure to ``reports/figures/<name>.png`` at 150 dpi."""
    path = figures_dir() / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.name)
    return path


# --------------------------------------------------------------------------- #
# A1
# --------------------------------------------------------------------------- #
def plot_pred_vs_true(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    subset: str,
    rul_cap: int = 125,
    name: str | None = None,
) -> Path:
    """Predicted against true RUL, with y=x and the cap marked.

    The cap line matters: everything at 125 on the x axis is an engine the target
    deliberately refuses to distinguish, so vertical spread there is expected and
    is not the model failing.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6))
    errors = y_pred - y_true
    scatter = ax.scatter(
        y_true,
        y_pred,
        c=errors,
        cmap="coolwarm",
        vmin=-40,
        vmax=40,
        s=45,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    limit = rul_cap * 1.08
    ax.plot([0, limit], [0, limit], "k--", lw=1.2, label="perfect (y = x)", zorder=2)
    ax.axvline(rul_cap, color=GREEN, ls=":", lw=1.5, label=f"cap = {rul_cap}", zorder=1)
    ax.axhline(rul_cap, color=GREEN, ls=":", lw=1.5, zorder=1)

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("true RUL (cycles)")
    ax.set_ylabel("predicted RUL (cycles)")
    ax.set_title(
        f"{subset}: predicted vs true RUL\none prediction per test engine",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(scatter, ax=ax, label="error (predicted - true)", shrink=0.85)

    rmse_value = float(np.sqrt(np.mean(errors**2)))
    late = float((errors > 0).mean())
    ax.annotate(
        f"RMSE {rmse_value:.2f}\n{len(y_true)} engines\n{late:.0%} late",
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        bbox={"boxstyle": "round", "fc": "white", "ec": "grey", "alpha": 0.9},
        fontsize=9,
    )
    fig.tight_layout()
    return save(fig, name or f"a1_{subset}_pred_vs_true")


def plot_rul_trajectories(
    trajectories: list[dict[str, Any]],
    subset: str,
    rul_cap: int = 125,
    name: str | None = None,
) -> Path:
    """RUL over cycles for a few engines, with the MC-dropout band shaded.

    Each entry needs ``unit``, ``cycles``, ``true``, ``mean`` and ``std``. The band
    is +/- 2 standard deviations, so it should contain the truth most of the time;
    where it does not, the model is confidently wrong, which is the failure mode
    that matters for a maintenance decision.
    """
    n = len(trajectories)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 8), squeeze=False)

    for ax, entry in zip(axes.ravel(), trajectories, strict=False):
        cycles = np.asarray(entry["cycles"])
        mean = np.asarray(entry["mean"])
        std = np.asarray(entry["std"])

        ax.plot(cycles, entry["true"], color="black", lw=2, label="true RUL", zorder=3)
        ax.plot(cycles, mean, color=BLUE, lw=1.8, label="MC-dropout mean", zorder=2)
        ax.fill_between(
            cycles,
            mean - 2 * std,
            mean + 2 * std,
            color=BLUE,
            alpha=0.25,
            label="+/- 2 sd",
            zorder=1,
        )
        ax.axhline(rul_cap, color=GREEN, ls=":", lw=1, label=f"cap = {rul_cap}")
        ax.set_title(f"engine {entry['unit']}")
        ax.set_xlabel("cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.set_ylim(0, rul_cap * 1.15)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(f"{subset}: RUL trajectories with MC-dropout uncertainty", fontweight="bold")
    fig.tight_layout()
    return save(fig, name or f"a1_{subset}_rul_trajectories")


def plot_attention_by_bucket(
    attention: dict[str, NDArray[np.float64]],
    subset: str,
    name: str | None = None,
) -> Path:
    """Mean attention weight per cycle position, grouped by RUL bucket.

    A flat line would mean attention pooling is doing nothing a mean could not do.
    A tilt towards the recent end as RUL falls means the model concentrates on the
    latest cycles when an engine is close to failure.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colours = [BLUE, GREEN, ORANGE, RED]

    for (label, weights), colour in zip(attention.items(), colours, strict=False):
        positions = np.arange(len(weights))
        ax.plot(positions, weights, marker="o", ms=3.5, lw=1.8, color=colour, label=f"RUL {label}")

    uniform = 1.0 / len(next(iter(attention.values())))
    ax.axhline(uniform, color="grey", ls="--", lw=1, label="uniform (mean pooling)")

    ax.set_xlabel("position in the 30-cycle window (0 = oldest)")
    ax.set_ylabel("mean attention weight")
    ax.set_title(
        f"{subset}: where A1 looks inside the window, by remaining life",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save(fig, name or f"a1_{subset}_attention_by_rul")


def plot_training_curves(
    histories: dict[int, list[dict[str, float]]],
    subset: str,
    name: str | None = None,
) -> Path:
    """Loss, its two components, validation RMSE and learning rate, per seed."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colours = [BLUE, GREEN, ORANGE, RED]

    panels = [
        ("train_loss", "total training loss", axes[0, 0]),
        ("val_rmse", "validation RMSE (cycles)", axes[0, 1]),
        ("train_mse", "RUL term (MSE, normalised)", axes[1, 0]),
        ("train_focal", "failure term (focal BCE)", axes[1, 1]),
    ]

    for key, title, ax in panels:
        for (seed, history), colour in zip(sorted(histories.items()), colours, strict=False):
            epochs = [h["epoch"] for h in history]
            ax.plot(epochs, [h[key] for h in history], color=colour, lw=1.6, label=f"seed {seed}")
            if key == "val_rmse":
                best = int(np.argmin([h[key] for h in history]))
                ax.plot(epochs[best], history[best][key], "*", color=colour, ms=14)
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)

    axes[0, 1].set_ylabel("cycles")
    fig.suptitle(
        f"{subset}: A1 training curves (stars mark the kept checkpoint)", fontweight="bold"
    )
    fig.tight_layout()
    return save(fig, name or f"a1_{subset}_training_curves")


# --------------------------------------------------------------------------- #
# A2
# --------------------------------------------------------------------------- #
def plot_pr_and_reliability(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    model_name: str = "a2_lightgbm",
    name: str | None = None,
) -> Path:
    """Precision-recall curve and reliability diagram side by side.

    Ranking and calibration are different properties and A2 needs both: the twin
    ranks machines to decide what to inspect, and turns the probability into a
    hazard, which only means something if it is calibrated.
    """
    from sklearn.metrics import precision_recall_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    base_rate = float(np.mean(y_true))
    axes[0].plot(recall, precision, color=BLUE, lw=2)
    axes[0].axhline(base_rate, color="grey", ls="--", lw=1, label=f"base rate {base_rate:.3f}")
    axes[0].axhline(0.80, color=RED, ls=":", lw=1.5, label="precision 0.80")

    feasible = precision >= 0.80
    if feasible.any():
        best_recall = float(np.max(recall[feasible]))
        axes[0].plot([best_recall], [0.80], "o", color=RED, ms=9, zorder=5)
        axes[0].annotate(
            f"recall {best_recall:.3f}\nat precision 0.80",
            xy=(best_recall, 0.80),
            xytext=(best_recall - 0.35, 0.55),
            arrowprops={"arrowstyle": "->", "color": RED},
            fontsize=9,
        )
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Precision-recall")
    axes[0].legend(fontsize=9, loc="upper right")

    predicted, observed, weights = reliability_curve(y_true, y_prob, n_bins=10)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1.2, label="perfectly calibrated")
    axes[1].plot(predicted, observed, marker="o", color=GREEN, lw=2, label="A2")
    sizes = 40 + 900 * weights
    axes[1].scatter(predicted, observed, s=sizes, color=GREEN, alpha=0.35, zorder=2)
    axes[1].set_xlabel("mean predicted probability")
    axes[1].set_ylabel("observed failure frequency")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Reliability (marker area = share of samples)")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"{model_name}: ranking and calibration", fontweight="bold")
    fig.tight_layout()
    return save(fig, name or f"{model_name}_pr_reliability")


# --------------------------------------------------------------------------- #
# Loading saved artefacts
# --------------------------------------------------------------------------- #
def load_histories(subset: str, results: Path) -> dict[int, list[dict[str, float]]]:
    """Read every ``history_a1_<subset>_seed*.json`` in a results directory."""
    histories: dict[int, list[dict[str, float]]] = {}
    for path in sorted(results.glob(f"history_a1_{subset}_seed*.json")):
        seed = int(path.stem.split("seed")[-1])
        histories[seed] = json.loads(path.read_text(encoding="utf-8"))
    return histories


def load_oof(model_name: str, seed: int, results: Path) -> tuple[NDArray[Any], NDArray[Any]]:
    """Read pooled out-of-fold predictions saved by the A2 trainer."""
    path = results / f"oof_{model_name}_seed{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        np.asarray(payload["y_true"], dtype="int64"),
        np.asarray(payload["y_prob"], dtype="float64"),
    )
