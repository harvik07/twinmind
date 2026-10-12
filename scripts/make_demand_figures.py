"""The four demand figures, plus the NASA delta-vs-scale figure from section 1.

python scripts/make_demand_figures.py --top-n 200
python scripts/make_demand_figures.py --only nasa
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from twinmind.data.features import build_demand_features, make_rolling_origin_splits  # noqa: E402
from twinmind.data.loaders import load_favorita  # noqa: E402
from twinmind.eval.registry import results_dir  # noqa: E402
from twinmind.models.demand import DemandData  # noqa: E402
from twinmind.models.demand_lgbm import LightGBMQuantileModel  # noqa: E402
from twinmind.utils.config import load_config  # noqa: E402
from twinmind.utils.logging import get_logger  # noqa: E402
from twinmind.utils.paths import figures_dir, models_dir  # noqa: E402

logger = get_logger("demand_figures")
HORIZON = 28
DPI = 150


def _save(fig: plt.Figure, name: str) -> None:
    path = figures_dir() / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.name)


# --------------------------------------------------------------------------- #
# 1. Fan chart
# --------------------------------------------------------------------------- #
def fan_chart(model: LightGBMQuantileModel, data: DemandData, origin: pd.Timestamp) -> None:
    """q10/q50/q90 against the realised series, for three representative series."""
    totals = data.frame.groupby("unique_id")["y"].sum().sort_values(ascending=False)
    chosen = [totals.index[0], totals.index[len(totals) // 2], totals.index[-1]]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for ax, series_id in zip(axes, chosen, strict=True):
        q = model.quantiles(series_id, origin, HORIZON)
        frame = q.as_frame(origin)
        history = data.history(series_id, origin).tail(56)
        actual = data.frame[
            (data.frame["unique_id"] == series_id)
            & (data.frame["ds"] > origin)
            & (data.frame["ds"] <= origin + pd.Timedelta(days=HORIZON))
        ]

        ax.plot(history["ds"], history["y"], color="#4C72B0", lw=1, label="history")
        ax.fill_between(
            frame["ds"], frame["q10"], frame["q90"], color="#DD8452", alpha=0.3, label="q10-q90"
        )
        ax.plot(frame["ds"], frame["q50"], color="#C44E52", lw=1.6, label="q50")
        ax.plot(actual["ds"], actual["y"], color="black", lw=1.2, ls="--", label="actual")
        ax.axvline(origin, color="grey", ls=":", lw=1)
        ax.set_title(
            f"{series_id}  (rank by volume: {list(totals.index).index(series_id) + 1})", fontsize=9
        )
        ax.set_ylabel("units")
        ax.legend(fontsize=7, ncol=4)

    fig.suptitle("Demand fan chart: 28-day forecast against realised sales", fontweight="bold")
    fig.tight_layout()
    _save(fig, "b_demand_fan_chart")


# --------------------------------------------------------------------------- #
# 2. Coverage calibration
# --------------------------------------------------------------------------- #
def coverage_calibration() -> None:
    """Nominal against empirical coverage, before and after conformal."""
    path = results_dir() / "demand_conformal.json"
    if not path.exists():
        logger.warning("no conformal record; skipping the calibration figure")
        return
    record = json.loads(path.read_text(encoding="utf-8"))

    buckets = ["coverage_d1_7", "coverage_d8_14", "coverage_d15_28"]
    labels = ["days 1-7", "days 8-14", "days 15-28"]
    before = [float(np.mean([f[b] for f in record["before"]])) for b in buckets]
    after = [float(np.mean([f[b] for f in record["after"]])) for b in buckets]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    ax.scatter(
        [0.8],
        [record["coverage_before"]],
        s=90,
        color="#4C72B0",
        label=f"before ({record['coverage_before']:.3f})",
        zorder=3,
    )
    ax.scatter(
        [0.8],
        [record["coverage_after"]],
        s=90,
        color="#55A868",
        marker="D",
        label=f"after ({record['coverage_after']:.3f})",
        zorder=3,
    )
    ax.axhspan(0.76, 0.84, color="#55A868", alpha=0.12, label="target band")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("80% interval, overall")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(buckets))
    ax.bar(x - 0.18, before, 0.36, label="before conformal", color="#4C72B0")
    ax.bar(x + 0.18, after, 0.36, label="after conformal", color="#55A868")
    ax.axhline(0.80, color="#C44E52", ls="--", lw=1.5, label="nominal 0.80")
    ax.axhspan(0.76, 0.84, color="#C44E52", alpha=0.10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1)
    ax.set_title("By horizon bucket")
    ax.legend(fontsize=8)

    fig.suptitle("Interval calibration, before and after split-conformal", fontweight="bold")
    fig.tight_layout()
    _save(fig, "b_demand_coverage_calibration")


# --------------------------------------------------------------------------- #
# 3. Pinball by model
# --------------------------------------------------------------------------- #
def pinball_by_model() -> None:
    """Pinball loss per model with fold-to-fold error bars."""
    path = results_dir() / "demand_folds.json"
    if not path.exists():
        logger.warning("no fold record; skipping the pinball figure")
        return
    folds = json.loads(path.read_text(encoding="utf-8"))

    quantile_entries = {
        "LGBM q10": [f["pinball_q10"] for f in folds],
        "LGBM q50": [f["pinball_q50"] for f in folds],
        "LGBM q90": [f["pinball_q90"] for f in folds],
    }
    point_entries = {"LGBM point\n(median pinball)": [f["point_median_pinball"] for f in folds]}

    fig, ax = plt.subplots(figsize=(9, 5))
    offset = 0
    for group, entries, colour in (
        ("quantile model", quantile_entries, "#4C72B0"),
        ("point model", point_entries, "#937860"),
    ):
        for label, values in entries.items():
            spread = np.std(values, ddof=1) if len(values) > 1 else 0.0
            ax.bar(
                offset,
                np.mean(values),
                yerr=spread,
                color=colour,
                capsize=4,
                width=0.6,
                label=group if label.startswith(("LGBM q10", "LGBM point")) else None,
            )
            ax.text(
                offset,
                np.mean(values),
                f"{np.mean(values):.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            offset += 1
        offset += 0.6  # visible gap between the two families

    ax.set_xticks([0, 1, 2, 3.6])
    ax.set_xticklabels([*quantile_entries, *point_entries], fontsize=8)
    ax.set_ylabel("pinball loss (units)")
    ax.set_title(
        "Pinball loss by model, error bars across folds\n"
        "the point model is scored at q50 only -- it has no q10 or q90 to compare",
        fontweight="bold",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "b_demand_pinball_by_model")


# --------------------------------------------------------------------------- #
# 4. Sampled trajectories
# --------------------------------------------------------------------------- #
def sampled_paths(model: LightGBMQuantileModel, data: DemandData, origin: pd.Timestamp) -> None:
    """50 sampled trajectories, with a lag-1 autocorrelation inset."""
    totals = data.frame.groupby("unique_id")["y"].sum().sort_values(ascending=False)
    series_id = totals.index[0]

    paths = model.sample_paths(series_id, origin, HORIZON, n_samples=50, seed=11)
    many = model.sample_paths(series_id, origin, HORIZON, n_samples=2000, seed=12)
    actual = data.frame[
        (data.frame["unique_id"] == series_id)
        & (data.frame["ds"] > origin)
        & (data.frame["ds"] <= origin + pd.Timedelta(days=HORIZON))
    ]
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON)

    fig, (ax, inset) = plt.subplots(1, 2, figsize=(13, 4.8), width_ratios=[2.4, 1])
    for row in paths:
        ax.plot(dates, row, color="#4C72B0", alpha=0.18, lw=0.8)
    ax.plot(dates, np.median(many, axis=0), color="#C44E52", lw=2, label="sampled median")
    if len(actual):
        ax.plot(actual["ds"], actual["y"], color="black", lw=1.6, ls="--", label="actual")
    ax.set_title(f"50 sampled trajectories -- {series_id}", fontsize=10)
    ax.set_ylabel("units")
    ax.legend(fontsize=8)

    # Lag-1 autocorrelation: sampled paths against the series' own history.
    sampled_acf = float(
        np.mean([np.corrcoef(many[:, i], many[:, i + 1])[0, 1] for i in range(HORIZON - 1)])
    )
    history = data.history(series_id, origin)["y"].to_numpy()[-365:]
    deviation = history - pd.Series(history).rolling(7, min_periods=1).mean().to_numpy()
    historical_acf = float(np.corrcoef(deviation[:-1], deviation[1:])[0, 1])

    inset.bar(
        ["sampled\npaths", "historical\nresiduals"],
        [sampled_acf, historical_acf],
        color=["#4C72B0", "#937860"],
        edgecolor="white",
    )
    inset.axhline(0, color="grey", lw=1)
    for i, v in enumerate([sampled_acf, historical_acf]):
        inset.text(i, v, f"{v:+.2f}", ha="center", va="bottom", fontsize=9)
    inset.set_ylabel("lag-1 autocorrelation")
    inset.set_title("Serial dependence preserved", fontsize=9)

    fig.suptitle("Copula-sampled demand trajectories", fontweight="bold")
    fig.tight_layout()
    _save(fig, "b_demand_sampled_paths")


# --------------------------------------------------------------------------- #
# NASA: RMSE delta against scale
# --------------------------------------------------------------------------- #
def nasa_delta_figure() -> None:
    """A1 minus random forest, against engine count, coloured by regime count.

    Four points, and the two axes are confounded, which the figure says on its
    face rather than in a caption nobody reads.
    """
    subsets = [
        {"name": "FD001", "engines": 100, "regimes": 1, "a1": 12.76, "rf": 12.72},
        {"name": "FD002", "engines": 260, "regimes": 6, "a1": 13.78, "rf": 14.36},
        {"name": "FD003", "engines": 100, "regimes": 1, "a1": 13.62, "rf": 13.32},
        {"name": "FD004", "engines": 249, "regimes": 6, "a1": 15.00, "rf": 15.68},
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for entry in subsets:
        delta = entry["a1"] - entry["rf"]
        colour = "#C44E52" if entry["regimes"] == 6 else "#4C72B0"
        ax.scatter(
            entry["engines"], delta, s=190, color=colour, zorder=3, edgecolor="white", linewidth=1.5
        )
        ax.annotate(
            f"{entry['name']}\n{delta:+.2f}",
            (entry["engines"], delta),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=9,
        )

    ax.axhline(0, color="black", lw=1.4)
    ax.text(
        178, 0.02, "A1 worse than the forest above this line", fontsize=8, color="grey", va="bottom"
    )
    ax.text(178, -0.02, "A1 better below it", fontsize=8, color="grey", va="top")

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color="#4C72B0",
                markersize=10,
                label="1 operating regime",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color="#C44E52",
                markersize=10,
                label="6 operating regimes",
            ),
        ],
        loc="upper right",
        fontsize=9,
    )

    ax.set_xlabel("training engines")
    ax.set_ylabel("RMSE delta: A1 (hct) minus random forest  (negative = A1 better)")
    ax.set_title(
        "A1's advantage over the forest grows with dataset size -- SUGGESTIVE ONLY",
        fontweight="bold",
    )
    ax.set_xlim(60, 300)
    # Headroom so the topmost point's label does not collide with the title.
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.08 * (high - low), high + 0.22 * (high - low))
    ax.grid(alpha=0.3)

    ax.annotate(
        "CONFOUNDED: the two 250+ engine subsets (FD002, FD004) are exactly the\n"
        "six-regime ones, and the two 100-engine subsets are exactly the\n"
        "single-regime ones. Engine count and regime count cannot be separated\n"
        "with these four points, so this shows an association, not a cause.",
        xy=(0.015, 0.03),
        xycoords="axes fraction",
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "#FFF3CD", "edgecolor": "#C44E52", "alpha": 0.95},
    )
    fig.tight_layout()
    _save(fig, "a1_rmse_delta_vs_scale")


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Render the demand and NASA figures.")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--only", default=None, choices=["nasa", "demand"])
    args = parser.parse_args()

    if args.only != "demand":
        nasa_delta_figure()
    if args.only == "nasa":
        return 0

    config = load_config("data")
    features = build_demand_features(load_favorita(), config.favorita, top_n=args.top_n)
    data = DemandData(features)
    folds = make_rolling_origin_splits(features, n_splits=3, horizon=HORIZON)
    origin = features.iloc[folds[-1][0]]["ds"].max()

    directory = models_dir() / "demand"
    if not (directory / "config.json").exists():
        logger.error("no saved demand model in %s; run scripts/train_demand.py", directory)
        return 1
    model = LightGBMQuantileModel.load(directory, data)

    fan_chart(model, data, origin)
    coverage_calibration()
    pinball_by_model()
    sampled_paths(model, data, origin)
    logger.info("done; see reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
