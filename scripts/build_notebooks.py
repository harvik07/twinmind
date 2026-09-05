"""Generate the EDA notebooks from source, then optionally execute them.

Notebooks are generated rather than hand-edited so their content stays in version
control as readable Python and cannot drift from the package API. Every notebook
imports from ``twinmind``; none of them redefines pipeline logic.

    python scripts/build_notebooks.py            # write the .ipynb files
    python scripts/build_notebooks.py --execute  # write and run them
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("build_notebooks")

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

PREAMBLE = '''\
import sys, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from twinmind.utils.config import load_config
from twinmind.utils.paths import figures_dir
from twinmind.utils.seed import set_seed

set_seed(42)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
FIGURES = figures_dir()
DPI = 150

def save(fig, name):
    """Save a figure to reports/figures as a 150-dpi PNG."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"saved {path.relative_to(path.parents[2])}")
    return path

config = load_config("data")
'''


# --------------------------------------------------------------------------- #
# 01 - C-MAPSS
# --------------------------------------------------------------------------- #
CMAPSS_CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# 01 - C-MAPSS exploration

NASA turbofan degradation, the training data for **Model A** (remaining useful life).

Four subsets differing in operating conditions and fault modes. Everything below
imports from `twinmind.data`; no preprocessing logic is redefined here.
""",
    ),
    ("code", PREAMBLE),
    (
        "code",
        """\
from twinmind.data.loaders import load_cmapss
from twinmind.data.windowing import (
    compute_rul, compute_test_rul, drop_constant_sensors,
    fit_operating_condition_normaliser, make_windows, split_by_unit,
)

cfg = config.cmapss
SENSORS, SETTINGS = cfg.sensor_columns, cfg.setting_columns

train = {s: load_cmapss(s, "train") for s in cfg.subsets}
test = {s: load_cmapss(s, "test") for s in cfg.subsets}

summary = pd.DataFrame(
    [
        {
            "subset": s,
            "train_rows": len(train[s]),
            "train_units": train[s]["unit"].nunique(),
            "test_rows": len(test[s]),
            "test_units": test[s]["unit"].nunique(),
            "min_len": int(train[s].groupby("unit").size().min()),
            "median_len": int(train[s].groupby("unit").size().median()),
            "max_len": int(train[s].groupby("unit").size().max()),
        }
        for s in cfg.subsets
    ]
)
summary""",
    ),
    ("md", "## Unit-length distribution per subset\n\nHow long each engine runs before failure."),
    (
        "code",
        """\
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, s in zip(axes.ravel(), cfg.subsets):
    lengths = train[s].groupby("unit").size()
    ax.hist(lengths, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(lengths.median(), color="#C44E52", ls="--",
               label=f"median {int(lengths.median())}")
    ax.axvline(cfg.window_length, color="#55A868", ls=":",
               label=f"window {cfg.window_length}")
    ax.set_title(f"{s} - {train[s]['unit'].nunique()} units")
    ax.set_xlabel("trajectory length (cycles)")
    ax.legend(fontsize=8)
fig.suptitle("C-MAPSS unit-length distribution by subset", fontweight="bold")
fig.tight_layout()
save(fig, "01_cmapss_unit_length_histogram");""",
    ),
    (
        "md",
        """## Per-sensor variance and which sensors get dropped

Constant sensors are **detected** from the training split, never hard-coded: the
flat set differs by subset, because FD002/FD004 vary across six operating regimes.
""",
    ),
    (
        "code",
        """\
dropped_by_subset = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, s in zip(axes.ravel(), cfg.subsets):
    kept, dropped = drop_constant_sensors(
        train[s], SENSORS, cfg.constant_variance_threshold
    )
    dropped_by_subset[s] = dropped
    var = train[s][SENSORS].var()
    colours = ["#C44E52" if c in dropped else "#4C72B0" for c in SENSORS]
    ax.bar(range(len(SENSORS)), var.reindex(SENSORS).clip(lower=1e-30), color=colours)
    ax.axhline(cfg.constant_variance_threshold, color="black", ls="--",
               label=f"threshold {cfg.constant_variance_threshold:g}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(SENSORS)))
    ax.set_xticklabels([c.split("_")[1] for c in SENSORS], fontsize=7)
    ax.set_title(f"{s} - dropping {len(dropped)}: "
                 f"{sorted(int(c.split('_')[1]) for c in dropped)}")
    ax.set_xlabel("sensor"); ax.set_ylabel("variance (log)")
    ax.legend(fontsize=8)
fig.suptitle("Per-sensor training variance; red bars are dropped", fontweight="bold")
fig.tight_layout()
save(fig, "01_cmapss_sensor_variance");

pd.Series({s: sorted(int(c.split("_")[1]) for c in d)
           for s, d in dropped_by_subset.items()}).to_frame("dropped sensors")""",
    ),
    (
        "md",
        """## The piecewise-linear RUL cap

Before the cap, the target claims a brand-new engine is "300 cycles from failure",
which the sensors cannot support. Capping at 125 asks the model only for what the
data can actually tell it.
""",
    ),
    (
        "code",
        """\
sub = cfg.primary_subset
raw_rul = (
    train[sub].groupby("unit")["cycle"].transform("max") - train[sub]["cycle"]
)
capped = compute_rul(train[sub], cap=cfg.rul_cap)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(raw_rul, bins=60, color="#4C72B0", edgecolor="white")
axes[0].set_title(f"{sub}: RUL before the cap")
axes[0].set_xlabel("RUL (cycles)"); axes[0].set_ylabel("rows")
axes[0].axvline(cfg.rul_cap, color="#C44E52", ls="--", label=f"cap {cfg.rul_cap}")
axes[0].legend()

axes[1].hist(capped, bins=60, color="#55A868", edgecolor="white")
axes[1].set_title(f"{sub}: RUL after the piecewise-linear cap")
axes[1].set_xlabel("RUL (cycles)")
pct = 100 * (capped == cfg.rul_cap).mean()
axes[1].annotate(f"{pct:.1f}% of rows pinned at {cfg.rul_cap}",
                 xy=(0.45, 0.85), xycoords="axes fraction",
                 bbox=dict(boxstyle="round", fc="white", ec="#C44E52"))
fig.suptitle("Piecewise-linear RUL target", fontweight="bold")
fig.tight_layout()
save(fig, "01_cmapss_rul_distribution_cap");""",
    ),
    ("md", "## Sensor trajectories for five sample units, coloured by RUL"),
    (
        "code",
        """\
kept, _ = drop_constant_sensors(train[sub], SENSORS, cfg.constant_variance_threshold)
show = kept[:6]
frame = train[sub].copy()
frame["rul"] = compute_rul(frame, cap=cfg.rul_cap)
sample_units = sorted(frame["unit"].unique())[:5]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, sensor in zip(axes.ravel(), show):
    for unit in sample_units:
        block = frame[frame["unit"] == unit]
        sc = ax.scatter(block["cycle"], block[sensor], c=block["rul"],
                        cmap="viridis_r", s=4)
    ax.set_title(sensor); ax.set_xlabel("cycle")
fig.colorbar(sc, ax=axes, label="RUL (cycles)", shrink=0.8)
fig.suptitle(f"{sub}: sensor trajectories for 5 units, coloured by RUL",
             fontweight="bold")
save(fig, "01_cmapss_sensor_trajectories");""",
    ),
    (
        "md",
        """## Regime structure on FD002

FD002 and FD004 are flown across six operating conditions. The op settings cluster
cleanly, which is what `fit_operating_condition_normaliser` exploits.
""",
    ),
    (
        "code",
        """\
multi = cfg.multi_regime_subsets[0]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(train[multi][SETTINGS[0]], train[multi][SETTINGS[1]], s=2, alpha=0.3)
axes[0].set_xlabel(SETTINGS[0]); axes[0].set_ylabel(SETTINGS[1])
axes[0].set_title(f"{multi}: operating settings form discrete regimes")

norm = fit_operating_condition_normaliser(
    train[multi], SENSORS, SETTINGS, n_conditions=cfg.n_operating_conditions
)
before = train[multi]["sensor_2"]
after = norm.transform(train[multi])["sensor_2"]
axes[1].hist(before, bins=60, alpha=0.6, label="raw", color="#4C72B0")
ax2 = axes[1].twiny()
ax2.hist(after, bins=60, alpha=0.6, label="within-regime z-scored", color="#55A868")
axes[1].set_title("sensor_2 before (blue) and after (green) regime normalisation")
fig.suptitle("Operating-condition structure", fontweight="bold")
fig.tight_layout()
save(fig, "01_cmapss_operating_regimes");""",
    ),
    ("md", "## End-to-end preprocessing check\n\nThe exact path `train_health.py` will take."),
    (
        "code",
        """\
frame = train[sub].copy()
frame["rul"] = compute_rul(frame, cap=cfg.rul_cap)
tr, va = split_by_unit(frame, cfg.val_fraction, seed=config.splits.seed)
assert not set(tr["unit"]) & set(va["unit"])

kept, dropped = drop_constant_sensors(tr, SENSORS, cfg.constant_variance_threshold)
norm = fit_operating_condition_normaliser(tr, kept, SETTINGS, n_conditions=1)
X, y, units = make_windows(norm.transform(tr), kept, window=cfg.window_length)
Xv, yv, _ = make_windows(norm.transform(va), kept, window=cfg.window_length)

print(f"kept {len(kept)} sensors, dropped {len(dropped)}")
print(f"train windows {X.shape}  targets {y.shape}  units {len(np.unique(units))}")
print(f"val   windows {Xv.shape}  targets {yv.shape}")
print(f"target range [{y.min()}, {y.max()}]")""",
    ),
]


# --------------------------------------------------------------------------- #
# 02 - AI4I
# --------------------------------------------------------------------------- #
AI4I_CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# 02 - AI4I 2020 exploration

UCI predictive-maintenance data, training the failure-probability head of **Model A**.

**The trap in this dataset:** `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are the individual
failure modes, and their OR reproduces the `machine_failure` target. Any model given
them scores ~1.0 and has learned nothing. `load_ai4i` therefore returns the features,
the target and the modes as three separate objects.
""",
    ),
    ("code", PREAMBLE),
    (
        "code",
        """\
from twinmind.data.loaders import load_ai4i

cfg = config.ai4i
features, target, modes = load_ai4i()

print(f"features {features.shape}: {list(features.columns)}")
print(f"target   {target.shape}, name={target.name!r}")
print(f"modes    {modes.shape}: {list(modes.columns)}")
assert not set(features.columns) & set(cfg.failure_mode_columns)
print("\\nconfirmed: no failure-mode column is in the feature frame")
features.describe().T""",
    ),
    ("md", "## Class balance\n\nSeverely imbalanced, which drives the choice of loss and metric."),
    (
        "code",
        """\
counts = target.value_counts().sort_index()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].bar(["no failure", "failure"], counts.to_numpy(),
            color=["#4C72B0", "#C44E52"], edgecolor="white")
for i, v in enumerate(counts.to_numpy()):
    axes[0].text(i, v, f"{v:,}\\n{v/len(target)*100:.2f}%", ha="center", va="bottom")
axes[0].set_title(f"Class balance - {target.mean()*100:.2f}% positive")
axes[0].set_ylabel("rows")

mode_counts = modes.sum().sort_values(ascending=False)
axes[1].bar(mode_counts.index.str.upper(), mode_counts.to_numpy(),
            color="#C44E52", edgecolor="white")
for i, v in enumerate(mode_counts.to_numpy()):
    axes[1].text(i, v, str(int(v)), ha="center", va="bottom")
axes[1].set_title("Failure count by mode")
axes[1].set_ylabel("occurrences")
fig.suptitle("AI4I 2020 class balance and failure modes", fontweight="bold")
fig.tight_layout()
save(fig, "02_ai4i_class_balance");""",
    ),
    (
        "md",
        """## Why the mode columns are quarantined

The OR of the five modes reproduces the target almost exactly. This cell measures
that agreement, so the separation in `load_ai4i` is justified by evidence.
""",
    ),
    (
        "code",
        """\
agreement = (modes.max(axis=1) == target).mean()
without_rnf = (modes.drop(columns=["rnf"]).max(axis=1) == target).mean()
print(f"OR(all five modes)      == target for {agreement*100:.2f}% of rows")
print(f"OR(modes without RNF)   == target for {without_rnf*100:.2f}% of rows")
print(f"\\nRNF ('random failure') is the only real discrepancy: "
      f"{int(modes['rnf'].sum())} rows flagged, "
      f"{int(((modes['rnf']==1) & (target==0)).sum())} of which are NOT failures.")

fig, ax = plt.subplots(figsize=(6, 4))
pd.Series({"OR(all modes)": agreement, "OR(no RNF)": without_rnf}).plot.barh(
    ax=ax, color="#C44E52")
ax.set_xlim(0.95, 1.0); ax.set_xlabel("agreement with target")
ax.set_title("Failure modes reconstruct the target", fontweight="bold")
fig.tight_layout()
save(fig, "02_ai4i_leakage_evidence");""",
    ),
    ("md", "## Feature distributions, split by outcome"),
    (
        "code",
        """\
numeric = list(cfg.feature_columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, column in zip(axes.ravel(), numeric):
    for label, colour, name in [(0, "#4C72B0", "no failure"), (1, "#C44E52", "failure")]:
        ax.hist(features.loc[target == label, column], bins=40, alpha=0.6,
                density=True, color=colour, label=name)
    ax.set_title(column, fontsize=10); ax.legend(fontsize=8)
axes.ravel()[-1].axis("off")
fig.suptitle("Feature distributions by outcome (density-normalised)", fontweight="bold")
fig.tight_layout()
save(fig, "02_ai4i_feature_distributions");""",
    ),
    (
        "md",
        """## Correlation

Including the modes and target here **for analysis only** — it makes the leakage
visible rather than leaving it as an assertion.
""",
    ),
    (
        "code",
        """\
combined = pd.concat(
    [features[numeric], modes, target.rename("machine_failure")], axis=1
)
fig, ax = plt.subplots(figsize=(11, 8.5))
sns.heatmap(combined.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, cbar_kws={"shrink": 0.7}, annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation - note the mode columns against the target",
             fontweight="bold")
fig.tight_layout()
save(fig, "02_ai4i_correlation_heatmap");

print("\\nCorrelation with the target:")
print(combined.corr()["machine_failure"].drop("machine_failure")
      .sort_values(ascending=False).to_string())""",
    ),
]


# --------------------------------------------------------------------------- #
# 03 - Favorita
# --------------------------------------------------------------------------- #
FAVORITA_CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# 03 - Favorita demand exploration

Kaggle Store Sales, the training data for **Model B** (multi-quantile demand).

This notebook runs against the real download when `data/raw/favorita` is present,
and otherwise falls back to the synthetic fixture used by the test suite, so it
always executes top-to-bottom. The banner below says which one you are looking at.
""",
    ),
    ("code", PREAMBLE),
    (
        "code",
        """\
from twinmind.data.features import (
    build_demand_features, make_rolling_origin_splits, prepare_oil,
)
from twinmind.data.loaders import DataNotDownloadedError, load_favorita

cfg = config.favorita

try:
    frames = load_favorita()
    SOURCE = "real"
except DataNotDownloadedError as exc:
    sys.path.insert(0, str(FIGURES.parents[1] / "tests"))
    from synthetic import make_favorita
    frames = make_favorita(n_stores=8, days=900)
    SOURCE = "synthetic"
    print(f"NOTE: {exc}\\n")

print(f"data source: {SOURCE.upper()}")
for name, frame in frames.items():
    print(f"  {name:<14} {frame.shape}")""",
    ),
    (
        "code",
        """\
features = build_demand_features(frames, cfg, top_n=cfg.top_series)
print(f"{len(features):,} rows | {features['unique_id'].nunique()} series "
      f"| {features.shape[1]} columns")
print(f"{features['ds'].min().date()} to {features['ds'].max().date()}")
features.head()""",
    ),
    ("md", "## Total sales over time"),
    (
        "code",
        """\
daily = features.groupby("ds")["y"].sum()
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(daily.index, daily.to_numpy(), lw=0.8, color="#4C72B0")
ax.plot(daily.index, daily.rolling(28, min_periods=1).mean().to_numpy(),
        lw=2, color="#C44E52", label="28-day mean")
ax.set_title(f"Total daily sales across {features['unique_id'].nunique()} series "
             f"({SOURCE} data)", fontweight="bold")
ax.set_xlabel("date"); ax.set_ylabel("units"); ax.legend()
fig.tight_layout()
save(fig, "03_favorita_total_sales");""",
    ),
    ("md", "## Weekly and yearly seasonality"),
    (
        "code",
        """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
by_dow = features.groupby("dow")["y"].mean()
axes[0].bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], by_dow.to_numpy(),
            color="#4C72B0", edgecolor="white")
axes[0].set_title("Mean sales by day of week"); axes[0].set_ylabel("units")

by_month = features.groupby("month")["y"].mean()
axes[1].plot(by_month.index, by_month.to_numpy(), marker="o", color="#55A868")
axes[1].set_xticks(range(1, 13))
axes[1].set_title("Mean sales by month"); axes[1].set_xlabel("month")
fig.suptitle("Seasonality", fontweight="bold")
fig.tight_layout()
save(fig, "03_favorita_seasonality");""",
    ),
    ("md", "## Promotion effect"),
    (
        "code",
        """\
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
promo = features["onpromotion"] > 0
axes[0].boxplot([features.loc[~promo, "y"], features.loc[promo, "y"]],
                labels=["no promo", "on promo"], showfliers=False)
axes[0].set_title("Sales with and without promotion"); axes[0].set_ylabel("units")

bins = pd.cut(features["onpromotion"], bins=[-0.1, 0, 1, 5, 20, 1e9],
              labels=["0", "1", "2-5", "6-20", "20+"])
# observed=False keeps empty bins, so the labels and heights stay the same length.
by_promo = features.groupby(bins, observed=False)["y"].mean().dropna()
axes[1].bar(by_promo.index.astype(str), by_promo.to_numpy(),
            color="#DD8452", edgecolor="white")
axes[1].set_title("Mean sales by promotion intensity")
axes[1].set_xlabel("items on promotion")

lift = features.loc[promo, "y"].mean() / max(features.loc[~promo, "y"].mean(), 1e-9)
fig.suptitle(f"Promotion effect (mean lift {lift:.2f}x)", fontweight="bold")
fig.tight_layout()
save(fig, "03_favorita_promotion_effect");""",
    ),
    (
        "md",
        """## Rolling-origin fold diagram

Folds walk forward in time. A random split would train on the future to predict
the past and report an accuracy the model can never achieve live.
""",
    ),
    (
        "code",
        """\
folds = make_rolling_origin_splits(features, n_splits=cfg.n_splits, horizon=cfg.horizon)

fig, ax = plt.subplots(figsize=(13, 3.2))
for i, (train_idx, val_idx) in enumerate(folds):
    tr, va = features["ds"].iloc[train_idx], features["ds"].iloc[val_idx]
    ax.barh(i, (tr.max() - tr.min()).days, left=tr.min(), height=0.55,
            color="#4C72B0", label="train" if i == 0 else None)
    ax.barh(i, (va.max() - va.min()).days + 1, left=va.min(), height=0.55,
            color="#C44E52", label="validation" if i == 0 else None)
    assert tr.max() < va.min()
ax.set_yticks(range(len(folds)))
ax.set_yticklabels([f"fold {i}" for i in range(len(folds))])
ax.set_title(f"Rolling-origin splits: {cfg.n_splits} folds, horizon {cfg.horizon} days",
             fontweight="bold")
ax.set_xlabel("date"); ax.legend(loc="lower left")
fig.tight_layout()
save(fig, "03_favorita_rolling_origin_folds");

pd.DataFrame([
    {"fold": i,
     "train_rows": len(t), "train_end": features["ds"].iloc[t].max().date(),
     "val_rows": len(v), "val_start": features["ds"].iloc[v].min().date(),
     "val_end": features["ds"].iloc[v].max().date()}
    for i, (t, v) in enumerate(folds)
])""",
    ),
    ("md", "## Oil, and the feature inventory"),
    (
        "code",
        """\
oil = prepare_oil(frames["oil"], features["ds"])
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.plot(oil["date"], oil["oil_price"], color="#937860", lw=1)
ax.set_title("WTI oil price, forward-filled across weekend gaps", fontweight="bold")
ax.set_xlabel("date"); ax.set_ylabel("USD/bbl")
fig.tight_layout()
save(fig, "03_favorita_oil_price");

groups = {
    "identifiers": ["unique_id", "ds", "y"],
    "calendar": [c for c in features if c in
                 {"dow","dom","month","week_of_year","year","is_weekend","is_payday"}],
    "fourier": [c for c in features if c.startswith("fourier_")],
    "holiday": [c for c in features if "holiday" in c or c == "is_work_day"],
    "lags/rolling": [c for c in features if c.startswith(("lag_", "roll_"))],
    "exogenous": [c for c in features if c in
                  {"onpromotion","oil_price","city","state","type","cluster",
                   "store_nbr","family"}],
}
for name, columns in groups.items():
    print(f"{name:<14} {len(columns):>2}  {', '.join(columns[:6])}"
          f"{' ...' if len(columns) > 6 else ''}")
print(f"\\nTOTAL {features.shape[1]} columns, {len(features):,} rows")""",
    ),
]


NOTEBOOKS = {
    "01_cmapss_eda.ipynb": CMAPSS_CELLS,
    "02_ai4i_eda.ipynb": AI4I_CELLS,
    "03_favorita_eda.ipynb": FAVORITA_CELLS,
}


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    """Write one notebook to notebooks/<name>."""
    notebook = new_notebook(
        cells=[
            new_markdown_cell(body) if kind == "md" else new_code_cell(body) for kind, body in cells
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
    )
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / name
    nbformat.write(notebook, path)
    logger.info("wrote %s (%d cells)", path.name, len(cells))
    return path


def execute(path: Path, timeout: int = 900) -> None:
    """Run a notebook in place, raising on the first error."""
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    logger.info("executed %s", path.name)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate (and optionally run) the EDA notebooks.")
    parser.add_argument("--execute", action="store_true", help="Run each notebook after writing.")
    parser.add_argument("--only", default=None, help="Build just one notebook, by filename.")
    args = parser.parse_args()

    targets = {k: v for k, v in NOTEBOOKS.items() if args.only in (None, k)}
    if not targets:
        logger.error("no notebook matches %r; choices: %s", args.only, list(NOTEBOOKS))
        return 1

    for name, cells in targets.items():
        path = build(name, cells)
        if args.execute:
            execute(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
