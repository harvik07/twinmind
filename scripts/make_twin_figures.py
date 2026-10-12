"""The four twin figures.

    python scripts/make_twin_figures.py

Exactly four, per the spec:

1. ``c_twin_machine_rul`` -- one pool engine's predicted RUL +- std against its
   true remaining life, with the warning and failure markers. This is the
   figure that shows the health model is doing real work on an engine it never
   trained on.
2. ``c_twin_replication`` -- one 28-day replication: raw and finished
   inventory, WIP, per-machine state bands, and demand against fulfilled.
3. ``c_twin_ci_scaling`` -- CI half-width against replication count with a
   1/sqrt(n) reference.
4. ``c_twin_sensitivity_tornado`` -- the one-at-a-time sensitivity, annotated
   with each assumption's rank-stability rho.

All four read saved artefacts (the health cache and
``reports/results/twin_validation.json``) or re-run the twin deterministically,
so each can be regenerated from what is in the repository.
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from twinmind.eval.figures import BLUE, GREEN, ORANGE, RED, save  # noqa: E402
from twinmind.eval.registry import results_dir  # noqa: E402
from twinmind.twin.runner import (  # noqa: E402
    Scenario,
    build_resources,
    last_detail,
    replicate,
    run_twin,
)
from twinmind.utils.logging import get_logger  # noqa: E402

logger = get_logger("make_twin_figures")


def figure_machine_rul(resources, slot: int = 0) -> None:  # type: ignore[no-untyped-def]
    """Predicted RUL against true remaining life for one held-out engine."""
    cache = resources.lookup.cache
    threshold = resources.config.health_replay.warning_threshold
    start, stop = int(cache.offsets[slot]), int(cache.offsets[slot + 1])
    mean = cache.rul_mean[start:stop]
    std = cache.rul_std[start:stop]
    probability = cache.p_fail[start:stop]
    failure_cycle = int(cache.failure_cycles[slot])
    engine_id = int(cache.engine_ids[slot])

    ages = np.arange(cache.window, cache.window + mean.size)
    true_rul = np.clip(failure_cycle - ages, 0, None)
    # The model was trained on a capped target, so its prediction saturates;
    # capping the truth the same way is the only fair comparison.
    capped_true = np.minimum(true_rul, 125.0)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    top.plot(ages, capped_true, color="black", lw=2, label="true RUL (capped at 125)")
    top.plot(ages, mean, color=BLUE, lw=1.8, label="predicted RUL")
    top.fill_between(
        ages, mean - std, mean + std, color=BLUE, alpha=0.25, label="+- 1 MC-dropout std"
    )

    crossed = np.flatnonzero(probability >= threshold)
    if crossed.size:
        warning_age = int(ages[crossed[0]])
        top.axvline(warning_age, color=ORANGE, ls="--", lw=1.8)
        top.annotate(
            f"warning at cycle {warning_age}\n({failure_cycle - warning_age} cycles of notice)",
            xy=(warning_age, 95),
            xytext=(warning_age - 95, 108),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": ORANGE},
            bbox={"boxstyle": "round", "fc": "white", "ec": ORANGE, "alpha": 0.9},
        )
    top.axvline(failure_cycle, color=RED, lw=2)
    top.annotate(
        f"failure at cycle {failure_cycle}",
        xy=(failure_cycle, 20),
        xytext=(failure_cycle - 80, 45),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": RED},
        bbox={"boxstyle": "round", "fc": "white", "ec": RED, "alpha": 0.9},
    )
    top.set_ylabel("remaining useful life (cycles)")
    top.set_title(
        f"A1 on held-out engine {engine_id} (FD002, seed 42) -- replayed ground truth, "
        "predicted health",
        fontsize=11,
    )
    top.legend(loc="upper right", fontsize=9)
    top.grid(alpha=0.3)

    bottom.plot(ages, probability, color=GREEN, lw=1.8)
    bottom.axhline(threshold, color=ORANGE, ls="--", lw=1.5, label=f"warning threshold {threshold}")
    bottom.set_ylabel("P(fail within 30 cycles)")
    bottom.set_xlabel("engine age (cycles)")
    bottom.set_ylim(-0.05, 1.05)
    bottom.legend(loc="upper left", fontsize=9)
    bottom.grid(alpha=0.3)

    fig.text(
        0.5,
        -0.02,
        "This engine was held out of A1's training set. The failure cycle is the "
        "dataset's, not the model's -- the model only supplies the warning.",
        ha="center",
        fontsize=8.5,
        style="italic",
    )
    save(fig, "c_twin_machine_rul")


def figure_replication(resources) -> None:  # type: ignore[no-untyped-def]
    """One 28-day replication: inventories, WIP, machine states, demand."""
    run_twin(Scenario(), 7, resources=resources)
    detail = last_detail()
    days = np.arange(1, len(detail["demand_trace"]) + 1)

    # Deliberately NOT sharex: three panels are in days and the station panel is
    # in hours. Sharing the axis squashes the day panels into the left tenth of
    # the figure, which is what the first version of this plot did.
    fig, axes = plt.subplots(4, 1, figsize=(11, 10))

    raw_trace = np.asarray(detail["raw_trace"], dtype="float64")
    axes[0].plot(
        np.arange(1, raw_trace.size + 1), raw_trace, color=BLUE, lw=1.8, label="raw material"
    )
    axes[0].set_ylabel("units")
    axes[0].set_xlim(0.5, days.size + 0.5)
    axes[0].set_xlabel("day")
    axes[0].set_title("Raw-material inventory, (s, Q) reviewed daily", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(days, detail["finished_trace"], color=GREEN, lw=1.8, label="finished stock")
    axes[1].set_ylabel("units")
    axes[1].set_xlim(0.5, days.size + 0.5)
    axes[1].set_xlabel("day")
    axes[1].set_title("Finished stock, after each day's despatch", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    states = detail["station_states"]
    labels = [f"M{i + 1}" for i in range(len(states))]
    processing = [s.processing_hours for s in states]
    starved = [s.starved_hours for s in states]
    blocked = [s.blocked_hours for s in states]
    down = [s.downtime_hours for s in states]
    positions = np.arange(len(states))
    axes[2].barh(positions, processing, color=GREEN, label="processing")
    axes[2].barh(positions, starved, left=processing, color=ORANGE, label="starved")
    axes[2].barh(
        positions,
        blocked,
        left=np.add(processing, starved),
        color=BLUE,
        label="blocked",
    )
    axes[2].barh(
        positions,
        down,
        left=np.add(np.add(processing, starved), blocked),
        color=RED,
        label="down",
    )
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels(labels)
    axes[2].set_xlabel("hours")
    axes[2].set_title(
        "Where each station's time went (blocking and starving emerge from the buffers)",
        fontsize=10,
    )
    axes[2].legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.95)
    axes[2].grid(alpha=0.3, axis="x")

    axes[3].bar(days, detail["demand_trace"], color=BLUE, alpha=0.45, label="demand")
    axes[3].bar(days, detail["delivered_trace"], color=GREEN, alpha=0.85, label="fulfilled")
    unmet = np.asarray(detail["demand_trace"]) - np.asarray(detail["delivered_trace"])
    stockout_days = days[unmet > 1e-9]
    for day in stockout_days:
        axes[3].axvline(day, color=RED, alpha=0.12, lw=6)
    axes[3].set_ylabel("units/day")
    axes[3].set_xlabel("day")
    axes[3].set_xlim(0.5, days.size + 0.5)
    axes[3].set_title(
        f"Demand against fulfilled -- {stockout_days.size} of {days.size} days with unmet demand "
        "(shaded); shortfall is lost, not backordered",
        fontsize=10,
    )
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].grid(alpha=0.3, axis="y")

    fig.suptitle("One replication, baseline scenario, seed 7", fontsize=12)
    fig.tight_layout()
    save(fig, "c_twin_replication")


def figure_ci_scaling(resources, counts=(5, 10, 20, 40, 80, 160)) -> None:  # type: ignore[no-untyped-def]
    """CI half-width against n, with a 1/sqrt(n) reference line."""
    scenario = Scenario()
    kpis = ("profit", "units_produced", "stockout_probability")
    widths: dict[str, list[float]] = {name: [] for name in kpis}
    for n in counts:
        estimate = replicate(scenario, n_reps=n, base_seed=5, resources=resources)
        for name in kpis:
            widths[name].append(estimate.get(name).width / 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, name, colour in zip(axes, kpis, (BLUE, GREEN, ORANGE), strict=True):
        values = np.asarray(widths[name], dtype="float64")
        axis.loglog(counts, values, "o-", color=colour, lw=1.8, label="measured")
        reference = values[0] * np.sqrt(counts[0] / np.asarray(counts, dtype="float64"))
        axis.loglog(counts, reference, "--", color="grey", lw=1.5, label=r"$1/\sqrt{n}$")
        slope = float(np.polyfit(np.log(counts), np.log(values), 1)[0])
        axis.set_title(f"{name}\nfitted exponent {slope:+.3f}", fontsize=10)
        axis.set_xlabel("replications")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=9)
    axes[0].set_ylabel("95% CI half-width")
    fig.suptitle(
        "Confidence-interval half-width shrinks as 1/sqrt(n), which is the check that "
        "replications are independent",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, "c_twin_ci_scaling")


def figure_sensitivity_tornado() -> None:
    """One-at-a-time sensitivity, annotated with the rank-stability rho."""
    path = results_dir() / "twin_validation.json"
    if not path.exists():
        logger.warning("no %s; run scripts/validate_twin.py --sensitivity first", path.name)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    sensitivity = payload.get("sensitivity")
    if not sensitivity:
        logger.warning("twin_validation.json has no sensitivity section")
        return

    rows = sensitivity["parameters"]
    names = sorted(rows, key=lambda n: rows[n]["max_abs_profit_pct_change"])
    low = [rows[n]["minus25"]["profit_pct_change"] for n in names]
    high = [rows[n]["plus25"]["profit_pct_change"] for n in names]
    rhos = [rows[n]["rank_spearman_rho"] for n in names]
    positions = np.arange(len(names))

    fig, axis = plt.subplots(figsize=(11, 0.55 * len(names) + 3))
    axis.barh(positions, low, color=BLUE, alpha=0.85, label="assumption -25%")
    axis.barh(positions, high, color=ORANGE, alpha=0.85, label="assumption +25%")
    axis.axvline(0, color="black", lw=1)

    span = max(max(np.abs(low), default=1.0), max(np.abs(high), default=1.0))
    # A parameter that is exactly inert at the baseline draws an empty bar, which
    # reads as "this does not matter" and is the opposite of the truth for
    # price_elasticity. Those rows carry their off-baseline effect instead.
    off_baseline = sensitivity.get("off_baseline", {}).get("parameters", {})
    for position, name, rho in zip(positions, names, rhos, strict=True):
        unstable = rho < 0.8
        label = f"rho = {rho:.3f}" + ("  RANK UNSTABLE" if unstable else "")
        axis.text(
            span * 1.08,
            position,
            label,
            va="center",
            fontsize=8.5,
            color=RED if unstable else "black",
            fontweight="bold" if unstable else "normal",
        )
        if rows[name]["max_abs_profit_pct_change"] < 0.01 and name in off_baseline:
            worst = max(
                abs(off_baseline[name]["minus25"]["profit_pct_change"]),
                abs(off_baseline[name]["plus25"]["profit_pct_change"]),
            )
            axis.text(
                span * 0.04,
                position,
                f"inert at baseline (speed = price = 1.0); {worst:.1f}% off-baseline",
                va="center",
                ha="left",
                fontsize=8,
                style="italic",
                color="dimgrey",
            )

    axis.set_yticks(positions)
    axis.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    axis.set_xlabel("change in mean profit (%)")
    axis.set_xlim(-span * 1.25, span * 1.75)
    axis.set_title(
        "Sensitivity to every assumed parameter, +-25%, with common random numbers\n"
        "rho is the Spearman rank correlation of scenario profit against the unperturbed "
        "ranking; below 0.8 the twin's advice depends on that assumption",
        fontsize=10.5,
    )
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save(fig, "c_twin_sensitivity_tornado")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Generate the four twin figures.")
    parser.add_argument("--slot", type=int, default=0, help="Pool slot for figure 1.")
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    resources = build_resources()
    figure_machine_rul(resources, args.slot)
    figure_replication(resources)
    figure_ci_scaling(resources)
    figure_sensitivity_tornado()
    return 0


if __name__ == "__main__":
    sys.exit(main())
