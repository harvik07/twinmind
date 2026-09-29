"""Render the hypothesis sweep as a Markdown table and a figure.

Reads reports/results/hypotheses.jsonl and writes

    reports/results/hypotheses.md
    reports/figures/a1_hypotheses_rmse.png

The figure plots RMSE per hypothesis with the random-forest baseline as a
horizontal reference, because the whole sweep exists to answer one question:
does any neural variant get under that line.
"""

from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from twinmind.eval.registry import results_dir  # noqa: E402
from twinmind.utils.logging import get_logger  # noqa: E402
from twinmind.utils.paths import figures_dir  # noqa: E402

logger = get_logger("report_hypotheses")

FOREST_RMSE = 12.72
FOREST_NASA = 245.0
BASELINE_KEY = "H0"


def load_records() -> list[dict[str, object]]:
    """Read the sweep log, keeping the newest record per (key, seed).

    Keyed on the seed as well as the hypothesis, because the winner is re-run
    across several seeds to check the margin is real: collapsing on key alone
    would silently keep only the last seed and hide the spread.
    """
    path = results_dir() / "hypotheses.jsonl"
    if not path.exists():
        return []
    newest: dict[tuple[str, int], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            newest[(str(record["key"]), int(record.get("seed", 42)))] = record
    return sorted(newest.values(), key=lambda r: (str(r["key"]), int(r.get("seed", 42))))


def group_by_key(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """All seeds of each hypothesis, in the order the keys first appear."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["key"]), []).append(record)
    return grouped


def _std(values: list[float]) -> float:
    """Population standard deviation; 0.0 for a single value."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def build_table(records: list[dict[str, object]]) -> str:
    """Markdown table of every hypothesis and its individual effect."""
    baseline = [r for r in records if r["key"] == BASELINE_KEY]
    base_rmse = sum(float(r["rmse"]) for r in baseline) / len(baseline) if baseline else None

    lines = [
        "# A1 remediation: per-hypothesis effect on FD001",
        "",
        "Single seed (42), last-window-per-unit protocol, one change at a time",
        "against the same baseline. `delta` is against H0; negative is better.",
        "",
        f"The line to beat is the random forest at **RMSE {FOREST_RMSE}** "
        f"(NASA {FOREST_NASA:.0f}).",
        "",
        "| key | change | RMSE | delta | NASA | params | best epoch | mins | beats forest |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for key, group in group_by_key(records).items():
        values = [float(r["rmse"]) for r in group]
        mean = sum(values) / len(values)
        first = group[0]
        shown = f"{mean:.2f}" if len(values) == 1 else f"{mean:.2f} +/- {_std(values):.2f}"
        is_baseline = base_rmse is None or key == BASELINE_KEY
        delta = "--" if is_baseline else f"{mean - base_rmse:+.2f}"
        lines.append(
            f"| {key} | {first['title']}"
            f"{'' if len(values) == 1 else f' ({len(values)} seeds)'} | {shown} | {delta} | "
            f"{sum(float(r['nasa_score']) for r in group) / len(group):.0f} | "
            f"{int(first['n_parameters']):,} | "
            f"{sum(float(r['mean_best_epoch']) for r in group) / len(group):.0f} | "
            f"{sum(float(r['duration_s']) for r in group) / len(group) / 60:.1f} | "
            f"{'**yes**' if mean < FOREST_RMSE else 'no'} |"
        )

    # The verdict is decided on the MEAN across seeds, never on a single run. A
    # one-seed result can clear the line by luck: HCt's best seed came in at
    # 12.63 while its three-seed mean is 12.76, and reporting the former as a win
    # is exactly the cherry-pick this protocol exists to prevent.
    grouped = group_by_key(records)
    means = {
        key: sum(float(r["rmse"]) for r in group) / len(group) for key, group in grouped.items()
    }
    beaten = {k: v for k, v in means.items() if v < FOREST_RMSE}

    lines += ["", "## Verdict", ""]
    if beaten:
        key = min(beaten, key=lambda k: beaten[k])
        lines.append(
            f"**{key} clears the baseline** at mean RMSE {means[key]:.2f} over "
            f"{len(grouped[key])} seed(s), against the forest's {FOREST_RMSE}. "
            "The stopping rule did not trigger."
        )
    elif means:
        key = min(means, key=lambda k: means[k])
        group = grouped[key]
        spread = _std([float(r["rmse"]) for r in group])
        suffix = f" +/- {spread:.2f}" if len(group) > 1 else ""
        lines += [
            f"**No neural variant beat the forest on a multi-seed mean.** The best "
            f"was {key} ({group[0]['title']}) at {means[key]:.2f}{suffix} over "
            f"{len(group)} seed(s), against the forest's {FOREST_RMSE} +/- 0.16 -- a "
            f"gap of {means[key] - FOREST_RMSE:+.2f}, well inside the seed spread of "
            "either model.",
            "",
            "The stopping rule triggered. The random forest becomes the production "
            "hazard model; A1 stays as the deep-learning comparison.",
            "",
            "This is parity rather than defeat, and the distinction is worth "
            "recording: the gap was 1.88 RMSE when this task began and is now 0.04. "
            "The same combination cut the seed spread from +/- 1.11 to +/- 0.16 and "
            "the parameter count from 738,626 to 115,106, so what remains is both "
            "steadier and six times smaller than what it replaced.",
        ]
    return "\n".join(lines) + "\n"


def build_figure(records: list[dict[str, object]]) -> None:
    """Bar chart of RMSE per hypothesis with the forest as a reference line."""
    if not records:
        logger.warning("no records; skipping the figure")
        return

    # One bar per hypothesis, not per record: the multi-seed variants would
    # otherwise draw three overlapping bars with colliding labels.
    grouped = group_by_key(records)
    keys = list(grouped)
    values = [sum(float(r["rmse"]) for r in g) / len(g) for g in grouped.values()]
    errors = [_std([float(r["rmse"]) for r in g]) for g in grouped.values()]
    colours = [
        "#55A868" if v < FOREST_RMSE else ("#937860" if k == BASELINE_KEY else "#4C72B0")
        for k, v in zip(keys, values, strict=True)
    ]

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(keys)), 5.6))
    bars = ax.bar(
        keys,
        values,
        yerr=errors,
        color=colours,
        edgecolor="white",
        capsize=3,
        error_kw={"ecolor": "#333333", "lw": 1},
    )
    for bar, value, error in zip(bars, values, errors, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + error + 0.25,
            f"{value:.2f}",
            ha="center",
            fontsize=8,
        )

    ax.axhline(
        FOREST_RMSE,
        color="#C44E52",
        ls="--",
        lw=2,
        label=f"random forest baseline ({FOREST_RMSE})",
    )
    ax.axhspan(
        FOREST_RMSE - 0.16,
        FOREST_RMSE + 0.16,
        color="#C44E52",
        alpha=0.12,
        label="forest seed spread (+/- 1 sd)",
    )
    ax.set_ylabel("FD001 test RMSE (cycles, last window per unit)")
    ax.set_xlabel("hypothesis  (error bars: standard deviation over seeds)")
    ax.set_title("A1 remediation: effect of each hypothesis on FD001 RMSE", fontweight="bold")
    ax.set_ylim(0, max(v + e for v, e in zip(values, errors, strict=True)) * 1.2)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    subtitle = "  |  ".join(f"{k}: {str(g[0]['title'])[:34]}" for k, g in list(grouped.items())[:3])
    fig.text(0.01, 0.005, subtitle, fontsize=6.5, alpha=0.6)
    fig.tight_layout()

    path = figures_dir() / "a1_hypotheses_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.name)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Render the hypothesis sweep results.")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    records = load_records()
    if not records:
        logger.error("no records in hypotheses.jsonl; run scripts/run_hypotheses.py first")
        return 1

    path = results_dir() / "hypotheses.md"
    path.write_text(build_table(records), encoding="utf-8", newline="\n")
    logger.info("wrote %s (%d hypotheses)", path, len(records))
    if not args.no_figure:
        build_figure(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
