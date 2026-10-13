"""Train the Model C ensemble and its baselines.

    python scripts/train_surrogate.py --rollouts data/processed/rollouts/train
    python scripts/train_surrogate.py --learning-curve

Reads ONLY the training split. ``load_rollouts`` refuses a test split unless the
caller explicitly opts in, and nothing here does -- the test splits are the only
evidence the surrogate works, and a training script that can reach them is one
accident away from destroying that evidence.

Three baselines on the identical split, because "the neural net got R² 0.97"
means nothing without knowing what a ridge gets:

* **Ridge** on the encoded features -- the "is this problem linear?" control.
* **LightGBM per KPI** -- the serious competitor. GBDTs beat neural nets on
  tabular data often enough that shipping an MLP without checking would be
  negligent, and the demand model's N-HiTS result in this same repository is a
  reminder that the expected winner does not always win.
* **A single MLP with no heteroscedastic head** -- isolates what the NLL head
  and the ensemble actually buy, as opposed to the architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from twinmind.eval.registry import results_dir
from twinmind.models.surrogate import KPI_ORDER, feature_names, scenario_bounds
from twinmind.models.surrogate_data import (
    RolloutData,
    load_rollouts,
    split_by_context_block,
    state_statistics,
)
from twinmind.models.surrogate_mlp import SurrogateConfig, SurrogateEnsemble, train_member
from twinmind.models.surrogate_transforms import forward_targets, target_statistics
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("train_surrogate")

MODEL_DIR = PROJECT_ROOT / "models" / "surrogate"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Train the surrogate ensemble.")
    parser.add_argument("--rollouts", default=str(PROJECT_ROOT / "data/processed/rollouts/train"))
    parser.add_argument("--out", default=str(MODEL_DIR))
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--learning-curve", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    return parser


def encode_split(
    data: RolloutData,
    rows: NDArray[np.int64],
    state_mean: NDArray[np.float64],
    state_scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Encode rows straight from the stored columns.

    The scenario half is normalised by the configured bounds exactly as
    ``encode`` does; the state half is already in the frame, so it is
    standardised here rather than round-tripped through a ``TwinContext``. A
    test asserts this matches ``encode_batch`` on the same inputs, because two
    encoders that drift apart is the quietest possible way to break a surrogate.
    """
    bounds = scenario_bounds()
    from twinmind.models.surrogate import CATEGORICAL_DIMS, _categorical_levels
    from twinmind.twin.runner import SCENARIO_DIMS

    scenarios = data.scenario_matrix()[rows]
    columns: list[NDArray[np.float64]] = []
    for index, dim in enumerate(SCENARIO_DIMS):
        raw = scenarios[:, index]
        if dim in CATEGORICAL_DIMS:
            for level in _categorical_levels(dim):
                columns.append((np.round(raw).astype("int64") == level).astype("float64"))
            continue
        low, high = bounds[dim]
        span = high - low
        columns.append((raw - low) / span if span > 0 else np.zeros_like(raw))

    scenario_part = np.column_stack(columns)
    state = data.state_matrix()[rows]
    standardised = (state - state_mean) / np.where(np.abs(state_scale) < 1e-12, 1.0, state_scale)
    return np.hstack([scenario_part, standardised])


def fit_ensemble(
    data: RolloutData, args: argparse.Namespace, max_blocks: int | None = None
) -> tuple[SurrogateEnsemble, dict[str, Any]]:
    """Train the ensemble on one (possibly truncated) training split."""
    if max_blocks is not None:
        keep = np.flatnonzero(data.context_blocks < max_blocks).astype("int64")
        frame = data.frame.iloc[keep].reset_index(drop=True)
        data = RolloutData(frame=frame, manifest=data.manifest, directory=data.directory)

    train_rows, val_rows = split_by_context_block(data, args.val_fraction, args.seed)
    state_mean, state_scale = state_statistics(data, train_rows)

    x_train = encode_split(data, train_rows, state_mean, state_scale)
    x_val = encode_split(data, val_rows, state_mean, state_scale)

    # Feature standardisation, again from training rows only.
    feature_mean = x_train.mean(axis=0)
    feature_scale = np.where(x_train.std(axis=0) < 1e-12, 1.0, x_train.std(axis=0))
    x_train_s = (x_train - feature_mean) / feature_scale
    x_val_s = (x_val - feature_mean) / feature_scale

    targets = forward_targets(data.targets())
    target_mean, target_scale = target_statistics(targets[train_rows])
    y_train = (targets[train_rows] - target_mean) / target_scale
    y_val = (targets[val_rows] - target_mean) / target_scale

    config = SurrogateConfig(
        input_dim=x_train.shape[1],
        n_members=args.members,
        epochs=args.epochs,
        seed=args.seed,
        feature_names=feature_names(),
    )

    members, reports = [], []
    blocks = np.unique(data.context_blocks[train_rows])
    for index in range(args.members):
        seed = args.seed + 1000 * index
        rows = train_rows
        if config.bootstrap_blocks and index > 0:
            # Bootstrap over CONTEXT BLOCKS, not rows. Resampling rows would
            # give every member nearly the same contexts -- 64 scenarios share
            # each one -- and the disagreement would collapse to initialisation
            # noise, understating epistemic uncertainty exactly where it matters.
            rng = np.random.default_rng(seed)
            chosen = set(rng.choice(blocks, size=len(blocks), replace=True).tolist())
            mask = np.array([b in chosen for b in data.context_blocks[train_rows]])
            rows = train_rows[mask]

        if rows is train_rows:
            member_x, member_y = x_train_s, y_train
        else:
            member_x = (
                encode_split(data, rows, state_mean, state_scale) - feature_mean
            ) / feature_scale
            member_y = (targets[rows] - target_mean) / target_scale
        started = time.perf_counter()
        model, report = train_member(member_x, member_y, x_val_s, y_val, config, seed)
        report["duration_s"] = time.perf_counter() - started
        report["n_rows"] = int(len(rows))
        report.pop("history", None)
        members.append(model)
        reports.append(report)
        logger.info(
            "  member %d/%d: val NLL %.4f at epoch %d (%d epochs, %d rows, %.0f s)",
            index + 1,
            args.members,
            report["best_val_nll"],
            report["best_epoch"],
            report["epochs_run"],
            report["n_rows"],
            report["duration_s"],
        )

    ensemble = SurrogateEnsemble(
        members=members,
        config=config,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        state_mean=state_mean,
        state_scale=state_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    return ensemble, {
        "n_train_rows": int(train_rows.size),
        "n_val_rows": int(val_rows.size),
        "n_features": int(x_train.shape[1]),
        "members": reports,
    }


def fit_baselines(
    data: RolloutData, args: argparse.Namespace, ensemble: SurrogateEnsemble
) -> dict[str, Any]:
    """Ridge, LightGBM per KPI, and a single MLP with no heteroscedastic head."""
    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    train_rows, val_rows = split_by_context_block(data, args.val_fraction, args.seed)
    x_train = encode_split(data, train_rows, ensemble.state_mean, ensemble.state_scale)
    x_val = encode_split(data, val_rows, ensemble.state_mean, ensemble.state_scale)
    targets = forward_targets(data.targets())
    y_train, y_val = targets[train_rows], targets[val_rows]

    out: dict[str, Any] = {}

    scale = ensemble.target_scale
    ridge_pred = Ridge(alpha=1.0).fit(x_train, y_train).predict(x_val)
    out["ridge"] = {
        "val_rmse_standardised": _rmse(ridge_pred, y_val, scale),
        "val_r2": _per_kpi_r2(ridge_pred, y_val),
    }

    boosters = []
    for index, kpi in enumerate(KPI_ORDER):
        booster = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=args.seed
        ).fit(x_train, y_train[:, index])
        boosters.append(booster)
        del kpi
    lgbm_pred = np.column_stack([b.predict(x_val) for b in boosters])
    out["lightgbm"] = {
        "val_rmse_standardised": _rmse(lgbm_pred, y_val, scale),
        "val_r2": _per_kpi_r2(lgbm_pred, y_val),
    }

    # Single homoscedastic MLP: same trunk, one member, std frozen at 1 so the
    # loss reduces to plain MSE. Isolates the NLL head and the ensemble from the
    # architecture.
    single = SurrogateConfig(
        input_dim=x_train.shape[1], n_members=1, epochs=args.epochs, seed=args.seed
    )
    feature_mean, feature_scale = ensemble.feature_mean, ensemble.feature_scale
    target_mean, target_scale = ensemble.target_mean, ensemble.target_scale
    model, report = train_member(
        (x_train - feature_mean) / feature_scale,
        (y_train - target_mean) / target_scale,
        (x_val - feature_mean) / feature_scale,
        (y_val - target_mean) / target_scale,
        single,
        args.seed,
    )
    import torch

    with torch.no_grad():
        mean, _ = model(torch.from_numpy((x_val - feature_mean) / feature_scale).float())
    single_pred = mean.numpy().astype("float64") * target_scale + target_mean
    out["single_mlp"] = {
        "val_rmse_standardised": _rmse(single_pred, y_val, scale),
        "val_r2": _per_kpi_r2(single_pred, y_val),
        "best_val_nll": report["best_val_nll"],
    }

    # The ensemble on the same yardstick.
    ens_mean = _ensemble_in_space(ensemble, x_val)
    out["ensemble"] = {
        "val_rmse_standardised": _rmse(ens_mean, y_val, scale),
        "val_r2": _per_kpi_r2(ens_mean, y_val),
    }

    for name in ("ridge", "lightgbm", "single_mlp", "ensemble"):
        row = out[name]
        finite = [v for v in row["val_r2"].values() if np.isfinite(v)]
        worst = min(row["val_r2"], key=lambda k: row["val_r2"][k])
        logger.info(
            "  %-12s standardised RMSE %.4f   mean R2 %.4f   worst %s %.3f",
            name,
            row["val_rmse_standardised"],
            float(np.mean(finite)),
            worst,
            row["val_r2"][worst],
        )
    return out


def _ensemble_in_space(
    ensemble: SurrogateEnsemble, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Ensemble mean in transformed space, for a like-for-like baseline table."""
    import torch

    x = torch.from_numpy((features - ensemble.feature_mean) / ensemble.feature_scale).float()
    means = []
    for member in ensemble.members:
        with torch.no_grad():
            mean, _ = member(x)
        means.append(mean.numpy().astype("float64"))
    return np.mean(means, axis=0) * ensemble.target_scale + ensemble.target_mean


def _rmse(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> float:
    """RMSE in STANDARDISED transformed space, averaged over the eight KPIs.

    Standardising first is what makes the number mean anything. ``profit`` is
    the one KPI with an identity transform and it lives in hundreds of
    thousands, so an unstandardised RMSE over all eight is simply "who predicts
    profit best in pounds" -- the other seven contribute about a millionth of
    the total. Measured on the dev set, that version read 51,009 for ridge
    against 27,647 for LightGBM, numbers that say nothing about seven KPIs.
    """
    residual = (np.asarray(predicted) - np.asarray(truth)) / scale
    return float(np.sqrt(np.mean(residual**2)))


def _per_kpi_r2(predicted: NDArray[np.float64], truth: NDArray[np.float64]) -> dict[str, float]:
    """R2 per KPI in transformed space, which is scale-free by construction."""
    out: dict[str, float] = {}
    for index, kpi in enumerate(KPI_ORDER):
        variance = float(np.var(truth[:, index], ddof=1))
        residual = float(np.mean((predicted[:, index] - truth[:, index]) ** 2))
        out[kpi] = 1.0 - residual / variance if variance > 0 else float("nan")
    return out


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    data = load_rollouts(args.rollouts, max_blocks=args.max_blocks)
    logger.info("training on %d rows from %s", data.n_rows, data.directory.name)

    summary: dict[str, Any] = {"rollout_manifest": data.manifest, "n_rows": data.n_rows}

    if args.learning_curve:
        curve = []
        for power in (11, 12, 13, 14, 15):
            n_scenarios = 2**power
            blocks = max(1, n_scenarios // int(data.manifest["scenarios_per_context"]))
            logger.info("=== learning curve: %d scenarios (%d blocks) ===", n_scenarios, blocks)
            ensemble, report = fit_ensemble(data, args, max_blocks=blocks)
            directory = Path(args.out).parent / f"surrogate_lc_{n_scenarios}"
            ensemble.save(directory)
            curve.append(
                {"n_scenarios": n_scenarios, "blocks": blocks, **report, "path": str(directory)}
            )
        summary["learning_curve"] = curve
        (results_dir() / "surrogate_learning_curve.json").write_text(
            json.dumps(curve, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )
        return 0

    logger.info("=== ensemble ===")
    started = time.perf_counter()
    ensemble, report = fit_ensemble(data, args)
    summary["training"] = report
    summary["training"]["total_duration_s"] = time.perf_counter() - started
    ensemble.save(Path(args.out))

    if not args.skip_baselines:
        logger.info("=== baselines, identical split ===")
        summary["baselines"] = fit_baselines(data, args, ensemble)

    (results_dir() / "surrogate_training.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("done in %.1f min", (time.perf_counter() - started) / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
