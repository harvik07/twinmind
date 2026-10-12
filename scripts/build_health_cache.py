"""Precompute A1's predictions over every replay-pool engine and cycle.

    python scripts/build_health_cache.py --subset FD002 --model-seed 42

The twin cannot call the network on its hot path: a 28-day run queries machine
health thousands of times, and A1 with 30 MC-dropout passes takes milliseconds
per window. Running it once offline over every (engine, cycle) the twin could
ever ask about turns that into an array lookup.

What gets cached is the real ``HealthModel.predict_batch`` output -- the same
code path the twin would call live -- so the cache is a performance device and
not a second implementation. ``tests/test_twin.py`` asserts a sample of cached
rows matches a live ``predict()`` call.
"""

from __future__ import annotations

import argparse
import sys
from typing import cast

import numpy as np

from twinmind.twin.health_replay import HealthCache, PoolSpec, cache_file, pool_engine_ids
from twinmind.utils.config import DataConfig, TwinConfig, load_config
from twinmind.utils.logging import get_logger
from twinmind.utils.paths import PROJECT_ROOT

logger = get_logger("build_health_cache")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for this script."""
    parser = argparse.ArgumentParser(description="Precompute A1 predictions for the twin.")
    parser.add_argument("--subset", default=None, help="Defaults to twin.yaml health_replay.")
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-passes", type=int, default=30, help="MC-dropout passes.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the file exists.")
    return parser


def main() -> int:
    """Entry point."""
    args = build_parser().parse_args()
    twin = cast(TwinConfig, load_config("twin"))
    data = cast(DataConfig, load_config("data"))
    replay = twin.health_replay

    subset = args.subset or replay.subset
    model_seed = args.model_seed if args.model_seed is not None else replay.model_seed
    checkpoint = PROJECT_ROOT / (args.checkpoint or replay.checkpoint)
    if not checkpoint.exists():
        logger.error("no checkpoint at %s", checkpoint)
        return 1

    window = data.cmapss.window_length
    spec = PoolSpec(
        subset=subset,
        model_seed=model_seed,
        val_fraction=data.cmapss.val_fraction,
        window=window,
        checkpoint=str(checkpoint),
    )
    destination = cache_file(PROJECT_ROOT / replay.cache_path, spec)
    if destination.exists() and not args.force:
        logger.info("cache already exists at %s (use --force to rebuild)", destination)
        return 0

    train_ids, pool_ids = pool_engine_ids(subset, model_seed, data.cmapss.val_fraction)
    logger.info(
        "%s seed %d: %d training engines, %d pool engines (key %s)",
        subset,
        model_seed,
        len(train_ids),
        len(pool_ids),
        spec.key(),
    )
    if len(pool_ids) < replay.min_pool_engines:
        logger.error(
            "pool has %d engines, below the configured minimum of %d",
            len(pool_ids),
            replay.min_pool_engines,
        )
        return 1

    model, normaliser, sensors = _load_model(checkpoint, args.n_passes, data)
    frame = _normalised_pool_frame(subset, pool_ids, normaliser)

    engine_ids: list[int] = []
    failure_cycles: list[int] = []
    offsets: list[int] = [0]
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    horizon_h = 0

    for position, engine in enumerate(pool_ids):
        block = frame[frame["unit"] == engine].sort_values("cycle")
        values = block[sensors].to_numpy(dtype="float32")
        n_cycles = values.shape[0]
        if n_cycles <= window:
            logger.warning("engine %d has only %d cycles, skipping", engine, n_cycles)
            continue

        # One window per cycle from `window` to the last: window t covers cycles
        # [t-window, t), so prediction i corresponds to age window+i. That is
        # exactly the indexing HealthCache.predictions_at assumes.
        starts = np.arange(0, n_cycles - window + 1)
        windows = np.stack([values[s : s + window] for s in starts])
        predictions = []
        for begin in range(0, windows.shape[0], args.batch_size):
            predictions.extend(model.predict_batch(windows[begin : begin + args.batch_size]))

        engine_ids.append(int(engine))
        failure_cycles.append(int(block["cycle"].max()))
        means.append(np.array([p.rul_mean for p in predictions], dtype="float64"))
        stds.append(np.array([p.rul_std for p in predictions], dtype="float64"))
        probabilities.append(np.array([p.p_fail_within_h for p in predictions], dtype="float64"))
        offsets.append(offsets[-1] + len(predictions))
        horizon_h = predictions[0].horizon_h
        if position % 10 == 0:
            logger.info("  %d/%d engines", position + 1, len(pool_ids))

    if not engine_ids:
        logger.error("no engine produced predictions")
        return 1

    # Sorted by engine id so HealthCache.index_of can binary-search.
    order = np.argsort(engine_ids)
    sorted_means = [means[i] for i in order]
    sorted_stds = [stds[i] for i in order]
    sorted_probabilities = [probabilities[i] for i in order]
    lengths = [m.size for m in sorted_means]

    cache = HealthCache(
        engine_ids=np.array([engine_ids[i] for i in order], dtype="int64"),
        failure_cycles=np.array([failure_cycles[i] for i in order], dtype="int64"),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype("int64"),
        rul_mean=np.concatenate(sorted_means),
        rul_std=np.concatenate(sorted_stds),
        p_fail=np.concatenate(sorted_probabilities),
        horizon_h=horizon_h,
        window=window,
        key=spec.key(),
        train_engine_ids=np.array(train_ids, dtype="int64"),
    )
    cache.save(destination)
    logger.info(
        "cached %d predictions over %d engines; lifetimes %d-%d cycles (mean %.1f)",
        cache.rul_mean.size,
        cache.n_engines,
        int(cache.failure_cycles.min()),
        int(cache.failure_cycles.max()),
        float(cache.failure_cycles.mean()),
    )
    return 0


def _load_model(checkpoint, n_passes, data):  # type: ignore[no-untyped-def]
    """Rebuild the trained A1 model and its normaliser from a checkpoint."""
    import torch

    from twinmind.data.windowing import OperatingConditionNormaliser
    from twinmind.models.a1_cnn_bilstm import A1Config, A1HealthModel, A1Net

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = A1Net(A1Config(**payload["a1_config"]))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    model = A1HealthModel(net, n_passes=n_passes, rul_cap=data.cmapss.rul_cap)
    # The checkpoint stores the normaliser as a plain dict (see a1_train's
    # `data.normaliser.to_dict()`), so it has to be rehydrated before use.
    normaliser = OperatingConditionNormaliser.from_dict(payload["normaliser"])
    logger.info(
        "loaded %s (subset %s, seed %s, %d sensors)",
        checkpoint.name,
        payload.get("subset"),
        payload.get("seed"),
        len(payload["sensors"]),
    )
    return model, normaliser, payload["sensors"]


def _normalised_pool_frame(subset, pool_ids, normaliser):  # type: ignore[no-untyped-def]
    """Pool rows, normalised exactly as training normalised them."""
    from twinmind.data.cmapss_prep import compute_rul
    from twinmind.data.loaders import load_cmapss

    raw = load_cmapss(subset, "train")
    twin_config = cast(DataConfig, load_config("data"))
    raw = raw.assign(rul=compute_rul(raw, cap=twin_config.cmapss.rul_cap))
    pool = raw[raw["unit"].isin(pool_ids)].reset_index(drop=True)
    return normaliser.transform(pool)


if __name__ == "__main__":
    sys.exit(main())
