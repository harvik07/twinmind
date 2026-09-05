"""Determinism helpers.

`set_seed` is the blunt instrument for scripts; `seeded` is a scoped version
that restores the previous RNG state on exit, so a seeded sub-computation
(e.g. one simulator replication) cannot silently reseed the caller.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator
from typing import Any

import numpy as np

try:  # torch is a hard dependency, but keep the import guarded for slim CI images
    import torch
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None  # type: ignore[assignment]


DEFAULT_SEED = 42


def set_seed(n: int, deterministic_torch: bool = True) -> None:
    """Seed `random`, numpy and torch (CPU + all CUDA devices) and PYTHONHASHSEED.

    PYTHONHASHSEED only affects interpreters started *after* this call; it is set
    so that subprocesses spawned by training scripts inherit it.
    """
    os.environ["PYTHONHASHSEED"] = str(n)
    random.seed(n)
    np.random.seed(n)
    if torch is not None:
        torch.manual_seed(n)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(n)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def seeded(n: int) -> Iterator[None]:
    """Run a block under seed `n`, then restore the prior RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state: Any = None
    cuda_state: Any = None
    if torch is not None:
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        set_seed(n)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed
