"""Seeding must make numpy and torch draws reproducible, and seeded() must restore."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from twinmind.utils.seed import seeded, set_seed


def _draw() -> tuple[np.ndarray, torch.Tensor, float]:
    return np.random.rand(5), torch.randn(5), random.random()


def test_set_seed_makes_draws_identical() -> None:
    set_seed(1234)
    np_a, torch_a, py_a = _draw()
    set_seed(1234)
    np_b, torch_b, py_b = _draw()

    assert np.array_equal(np_a, np_b)
    assert torch.equal(torch_a, torch_b)
    assert py_a == py_b


def test_different_seeds_differ() -> None:
    set_seed(1)
    np_a, torch_a, _ = _draw()
    set_seed(2)
    np_b, torch_b, _ = _draw()

    assert not np.array_equal(np_a, np_b)
    assert not torch.equal(torch_a, torch_b)


def test_set_seed_sets_pythonhashseed() -> None:
    set_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_seeded_restores_the_previous_state() -> None:
    set_seed(99)
    expected_np, expected_torch, expected_py = _draw()

    set_seed(99)
    with seeded(12345):
        _draw()  # burn RNG state inside the block
    actual_np, actual_torch, actual_py = _draw()

    assert np.array_equal(expected_np, actual_np)
    assert torch.equal(expected_torch, actual_torch)
    assert expected_py == actual_py


def test_seeded_is_deterministic_inside_the_block() -> None:
    with seeded(555):
        inside_a = np.random.rand(3)
    with seeded(555):
        inside_b = np.random.rand(3)
    assert np.array_equal(inside_a, inside_b)
