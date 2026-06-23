"""Unit tests for ``openwam.train.utils.seeding``.

Pure CPU; no distributed init or GPU required. Verifies the opt-in
deterministic primitives consumed by ``OpenWAMTrainer``:

1. ``seed_everything`` makes ``torch.randn`` reproducible across calls.
2. ``make_dataloader_generator`` is per-rank reproducible.
3. ``per_step_seed`` gives each (rank, step) pair its own seed.
"""

from __future__ import annotations

import numpy as np
import torch

from openwam.train.utils.seeding import (
    make_dataloader_generator,
    per_step_seed,
    seed_everything,
)


def test_seed_everything_reproducible_torch_randn():
    seed_everything(42)
    a = torch.randn(10)

    seed_everything(42)
    b = torch.randn(10)

    assert torch.equal(a, b), "seed_everything must make torch.randn reproducible"


def test_seed_everything_reproducible_numpy_and_python():
    import random as _random

    seed_everything(7)
    np_a = np.random.rand(5)
    py_a = [_random.random() for _ in range(5)]

    seed_everything(7)
    np_b = np.random.rand(5)
    py_b = [_random.random() for _ in range(5)]

    assert np.allclose(np_a, np_b)
    assert py_a == py_b


def test_dataloader_generator_reproducible_when_freshly_made():
    g1 = make_dataloader_generator(42, rank=0)
    g2 = make_dataloader_generator(42, rank=0)

    a = torch.randperm(100, generator=g1)
    b = torch.randperm(100, generator=g2)

    assert torch.equal(a, b), "freshly-made generators with same seed must agree"


def test_dataloader_generator_is_stateful():
    g = make_dataloader_generator(42, rank=0)
    a = torch.randperm(100, generator=g)
    b = torch.randperm(100, generator=g)

    assert not torch.equal(a, b), "consecutive draws on a stateful generator must differ"


def test_per_step_seed_unique_per_step_and_rank():
    base = 42
    s00 = per_step_seed(base, rank=0, step=0)
    s01 = per_step_seed(base, rank=0, step=1)
    s10 = per_step_seed(base, rank=1, step=0)
    assert s00 != s01
    assert s00 != s10
    assert s01 != s10
