"""Unit tests for ``openwam.train.utils.seeding``.

Pure CPU; no distributed init or GPU required.  Verifies the three opt-in
deterministic primitives consumed by ``OpenWAMTrainer``:

1. ``seed_everything`` makes ``torch.randn`` reproducible across calls.
2. ``make_dataloader_generator`` is per-rank reproducible.
3. ``make_noise_generator`` produces *different* streams for different ranks
   given the same base seed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openwam.train.utils.seeding import (
    make_dataloader_generator,
    make_noise_generator,
    per_step_seed,
    read_env_seed,
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


def test_noise_generator_per_rank_streams_diverge():
    g0 = make_noise_generator(42, device="cpu", rank=0)
    g1 = make_noise_generator(42, device="cpu", rank=1)

    n0 = torch.randn(20, generator=g0)
    n1 = torch.randn(20, generator=g1)

    assert not torch.equal(n0, n1), "different ranks must produce independent streams"


def test_noise_generator_same_rank_same_seed_agrees():
    g0 = make_noise_generator(42, device="cpu", rank=0)
    g1 = make_noise_generator(42, device="cpu", rank=0)

    n0 = torch.randn(20, generator=g0)
    n1 = torch.randn(20, generator=g1)

    assert torch.equal(n0, n1)


def test_per_step_seed_unique_per_step_and_rank():
    base = 42
    s00 = per_step_seed(base, rank=0, step=0)
    s01 = per_step_seed(base, rank=0, step=1)
    s10 = per_step_seed(base, rank=1, step=0)
    assert s00 != s01
    assert s00 != s10
    assert s01 != s10


def test_read_env_seed(monkeypatch):
    monkeypatch.delenv("OPENWAM_SEED", raising=False)
    assert read_env_seed() is None

    monkeypatch.setenv("OPENWAM_SEED", "")
    assert read_env_seed() is None

    monkeypatch.setenv("OPENWAM_SEED", "42")
    assert read_env_seed() == 42


def test_read_env_seed_invalid_raises(monkeypatch):
    monkeypatch.setenv("OPENWAM_SEED", "not-an-int")
    with pytest.raises(ValueError):
        read_env_seed()
