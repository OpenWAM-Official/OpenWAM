"""Tests for the dataloader-side seeding wiring (``Group 1`` of the
``cfg.project.seed`` reproducibility plan).

What is being guarded here, in order of priority:

1. **Worker RNGs are seeded** — ``dataloader_worker_init_fn`` actually seeds
   Python ``random`` and NumPy inside each worker process. Without this,
   the augmentations in ``openwam/dataloader/transforms/video.py`` use the
   worker's default time/PID-derived RNG and ablations are not comparable.

2. **Augmentations differ across epochs (single run)** — the *most important*
   correctness check. Re-iterating the DataLoader must NOT replay the same
   augmentation values, because that would silently collapse training-time
   data diversity and overfit faster than expected. This relies on PyTorch
   advancing ``loader.generator`` on every ``__iter__`` so each worker spawn
   sees a fresh ``info.seed``.

3. **Same seed -> same output across runs** — two independent runs with the
   same ``cfg.project.seed`` see the exact same per-sample augmentation
   values within the same epoch.

4. **DistributedSampler.seed wiring works** — modifying ``sampler.seed`` and
   advancing ``set_epoch`` produces a deterministic, epoch-dependent
   permutation that matches what ``OpenWAMTrainer`` does after
   ``accelerator.prepare``.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from openwam.train.utils.seeding import (
    dataloader_worker_init_fn,
    make_dataloader_generator,
)

# ---------------------------------------------------------------------------
# Test dataset — every __getitem__ pulls from both Python random and NumPy,
# so the recorded payload is sensitive to whether the worker's RNGs were
# seeded by ``dataloader_worker_init_fn``.
# ---------------------------------------------------------------------------


class _RNGSensitiveDataset(Dataset):
    def __init__(self, size: int = 16):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Mirror what openwam/dataloader/transforms/video.py does:
        # random.uniform / random.random + np.random.random.
        return {
            "idx": idx,
            "py_random": random.random(),
            "py_uniform": random.uniform(-1.0, 1.0),
            "np_random": float(np.random.random()),
        }


def _collect_epoch(loader: DataLoader) -> list[dict[str, Any]]:
    """Run one full pass through ``loader`` and return per-sample dicts."""
    out: list[dict[str, Any]] = []
    for batch in loader:
        out.extend(batch)
    return out


def _build_loader(seed: int | None, num_workers: int = 2, dataset_size: int = 16) -> DataLoader:
    kwargs: dict = dict(
        batch_size=4,
        shuffle=False,  # disable shuffle so idx order is fixed; isolate worker-RNG behaviour
        num_workers=num_workers,
        collate_fn=list,
        # Python 3.12 deprecates fork on Linux; spawn is the forward-compatible default.
        multiprocessing_context="spawn",
    )
    if seed is not None:
        kwargs["generator"] = make_dataloader_generator(seed, rank=0)
        kwargs["worker_init_fn"] = dataloader_worker_init_fn
    return DataLoader(_RNGSensitiveDataset(size=dataset_size), **kwargs)


# ---------------------------------------------------------------------------
# 1. Worker RNGs are seeded (unit-level)
# ---------------------------------------------------------------------------


def test_worker_init_fn_seeds_python_and_numpy(monkeypatch: pytest.MonkeyPatch):
    """``dataloader_worker_init_fn`` must seed both ``random`` and ``numpy``.

    Without this, augmentations like ``random.uniform(...)`` in
    transforms/video.py keep using the worker's default RNG.
    """

    # Drive the helper as if we were inside a worker with info.seed=12345.
    class _FakeWorkerInfo:
        seed = 12345

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: _FakeWorkerInfo())

    # Run the helper twice with the same fake seed; both Python random and
    # numpy should emit the same first value.
    dataloader_worker_init_fn(worker_id=0)
    first_py = random.random()
    first_np = np.random.random()

    dataloader_worker_init_fn(worker_id=0)
    second_py = random.random()
    second_np = np.random.random()

    assert first_py == second_py, "Python random was not re-seeded by helper"
    assert first_np == second_np, "NumPy random was not re-seeded by helper"


def test_worker_init_fn_is_safe_outside_worker(monkeypatch: pytest.MonkeyPatch):
    """When called in the main process (``info is None``), the helper must
    be a no-op so it doesn't accidentally clobber the main RNG."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    random.seed(999)
    np.random.seed(999)
    py_before = random.random()
    np_before = np.random.random()

    random.seed(999)
    np.random.seed(999)
    # If the helper clobbered RNG state, the subsequent draws would differ.
    dataloader_worker_init_fn(worker_id=0)
    py_after = random.random()
    np_after = np.random.random()

    assert py_before == py_after
    assert np_before == np_after


# ---------------------------------------------------------------------------
# 2. CRITICAL: augmentations differ across epochs (single run)
#
# This is the test that would have caught the "fixed worker seed" bug from
# the design discussion. If anyone ever changes the helper to a fixed-seed
# pattern, this test fails.
# ---------------------------------------------------------------------------


def test_augmentation_differs_across_epochs_same_run():
    """Re-iterating the same DataLoader must NOT replay the same per-sample
    random values. PyTorch advances ``loader.generator`` on each ``__iter__``,
    so workers spawned in epoch 1 see a different ``info.seed`` from workers
    spawned in epoch 0."""
    loader = _build_loader(seed=42, num_workers=2)
    epoch0 = _collect_epoch(loader)
    epoch1 = _collect_epoch(loader)

    # Same sample ordering (shuffle=False), but the *random* values must
    # differ between epochs. Allow occasional accidental equality on a
    # single value (Gaussian RNG can collide) — at least 75% must differ.
    differing = sum(
        1 for a, b in zip(epoch0, epoch1) if a["py_random"] != b["py_random"] or a["np_random"] != b["np_random"]
    )
    ratio = differing / len(epoch0)
    assert ratio > 0.75, (
        f"Only {differing}/{len(epoch0)} samples differed across epochs — "
        "worker init seed appears to be fixed across epochs, which would silently "
        "collapse augmentation diversity. Make sure dataloader_worker_init_fn "
        "uses get_worker_info().seed (PyTorch-derived per-epoch seed), not a "
        "fixed (base_seed + worker_id) value."
    )


# ---------------------------------------------------------------------------
# 3. Same seed -> same output across runs
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_across_runs():
    """Two independent DataLoader constructions with the same seed must
    produce identical per-sample random values for the same epoch."""
    loader_a = _build_loader(seed=42, num_workers=2)
    loader_b = _build_loader(seed=42, num_workers=2)
    out_a = _collect_epoch(loader_a)
    out_b = _collect_epoch(loader_b)

    assert len(out_a) == len(out_b)
    for sa, sb in zip(out_a, out_b):
        assert sa["idx"] == sb["idx"]
        assert sa["py_random"] == sb["py_random"], f"Python random diverged at idx={sa['idx']}"
        assert sa["np_random"] == sb["np_random"], f"NumPy random diverged at idx={sa['idx']}"


def test_different_seeds_produce_different_outputs():
    """Different seeds must produce different per-sample random values —
    otherwise the seed knob does nothing observable."""
    out_42 = _collect_epoch(_build_loader(seed=42, num_workers=2))
    out_43 = _collect_epoch(_build_loader(seed=43, num_workers=2))

    differing = sum(
        1 for a, b in zip(out_42, out_43) if a["py_random"] != b["py_random"] or a["np_random"] != b["np_random"]
    )
    assert differing / len(out_42) > 0.75


def test_no_seed_keeps_legacy_non_determinism():
    """Without ``seed`` (the production path before this change), runs
    should NOT be reproducible. Catches a regression where the new code
    accidentally seeds workers even when the user didn't ask for it."""
    out_a = _collect_epoch(_build_loader(seed=None, num_workers=2))
    out_b = _collect_epoch(_build_loader(seed=None, num_workers=2))
    # Some samples may collide by luck; require <75% identity.
    identical = sum(
        1 for a, b in zip(out_a, out_b) if a["py_random"] == b["py_random"] and a["np_random"] == b["np_random"]
    )
    assert identical / len(out_a) < 0.75


# ---------------------------------------------------------------------------
# 4. DistributedSampler.seed wiring (the post-prepare patch in trainer)
# ---------------------------------------------------------------------------


def test_distributed_sampler_seed_takes_effect():
    """The post-``prepare`` patch sets ``sampler.seed = cfg.project.seed``.
    Verify a DistributedSampler honours that and that ``set_epoch`` still
    produces a different permutation for each epoch."""
    ds = _RNGSensitiveDataset(size=64)
    sampler_a = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    sampler_b = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)

    # Same default seed=0 => identical permutations.
    sampler_a.set_epoch(0)
    sampler_b.set_epoch(0)
    assert list(sampler_a) == list(sampler_b)

    # Patch sampler_b.seed (this mirrors what the trainer does after prepare).
    sampler_b.seed = 42
    sampler_b.set_epoch(0)
    sampler_a.set_epoch(0)
    perm_default = list(sampler_a)
    perm_patched = list(sampler_b)
    assert perm_default != perm_patched, "Patching sampler.seed had no effect on permutation"

    # Same seed=42, two different epochs => different permutations (the
    # ``seed + epoch`` mechanism from PyTorch DistributedSampler).
    sampler_b.set_epoch(0)
    perm_epoch0 = list(sampler_b)
    sampler_b.set_epoch(1)
    perm_epoch1 = list(sampler_b)
    assert perm_epoch0 != perm_epoch1, "set_epoch did not advance the shuffle permutation"


def test_distributed_sampler_seed_reproduces_across_constructions():
    """Two DistributedSamplers built with the same patched seed must yield
    bit-exact identical permutations for matching epochs."""
    ds = _RNGSensitiveDataset(size=64)
    sampler_a = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    sampler_b = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    sampler_a.seed = 42
    sampler_b.seed = 42

    for epoch in range(3):
        sampler_a.set_epoch(epoch)
        sampler_b.set_epoch(epoch)
        assert list(sampler_a) == list(sampler_b), f"Epoch {epoch} permutations diverged"


# ---------------------------------------------------------------------------
# 5. _wire_sampler_seed helper — production path + warn fallback
#
# These guard the "silent failure" Issue 1 from the review: if the prepared
# DataLoader doesn't expose a sampler with a ``.seed`` attribute, the user
# must see a WARNING, not a successful-looking log line.
# ---------------------------------------------------------------------------


class _FakeSamplerWithSeed:
    def __init__(self, seed: int = 0):
        self.seed = seed


class _FakeSamplerNoSeed:
    """Mirrors e.g. a non-shuffled sampler or some accelerate wrappers that
    don't expose a per-instance seed attribute."""


class _FakeDataLoader:
    def __init__(self, sampler=None, batch_sampler=None):
        self.sampler = sampler
        self.batch_sampler = batch_sampler


class _FakeBatchSampler:
    def __init__(self, sampler):
        self.sampler = sampler


def test_wire_sampler_seed_patches_top_level_sampler(caplog: pytest.LogCaptureFixture):
    """Happy path: dataloader.sampler exposes ``seed`` -> patched, INFO logged."""
    import logging

    from openwam.train.utils.seeding import wire_sampler_seed

    sampler = _FakeSamplerWithSeed(seed=0)
    dataloader = _FakeDataLoader(sampler=sampler)
    with caplog.at_level(logging.INFO, logger="openwam.train.utils.seeding"):
        wire_sampler_seed(dataloader, run_seed=42)
    assert sampler.seed == 42
    assert any("wired to cfg.project.seed" in rec.message for rec in caplog.records)
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_wire_sampler_seed_falls_through_to_batch_sampler(caplog: pytest.LogCaptureFixture):
    """When ``dataloader.sampler`` is None, the helper falls back to
    ``dataloader.batch_sampler.sampler``. Mirrors accelerate's wrapping."""
    import logging

    from openwam.train.utils.seeding import wire_sampler_seed

    inner = _FakeSamplerWithSeed(seed=0)
    dataloader = _FakeDataLoader(sampler=None, batch_sampler=_FakeBatchSampler(inner))
    with caplog.at_level(logging.INFO, logger="openwam.train.utils.seeding"):
        wire_sampler_seed(dataloader, run_seed=7)
    assert inner.seed == 7
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)


def test_wire_sampler_seed_warns_when_no_sampler_with_seed(caplog: pytest.LogCaptureFixture):
    """CORE Issue 1 guard: if neither sampler nor batch_sampler.sampler exposes
    ``.seed``, the helper must WARN — otherwise the user gets shuffle order
    pinned to the upstream default while believing project.seed took effect."""
    import logging

    from openwam.train.utils.seeding import wire_sampler_seed

    # Case A: sampler exists but lacks ``.seed`` (e.g. plain RandomSampler).
    dataloader = _FakeDataLoader(sampler=_FakeSamplerNoSeed())
    with caplog.at_level(logging.WARNING, logger="openwam.train.utils.seeding"):
        wire_sampler_seed(dataloader, run_seed=42)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "Expected a WARNING when sampler has no .seed attribute"
    assert "_FakeSamplerNoSeed" in warnings[-1].message
    caplog.clear()

    # Case B: no sampler at all (e.g. IterableDataset).
    dataloader = _FakeDataLoader(sampler=None, batch_sampler=None)
    with caplog.at_level(logging.WARNING, logger="openwam.train.utils.seeding"):
        wire_sampler_seed(dataloader, run_seed=42)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "Expected a WARNING when sampler is None"
    assert "None" in warnings[-1].message


def test_wire_sampler_seed_with_real_distributed_sampler(caplog: pytest.LogCaptureFixture):
    """End-to-end sanity: a real PyTorch DistributedSampler wired by the
    helper produces the same epoch-0 permutation as one constructed with the
    target seed directly. Guards against accidental drift between the helper
    and what users would get if they passed seed=N to DistributedSampler."""
    import logging

    from openwam.train.utils.seeding import wire_sampler_seed

    ds = _RNGSensitiveDataset(size=32)
    wired_sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    dataloader = _FakeDataLoader(sampler=wired_sampler)

    with caplog.at_level(logging.INFO, logger="openwam.train.utils.seeding"):
        wire_sampler_seed(dataloader, run_seed=42)

    reference = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=42)
    wired_sampler.set_epoch(0)
    reference.set_epoch(0)
    assert list(wired_sampler) == list(reference)


def test_wire_sampler_seed_accepts_generator_driven_random_sampler(caplog: pytest.LogCaptureFixture):
    """Production path: accelerate's prepared loader carries a vestigial
    SequentialSampler on ``.sampler`` while the real generator-driven
    RandomSampler sits at ``.batch_sampler.batch_sampler.sampler``. When that
    generator was seeded by make_dataloader_generator(run_seed, rank), the
    helper must recognize it (INFO) and must NOT warn."""
    import logging

    from openwam.train.utils.seeding import RANK_OFFSET, make_dataloader_generator, wire_sampler_seed

    ds = list(range(64))
    rank = 2
    inner_sampler = torch.utils.data.RandomSampler(ds, generator=make_dataloader_generator(42, rank=rank))
    inner_batch = torch.utils.data.BatchSampler(inner_sampler, batch_size=8, drop_last=False)

    class _FakeShard:  # mirrors accelerate.BatchSamplerShard
        def __init__(self, batch_sampler):
            self.batch_sampler = batch_sampler

    dataloader = _FakeDataLoader(
        sampler=torch.utils.data.SequentialSampler(ds),  # torch's vestigial default
        batch_sampler=_FakeShard(inner_batch),
    )
    with caplog.at_level(logging.INFO, logger="openwam.train.utils.seeding"):
        wire_sampler_seed(dataloader, run_seed=42, rank=rank)
    assert any("already follows cfg.project.seed" in rec.message for rec in caplog.records)
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    assert inner_sampler.generator.initial_seed() == 42 + RANK_OFFSET * rank


def test_wire_sampler_seed_warns_on_foreign_generator(caplog: pytest.LogCaptureFixture):
    """A generator NOT seeded from run_seed (e.g. accelerate injecting a
    randomly-seeded one when none was passed) must still trigger the warning —
    shuffle order would not follow cfg.project.seed."""
    import logging

    from openwam.train.utils.seeding import wire_sampler_seed

    ds = list(range(64))
    foreign = torch.Generator()
    foreign.manual_seed(123456789)
    sampler = torch.utils.data.RandomSampler(ds, generator=foreign)
    dataloader = _FakeDataLoader(sampler=sampler)
    with caplog.at_level(logging.WARNING, logger="openwam.train.utils.seeding"):
        wire_sampler_seed(dataloader, run_seed=42, rank=0)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "Expected a WARNING for a generator not derived from run_seed"
    assert "RandomSampler" in warnings[-1].message
