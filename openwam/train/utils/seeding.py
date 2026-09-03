"""Deterministic-seeding helpers used by the OpenWAM trainer.

``OpenWAMTrainer`` (dual_system / single_system / tri_system path) reads
the ``cfg.project.seed`` yaml field (or the ``OPENWAM_SEED`` env var) to opt
in to deterministic mode. The FastWAM-style seeding intentionally does NOT
touch cudnn or cuBLAS, so FSDP / fused-attention kernels stay usable. It
reaches into the helpers below for ``make_dataloader_generator`` and
``dataloader_worker_init_fn`` to make dataset-side randomness reproducible.

Default training behaviour (neither switch set) is unchanged so production
runs keep their stochasticity.

Helpers:

- ``seed_process`` seeds Python ``random``, NumPy and PyTorch (CPU + CUDA)
  global RNGs for the local rank, leaving cudnn/cuBLAS untouched — the trainer
  uses it for per-process model-init seeding without disabling autotuning.
- ``seed_everything`` is ``seed_process`` plus cudnn-deterministic + a fixed
  cuBLAS workspace. Run once at trainer construction *before* the model and
  dataset are built so DiT weight init becomes deterministic.
- ``dataloader_worker_init_fn`` is the ``worker_init_fn`` for
  ``DataLoader``; it seeds Python ``random`` and NumPy inside each worker
  process from PyTorch's auto-derived ``info.seed`` (which advances per
  epoch), so dataset transforms become reproducible across runs without
  replaying identical augmentations every epoch.
- ``make_dataloader_generator`` returns a fresh ``torch.Generator`` seeded
  for the local rank, intended to be passed to ``DataLoader(generator=...)``.
- ``per_step_seed`` derives a deterministic ``int`` seed from the run seed,
  the rank and a step counter; useful for ``torch.manual_seed`` calls done
  inside the forward pass when threading a generator all the way down to
  ``q_sample`` would require invasive changes.
- ``RANK_OFFSET`` keeps each rank's RNG stream disjoint; export so callers
  picking up state from other tools agree on the convention.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Each rank gets its own RNG stream by adding ``RANK_OFFSET * rank`` to the
# base seed.  Large enough that the per-step counter used in ``per_step_seed``
# (which adds ``step`` on top) cannot overflow into the next rank's window
# within any plausible single-run step budget — 1M steps × 1k+ ranks is far
# beyond any real training run.
RANK_OFFSET: int = 1_000_000


def seed_process(seed: int, *, rank: int = 0) -> None:
    """Seed Python / NumPy / torch (CPU + CUDA) RNGs for ``rank``, WITHOUT touching
    cudnn or cuBLAS.

    Ranks get disjoint streams via ``RANK_OFFSET`` — the same stride
    ``per_step_seed`` uses — so a process's init-time and per-step seeds share one
    rank window. Deliberately leaves cudnn/cuBLAS alone (keeps autotuning + FSDP /
    fused-attention compat); callers that also want deterministic matmul kernels
    use ``seed_everything`` instead.
    """
    rank_seed = int(seed) + RANK_OFFSET * int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32 - 1))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def seed_everything(seed: int, *, rank: int = 0) -> None:
    """``seed_process`` plus cudnn-deterministic and a fixed cuBLAS workspace.

    Adds ``torch.backends.cudnn.deterministic = True``, disables cudnn benchmark,
    and (when CUDA is available) pins the cuBLAS workspace so deterministic matmul
    kernels can be selected. Must run *before* model construction so DiT weight
    initialisation lands in deterministic territory.

    Note: we deliberately do *not* call ``torch.use_deterministic_algorithms``
    because several FSDP / attention paths used in training fall back to
    non-deterministic kernels and would crash with
    ``RuntimeError: not implemented for deterministic``.  Loss reproducibility
    therefore relies on cudnn-deterministic + a fixed cuBLAS workspace, which
    leaves only small bf16 reduction noise on non-deterministic paths.
    """
    seed_process(seed, rank=rank)
    if torch.cuda.is_available():
        # Required for deterministic cuBLAS matmul (CUDA >= 10.2); set once at startup.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Per-worker init function for ``DataLoader(worker_init_fn=...)``.

    PyTorch already derives a per-worker seed from the main process's
    ``base_seed`` and the worker_id, but it only seeds ``torch.manual_seed``
    inside the worker.  This helper additionally seeds Python ``random`` and
    NumPy with the same value so any dataset transform that reaches into
    those RNGs is reproducible too.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    base_seed = info.seed % (2**32 - 1)
    random.seed(base_seed)
    np.random.seed(base_seed)


def make_dataloader_generator(seed: int, *, rank: int = 0) -> torch.Generator:
    """Build a CPU ``torch.Generator`` for ``DataLoader(generator=...)``.

    Each rank gets a different stream so independent ranks shuffle their
    local indices independently while still being reproducible.

    Note (accelerate multi-GPU path): ``accelerator.prepare`` registers this
    generator as the prepared loader's ``synchronized_generator`` and
    re-broadcasts rank 0's generator state to every rank at each epoch start,
    so the per-rank offset does NOT diversify the epoch shuffle order there —
    all ranks intentionally share rank 0's permutation (i.e. the one seeded
    with ``seed`` itself) and ``BatchSamplerShard`` hands each rank a disjoint
    slice of it. The offset still applies to loaders that are never
    ``prepare()``-d.
    """
    g = torch.Generator()
    g.manual_seed(int(seed) + RANK_OFFSET * int(rank))
    return g


def per_step_seed(seed: int, *, rank: int = 0, step: int = 0) -> int:
    """Derive a deterministic per-step seed used for in-forward ``manual_seed`` calls.

    Combining seed, rank and step gives every (rank, step) pair its own RNG
    starting point, which is what we want when forward passes share the
    global RNG.

    Caveat — additive collision domain: this returns
    ``seed + RANK_OFFSET * rank + step``, so for two run seeds ``S`` and ``S+k``
    on the same rank, step ``j`` of one run shares its RNG state with step
    ``j+k`` of the other. For multi-seed ablation studies, prefer widely
    spaced seeds (e.g. 0 / 1000 / 2000) over a dense grid (42, 43, 44) so the
    overlap window is pushed past any step count you care about.
    """
    return int(seed) + RANK_OFFSET * int(rank) + int(step)


def _candidate_samplers(dataloader) -> list:
    """Collect every sampler reachable from ``dataloader``, outermost first.

    Handles the shapes produced by ``accelerator.prepare``: the re-created
    DataLoader is constructed with ``batch_sampler=BatchSamplerShard(...)``,
    which leaves torch's vestigial default ``SequentialSampler`` on
    ``.sampler`` (never used for iteration) while the real sampler sits at
    ``.batch_sampler.batch_sampler.sampler``. Walks a bounded number of
    ``batch_sampler`` nesting levels so both the plain and the wrapped
    layouts are covered.
    """
    samplers = []
    top = getattr(dataloader, "sampler", None)
    if top is not None:
        samplers.append(top)
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    for _ in range(4):  # bounded walk; accelerate nests exactly one level
        if batch_sampler is None:
            break
        inner = getattr(batch_sampler, "sampler", None)
        if inner is not None and inner not in samplers:
            samplers.append(inner)
        batch_sampler = getattr(batch_sampler, "batch_sampler", None)
    return samplers


def wire_sampler_seed(dataloader, run_seed: int, *, rank: int = 0) -> None:
    """Ensure the prepared dataloader's shuffle order follows ``run_seed``.

    Two supported seeding mechanisms, checked in order over every sampler
    reachable via :func:`_candidate_samplers`:

    1. A sampler exposing ``.seed`` (DistributedSampler, accelerate's
       SeedableRandomSampler): patch it to ``run_seed``. Without this,
       ``accelerator.prepare``'s auto-wrapped DistributedSampler keeps the
       upstream default ``seed=0`` and per-epoch shuffle order is identical
       regardless of ``cfg.project.seed``.
    2. A generator-driven ``RandomSampler`` (this repo's production path:
       ``build_dataloader`` passes ``make_dataloader_generator(run_seed,
       rank)``): if the generator's ``initial_seed()`` matches the expected
       per-rank seed, the shuffle order already follows ``run_seed`` —
       accelerate registers that generator as ``synchronized_generator`` and
       re-broadcasts rank 0's state each epoch, so nothing needs wiring and
       no warning is emitted.

    Warns only when neither mechanism is in effect (shuffle order then falls
    back to library defaults and will NOT vary with ``cfg.project.seed``).

    Must be called BEFORE the loader's first iteration: accelerate re-broadcasts
    rank 0's generator state via ``set_state`` on every ``__iter__``, and
    ``set_state`` overwrites ``initial_seed()`` — so mechanism 2's
    ``initial_seed()`` check is only valid pre-iteration (calling this later
    would spuriously warn on rank > 0).
    """
    samplers = _candidate_samplers(dataloader)
    for sampler in samplers:
        if hasattr(sampler, "seed"):
            old = sampler.seed
            sampler.seed = int(run_seed)
            logger.info("%s.seed wired to cfg.project.seed: %s -> %d", type(sampler).__name__, old, run_seed)
            return
    # Single source of truth for the expected per-rank seed: derive it through
    # make_dataloader_generator itself (also normalizes negative seeds the same
    # way torch.Generator.manual_seed does, so e.g. seed=-1 still matches).
    expected_seed = make_dataloader_generator(run_seed, rank=rank).initial_seed()
    for sampler in samplers:
        generator = getattr(sampler, "generator", None)
        if generator is not None and generator.initial_seed() == expected_seed:
            logger.info(
                "shuffle order already follows cfg.project.seed via %s.generator "
                "(initial_seed=%d); accelerate re-syncs this generator from rank 0 "
                "each epoch — nothing to wire.",
                type(sampler).__name__,
                expected_seed,
            )
            return
    logger.warning(
        "cfg.project.seed=%d is set but the prepared dataloader has no sampler with a "
        "``.seed`` attribute and no generator seeded from it (found %s). Per-epoch "
        "shuffle order falls back to the library default and will NOT vary with "
        "cfg.project.seed.",
        run_seed,
        "[" + ", ".join(type(s).__name__ for s in samplers) + "]" if samplers else "None",
    )
