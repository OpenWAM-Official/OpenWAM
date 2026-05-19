"""Deterministic-seeding helpers used by the OpenWAM trainer.

``OpenWAMTrainer`` (dual_system / shared_backbone / tri_system path) reads
the ``cfg.project.seed`` yaml field (or the ``OPENWAM_SEED`` env var) to opt
in to deterministic mode. The FastWAM-style seeding intentionally does NOT
touch cudnn or cuBLAS, so FSDP / fused-attention kernels stay usable. It
reaches into the helpers below for ``make_dataloader_generator`` and
``dataloader_worker_init_fn`` to make dataset-side randomness reproducible.

Default training behaviour (neither switch set) is unchanged so production
runs keep their stochasticity.

Helpers:

- ``seed_everything`` seeds Python ``random``, NumPy and PyTorch (CPU + CUDA)
  global RNGs and turns on cudnn-deterministic. Run once at trainer
  construction *before* the model and dataset are built so DiT weight init
  and any other module-construction-time randomness become deterministic.
- ``dataloader_worker_init_fn`` is the ``worker_init_fn`` for
  ``DataLoader``; it seeds Python ``random`` and NumPy inside each worker
  process from PyTorch's auto-derived ``info.seed`` (which advances per
  epoch), so dataset transforms become reproducible across runs without
  replaying identical augmentations every epoch.
- ``make_dataloader_generator`` returns a fresh ``torch.Generator`` seeded
  for the local rank, intended to be passed to ``DataLoader(generator=...)``.
- ``make_noise_generator`` returns a per-rank ``torch.Generator`` on the
  requested device, intended for diffusion noise sampling.
- ``per_step_seed`` derives a deterministic ``int`` seed from the run seed,
  the rank and a step counter; useful for ``torch.manual_seed`` calls done
  inside the forward pass when threading a generator all the way down to
  ``q_sample`` would require invasive changes.
- ``read_env_seed`` reads an integer seed from an environment variable
  (default ``OPENWAM_SEED``), returning ``None`` when unset or empty.
- ``RANK_OFFSET`` keeps each rank's RNG stream disjoint; export so callers
  picking up state from other tools agree on the convention.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

# Each rank gets its own RNG stream by adding ``RANK_OFFSET * rank`` to the
# base seed.  Large enough that the per-step counter used in ``per_step_seed``
# (which adds ``step`` on top) cannot overflow into the next rank's window
# within any plausible single-run step budget — 1M steps × 1k+ ranks is far
# beyond any real training run.
RANK_OFFSET: int = 1_000_000


def seed_everything(seed: int, *, rank: int = 0) -> None:
    """Seed Python random / numpy / torch (CPU + CUDA) for reproducible runs.

    Sets ``torch.backends.cudnn.deterministic = True``, disables cudnn
    benchmark, and (when CUDA is available) configures the cuBLAS workspace
    so deterministic matmul kernels can be selected.  Must be invoked
    *before* model construction so that DiT weight initialisation lands in
    deterministic territory.

    Note: we deliberately do *not* call ``torch.use_deterministic_algorithms``
    because several FSDP / attention paths used in training fall back to
    non-deterministic kernels and would crash with
    ``RuntimeError: not implemented for deterministic``.  Loss reproducibility
    therefore relies on cudnn-deterministic + a fixed cuBLAS workspace, which
    bounds residual non-determinism to bf16 reduction noise (small across a
    short validation run).
    """
    rank_seed = int(seed) + RANK_OFFSET * int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32 - 1))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
        # Required for deterministic cuBLAS matmul (CUDA >= 10.2). Setting
        # this once at process startup is sufficient.
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
    """
    g = torch.Generator()
    g.manual_seed(int(seed) + RANK_OFFSET * int(rank))
    return g


def make_noise_generator(
    seed: int,
    *,
    device: torch.device | str = "cpu",
    rank: int = 0,
) -> torch.Generator:
    """Build a device-bound ``torch.Generator`` for diffusion noise sampling.

    Use a per-rank stream offset so each FSDP rank generates an independent
    yet deterministic noise sequence.
    """
    g = torch.Generator(device=device)
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


def read_env_seed(env_var: str = "OPENWAM_SEED") -> Optional[int]:
    """Return the integer seed from ``env_var`` when set, else ``None``.

    Empty string is treated the same as unset so a no-op
    ``export OPENWAM_SEED=`` keeps the default non-deterministic behaviour.
    """
    raw = os.environ.get(env_var, "")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer, got {raw!r}") from exc
