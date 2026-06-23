"""Device/dtype movement helpers for the Cosmos25 plain-object submodules.

``Wan2pt1VAEInterface`` (the upstream Cosmos VAE wrapper) and
:class:`Reason1LiveTextEncoder` are plain Python objects, not ``nn.Module`` s,
so ``nn.Module.to(...)`` on the surrounding pipeline wrapper does not reach
them. :meth:`Cosmos25VideoBackbone.set_dtype_device` calls the helpers here to
move their inner ``nn.Module`` plus the auxiliary tensors explicitly.

Lives in its own module (rather than in ``cosmos25_backbone.py``) so both the
backbone and :mod:`pipeline_builder` can import it without an import cycle and
without triggering the lazy ``cosmos_predict2`` import.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

# The actual nn.Module lives at `iface.model.model` (a `WanVAE_`). Six mean/std
# tensors sit alongside it on `iface.model`: both the parameters and these
# tensors need explicit moves when set_dtype_device is called.
_COSMOS_VAE_TENSOR_ATTRS: tuple = (
    "mean",
    "std",
    "img_mean",
    "img_std",
    "video_mean",
    "video_std",
)


def _vae_inner_module(vae: Any) -> Optional[nn.Module]:
    """Return the inner nn.Module of a Cosmos VAE wrapper, or None."""
    if vae is None:
        return None
    outer = getattr(vae, "model", None)
    inner = getattr(outer, "model", None) if outer is not None else None
    return inner if isinstance(inner, nn.Module) else None


def _move_cosmos_vae(vae: Any, *, dtype: torch.dtype, device: torch.device) -> None:
    """Move the Cosmos VAE inner nn.Module + mean/std tensors to (dtype, device)."""
    if vae is None:
        return
    outer = getattr(vae, "model", None)
    if outer is None:
        return
    inner = getattr(outer, "model", None)
    if isinstance(inner, nn.Module):
        inner.to(dtype=dtype, device=device)
    # Also update the cached `WanVAE.device` / `WanVAE.dtype` attrs so internal
    # encode/decode paths that read them stay consistent.
    if hasattr(outer, "device"):
        outer.device = device
    if hasattr(outer, "dtype"):
        outer.dtype = dtype
    for attr in _COSMOS_VAE_TENSOR_ATTRS:
        t = getattr(outer, attr, None)
        if isinstance(t, torch.Tensor):
            setattr(outer, attr, t.to(dtype=dtype, device=device))
    # Upstream `Wan2pt1VAEInterface.__init__` caches `self.scale = [self.mean,
    # 1.0 / self.std]` (wan2pt1.py:764) — a plain Python list that captured the
    # original tensors. The `setattr` loop above rebinds `outer.mean`/`outer.std`
    # to moved tensors but leaves the list pointing at the stale references;
    # `encode()` then mixes the moved latents with the stale scale and crashes
    # "Expected all tensors to be on the same device". Rebuild the list from the
    # freshly-moved tensors, mirroring the upstream init pattern exactly.
    if isinstance(getattr(outer, "scale", None), list) and hasattr(outer, "mean") and hasattr(outer, "std"):
        outer.scale = [outer.mean, 1.0 / outer.std]


def _move_cosmos_reason1(te: Any, *, dtype: torch.dtype, device: torch.device) -> None:
    """Move the plain-class :class:`Reason1LiveTextEncoder` to (dtype, device).

    Mirrors :func:`_move_cosmos_vae` — the encoder facade is intentionally not
    an ``nn.Module``. The wrapper separately registers the inner Qwen module as
    ``_reason1_inner`` so the weights still enter ``state_dict()``, but
    dtype/device bookkeeping lives on the facade. Delegate to the encoder's own
    ``to`` shim to keep both in sync.
    """
    if te is None:
        return
    mover = getattr(te, "to", None)
    if callable(mover):
        mover(dtype=dtype, device=device)
