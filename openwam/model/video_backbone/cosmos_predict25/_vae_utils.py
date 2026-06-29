"""Device/dtype movement helpers for the CosmosPredict25 plain-object submodules.

``Wan2pt1VAEInterface`` (the upstream Cosmos VAE wrapper) and
:class:`Reason1LiveTextEncoder` are plain Python objects, not ``nn.Module`` s,
so ``nn.Module.to(...)`` on the surrounding pipeline wrapper does not reach
them. :meth:`CosmosPredict25VideoBackbone.set_dtype_device` calls the helpers here to
move their inner ``nn.Module`` plus the auxiliary tensors explicitly.

Lives in its own module (rather than in ``cosmos_predict25_backbone.py``) so both the
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
    ``reason1`` so the weights still enter ``state_dict()``, but
    dtype/device bookkeeping lives on the facade. Delegate to the encoder's own
    ``to`` shim to keep both in sync.
    """
    if te is None:
        return
    mover = getattr(te, "to", None)
    if callable(mover):
        mover(dtype=dtype, device=device)


# ----------------------------------------------------------------------
# Frame <-> tensor + VAE device helpers (relocated from the deleted
# pipeline_wrapper.py). Kept here, not on the backbone, so they stay pure /
# import-cycle-free (this module must NOT import cosmos_predict25_backbone).
# ----------------------------------------------------------------------


def _pil_video_to_tensor(frames: Any) -> "torch.Tensor":
    """Convert ``list[list[PIL.Image]]`` → ``(B, 3, T, H, W)`` float in ``[-1, 1]``.

    uint8 RGB → ``float / 127.5 - 1``. Self-contained (no Wan imports) so
    CosmosPredict25 works without the Wan backbone installed.
    """
    import numpy as np

    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError(
            f"Expected `frames` as a non-empty list of clips (each a list of PIL frames); got {type(frames).__name__}."
        )
    arrs = []
    for clip in frames:
        if not isinstance(clip, (list, tuple)) or not clip:
            raise ValueError("Each per-sample entry in `frames` must be a non-empty list of PIL.Image frames.")
        clip_arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in clip], axis=0)
        arrs.append(clip_arr)
    stack = np.stack(arrs, axis=0)  # (B, T, H, W, 3) uint8
    t = torch.from_numpy(stack).to(dtype=torch.float32)
    t = t / 127.5 - 1.0
    t = t.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)
    return t


def _video_tensor_to_pil(video: "torch.Tensor") -> list:
    """Convert ``(B, 3, T, H, W)`` in ``[-1, 1]`` → ``list[PIL.Image]`` (B=1 only)."""
    from PIL import Image  # local import; not always present on minimal CI

    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"_video_tensor_to_pil expected (B, 3, T, H, W); got shape {tuple(video.shape)}.")
    if video.shape[0] != 1:
        raise NotImplementedError(
            f"CosmosPredict25 decode currently supports B=1 only; got B={video.shape[0]}. Deploy paths call decode per-sample."
        )
    frame_uint8 = (
        ((video[0].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )  # (T, H, W, 3) uint8
    return [Image.fromarray(frame) for frame in frame_uint8]


def _vae_device(vae: Any) -> "torch.device":
    """Return the device of a ``Wan2pt1VAEInterface``-like VAE."""
    inner = getattr(getattr(vae, "model", None), "model", None)
    if isinstance(inner, nn.Module):
        try:
            return next(inner.parameters()).device
        except StopIteration:
            pass
    # Fallback: WanVAE caches its constructor `device=` attribute (wan2pt1.py:718).
    cached = getattr(getattr(vae, "model", None), "device", None)
    if isinstance(cached, torch.device):
        return cached
    if isinstance(cached, str):
        return torch.device(cached)
    return torch.device("cpu")
