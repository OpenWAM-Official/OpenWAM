"""Architecture-side helpers shared by multiple WAM frameworks."""

from __future__ import annotations

from typing import Any, Optional

import torch

# Wan VAE encodes every 4 video frames into 1 latent time step.
VAE_TEMPORAL_FACTOR = 4


def downsample_video_mask_to_latent(
    video_is_pad: torch.Tensor, *, temporal_factor: int = VAE_TEMPORAL_FACTOR
) -> torch.Tensor:
    """Downsample frame-level padding mask to VAE latent temporal dimension.

    Following FastWAM: separate frame 0 (conditioning, excluded from loss),
    then group the tail frames by ``temporal_factor``. A latent step is
    padded only if ALL frames in the group are padded.

    The returned mask covers tail latent steps only (frame 0 excluded),
    matching the loss which trims pred/target via ``[:, :, 1:]``.

    Args:
        video_is_pad: (..., T_video) bool, True=padded. The leading dims are
            preserved (typically ``(B, T_video)``).
        temporal_factor: VAE temporal compression factor.

    Returns:
        (..., T_latent_tail) bool mask where
        T_latent_tail = ceil((T_video - 1) / temporal_factor).
    """
    T = video_is_pad.shape[-1]
    if T <= 1:
        leading_shape = video_is_pad.shape[:-1]
        return torch.zeros((*leading_shape, 0), dtype=torch.bool, device=video_is_pad.device)

    tail_is_pad = video_is_pad[..., 1:]
    T_tail = tail_is_pad.shape[-1]
    pad_len = (temporal_factor - T_tail % temporal_factor) % temporal_factor
    if pad_len > 0:
        leading_shape = tail_is_pad.shape[:-1]
        pad_block = torch.ones((*leading_shape, pad_len), dtype=torch.bool, device=tail_is_pad.device)
        tail_is_pad = torch.cat([tail_is_pad, pad_block], dim=-1)
    grouped = tail_is_pad.view(*tail_is_pad.shape[:-1], -1, temporal_factor)
    return grouped.all(dim=-1)


def resolve_bridge_layers(cfg: Any, *, num_layers: Optional[int] = None) -> tuple:
    """Parse ``bridge_layers`` indices from an architecture config.

    Two input modes:
      - Explicit: ``cfg.bridge_layers`` is a list / tuple / comma-separated str.
      - Interval: ``cfg.bridge_layers`` is None and ``cfg.bridge_interval`` is set;
        the indices are ``range(0, num_layers, bridge_interval)``.

    ``num_layers`` should be passed by the caller (typically the video backbone's
    ``num_layers``). It falls back to ``cfg.num_dit_layers`` for backward
    compatibility.

    The output is always sorted with no duplicates.
    """
    bl_raw = cfg.get("bridge_layers", None) if isinstance(cfg, dict) else getattr(cfg, "bridge_layers", None)

    if bl_raw is None:
        interval_raw = (
            cfg.get("bridge_interval", None) if isinstance(cfg, dict) else getattr(cfg, "bridge_interval", None)
        )
        if interval_raw is None:
            raise ValueError("bridge_layers is null but bridge_interval is not set")

        if num_layers is None:
            num_layers = (
                cfg.get("num_dit_layers", None) if isinstance(cfg, dict) else getattr(cfg, "num_dit_layers", None)
            )
        if num_layers is None:
            raise ValueError("num_layers must be provided (or num_dit_layers set on cfg) when bridge_layers is null")
        num_layers = int(num_layers)

        interval = int(interval_raw)
        assert interval >= 1, f"bridge_interval must be >= 1, got {interval}"
        bl = tuple(range(0, num_layers, interval))
    elif isinstance(bl_raw, str):
        bl = tuple(int(x) for x in bl_raw.split(","))
    elif isinstance(bl_raw, tuple):
        bl = bl_raw
    else:
        bl = tuple(bl_raw)

    bl = tuple(sorted(bl))
    assert len(set(bl)) == len(bl), f"bridge_layers must be unique, got {bl}"
    return bl


__all__ = ["resolve_bridge_layers", "downsample_video_mask_to_latent", "VAE_TEMPORAL_FACTOR"]
