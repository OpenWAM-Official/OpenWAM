'Public implementation.'

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoEncoderSpec:
    'Public implementation.'

    z_dim: int
    spatial_compression: int
    temporal_compression: int
    causal_temporal: bool
    pixel_range: tuple[float, float] = (-1.0, 1.0)
    is_reversible: bool = True
    dit_patch_size: tuple[int, int, int] = (1, 2, 2)



__all__ = ["VideoEncoderSpec"]
