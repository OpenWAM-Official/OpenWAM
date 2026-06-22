"""Architecture-side shared utilities: common config/mask helpers."""

from openwam.model.architectures.utils.common import (
    VAE_TEMPORAL_FACTOR,
    compute_video_tokens_per_frame,
    downsample_video_mask_to_latent,
    resolve_bridge_layers,
)

__all__ = [
    "VAE_TEMPORAL_FACTOR",
    "compute_video_tokens_per_frame",
    "downsample_video_mask_to_latent",
    "resolve_bridge_layers",
]
