"""S-VAE: optional frozen feature reducer for external video-encoder latents.

``model`` holds the :class:`SVAE` network + train/load helpers; ``reducer``
holds the host-encoder wiring (opt-in build / apply / deploy sidecar). The
public names are re-exported here so ``encoder.svae`` stays the import path.
"""

from openwam.model.video_backbone.encoder.svae import reducer
from openwam.model.video_backbone.encoder.svae.model import (
    _CHECKPOINT_FORMAT_VERSION,
    SVAE,
    DiagonalGaussian,
    build_svae,
    load_svae,
    svae_loss,
)

__all__ = [
    "SVAE",
    "DiagonalGaussian",
    "build_svae",
    "load_svae",
    "svae_loss",
    "reducer",
    "_CHECKPOINT_FORMAT_VERSION",
]
