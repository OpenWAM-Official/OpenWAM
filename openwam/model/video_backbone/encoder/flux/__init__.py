"""FLUX.2 VAE extracted source (encoder-only) + diffusers->BFL converter.

Vendored from ``references/flux2`` and isolated in its own subpackage so the
bulky extracted model code stays separate from the thin :class:`VideoEncoder`
wrapper in ``encoder/flux_vae.py``. Import the public API from this package:

    from openwam.model.video_backbone.encoder.flux import FluxVaeEncoderCore
"""

from openwam.model.video_backbone.encoder.flux.flux_vae_src import (
    Encoder,
    FluxVaeEncoderCore,
    convert_attn,
    convert_diffusers_encoder_sd,
    convert_resnet,
)

__all__ = [
    "Encoder",
    "FluxVaeEncoderCore",
    "convert_attn",
    "convert_diffusers_encoder_sd",
    "convert_resnet",
]
