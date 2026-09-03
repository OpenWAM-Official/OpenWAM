"""Pluggable video encoder subsystem.

Only activated when ``video_backbone.from_scratch=true`` AND
``video_backbone.encoder`` is set in yaml. Under any other combination the
backbone keeps using its built-in ``pipe.vae`` and the encoder package is
inert (registration still runs, but no encoder is instantiated).

This package is the public facade. The pieces live in dedicated modules:
  - ``base`` — :class:`VideoEncoder` ABC + :class:`VideoEncoderProperties`
  - ``registry``          — the registry dict + ``register_video_encoder`` /
                            ``build_video_encoder``
This file only re-exports them and imports each implementation at the bottom
to trigger registration.

Adding a new encoder:
  1. Create ``encoder/<name>.py`` with ``class XxxEncoder(VideoEncoder)``
     decorated by ``@register_video_encoder("xxx")``.
  2. Implement the four abstract methods: ``spec`` (property),
     ``preprocess_video``, ``batch_encode``, ``from_pretrained``.
  3. Optionally override ``build_dit_input_proj`` / ``build_dit_output_proj``
     when the default Wan-style ``nn.Conv3d`` / ``nn.Linear`` does not fit.
  4. Add ``from .xxx import XxxEncoder  # noqa: F401`` at the bottom of this
     file to trigger registration on import.

The author NEVER needs to touch ``wan_backbone.py`` / ``dit.py``; the public
encoder contract is defined in this package's ``base.py``.
"""

from __future__ import annotations

from openwam.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from openwam.model.video_backbone.encoder.registry import (
    _VIDEO_ENCODER_REGISTRY,
    build_video_encoder,
    register_video_encoder,
)

__all__ = [
    "VideoEncoder",
    "VideoEncoderProperties",
    "DinoV3VideoEncoder",
    "FluxVAEVideoEncoder",
    "VJEPA21VideoEncoder",
    "WanVideoVAEEncoder",
    "build_video_encoder",
    "register_video_encoder",
    "_VIDEO_ENCODER_REGISTRY",
]

# Built-in registrations (kept at the bottom so implementations can import from
# ``registry`` / ``base`` without circular issues). Adding a new
# encoder = adding a new line here and a new file alongside.
from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder  # noqa: E402, F401
from openwam.model.video_backbone.encoder.flux2_vae import FluxVAEVideoEncoder  # noqa: E402, F401
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder  # noqa: E402, F401
from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder  # noqa: E402, F401
