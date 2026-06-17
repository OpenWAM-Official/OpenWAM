"""Video backbone package for OpenWAM.

This package is the public facade. The pieces live in dedicated modules:
  - ``videobackbone_base`` — :class:`VideoBackbone` ABC + :class:`BlockLoopState`
  - ``registry``          — the registry dict + ``register_video_backbone`` /
                            ``build_video_backbone``
This file re-exports them and registers the built-in Wan backbone at the bottom::

    from openwam.model.video_backbone import build_video_backbone
    backbone = build_video_backbone("wan22_ti2v_5b", cfg)

Adding a new backbone:
1. Subclass :class:`VideoBackbone` (implement prepare/run_block/finalize).
2. Register it at the bottom of this file via ``register_video_backbone("name")(YourClass)``.
3. Set ``video_backbone.name: your_name`` in the model config yaml.
"""

from __future__ import annotations

from openwam.model.video_backbone.registry import (
    _VIDEO_BACKBONE_REGISTRY,
    build_video_backbone,
    register_video_backbone,
)
from openwam.model.video_backbone.videobackbone_base import BlockLoopState, VideoBackbone

__all__ = [
    "BlockLoopState",
    "VideoBackbone",
    "Wan22Ti2vBackbone",
    "Wan21Backbone",
    "build_video_backbone",
    "register_video_backbone",
    "_VIDEO_BACKBONE_REGISTRY",
]

# Built-in registrations (kept at the bottom so the implementation can import
# from ``registry`` / ``videobackbone_base`` without circular issues).
from openwam.model.video_backbone.wan_videobackbone import Wan21Backbone, Wan22Ti2vBackbone  # noqa: E402

register_video_backbone("wan22_ti2v_5b")(Wan22Ti2vBackbone)
register_video_backbone("wan21_vace_1_3b")(Wan21Backbone)
register_video_backbone("wan21_i2v_14b_480p")(Wan21Backbone)
