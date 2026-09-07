"""Video backbone package for OpenWAM.

This package is the public facade. The pieces live in dedicated modules:
  - ``base`` — :class:`VideoBackbone` ABC + :class:`BlockLoopState`
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

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.registry import (
    _VIDEO_BACKBONE_REGISTRY,
    build_video_backbone,
    register_video_backbone,
)

__all__ = [
    "BlockLoopState",
    "VideoBackbone",
    "Wan22Ti2v",
    "Wan21",
    "CosmosPredict25VideoBackbone",
    "Cosmos3EdgeVideoBackbone",
    "build_video_backbone",
    "register_video_backbone",
    "_VIDEO_BACKBONE_REGISTRY",
]

# Built-in registrations (kept at the bottom so the implementation can import
# from ``registry`` / ``base`` without circular issues).
from openwam.model.video_backbone.wan_backbone import Wan21, Wan22Ti2v  # noqa: E402

register_video_backbone("wan22_ti2v_5b")(Wan22Ti2v)
register_video_backbone("wan21_vace_1_3b")(Wan21)
register_video_backbone("wan21_i2v_14b_480p")(Wan21)

# Cosmos-Predict2.5 backbone. Importing the module is CPU-only-CI safe — the
# heavy ``cosmos_predict2`` import is deferred to ``from_pretrained``.
from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone  # noqa: E402

register_video_backbone("cosmos_predict25_2b")(CosmosPredict25VideoBackbone)

# Cosmos3-Edge backbone. CPU-only-CI safe — the vendored transformer (which
# pulls ``diffusers``) is deferred to ``from_pretrained``.
from openwam.model.video_backbone.cosmos3_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402

register_video_backbone("cosmos3_edge")(Cosmos3EdgeVideoBackbone)
