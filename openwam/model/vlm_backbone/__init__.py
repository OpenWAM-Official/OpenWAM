"""VLM backbone package: the VlmBackbone ABC + its Qwen3-VL implementation.

    from openwam.model.vlm_backbone import build_vlm_backbone
    backbone = build_vlm_backbone("qwen3_vl_2b", checkpoint_path=..., dtype=...)

Adding a VLM backbone: subclass :class:`VlmBackbone`, then register it at the
bottom of this file via ``register_vlm_backbone("name")(YourClass)`` and set
``vlm_backbone.name: your_name`` in the model config yaml.
"""

from openwam.model.vlm_backbone.base import VlmBackbone
from openwam.model.vlm_backbone.registry import (
    _VLM_BACKBONE_REGISTRY,
    build_vlm_backbone,
    register_vlm_backbone,
)

__all__ = [
    "VlmBackbone",
    "Qwen3VLBackbone",
    "build_vlm_backbone",
    "register_vlm_backbone",
    "_VLM_BACKBONE_REGISTRY",
]

# Built-in registration (at the bottom so the implementation imports only from
# base/registry, with no circular dependency).
from openwam.model.vlm_backbone.qwen3_vl_backbone import Qwen3VLBackbone  # noqa: E402

register_vlm_backbone("qwen3_vl_2b")(Qwen3VLBackbone)
