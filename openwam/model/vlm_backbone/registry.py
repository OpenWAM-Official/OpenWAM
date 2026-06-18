"""VLM backbone registry + factory.

Mirrors ``video_backbone/registry.py``. The host architecture (tri_system) builds
its frozen VLM through ``build_vlm_backbone`` instead of importing a concrete
class, so adding a VLM implementation needs no edit to the architecture layer
(C2). Registrations live at the bottom of the package ``__init__`` so subclasses
stay free of registry imports.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from openwam.model.vlm_backbone.base import VlmBackbone

_VLM_BACKBONE_REGISTRY: Dict[str, Type[VlmBackbone]] = {}


def register_vlm_backbone(name: str):
    """Register a VlmBackbone implementation under ``name``."""

    def _wrap(cls: Type[VlmBackbone]) -> Type[VlmBackbone]:
        if name in _VLM_BACKBONE_REGISTRY:
            raise ValueError(f"VLM backbone '{name}' already registered")
        _VLM_BACKBONE_REGISTRY[name] = cls
        return cls

    return _wrap


def build_vlm_backbone(name: str, **kwargs: Any) -> VlmBackbone:
    """Instantiate a registered VlmBackbone by name.

    The subclass loads its frozen pretrained VLM in ``__init__`` (no two-stage
    ``from_pretrained``), so yaml fields are forwarded straight as kwargs.
    """
    if name not in _VLM_BACKBONE_REGISTRY:
        available = ", ".join(sorted(_VLM_BACKBONE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown VLM backbone '{name}'. Available: {available}")
    return _VLM_BACKBONE_REGISTRY[name](**kwargs)
