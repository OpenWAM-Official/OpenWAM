"""Video backbone package for OpenWAM.

Provides the :class:`VideoBackbone` ABC and a lightweight registry so that
architecture code can instantiate the correct backbone from config::

    from openwam.model.video_backbone import build_video_backbone
    backbone = build_video_backbone("wan22_ti2v_5b", cfg)

Adding a new backbone:
1. Subclass :class:`VideoBackbone` (implement prepare/run_block/finalize).
2. Call ``register_video_backbone("name")(YourClass)``.
3. Set ``video_backbone.name: your_name`` in the model config yaml.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from openwam.model.video_backbone.adapter import (
    BlockLoopState,
    VideoBackbone,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_VIDEO_BACKBONE_REGISTRY: Dict[str, Type[VideoBackbone]] = {}


def register_video_backbone(name: str):
    """Decorator to register a VideoBackbone implementation by name."""

    def _wrap(cls: Type[VideoBackbone]) -> Type[VideoBackbone]:
        if name in _VIDEO_BACKBONE_REGISTRY:
            raise ValueError(f"Video backbone '{name}' already registered")
        _VIDEO_BACKBONE_REGISTRY[name] = cls
        return cls

    return _wrap


def build_video_backbone(
    name: Optional[str],
    cfg: Any,
    *,
    source: Any = None,
    device: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
) -> VideoBackbone:
    """Instantiate a VideoBackbone from the registry.

    Two call modes:
      - **Training (default)**: pass ``name`` and full ``cfg``; the backbone
        reads what it needs from ``cfg`` via ``cls.from_pretrained(cfg)``.
      - **Deployment**: pass ``source`` (a model directory or components dict)
        plus optional ``device`` and ``ckpt_dir``; the call
        becomes ``cls.from_pretrained(source, device=..., ckpt_dir=...)``. This
        is used by :class:`BaseWAMArchitecture` when the saved config carries a
        ``video_backbone._source`` field.

    When ``source`` is given but ``name`` is missing or unregistered, the first
    registered class is used as a fallback (preserves backward compatibility
    with older checkpoints).

    Args:
        name:     Registry key (e.g. ``"wan22_ti2v_5b"``). Required unless
                  ``source`` is given.
        cfg:      Full Hydra config (only consumed when ``source is None``).
        source:   Optional explicit deploy-time source.
        device:   Forwarded to ``from_pretrained`` when ``source`` is set.
        ckpt_dir: Forwarded to ``from_pretrained`` when ``source`` is set.
    """
    if name and name in _VIDEO_BACKBONE_REGISTRY:
        cls = _VIDEO_BACKBONE_REGISTRY[name]
    elif source is not None and _VIDEO_BACKBONE_REGISTRY:
        # Older checkpoints may lack a registry-aligned name; pick any class.
        cls = next(iter(_VIDEO_BACKBONE_REGISTRY.values()))
    else:
        available = ", ".join(sorted(_VIDEO_BACKBONE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown video backbone '{name}'. Available: {available}")

    if source is not None:
        kw: Dict[str, Any] = {}
        if device is not None:
            kw["device"] = device
        if ckpt_dir is not None:
            kw["ckpt_dir"] = ckpt_dir
        return cls.from_pretrained(source, **kw)
    return cls.from_pretrained(cfg)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

from openwam.model.video_backbone.wan_adapter import WanVideoBackbone  # noqa: E402

register_video_backbone("wan22_ti2v_5b")(WanVideoBackbone)
register_video_backbone("wan21_vace_1_3b")(WanVideoBackbone)
register_video_backbone("wan21_i2v_14b_480p")(WanVideoBackbone)

__all__ = [
    "BlockLoopState",
    "VideoBackbone",
    "WanVideoBackbone",
    "build_video_backbone",
    "register_video_backbone",
]
