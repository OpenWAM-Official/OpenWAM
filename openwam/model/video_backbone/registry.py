"""Video backbone registry + factory.

Decorator-based registration, mirroring ``openwam/model/registry.py`` for
architectures. A backbone class registers under one or more names (a single
``WanVideoBackbone`` backs several checkpoint variants); ``build_video_backbone``
is the factory used by ``BaseWAMArchitecture`` for both training and deploy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from openwam.model.video_backbone.videobackbone_base import VideoBackbone

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
    external_encoder: Any = None,
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
        external_encoder: Optional pre-built :class:`VideoEncoder` to swap in
                  for the backbone's native VAE. Forwarded to
                  ``cls.from_pretrained`` on BOTH paths — training (built
                  from yaml + model_path) and deploy (built from the saved
                  components entry via :meth:`VideoEncoder.from_skeleton`,
                  weights filled in by the architecture's checkpoint load).
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
        if external_encoder is not None:
            kw["external_encoder"] = external_encoder
        return cls.from_pretrained(source, **kw)
    if external_encoder is not None:
        return cls.from_pretrained(cfg, external_encoder=external_encoder)
    return cls.from_pretrained(cfg)
