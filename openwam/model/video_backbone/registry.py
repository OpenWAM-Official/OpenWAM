"""Video backbone registry + factory.

Decorator-based registration, mirroring ``openwam/model/architectures/registry.py`` for
architectures. A backbone class registers under one or more names (a Wan
backbone class backs several checkpoint variants); ``build_video_backbone``
is the factory used by ``BaseWAMArchitecture`` for both training and deploy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from openwam.model.video_backbone.base import VideoBackbone

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
    materialize_weights: bool = False,
    external_encoder: Any = None,
    text_dim: Optional[int] = None,
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

    Args:
        name:     Registry key (e.g. ``"wan22_ti2v_5b"``). Required unless
                  ``source`` is given.
        cfg:      Full Hydra config (only consumed when ``source is None``).
        source:   Optional explicit deploy-time source.
        device:   Forwarded to ``from_pretrained`` when ``source`` is set.
        ckpt_dir: Forwarded to ``from_pretrained`` when ``source`` is set.
        materialize_weights: Forwarded when ``source`` is set. Tells a backbone
                  that builds empty shells from ``ckpt_dir`` that nothing will
                  load a state_dict into it, so it must allocate real storage
                  rather than leaving parameters on ``meta``. Set by the
                  training RESUME path, where accelerate's ``load_state``
                  fills the weights only after ``prepare``.
        external_encoder: Optional pre-built :class:`VideoEncoder` to swap in
                  for the backbone's native VAE. Forwarded to
                  ``cls.from_pretrained`` on BOTH paths — training (built
                  from yaml + model_path) and deploy (built from the saved
                  components entry via :meth:`VideoEncoder.from_skeleton`,
                  weights filled in by the architecture's checkpoint load).
        text_dim: Optional architecture-level raw text/context dimension.
                  Forwarded to the backbone so it can validate that its text
                  embedding expects the same context width as the action stream.
    """
    if name and name in _VIDEO_BACKBONE_REGISTRY:
        cls = _VIDEO_BACKBONE_REGISTRY[name]
    else:
        available = ", ".join(sorted(_VIDEO_BACKBONE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown video backbone '{name}'. Available: {available}")

    if source is not None:
        kw: Dict[str, Any] = {}
        if device is not None:
            kw["device"] = device
        if ckpt_dir is not None:
            kw["ckpt_dir"] = ckpt_dir
        if materialize_weights:
            kw["materialize_weights"] = True
        if external_encoder is not None:
            kw["external_encoder"] = external_encoder
        if text_dim is not None:
            kw["text_dim"] = int(text_dim)
        return cls.from_pretrained(source, **kw)
    if external_encoder is not None:
        if text_dim is not None:
            return cls.from_pretrained(cfg, external_encoder=external_encoder, text_dim=int(text_dim))
        return cls.from_pretrained(cfg, external_encoder=external_encoder)
    if text_dim is not None:
        return cls.from_pretrained(cfg, text_dim=int(text_dim))
    return cls.from_pretrained(cfg)
