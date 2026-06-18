"""Video encoder registry + factory.

Decorator-based registration, mirroring ``openwam/model/architectures/registry.py`` for
architectures. Encoder implementations self-register via
``@register_video_encoder("name")``; the package ``__init__`` imports them to
trigger registration. ``build_video_encoder`` is the config-driven factory.
"""

from __future__ import annotations

from openwam.model.video_backbone.encoder.base import VideoEncoder

_VIDEO_ENCODER_REGISTRY: dict[str, type[VideoEncoder]] = {}


def register_video_encoder(name: str):
    """Decorator that registers a :class:`VideoEncoder` subclass under ``name``.

    Raises:
        ValueError: If ``name`` is already taken.
        TypeError:  If the decorated class is not a :class:`VideoEncoder` subclass.
    """

    def _wrap(cls):
        if not isinstance(cls, type) or not issubclass(cls, VideoEncoder):
            raise TypeError(f"register_video_encoder('{name}') expects a VideoEncoder subclass, got {cls!r}.")
        if name in _VIDEO_ENCODER_REGISTRY:
            raise ValueError(f"Video encoder '{name}' is already registered.")
        _VIDEO_ENCODER_REGISTRY[name] = cls
        return cls

    return _wrap


def build_video_encoder(cfg) -> VideoEncoder:
    """Build a :class:`VideoEncoder` from a config dict / DictConfig.

    Requires ``name`` / ``model_path``; every other non-null field is forwarded
    to the picked encoder's :meth:`VideoEncoder.from_pretrained` as a kwarg. The
    encoder signature is the contract — an explicit-signature encoder (e.g.
    V-JEPA 2.1) raises ``TypeError`` on an unknown / typo'd field, while a
    ``**kw`` encoder ignores extras.
    """

    def _read(key: str):
        # Both dict and OmegaConf DictConfig support indexing + attribute access;
        # ``isinstance(cfg, dict)`` distinguishes them.
        if isinstance(cfg, dict):
            value = cfg.get(key)
        else:
            value = getattr(cfg, key, None)
        if value is None:
            raise ValueError(
                f"video_backbone.encoder.{key} is required (got cfg={dict(cfg) if hasattr(cfg, 'keys') else cfg!r})."
            )
        return value

    name = _read("name")
    model_path = _read("model_path")
    if name not in _VIDEO_ENCODER_REGISTRY:
        available = ", ".join(sorted(_VIDEO_ENCODER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown video encoder '{name}'. Available: {available}")
    encoder_cls = _VIDEO_ENCODER_REGISTRY[name]
    extras = {k: v for k, v in cfg.items() if k not in ("name", "model_path") and v is not None}
    return encoder_cls.from_pretrained(str(model_path), **extras)
