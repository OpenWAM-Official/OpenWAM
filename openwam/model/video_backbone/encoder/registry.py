"""Video encoder registry + factory.

Decorator-based registration, mirroring ``openwam/model/architectures/registry.py`` for
architectures. Encoder implementations self-register via
``@register_video_encoder("name")``; the package ``__init__`` imports them to
trigger registration. ``build_video_encoder`` is the config-driven factory.
"""

from __future__ import annotations

from typing import Any

from openwam.model.video_backbone.encoder.videoencoder_base import VideoEncoder

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

    Accepts the two required fields ``name`` / ``model_path`` plus any
    optional fields the picked encoder class exposes through
    :meth:`VideoEncoder.optional_yaml_keys` (e.g. ``vjepa2_1_forward``).
    The base-side gate in :meth:`BaseWAMArchitecture._init_video_backbone`
    enforces the same whitelist; this function adds friendly errors for the
    "field present but empty" case (which the whitelist wouldn't catch) and
    forwards the optional fields into :meth:`from_pretrained` as kwargs.
    """

    def _read(key: str):
        # Both dict and OmegaConf DictConfig support both indexing and
        # attribute access; ``isinstance(cfg, dict)`` distinguishes them.
        # OmegaConf with struct mode returns the default from getattr
        # rather than raising, so the second branch is safe.
        if isinstance(cfg, dict):
            value = cfg.get(key)
        else:
            value = getattr(cfg, key, None)
        if value is None:
            raise ValueError(
                f"video_backbone.encoder.{key} is required (got cfg={dict(cfg) if hasattr(cfg, 'keys') else cfg!r})."
            )
        return value

    def _read_optional(key: str):
        if isinstance(cfg, dict):
            return cfg.get(key)
        return getattr(cfg, key, None)

    name = _read("name")
    model_path = _read("model_path")
    if name not in _VIDEO_ENCODER_REGISTRY:
        available = ", ".join(sorted(_VIDEO_ENCODER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown video encoder '{name}'. Available: {available}")
    encoder_cls = _VIDEO_ENCODER_REGISTRY[name]
    # Optional fields propagated as keyword arguments to ``from_pretrained``
    # so each encoder can pick up the ones it cares about and ignore the
    # rest. The picked encoder declares which optional keys it accepts via
    # :meth:`VideoEncoder.optional_yaml_keys` (e.g. ``vjepa2_1_forward`` on
    # V-JEPA 2.1). Absent / ``None`` values are treated as "use the encoder
    # default".
    optional_kwargs: dict[str, Any] = {}
    for k in encoder_cls.optional_yaml_keys():
        v = _read_optional(k)
        if v is not None:
            optional_kwargs[k] = v
    return encoder_cls.from_pretrained(str(model_path), **optional_kwargs)
