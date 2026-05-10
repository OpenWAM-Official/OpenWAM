"""Helpers for deploy-time ``torch.compile`` configuration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

COMPILE_MODES = ("none", "default", "mot_loop")


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or attribute-like configs."""

    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return getattr(cfg, key)
    except (AttributeError, KeyError):
        return default


def as_bool(value: Any, default: bool = True) -> bool:
    """Interpret OmegaConf/string/bool values as a Python bool."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def cfg_namespace(cfg: Any, **overrides: Any) -> SimpleNamespace:
    """Return a shallow namespace copy of a config section."""

    data: dict[str, Any] = {}
    if cfg is None:
        data = {}
    elif isinstance(cfg, dict):
        data = dict(cfg)
    elif hasattr(cfg, "items"):
        data = {str(k): v for k, v in cfg.items()}
    else:
        for key in ("enabled", "video_dit", "vae", "mode", "torch_mode", "dynamic"):
            value = cfg_get(cfg, key, None)
            if value is not None:
                data[key] = value
    data.update(overrides)
    return SimpleNamespace(**data)


def compile_mode(compile_cfg: Any, default: str | None = None, *, strict: bool = False) -> str | None:
    """Return the high-level compile mode when one is configured."""

    value = cfg_get(compile_cfg, "mode", default)
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in COMPILE_MODES:
        return normalized
    if strict:
        raise ValueError(f"Unknown compile mode '{value}'. Choose from: {', '.join(COMPILE_MODES)}")
    return default


def default_compile_cfg(compile_cfg: Any) -> Any:
    """Return the broad/default compile section.

    ``optimization.compile.mode`` owns the user-facing strategy. ``default``
    selects this section, while ``none`` and ``mot_loop`` disable it. An
    explicit ``default.enabled`` still only controls ActionDiT compilation;
    backbone flags such as ``video_dit`` and ``vae`` remain independent.
    Falling back to the root keeps direct unit tests and older ad-hoc configs
    readable.
    """

    section = cfg_get(compile_cfg, "default", compile_cfg)
    mode = compile_mode(compile_cfg, default=None)
    if mode is None:
        return section
    if mode == "default":
        return cfg_namespace(section, enabled=cfg_get(section, "enabled", True))
    return cfg_namespace(section, enabled=False)


def mot_loop_compile_cfg(compile_cfg: Any) -> Any:
    """Return the narrow MoT-loop compile section.

    ``optimization.compile.mode=mot_loop`` enables this section. A small
    legacy fallback accepts the earlier flat ``mot_loop`` / ``mot_loop_mode``
    fields so local experiment configs do not fail mysteriously.
    """

    mode = compile_mode(compile_cfg, default=None)
    section = cfg_get(compile_cfg, "mot_loop", None)
    if mode is not None:
        return cfg_namespace(section, enabled=(mode == "mot_loop"))
    if section is not None and not isinstance(section, bool):
        return section
    return SimpleNamespace(
        enabled=as_bool(section, default=False),
        mode=cfg_get(compile_cfg, "mot_loop_mode", cfg_get(compile_cfg, "mode", "reduce-overhead")),
        dynamic=cfg_get(compile_cfg, "mot_loop_dynamic", False),
    )


def section_enabled(cfg: Any, default: bool = False) -> bool:
    """Return whether a compile section is enabled."""

    return as_bool(cfg_get(cfg, "enabled", default), default=default)


def torch_compile_kwargs(compile_cfg: Any, *, default_mode: str | None = None) -> dict[str, Any]:
    """Build kwargs for ``torch.compile`` from a compile section."""

    dynamic = cfg_get(compile_cfg, "dynamic", True)
    mode = cfg_get(compile_cfg, "torch_mode", cfg_get(compile_cfg, "mode", default_mode))

    kwargs: dict[str, Any] = {"dynamic": as_bool(dynamic, default=True)}
    if mode is None:
        mode = default_mode
    if mode not in (None, "", "none", "None", "null", "Null"):
        kwargs["mode"] = str(mode)
    return kwargs


__all__ = [
    "COMPILE_MODES",
    "as_bool",
    "compile_mode",
    "cfg_get",
    "cfg_namespace",
    "default_compile_cfg",
    "mot_loop_compile_cfg",
    "section_enabled",
    "torch_compile_kwargs",
]
