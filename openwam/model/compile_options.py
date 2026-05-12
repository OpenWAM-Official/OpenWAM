"""Helpers for deploy-time ``torch.compile`` configuration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

COMPILE_MODES = ("none", "self_attn", "cross_attn")


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
        for key in ("enabled", "mode", "torch_mode", "dynamic"):
            value = cfg_get(cfg, key, None)
            if value is not None:
                data[key] = value
    data.update(overrides)
    return SimpleNamespace(**data)


def normalize_compile_mode(value: Any) -> str:
    """Normalize and validate the public compile mode spelling."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in COMPILE_MODES:
        raise ValueError(f"Unknown compile mode '{value}'. Choose from: {', '.join(COMPILE_MODES)}")
    return normalized


def compile_mode(compile_cfg: Any, default: str | None = None, *, strict: bool = False) -> str | None:
    """Return the high-level compile mode when one is configured."""

    value = cfg_get(compile_cfg, "mode", default)
    if value is None:
        return None
    try:
        return normalize_compile_mode(value)
    except ValueError:
        if strict:
            raise
    return default


def _fast_path_compile_cfg(compile_cfg: Any, section_name: str, mode_name: str) -> SimpleNamespace:
    """Resolve an architecture-specific fixed-shape compile section."""

    mode = compile_mode(compile_cfg, default=None, strict=True)
    section = cfg_namespace(cfg_get(compile_cfg, section_name, None))
    section_enabled_value = cfg_get(section, "enabled", None)
    if cfg_get(section, "torch_mode", None) is None:
        section.torch_mode = "reduce-overhead"
    if cfg_get(section, "dynamic", None) is None:
        section.dynamic = False
    section.enabled = (mode == mode_name) and as_bool(section_enabled_value, default=True)
    return section


def self_attn_compile_cfg(compile_cfg: Any) -> Any:
    """Return the narrow self-attention compile section.

    ``optimization.compile.mode=self_attn`` enables the existing MoT-loop
    helper for ``dual_system_self_attn``. The section name is public-facing;
    internal class/file names may still use MoT where that is the actual
    mechanism.
    """

    return _fast_path_compile_cfg(compile_cfg, "self_attn", "self_attn")


def cross_attn_compile_cfg(compile_cfg: Any) -> Any:
    """Return the narrow cross-attention compile section."""

    return _fast_path_compile_cfg(compile_cfg, "cross_attn", "cross_attn")


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
    "cross_attn_compile_cfg",
    "cfg_get",
    "cfg_namespace",
    "normalize_compile_mode",
    "section_enabled",
    "self_attn_compile_cfg",
    "torch_compile_kwargs",
]
