"""Helpers for deploy-time ``torch.compile`` configuration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}


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


def normalize_compile_enabled(value: Any) -> bool:
    """Normalize and validate the public compile enabled flag."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Unknown compile enabled value '{value}'. Choose true or false.")


def _legacy_mode_enabled(value: Any) -> bool:
    """Map the pre-merge mode spelling to the enabled flag."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        return True
    if normalized == "none":
        return False
    raise ValueError(f"Unknown legacy compile mode '{value}'. Choose from: auto, none")


def compile_enabled(compile_cfg: Any, default: bool = False, *, strict: bool = False) -> bool:
    """Return whether high-level deploy compile is enabled."""

    value = cfg_get(compile_cfg, "enabled", None)
    if value is None:
        mode = cfg_get(compile_cfg, "mode", None)
        if mode is None:
            value = default
        else:
            try:
                return _legacy_mode_enabled(mode)
            except ValueError:
                if strict:
                    raise
                return default
    try:
        return normalize_compile_enabled(value)
    except ValueError:
        if strict:
            raise
    return default


def _fast_path_compile_cfg(
    compile_cfg: Any,
    section_name: str,
    *,
    default_torch_mode: str = "reduce-overhead",
) -> SimpleNamespace:
    """Resolve an architecture-specific fixed-shape compile section."""

    section = cfg_namespace(cfg_get(compile_cfg, section_name, None))
    section_enabled_value = cfg_get(section, "enabled", None)
    if cfg_get(section, "torch_mode", None) is None:
        section.torch_mode = default_torch_mode
    if cfg_get(section, "dynamic", None) is None:
        section.dynamic = False
    section.enabled = as_bool(section_enabled_value, default=True)
    return section


def self_attn_compile_cfg(compile_cfg: Any) -> Any:
    """Return the narrow self-attention compile section.

    ``optimization.compile.enabled=true`` lets ``dual_system_self_attn`` select
    the existing MoT-loop helper. The section name remains explicit so the
    helper can keep its own torch.compile options.
    """

    return _fast_path_compile_cfg(compile_cfg, "self_attn")


def cross_attn_compile_cfg(compile_cfg: Any) -> Any:
    """Return the narrow cross-attention compile section."""

    return _fast_path_compile_cfg(compile_cfg, "cross_attn")


def _inherit_fast_path_defaults(parent: Any, section_name: str) -> SimpleNamespace:
    """Return a child compile section inheriting IDM-level defaults."""

    section = cfg_namespace(cfg_get(parent, section_name, None))
    enabled = cfg_get(section, "enabled", None)
    if enabled is None:
        section.enabled = as_bool(cfg_get(parent, "enabled", True), default=True)
    else:
        section.enabled = as_bool(enabled, default=True)
    if cfg_get(section, "torch_mode", None) is None:
        section.torch_mode = cfg_get(parent, "torch_mode", "reduce-overhead")
    if cfg_get(section, "dynamic", None) is None:
        section.dynamic = cfg_get(parent, "dynamic", False)
    return section


def idm_compile_cfg(compile_cfg: Any) -> Any:
    """Return the narrow IDM compile section.

    IDM has two inference hot loops, so the top-level section controls both
    helpers while optional children can disable or tune each one independently.
    """

    section = _fast_path_compile_cfg(compile_cfg, "idm")
    section.video_loop = _inherit_fast_path_defaults(section, "video_loop")
    section.action_cache = _inherit_fast_path_defaults(section, "action_cache")
    return section


def tri_system_compile_cfg(compile_cfg: Any) -> Any:
    """Return the narrow tri-system trimodal MoT compile section."""

    return _fast_path_compile_cfg(compile_cfg, "tri_system")


def wan_blocks_compile_cfg(compile_cfg: Any) -> Any:
    """Return the Wan per-block compile section.

    Per-block Wan compile intentionally defaults to ``torch_mode=default``.
    ``reduce-overhead`` can use CUDAGraphs at this boundary, where block outputs
    are consumed by later blocks and can hit output lifetime errors.
    """

    return _fast_path_compile_cfg(compile_cfg, "wan_blocks", default_torch_mode="default")


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
    "as_bool",
    "compile_enabled",
    "cross_attn_compile_cfg",
    "cfg_get",
    "cfg_namespace",
    "idm_compile_cfg",
    "normalize_compile_enabled",
    "section_enabled",
    "self_attn_compile_cfg",
    "torch_compile_kwargs",
    "tri_system_compile_cfg",
    "wan_blocks_compile_cfg",
]
