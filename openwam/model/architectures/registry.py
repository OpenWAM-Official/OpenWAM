"""WAM Architecture registry.

Provides a decorator-based registration pattern for discovering and
instantiating architectures from configuration.

WAM architectures implement BaseWAMArchitecture (3-hook interface for
video DiT integration).

Canonical registry names (one per concrete architecture class):
    dual_system_cross_attn  / DualSystemCrossAttnArchitecture
    dual_system_self_attn   / DualSystemSelfAttnArchitecture
    dual_system_idm         / DualSystemIDMArchitecture
    single_system_vanilla / SingleSystemVanillaArchitecture
    single_system_moe     / SingleSystemMoEArchitecture
    tri_system_joint_self_attn / TriSystemJointSelfAttnArchitecture

Usage:
    @register_architecture(
        "dual_system_cross_attn",
        framework="dual_system",
        variant="joint_cross_attn",
    )
    class DualSystemCrossAttnArchitecture(BaseWAMArchitecture):
        ...

    arch = build_architecture("dual_system_cross_attn", cfg)
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from openwam.model.architectures.base import BaseWAMArchitecture

ARCHITECTURE_REGISTRY: Dict[str, Type[BaseWAMArchitecture]] = {}
ARCHITECTURE_SUPPORT: Dict[str, "ArchitectureSupport"] = {}
ARCHITECTURE_METADATA: Dict[str, "ArchitectureMetadataEntry"] = {}

_FRAMEWORK_VARIANT_INDEX: Dict[tuple, str] = {}


@dataclass(frozen=True)
class ArchitectureMetadataEntry:
    """Metadata declared by each architecture at registration time."""

    framework: str
    variant: str
    options_from_cfg: Optional[Callable] = None


@dataclass(frozen=True)
class CanonicalArchitectureSpec:
    """Normalized architecture descriptor used across training and deployment."""

    framework: str
    variant: str
    options: dict
    original_name: str


@dataclass(frozen=True)
class ResolvedArchitectureConfig:
    """Canonicalized architecture build input shared by training and deployment."""

    registry_name: str
    canonical: CanonicalArchitectureSpec
    params: dict


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def normalize_architecture_spec(name: str, cfg=None) -> CanonicalArchitectureSpec:
    """Map a canonical architecture name into its framework/variant pair."""
    if name not in ARCHITECTURE_METADATA:
        raise KeyError(f"Unknown architecture '{name}'")

    meta = ARCHITECTURE_METADATA[name]
    options = {}
    if meta.options_from_cfg is not None:
        options = meta.options_from_cfg(cfg) or {}

    return CanonicalArchitectureSpec(
        framework=meta.framework,
        variant=meta.variant,
        options=options,
        original_name=name,
    )


def resolve_architecture_config(
    model_cfg, *, video_dim: int = 0, num_dit_layers: int | None = None
) -> ResolvedArchitectureConfig:
    """Resolve model config into one canonical architecture build payload.

    ``video_dim`` and ``num_dit_layers`` are optional — when omitted the
    architecture will derive them from ``self.video_backbone`` at init time.
    """
    arch_cfg = getattr(model_cfg, "architecture", {})
    action_cfg = getattr(model_cfg, "action_backbone", {})

    framework = _cfg_get(arch_cfg, "framework", None)
    variant = _cfg_get(arch_cfg, "variant", None)

    lookup_key = (framework, variant)
    if lookup_key not in _FRAMEWORK_VARIANT_INDEX:
        # Fallback: variant=None means "use the default variant for this framework"
        if variant is None:
            candidates = [k for k in _FRAMEWORK_VARIANT_INDEX if k[0] == framework]
            if len(candidates) == 1:
                lookup_key = candidates[0]
            else:
                available = sorted(_FRAMEWORK_VARIANT_INDEX.keys())
                raise KeyError(
                    f"Ambiguous canonical architecture config framework={framework!r} variant={variant!r}. "
                    f"Available: {available}"
                )
        else:
            available = sorted(_FRAMEWORK_VARIANT_INDEX.keys())
            raise KeyError(
                f"Unsupported canonical architecture config framework={framework!r} variant={variant!r}. "
                f"Available: {available}"
            )

    registry_name = _FRAMEWORK_VARIANT_INDEX[lookup_key]

    params = {k: v for k, v in arch_cfg.items() if k not in {"framework", "variant"}}
    params["framework"] = framework
    params["variant"] = variant
    if action_cfg:
        # ``text_dim`` is architecture-owned (raw context shared by both streams);
        # ``variant`` is resolved above. Everything else forwards to the architecture.
        _non_arch = {"text_dim", "variant"}
        params.update({k: v for k, v in action_cfg.items() if k not in _non_arch})

    vb_cfg = getattr(model_cfg, "video_backbone", None)
    if vb_cfg is not None:
        params["video_backbone"] = vb_cfg
    vlm_cfg = getattr(model_cfg, "vlm_backbone", None)
    if vlm_cfg is not None:
        params["vlm_backbone"] = vlm_cfg
    if video_dim:
        params["video_dim"] = video_dim
    if num_dit_layers is not None:
        params.setdefault("num_dit_layers", int(num_dit_layers))

    normalized = normalize_architecture_spec(registry_name, params)
    for key, value in normalized.options.items():
        params.setdefault(key, value)

    return ResolvedArchitectureConfig(
        registry_name=registry_name,
        canonical=normalized,
        params=params,
    )


@dataclass(frozen=True)
class ArchitectureSupport:
    """Support metadata for a registered architecture."""

    status: str
    note: str = ""

    @property
    def supported(self) -> bool:
        return self.status == "supported"


def register_architecture(
    name: str,
    *,
    status: str = "supported",
    note: str = "",
    framework: str = "",
    variant: str = "",
    options_from_cfg: Optional[Callable] = None,
):
    """Decorator to register a WAM architecture class.

    Args:
        name: Canonical registry key (e.g. ``"dual_system_cross_attn"``).
        status: ``"supported"`` or ``"experimental"``.
        note: Human-readable note.
        framework: Architecture family (e.g. ``"dual_system"``).
        variant: Variant within the family (e.g. ``"joint_cross_attn"``).
        options_from_cfg: Optional callable ``(cfg) -> dict`` that extracts
            architecture-specific options from the build config.
    """

    def decorator(cls: Type[BaseWAMArchitecture]):
        if name in ARCHITECTURE_REGISTRY:
            raise ValueError(f"Architecture '{name}' already registered")
        if status not in {"supported", "experimental"}:
            raise ValueError(f"Unsupported architecture status '{status}'")
        ARCHITECTURE_REGISTRY[name] = cls
        ARCHITECTURE_SUPPORT[name] = ArchitectureSupport(status=status, note=note)
        ARCHITECTURE_METADATA[name] = ArchitectureMetadataEntry(
            framework=framework,
            variant=variant,
            options_from_cfg=options_from_cfg,
        )
        if framework:
            _FRAMEWORK_VARIANT_INDEX[(framework, variant or None)] = name

        return cls

    return decorator


def get_architecture_support(name: str) -> ArchitectureSupport:
    """Return support metadata for a registered architecture."""
    if name not in ARCHITECTURE_SUPPORT:
        available = ", ".join(sorted(ARCHITECTURE_SUPPORT.keys()))
        raise KeyError(f"Unknown architecture '{name}'. Available: {available}")
    return ARCHITECTURE_SUPPORT[name]


def list_supported_architectures() -> tuple[str, ...]:
    """List architecture names that are part of the supported matrix."""
    return tuple(name for name in sorted(ARCHITECTURE_REGISTRY.keys()) if ARCHITECTURE_SUPPORT[name].supported)


def build_architecture(name: str, cfg=None, *, allow_experimental: bool = False) -> BaseWAMArchitecture:
    """
    Instantiate a registered WAM architecture by registry key.

    Parameters:
        name (str): Registry key of the architecture (e.g., "dual_system_cross_attn").
        cfg: Configuration object passed to the architecture constructor.
            The architecture creates its own ``video_backbone`` internally
            from ``cfg.video_backbone.name`` via the video backbone registry.
        allow_experimental (bool): If False, prevent instantiation of architectures marked as experimental.

    Returns:
        BaseWAMArchitecture: An instance of the registered architecture class.
    """
    canonical_name = name
    if canonical_name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise KeyError(f"Unknown architecture '{name}'. Available: {available}")
    support = ARCHITECTURE_SUPPORT[canonical_name]
    if not support.supported and not allow_experimental:
        detail = f" {support.note}" if support.note else ""
        raise NotImplementedError(
            f"Architecture '{canonical_name}' is experimental and not part of the supported OpenWAM matrix.{detail}"
        )

    normalized = normalize_architecture_spec(name, cfg)
    cfg_out = deepcopy(cfg) if isinstance(cfg, dict) else cfg
    if isinstance(cfg_out, dict):
        cfg_out.setdefault("framework", normalized.framework)
        cfg_out.setdefault("variant", normalized.variant)
        for key, value in normalized.options.items():
            cfg_out.setdefault(key, value)
    return ARCHITECTURE_REGISTRY[canonical_name](cfg=cfg_out)
