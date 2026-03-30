"""WAM Architecture registry.

Provides a decorator-based registration pattern (inspired by StarVLA's
FRAMEWORK_REGISTRY) for discovering and instantiating WAM architectures
from configuration.

Usage:
    @register_architecture("dual_system")
    class DualSystemArchitecture(BaseWAMArchitecture):
        ...

    arch = build_architecture("dual_system", cfg)
"""

from dataclasses import dataclass
from typing import Dict, Type

from open_wam.models.architectures.base import BaseWAMArchitecture

ARCHITECTURE_REGISTRY: Dict[str, Type[BaseWAMArchitecture]] = {}
ARCHITECTURE_SUPPORT: Dict[str, "ArchitectureSupport"] = {}


@dataclass(frozen=True)
class ArchitectureSupport:
    """Support metadata for a registered architecture."""

    status: str
    note: str = ""

    @property
    def supported(self) -> bool:
        return self.status == "supported"


def register_architecture(name: str, *, status: str = "supported", note: str = ""):
    """Decorator to register a WAM architecture class."""
    def decorator(cls: Type[BaseWAMArchitecture]):
        if name in ARCHITECTURE_REGISTRY:
            raise ValueError(f"Architecture '{name}' already registered")
        if status not in {"supported", "experimental"}:
            raise ValueError(f"Unsupported architecture status '{status}'")
        ARCHITECTURE_REGISTRY[name] = cls
        ARCHITECTURE_SUPPORT[name] = ArchitectureSupport(status=status, note=note)
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
    return tuple(
        name for name in sorted(ARCHITECTURE_REGISTRY.keys())
        if ARCHITECTURE_SUPPORT[name].supported
    )


def list_experimental_architectures() -> tuple[str, ...]:
    """List architecture names that remain explicitly experimental."""
    return tuple(
        name for name in sorted(ARCHITECTURE_REGISTRY.keys())
        if not ARCHITECTURE_SUPPORT[name].supported
    )


def build_architecture(name: str, cfg=None, *, allow_experimental: bool = False) -> BaseWAMArchitecture:
    """Instantiate a registered WAM architecture by name.

    Args:
        name: Registry key (e.g. "dual_system", "moe_expert", "shared_backbone").
        cfg: Architecture configuration (passed to constructor).
        allow_experimental: Allow experimental architectures to instantiate.

    Returns:
        Instantiated BaseWAMArchitecture subclass.
    """
    if name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown architecture '{name}'. Available: {available}"
        )
    support = ARCHITECTURE_SUPPORT[name]
    if not support.supported and not allow_experimental:
        detail = f" {support.note}" if support.note else ""
        raise NotImplementedError(
            f"Architecture '{name}' is experimental and not part of the supported "
            f"OpenWAM matrix.{detail}"
        )
    return ARCHITECTURE_REGISTRY[name](cfg)
