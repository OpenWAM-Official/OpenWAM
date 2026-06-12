"""Optional deploy-time accelerators (toggled via ``cfg.optimization``).

- :class:`DiTVelocityCache`: Skip redundant DiT forward passes

The sync/async execution mechanisms live in ``openwam.deploy.executors``.
"""

from openwam.deploy.optimizations.dit_cache import DiTVelocityCache

__all__ = [
    "DiTVelocityCache",
]
