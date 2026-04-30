"""Deployment optimizations for WAM inference (DreamZero-inspired).

- :class:`DiTVelocityCache`: Skip redundant DiT forward passes
- :func:`schedule_decoupled_flash`: 1-4 step action inference
- :class:`AsyncInferenceExecutor`: Overlap inference with execution
"""

from openwam.deploy.optimizations.async_executor import AsyncInferenceExecutor
from openwam.deploy.optimizations.decoupled_schedule import (
    schedule_decoupled_asymmetric,
    schedule_decoupled_flash,
)
from openwam.deploy.optimizations.dit_cache import DiTVelocityCache

__all__ = [
    "DiTVelocityCache",
    "schedule_decoupled_flash",
    "schedule_decoupled_asymmetric",
    "AsyncInferenceExecutor",
]
