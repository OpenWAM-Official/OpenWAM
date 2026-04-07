"""Deployment optimizations for WAM inference (DreamZero-inspired).

- :class:`DiTVelocityCache`: Skip redundant DiT forward passes
- :class:`CFGBatchMerger`: Merge CFG passes into single batched call
- :class:`CFGParallelExecutor`: Distribute CFG across GPUs
- :func:`schedule_decoupled_flash`: 1-4 step action inference
- :class:`AsyncInferenceExecutor`: Overlap inference with execution
"""

from open_wam.inference.optimizations.async_executor import AsyncInferenceExecutor
from open_wam.inference.optimizations.cfg_parallel import (
    CFGBatchMerger,
    CFGParallelExecutor,
)
from open_wam.inference.optimizations.decoupled_schedule import (
    sample_decoupled_timesteps,
    schedule_decoupled_asymmetric,
    schedule_decoupled_flash,
)
from open_wam.inference.optimizations.dit_cache import DiTVelocityCache

__all__ = [
    "DiTVelocityCache",
    "CFGBatchMerger",
    "CFGParallelExecutor",
    "schedule_decoupled_flash",
    "schedule_decoupled_asymmetric",
    "sample_decoupled_timesteps",
    "AsyncInferenceExecutor",
]
