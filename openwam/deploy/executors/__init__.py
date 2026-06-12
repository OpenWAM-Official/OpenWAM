"""Inference executors: how engine-generated action chunks reach the robot.

Two interchangeable execution mechanisms behind one interface
(``predict_action(conditions)`` / ``reset()`` / ``shutdown()``):

- :class:`SyncInferenceExecutor` — default; buffer-and-replan with
  receding horizon + temporal ensembling.
- :class:`AsyncInferenceExecutor` — double-buffered background inference
  that overlaps generation with execution (DreamZero-inspired).

The async-inference config surface (:class:`AsyncInferenceConfig` and the
normalize / resolve / CLI-override helpers) lives in
:mod:`~openwam.deploy.executors.async_executor` next to its consumer.
"""

from openwam.deploy.executors.async_executor import (
    ASYNC_CLI_NUMERIC_OVERRIDES,
    AsyncInferenceConfig,
    AsyncInferenceExecutor,
    apply_async_cli_overrides,
    normalize_async_inference_config,
    resolve_async_inference_config,
)
from openwam.deploy.executors.sync_executor import SyncInferenceExecutor

__all__ = [
    "SyncInferenceExecutor",
    "AsyncInferenceExecutor",
    "AsyncInferenceConfig",
    "ASYNC_CLI_NUMERIC_OVERRIDES",
    "apply_async_cli_overrides",
    "normalize_async_inference_config",
    "resolve_async_inference_config",
]
