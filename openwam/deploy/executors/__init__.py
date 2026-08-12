"""Inference executors: how engine-generated action chunks reach the robot.

Two interchangeable mechanisms behind one interface
(``predict_action(conditions)`` / ``reset()`` / ``shutdown()``):
:class:`SyncInferenceExecutor` (default; buffer-and-replan) and
:class:`AsyncInferenceExecutor` (threaded prefetch). Selected via
``inference.inference_mode``.
"""

from openwam.deploy.executors.async_executor import (
    EXECUTION_CLI_NUMERIC_OVERRIDES,
    AsyncInferenceExecutor,
    ExecutionConfig,
    apply_execution_cli_overrides,
    normalize_execution_config,
    resolve_execution_config,
)
from openwam.deploy.executors.sync_executor import SyncInferenceExecutor

__all__ = [
    "SyncInferenceExecutor",
    "AsyncInferenceExecutor",
    "ExecutionConfig",
    "EXECUTION_CLI_NUMERIC_OVERRIDES",
    "apply_execution_cli_overrides",
    "normalize_execution_config",
    "resolve_execution_config",
]
