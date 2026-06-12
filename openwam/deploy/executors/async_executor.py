"""Asynchronous inference executor for closed-loop robot control.

The executor overlaps generation of the next action chunk with execution of
the current executable horizon. This is a runtime scheduling optimization
only; it does not change the model API or the denoising algorithm.

Also hosts the async-inference configuration surface
(:class:`AsyncInferenceConfig` + normalize / resolve / CLI-override helpers):
the config describes exactly this executor's constructor knobs, so they live
together.
"""

import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import numpy as np
import torch

from openwam.deploy.engine import BaseInferenceEngine

logger = logging.getLogger(__name__)

VALID_ASYNC_MODES = ("none", "vanilla")
ASYNC_CLI_NUMERIC_OVERRIDES = ("async_execution_horizon", "async_inference_delay_steps")


@dataclass(frozen=True)
class AsyncInferenceConfig:
    """Normalized async inference config used by deployment policy code."""

    mode: str = "none"
    execution_horizon: Optional[int] = None
    inference_delay_steps: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "execution_horizon": self.execution_horizon,
            "inference_delay_steps": self.inference_delay_steps,
        }


def _select(cfg, path: str, default=None):
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.select(cfg, path, default=default)
    except ImportError:
        pass

    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _coerce_optional_int(value, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
    raise ValueError(f"{name} must be an integer, got {value!r}")


def normalize_async_inference_config(async_cfg=None, policy_cfg=None) -> AsyncInferenceConfig:
    """Normalize async inference config into a stable dataclass."""
    if async_cfg is None:
        return AsyncInferenceConfig()

    mode = _select(async_cfg, "mode", default=None)
    if mode is None:
        mode = "vanilla" if bool(_select(async_cfg, "enabled", default=False)) else "none"
    mode = str(mode).strip().lower()
    if mode not in VALID_ASYNC_MODES:
        raise ValueError(f"Unsupported async inference mode {mode!r}; expected one of {VALID_ASYNC_MODES}")

    vanilla_cfg = _select(async_cfg, "vanilla", default=None)
    execution_horizon = _select(vanilla_cfg, "execution_horizon", default=None)
    if execution_horizon is None:
        execution_horizon = _select(async_cfg, "execution_horizon", default=None)
    if execution_horizon is None:
        execution_horizon = _select(policy_cfg, "execute_horizon", default=None)

    inference_delay_steps = _select(vanilla_cfg, "inference_delay_steps", default=None)
    if inference_delay_steps is None:
        inference_delay_steps = _select(async_cfg, "inference_delay_steps", default=None)

    execution_horizon = _coerce_optional_int(execution_horizon, "execution_horizon")
    inference_delay_steps = _coerce_optional_int(inference_delay_steps, "inference_delay_steps")

    if mode == "vanilla":
        if execution_horizon is not None and execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if inference_delay_steps is not None and inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be non-negative")
        if (
            execution_horizon is not None
            and inference_delay_steps is not None
            and inference_delay_steps >= execution_horizon
        ):
            raise ValueError("inference_delay_steps must be < execution_horizon")

    return AsyncInferenceConfig(
        mode=mode,
        execution_horizon=execution_horizon,
        inference_delay_steps=inference_delay_steps,
    )


def _arg_value(args, name: str, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def apply_async_cli_overrides(root_cfg, args):
    """Apply async CLI overrides and validate the resulting nested config."""
    from omegaconf import OmegaConf

    async_mode = _arg_value(args, "async_mode")
    if async_mode is not None:
        mode = str(async_mode).strip().lower()
        if mode not in VALID_ASYNC_MODES:
            raise ValueError(f"Unsupported async inference mode {mode!r}; expected one of {VALID_ASYNC_MODES}")
        OmegaConf.update(root_cfg, "optimization.async_inference.mode", mode, merge=False)

    has_timing_override = any(_arg_value(args, name) is not None for name in ASYNC_CLI_NUMERIC_OVERRIDES)
    if has_timing_override:
        resolved = resolve_async_inference_config(root_cfg)
        if resolved.mode != "vanilla":
            raise ValueError(
                "--async-execution-horizon and --async-inference-delay-steps require "
                "--async-mode vanilla or optimization.async_inference.mode=vanilla"
            )

    execution_horizon = _arg_value(args, "async_execution_horizon")
    if execution_horizon is not None:
        OmegaConf.update(
            root_cfg,
            "optimization.async_inference.vanilla.execution_horizon",
            execution_horizon,
            merge=False,
        )

    inference_delay_steps = _arg_value(args, "async_inference_delay_steps")
    if inference_delay_steps is not None:
        OmegaConf.update(
            root_cfg,
            "optimization.async_inference.vanilla.inference_delay_steps",
            inference_delay_steps,
            merge=False,
        )

    if async_mode is not None or has_timing_override:
        resolve_async_inference_config(root_cfg)

    return root_cfg


def resolve_async_inference_config(root_cfg, policy_cfg=None) -> AsyncInferenceConfig:
    """Resolve async inference config from the deploy config tree.

    Async inference is configured through ``optimization.async_inference``.
    """
    async_cfg = _select(root_cfg, "optimization.async_inference", default=None)
    return normalize_async_inference_config(async_cfg, policy_cfg=policy_cfg)


class AsyncInferenceExecutor:
    """Asynchronous inference executor with a single background worker.

    Args:
        engine: Inference engine to run.
        execution_horizon: Number of actions to execute from each generated
            chunk before switching to a fresh chunk. ``None`` means use the
            generated action horizon.
        inference_delay_steps: Expected inference latency expressed in action
            steps. ``None`` chooses a conservative auto threshold of half the
            resolved execution horizon.
        chunk_size: Legacy alias for ``execution_horizon``.
        prefetch: Legacy switch. ``False`` disables background generation.
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        execution_horizon: Optional[int] = None,
        inference_delay_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
        prefetch: Optional[bool] = None,
    ):
        self.engine = engine
        if execution_horizon is None and chunk_size is not None:
            execution_horizon = chunk_size
        if execution_horizon is not None and execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if inference_delay_steps is not None and inference_delay_steps < 0:
            raise ValueError("inference_delay_steps must be non-negative")

        self.execution_horizon = execution_horizon
        self.inference_delay_steps = inference_delay_steps
        self._background_enabled = prefetch is not False

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._action_buffer: deque = deque()
        self._pending_future: Optional[Future] = None
        self._pending_start_step: Optional[int] = None
        self._last_conditions: Optional[dict] = None
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._action_horizon: Optional[int] = None
        self._resolved_execution_horizon: Optional[int] = execution_horizon
        self._resolved_inference_delay_steps: Optional[int] = inference_delay_steps
        self._current_step = 0
        self._last_skip_steps = 0

        self._num_sync_inferences = 0
        self._num_background_inferences = 0

    def predict_action(self, conditions: dict) -> np.ndarray:
        """Get the next action, starting the next chunk before this horizon ends."""
        self._last_conditions = conditions

        with self._lock:
            if len(self._action_buffer) == 0:
                self._refill_buffer(conditions)

            if len(self._action_buffer) == 0:
                raise RuntimeError("Failed to generate actions")

            self._maybe_start_async_inference(conditions)
            action = self._action_buffer.popleft()
            self._current_step += 1

        return action

    def _refill_buffer(self, conditions: dict):
        """Fill the action buffer, waiting for pending inference if needed."""
        if self._pending_future is not None:
            self._adopt_pending_result()
        else:
            with torch.no_grad():
                result = self.engine.generate(conditions)
            self._record_sync_inference()
            self._unpack_result(result)

    def _adopt_pending_result(self):
        """Adopt a pending result and skip actions made stale by elapsed steps."""
        future = self._pending_future
        if future is None:
            return

        start_step = self._pending_start_step
        try:
            result = future.result()
        finally:
            self._pending_future = None
            self._pending_start_step = None
        skip_steps = 0 if start_step is None else max(0, self._current_step - start_step)
        self._unpack_result(result, skip_steps=skip_steps)

    def _unpack_result(self, result: dict, skip_steps: int = 0):
        """Extract the executable action horizon from an inference result."""
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()

        action_horizon = len(actions)
        if action_horizon <= 0:
            raise RuntimeError("Inference result did not contain any actions")

        execution_horizon = self._resolve_execution_horizon(action_horizon)
        inference_delay_steps = self._resolve_inference_delay_steps(execution_horizon)

        self._action_horizon = action_horizon
        self._resolved_execution_horizon = execution_horizon
        self._resolved_inference_delay_steps = inference_delay_steps
        self._last_skip_steps = int(skip_steps)
        self._action_buffer.clear()
        start = int(skip_steps)
        end = min(action_horizon, start + execution_horizon)
        if start >= end:
            raise RuntimeError(f"Async inference result is stale: skip_steps={start}, action_horizon={action_horizon}")
        for action in actions[start:end]:
            self._action_buffer.append(action)

    def _resolve_execution_horizon(self, action_horizon: int) -> int:
        execution_horizon = self.execution_horizon if self.execution_horizon is not None else action_horizon
        if execution_horizon > action_horizon:
            raise ValueError(f"execution_horizon ({execution_horizon}) must be <= action horizon ({action_horizon})")
        return execution_horizon

    def _resolve_inference_delay_steps(self, execution_horizon: int) -> int:
        if self.inference_delay_steps is None:
            return max(0, execution_horizon // 2)
        if self.inference_delay_steps >= execution_horizon:
            raise ValueError("inference_delay_steps must be < execution_horizon")
        return self.inference_delay_steps

    def _maybe_start_async_inference(self, conditions: dict):
        if not self._background_enabled or self._pending_future is not None:
            return
        if self._resolved_inference_delay_steps is None:
            return
        lead_time_steps = self._resolved_inference_delay_steps
        if len(self._action_buffer) <= lead_time_steps:
            self._start_async_inference(conditions)

    def _start_async_inference(self, conditions: dict):
        """Start inference in the background thread pool."""

        def _infer():
            with torch.no_grad():
                result = self.engine.generate(conditions)
            self._record_background_inference()
            return result

        self._pending_future = self._executor.submit(_infer)
        self._pending_start_step = self._current_step

    def _record_sync_inference(self):
        with self._stats_lock:
            self._num_sync_inferences += 1

    def _record_background_inference(self):
        with self._stats_lock:
            self._num_background_inferences += 1

    def reset(self):
        """Clear buffers and cancel pending inference. Call between episodes."""
        with self._lock:
            self._action_buffer.clear()
            if self._pending_future is not None:
                future = self._pending_future
                self._pending_future = None
                if not future.cancel():
                    try:
                        future.result()
                    except Exception as exc:
                        logger.debug("Discarding failed async inference during reset: %s", exc)
            self._last_conditions = None
            self._pending_start_step = None
            self._current_step = 0
            self._last_skip_steps = 0

    def shutdown(self):
        """Clean up the thread pool."""
        self.reset()
        self._executor.shutdown(wait=True)

    @property
    def stats(self) -> dict:
        """Return scheduling statistics."""
        with self._lock:
            buffer_size = len(self._action_buffer)
            pending = self._pending_future is not None
            pending_start_step = self._pending_start_step
            current_step = self._current_step
            action_horizon = self._action_horizon
            execution_horizon = self._resolved_execution_horizon
            inference_delay_steps = self.inference_delay_steps
            resolved_inference_delay_steps = self._resolved_inference_delay_steps
            last_skip_steps = self._last_skip_steps
        with self._stats_lock:
            num_sync_inferences = self._num_sync_inferences
            num_background_inferences = self._num_background_inferences
        return {
            "num_inferences": num_sync_inferences + num_background_inferences,
            "num_sync_inferences": num_sync_inferences,
            "num_background_inferences": num_background_inferences,
            "buffer_size": buffer_size,
            "pending": pending,
            "pending_start_step": pending_start_step,
            "current_step": current_step,
            "action_horizon": action_horizon,
            "execution_horizon": execution_horizon,
            "inference_delay_steps": inference_delay_steps,
            "resolved_inference_delay_steps": resolved_inference_delay_steps,
            "lead_time_steps": resolved_inference_delay_steps,
            "last_skip_steps": last_skip_steps,
        }
