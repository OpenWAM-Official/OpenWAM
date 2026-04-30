"""Asynchronous inference executor for real-time closed-loop robot control.

DreamZero-inspired: overlaps inference of the next action chunk with
execution of the current one. While the robot executes actions from
chunk N, the model concurrently computes chunk N+1 in a background thread.

This hides inference latency and enables tighter control loops,
approaching real-time performance even with large video diffusion models.
"""

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from openwam.deploy.base import BaseInferenceEngine


class AsyncInferenceExecutor:
    """Asynchronous inference executor with double buffering.

    Maintains two buffers: one being executed by the robot, one being
    filled by the inference engine. When the execution buffer is exhausted,
    it swaps with the inference buffer.

    Args:
        engine: The inference engine to run asynchronously.
        chunk_size: Number of actions per inference chunk.
            Should match the model's ``num_frames`` parameter.
        prefetch: If True, start computing the next chunk as soon as
            the current one begins execution (before it's exhausted).
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        chunk_size: int = 49,
        prefetch: bool = True,
    ):
        self.engine = engine
        self.chunk_size = chunk_size
        self.prefetch = prefetch

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._action_buffer: deque = deque()
        self._pending_future: Optional[Future] = None
        self._last_conditions: Optional[dict] = None
        self._lock = threading.Lock()

        # Timing stats
        self._inference_times: list = []
        self._wait_times: list = []

    def predict_action(self, conditions: dict) -> np.ndarray:
        """Get the next action, triggering async inference if needed.

        If actions are buffered, returns immediately. If the buffer is
        empty, waits for the pending inference to complete. In either
        case, if prefetch is enabled, starts the next inference in the
        background.

        Args:
            conditions: Observation conditions for the inference engine.
                Should include current observation, prompt, etc.

        Returns:
            (action_dim,) numpy array — the next action to execute.
        """
        self._last_conditions = conditions

        with self._lock:
            if len(self._action_buffer) == 0:
                self._refill_buffer(conditions)

            if len(self._action_buffer) == 0:
                raise RuntimeError("Failed to generate actions")

            action = self._action_buffer.popleft()

            # Prefetch: start next inference when buffer is half empty
            if self.prefetch and self._pending_future is None and len(self._action_buffer) <= self.chunk_size // 2:
                self._start_async_inference(conditions)

        return action

    def _refill_buffer(self, conditions: dict):
        """Fill the action buffer, waiting for pending inference if needed."""
        if self._pending_future is not None:
            # Wait for the async inference to complete
            t0 = time.monotonic()
            result = self._pending_future.result()
            self._wait_times.append(time.monotonic() - t0)
            self._pending_future = None
            self._unpack_result(result)
        else:
            # Synchronous inference
            t0 = time.monotonic()
            result = self.engine.generate(conditions)
            self._inference_times.append(time.monotonic() - t0)
            self._unpack_result(result)

    def _unpack_result(self, result: dict):
        """Extract actions from inference result into the buffer."""
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()
        for a in actions:
            self._action_buffer.append(a)

    def _start_async_inference(self, conditions: dict):
        """Start inference in the background thread pool."""

        def _infer():
            t0 = time.monotonic()
            with torch.no_grad():
                result = self.engine.generate(conditions)
            self._inference_times.append(time.monotonic() - t0)
            return result

        self._pending_future = self._executor.submit(_infer)

    def reset(self):
        """Clear buffers and cancel pending inference. Call between episodes."""
        with self._lock:
            self._action_buffer.clear()
            if self._pending_future is not None:
                self._pending_future.cancel()
                self._pending_future = None
            self._last_conditions = None

    def shutdown(self):
        """Clean up the thread pool."""
        self.reset()
        self._executor.shutdown(wait=False)

    @property
    def stats(self) -> dict:
        """Return timing statistics."""
        inf_times = self._inference_times
        wait_times = self._wait_times
        return {
            "num_inferences": len(inf_times),
            "avg_inference_time_ms": (1000 * sum(inf_times) / len(inf_times) if inf_times else 0.0),
            "avg_wait_time_ms": (1000 * sum(wait_times) / len(wait_times) if wait_times else 0.0),
            "buffer_size": len(self._action_buffer),
        }
