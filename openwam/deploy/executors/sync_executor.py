"""Synchronous inference executor: buffer-and-replan over engine chunks.

The default execution mode. Generates an action chunk when the executable
buffer runs out and returns one action per control step. If
``inference_horizon`` is shorter than the generated chunk, the unused tail is
discarded rather than mixed into later predictions.

Mirrors :class:`AsyncInferenceExecutor`'s interface
(``predict_action(conditions)`` / ``reset()`` / ``shutdown()``) so
:class:`~openwam.deploy.policy.WAMPolicy` can pick either executor at
construction time.
"""

from collections import deque
from typing import Optional

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine


class SyncInferenceExecutor:
    """Receding-horizon execution of engine-generated action chunks.

    Args:
        engine: Inference engine that generates action chunks.
        inference_horizon: Number of actions to execute before re-generating.
            ``None`` means consume the full chunk (greedy).
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        inference_horizon: Optional[int] = None,
    ):
        self.engine = engine
        if inference_horizon is not None and inference_horizon <= 0:
            raise ValueError("inference_horizon must be positive")
        self.inference_horizon = inference_horizon

        self._action_buffer: deque = deque()

    def predict_action(self, conditions: dict) -> np.ndarray:
        """Pop the next action, regenerating after the configured horizon."""
        if len(self._action_buffer) == 0:
            self._generate_and_enqueue(conditions)

        action = self._action_buffer.popleft()
        return action

    def _generate_and_enqueue(self, conditions: dict):
        """Run inference and keep only the executable action horizon."""
        result = self.engine.generate(conditions)
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()

        chunk_len = len(actions)
        if chunk_len <= 0:
            raise RuntimeError("Inference result did not contain any actions")

        inference_horizon = self.inference_horizon if self.inference_horizon is not None else chunk_len
        if inference_horizon > chunk_len:
            raise ValueError(f"inference_horizon ({inference_horizon}) must be <= action horizon ({chunk_len})")

        self._action_buffer.clear()
        self._action_buffer.extend(actions[:inference_horizon])

    def reset(self):
        """Clear state between episodes."""
        self._action_buffer.clear()

    def shutdown(self):
        """No background resources; present for executor-interface symmetry."""
