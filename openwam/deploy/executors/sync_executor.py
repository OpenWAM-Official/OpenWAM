"""Synchronous inference executor: buffer-and-replan over engine chunks.

The default execution mode. Generates a full action chunk only when the
buffer runs out (or the receding horizon elapses), pops one action per
control step in between, and fuses overlapping predictions from consecutive
generations via temporal ensembling (exponential weighting) — the standard
approach used in ACT, Diffusion Policy, and similar action-chunking policies.

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
        execute_horizon: Number of actions to execute before re-generating.
            ``None`` means consume the full chunk (greedy).
        temporal_ensemble: Enable temporal ensembling of overlapping
            predictions (effective only with a receding horizon).
        ensemble_decay: Exponential decay weight for older predictions.
            Lower = trust newer predictions more.
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        execute_horizon: Optional[int] = None,
        temporal_ensemble: bool = True,
        ensemble_decay: float = 0.5,
    ):
        self.engine = engine
        self.execute_horizon = execute_horizon
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_decay = ensemble_decay

        # Action buffer for the current window
        self._action_buffer: deque = deque()
        # Ensemble accumulator: _ensemble_buffer[t] = list of (weight, action)
        # for absolute timestep t, from multiple overlapping predictions
        self._ensemble_buffer: dict = {}
        self._current_step: int = 0
        self._steps_since_generate: int = 0

    def predict_action(self, conditions: dict) -> np.ndarray:
        """Pop the next action, regenerating from the engine when needed."""
        need_generate = len(self._action_buffer) == 0 or (
            self.execute_horizon is not None and self._steps_since_generate >= self.execute_horizon
        )

        if need_generate:
            self._generate_and_enqueue(conditions)
            self._steps_since_generate = 0

        action = self._action_buffer.popleft()
        self._current_step += 1
        self._steps_since_generate += 1
        return action

    def _generate_and_enqueue(self, conditions: dict):
        """Run inference and populate the action buffer.

        When temporal ensembling is active, new predictions are merged
        with any remaining buffered predictions for overlapping timesteps.
        """
        result = self.engine.generate(conditions)
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.cpu().numpy()

        chunk_len = len(actions)
        t_start = self._current_step

        if self.temporal_ensemble and self.execute_horizon is not None:
            # Add new predictions to ensemble buffer with full weight
            for i, a in enumerate(actions):
                t = t_start + i
                if t not in self._ensemble_buffer:
                    self._ensemble_buffer[t] = []
                self._ensemble_buffer[t].append((1.0, a))

            # Reweight by generation age: entry at age k gets weight decay^k.
            # Newest (last) entry always has weight 1.0 (age 0).
            for t in list(self._ensemble_buffer.keys()):
                entries = self._ensemble_buffer[t]
                if len(entries) > 1:
                    n = len(entries)
                    for j in range(n):
                        age = n - 1 - j
                        _, a = entries[j]
                        entries[j] = (self.ensemble_decay**age, a)

            # Build fused action buffer for the upcoming steps
            self._action_buffer.clear()
            for i in range(chunk_len):
                t = t_start + i
                entries = self._ensemble_buffer.get(t, [])
                if entries:
                    fused = self._weighted_average(entries)
                    self._action_buffer.append(fused)

            # Cleanup old timesteps we've already passed
            for t in list(self._ensemble_buffer.keys()):
                if t < t_start:
                    del self._ensemble_buffer[t]
        else:
            # Greedy mode: just fill the buffer
            self._action_buffer.clear()
            for a in actions:
                self._action_buffer.append(a)

    @staticmethod
    def _weighted_average(entries: list) -> np.ndarray:
        """Compute weighted average of (weight, action) pairs."""
        total_w = sum(w for w, _ in entries)
        if total_w == 0:
            return entries[-1][1]
        result = sum(w * a for w, a in entries) / total_w
        return result

    def reset(self):
        """Clear state between episodes."""
        self._action_buffer.clear()
        self._ensemble_buffer.clear()
        self._current_step = 0
        self._steps_since_generate = 0

    def shutdown(self):
        """No background resources; present for executor-interface symmetry."""
