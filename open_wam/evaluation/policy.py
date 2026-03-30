"""Generic WAM policy interface for closed-loop evaluation.

Supports two execution modes:

1. **Greedy** (``execute_horizon=None``): Generate a full action chunk, consume
   all actions, then re-generate.  Simple but the tail of the chunk degrades.

2. **Receding-horizon** (``execute_horizon=K``): Generate a full chunk but only
   execute the first K actions, then re-generate with a fresh observation.
   Overlapping predictions are fused via temporal ensembling (exponential
   weighting) to reduce jitter.  This is the standard approach used in
   ACT, Diffusion Policy, and similar action-chunking policies.
"""

from collections import deque
from typing import Optional

import numpy as np

from open_wam.inference.base import BaseInferenceEngine


class WAMPolicy:
    """WAM policy adapter with receding-horizon action execution.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Config object with optional fields:
            - ``history_len``: Observation history length (default 10).
            - ``execute_horizon``: Number of actions to execute before
              re-generating.  ``None`` means use the full chunk (greedy).
            - ``temporal_ensemble``: Enable temporal ensembling of
              overlapping predictions (default True when receding-horizon).
            - ``ensemble_decay``: Exponential decay weight for older
              predictions.  Lower = trust newer predictions more (default 0.5).
    """

    def __init__(self, engine: BaseInferenceEngine, cfg):
        self.engine = engine
        self.cfg = cfg

        history_len = getattr(cfg, "history_len", 10)
        self.obs_history: deque = deque(maxlen=history_len)

        # Receding-horizon config
        self.execute_horizon: Optional[int] = getattr(cfg, "execute_horizon", None)
        self.temporal_ensemble: bool = getattr(cfg, "temporal_ensemble", True)
        self.ensemble_decay: float = getattr(cfg, "ensemble_decay", 0.5)

        # Action buffer: list of (timestep, action) for the current window
        self._action_buffer: deque = deque()
        # Ensemble accumulator: pending_actions[t] = list of (weight, action)
        # for absolute timestep t, from multiple overlapping predictions
        self._ensemble_buffer: dict = {}  # timestep -> list of (weight, action)
        self._current_step: int = 0
        self._steps_since_generate: int = 0

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given observation.

        In receding-horizon mode, triggers re-generation every
        ``execute_horizon`` steps and fuses overlapping predictions.
        """
        self.obs_history.append(obs)

        need_generate = (
            len(self._action_buffer) == 0
            or (
                self.execute_horizon is not None
                and self._steps_since_generate >= self.execute_horizon
            )
        )

        if need_generate:
            self._generate_and_enqueue(obs)
            self._steps_since_generate = 0

        action = self._action_buffer.popleft()
        self._current_step += 1
        self._steps_since_generate += 1
        return action

    def _generate_and_enqueue(self, obs: dict):
        """Run inference and populate the action buffer.

        When temporal ensembling is active, new predictions are merged
        with any remaining buffered predictions for overlapping timesteps.
        """
        conditions = self._build_conditions(obs)
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

            # Decay older predictions: each older generation gets decay^k weight
            # The newest generation always has weight 1.0
            for t in list(self._ensemble_buffer.keys()):
                entries = self._ensemble_buffer[t]
                if len(entries) > 1:
                    # Re-weight: newest is last, oldest is first
                    n = len(entries)
                    for j in range(n - 1):
                        w, a = entries[j]
                        entries[j] = (w * self.ensemble_decay, a)

            # Build fused action buffer for the next execute_horizon steps
            self._action_buffer.clear()
            horizon = self.execute_horizon if self.execute_horizon else chunk_len
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
        self.obs_history.clear()
        self._current_step = 0
        self._steps_since_generate = 0

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from current observation + history."""
        return {
            "observation": obs,
            "obs_history": list(self.obs_history),
        }
