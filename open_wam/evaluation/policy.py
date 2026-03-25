"""Generic WAM policy interface for closed-loop evaluation."""

from collections import deque
from typing import Optional

import numpy as np

from open_wam.inference.base import BaseInferenceEngine


class WAMPolicy:
    """WAM policy adapter with action caching for closed-loop control.

    Generates a trajectory chunk via the inference engine, caches actions,
    and returns one action per ``predict_action`` call.  When the cache is
    exhausted, a new trajectory is generated (receding-horizon execution).
    """

    def __init__(self, engine: BaseInferenceEngine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.action_buffer: deque = deque()
        history_len = getattr(cfg, "history_len", 10)
        self.obs_history: deque = deque(maxlen=history_len)

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given observation.

        Refills the action buffer when empty by calling the inference engine.
        """
        self.obs_history.append(obs)
        if len(self.action_buffer) == 0:
            conditions = self._build_conditions(obs)
            result = self.engine.generate(conditions)
            actions = result["actions"]
            if hasattr(actions, "cpu"):
                actions = actions.cpu().numpy()
            for a in actions:
                self.action_buffer.append(a)
        return self.action_buffer.popleft()

    def reset(self):
        """Clear state between episodes."""
        self.action_buffer.clear()
        self.obs_history.clear()

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from current observation + history."""
        return {
            "observation": obs,
            "obs_history": list(self.obs_history),
        }
