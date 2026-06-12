"""Generic WAM policy interface for closed-loop evaluation.

Supports two execution modes:

1. **Greedy** (``execute_horizon=None``): Generate a full action chunk, consume
   all actions, then re-generate.  Simple but the tail of the chunk degrades.

2. **Receding-horizon** (``execute_horizon=K``): Generate a full chunk but only
   execute the first K actions, then re-generate with a fresh observation.
   Overlapping predictions are fused via temporal ensembling (exponential
   weighting) to reduce jitter.  This is the standard approach used in
   ACT, Diffusion Policy, and similar action-chunking policies.

Optionally wraps inference in an :class:`AsyncInferenceExecutor` for
double-buffered closed-loop control that overlaps computation with execution.
"""

from collections import deque
from typing import Optional

import numpy as np

from openwam.deploy.base import BaseInferenceEngine


def build_async_info(async_config, policy_cfg, async_executor=None) -> dict:
    """Return a stable async inference info payload for get_info() and policies."""
    info = async_config.as_dict()
    resolved_delay = info["inference_delay_steps"]
    if info["enabled"] and resolved_delay is None and info["execution_horizon"] is not None:
        resolved_delay = max(0, info["execution_horizon"] // 2)
    info["effective_temporal_ensemble"] = (
        False
        if info["enabled"]
        else bool(
            getattr(policy_cfg, "temporal_ensemble", True) and getattr(policy_cfg, "execute_horizon", None) is not None
        )
    )
    info.update(
        {
            "num_inferences": 0,
            "num_sync_inferences": 0,
            "num_background_inferences": 0,
            "buffer_size": 0,
            "pending": False,
            "pending_start_step": None,
            "current_step": 0,
            "action_horizon": None,
            "execution_horizon": info["execution_horizon"],
            "inference_delay_steps": info["inference_delay_steps"],
            "resolved_inference_delay_steps": resolved_delay,
            "lead_time_steps": resolved_delay,
            "last_skip_steps": 0,
        }
    )
    if async_executor is not None:
        info.update(async_executor.stats)
    return info


class WAMPolicy:
    """WAM policy adapter with receding-horizon action execution.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Config object with optional fields:
            - ``execute_horizon``: Number of actions to execute before
              re-generating.  ``None`` means use the full chunk (greedy).
            - ``temporal_ensemble``: Enable temporal ensembling of
              overlapping predictions (default True when receding-horizon).
            - ``ensemble_decay``: Exponential decay weight for older
              predictions.  Lower = trust newer predictions more (default 0.5).
        async_config: Optional config for async inference. When enabled,
            wraps the engine in AsyncInferenceExecutor for double-buffered
            closed-loop execution. Expected fields:
            - ``mode``: ``none`` or ``vanilla``.
            - ``execution_horizon``: actions executed from each generated chunk.
            - ``inference_delay_steps``: expected latency in action steps.
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, async_config=None):
        self.cfg = cfg
        self.engine = engine

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

        from openwam.deploy.optimizations import AsyncInferenceExecutor, normalize_async_inference_config

        self._async_config = normalize_async_inference_config(async_config, policy_cfg=cfg)
        self._async = self._async_config.enabled
        self._async_executor = None
        if self._async:
            self._async_executor = AsyncInferenceExecutor(
                engine=engine,
                execution_horizon=self._async_config.execution_horizon,
                inference_delay_steps=self._async_config.inference_delay_steps,
            )

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given observation.

        In async mode, delegates to the AsyncInferenceExecutor's buffer
        management. Otherwise uses receding-horizon or greedy mode.
        """
        if self._async:
            conditions = self._build_conditions(obs)
            action = self._async_executor.predict_action(conditions)
            self._current_step += 1
            return action

        need_generate = len(self._action_buffer) == 0 or (
            self.execute_horizon is not None and self._steps_since_generate >= self.execute_horizon
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

            # Build fused action buffer for the next execute_horizon steps
            self._action_buffer.clear()
            _horizon = self.execute_horizon if self.execute_horizon else chunk_len  # noqa: F841
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
        if self._async_executor is not None:
            self._async_executor.reset()

    def shutdown(self):
        """Clean up async resources."""
        if self._async_executor is not None:
            self._async_executor.shutdown()

    @property
    def async_info(self) -> dict:
        """Return normalized async mode and runtime stats."""
        return build_async_info(self._async_config, self.cfg, self._async_executor)

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from the current observation.

        Populates the engine-facing fields (``first_frame_image``,
        ``prompt``) from the server-preprocessed observation so the
        pipeline receives images without any further client-side work.
        """
        conditions = {
            "observation": obs,
        }
        img = obs.get("image")
        if img is not None:
            # Single first frame — pipeline expects list[PIL.Image]
            conditions["first_frame_image"] = [img]
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio_state"] = obs["state"]
        return conditions
