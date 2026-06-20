"""WAM policy facade: one obs→action entry point over the two executors.

``WAMPolicy`` is the seam between the server (which hands it preprocessed
observations) and the execution mechanism (which schedules engine calls):

- sync mode (default): :class:`SyncInferenceExecutor` — buffer-and-replan
  with receding horizon + temporal ensembling.
- async mode: :class:`AsyncInferenceExecutor` — double-buffered background
  inference overlapping generation with execution.

The executor is chosen once at construction from the normalized async
config; per-step dispatch is plain delegation.
"""

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.executors import (
    AsyncInferenceExecutor,
    SyncInferenceExecutor,
    normalize_execution_config,
)


class WAMPolicy:
    """Unified policy facade over the sync / async execution mechanisms.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Config object with optional fields (sync mode):
            - ``execute_horizon``: Number of actions to execute before
              re-generating.  ``None`` means use the full chunk (greedy).
            - ``temporal_ensemble``: Enable temporal ensembling of
              overlapping predictions (default True when receding-horizon).
            - ``ensemble_decay``: Exponential decay weight for older
              predictions.  Lower = trust newer predictions more (default 0.5).
        execution_config: ExecutionConfig-like (mode sync|async +
            execution_horizon / inference_delay_steps, async-only).
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, execution_config=None):
        self.cfg = cfg
        self.engine = engine

        self._execution_config = normalize_execution_config(execution_config, policy_cfg=cfg)
        self._async = self._execution_config.enabled
        if self._async:
            self._executor = AsyncInferenceExecutor(
                engine=engine,
                execution_horizon=self._execution_config.execution_horizon,
                inference_delay_steps=self._execution_config.inference_delay_steps,
            )
        else:
            self._executor = SyncInferenceExecutor(
                engine=engine,
                execute_horizon=getattr(cfg, "execute_horizon", None),
                temporal_ensemble=getattr(cfg, "temporal_ensemble", True),
                ensemble_decay=getattr(cfg, "ensemble_decay", 0.5),
            )

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given (already preprocessed) observation."""
        return self._executor.predict_action(self._build_conditions(obs))

    def reset(self):
        """Clear executor state between episodes."""
        self._executor.reset()

    def shutdown(self):
        """Release executor resources (background threads in async mode)."""
        self._executor.shutdown()

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
            conditions["proprio"] = obs["state"]
        return conditions
