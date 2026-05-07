"""Mock inference engine for testing and development without model weights.

Implements the same BaseInferenceEngine interface as JointInferenceEngine but
returns random Gaussian actions immediately, optionally sleeping to simulate
realistic inference latency.

Typical usage:

    # Start a mock server (no GPU or weights needed)
    python scripts/deploy.py --mock --mock-action-dim 20 --mock-latency-ms 2000

    # Then test against it normally
    python scripts/inference_single_test.py --test
"""

import time

import numpy as np

from openwam.deploy.base import BaseInferenceEngine


class MockInferenceEngine(BaseInferenceEngine):
    """Inference engine that returns random actions without loading any model.

    Useful for:
    - Integration testing the server protocol without a GPU
    - Benchmarking client-side code
    - CI environments where model weights are unavailable

    Args:
        cfg: Config object (only ``cfg.inference.num_frames`` is read, if present).
        action_dim: Dimensionality of the returned action vector.
        latency_ms: Simulated inference latency in milliseconds.  The engine
            sleeps for this duration before returning, mimicking a real model.
        seed: Optional random seed for reproducible outputs.
    """

    def __init__(
        self,
        cfg=None,
        action_dim: int = 20,
        latency_ms: float = 0.0,
        seed: int | None = None,
    ):
        # Pass None for pipeline/action_backbone — mock needs no real model
        super().__init__(cfg=cfg)
        self.action_dim = action_dim
        self.latency_ms = latency_ms
        self._rng = np.random.default_rng(seed)

    def _num_frames(self, conditions: dict) -> int:
        """Resolve chunk length from conditions or config."""
        if "num_frames" in conditions:
            return int(conditions["num_frames"])
        try:
            return int(self.cfg.inference.num_frames)
        except (AttributeError, TypeError):
            return 33  # training default

    def generate(self, conditions: dict) -> dict:
        """Return random Gaussian actions after an optional sleep.

        Args:
            conditions: Observation conditions dict (mostly ignored by mock).
                ``num_frames`` key is respected if present.

        Returns:
            dict with:
                - ``video``: ``None`` (no video generated)
                - ``actions``: ``np.ndarray`` of shape ``(num_frames - 1, action_dim)``,
                  aligned with the real engine's output: action latents carry
                  ``num_frames - 1`` steps because frame 0 is the conditioning
                  frame excluded from the loss.
        """
        if self.latency_ms > 0.0:
            time.sleep(self.latency_ms / 1000.0)

        num_frames = self._num_frames(conditions)
        action_steps = max(num_frames - 1, 1)
        actions = self._rng.standard_normal((action_steps, self.action_dim)).astype(np.float32) * 0.1

        return {"video": None, "actions": actions}
