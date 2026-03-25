"""Abstract base class for inference engines."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseInferenceEngine(ABC):
    """Inference engine base class.

    Concrete engines implement ``generate`` which accepts observation
    conditions and produces video frames and/or action trajectories.
    """

    def __init__(self, cfg, pipeline, action_dit):
        self.cfg = cfg
        self.pipeline = pipeline
        self.action_dit = action_dit

    @abstractmethod
    def generate(self, conditions: dict) -> dict:
        """Generate video and/or actions from conditions.

        Args:
            conditions: dict with keys like ``prompt``, ``reference_image``,
                ``context_video``, ``seed``, etc.

        Returns:
            dict with ``video`` (Tensor) and ``actions`` (Tensor).
        """
        ...
