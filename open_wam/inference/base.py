"""Abstract base class for inference engines."""

from abc import ABC, abstractmethod
from typing import Optional

from open_wam.models.architectures.base import BaseWAMArchitecture


class BaseInferenceEngine(ABC):
    """Inference engine base class.

    Concrete engines implement ``generate`` which accepts observation
    conditions and produces video frames and/or action trajectories.

    Args:
        cfg: Hydra config.
        pipeline: Loaded video pipeline.
        action_dit: Raw ActionDiT model (legacy, prefer ``architecture``).
        architecture: WAM architecture wrapping the action model.
    """

    def __init__(self, cfg, pipeline, action_dit=None, architecture: Optional[BaseWAMArchitecture] = None):
        self.cfg = cfg
        self.pipeline = pipeline
        self.action_dit = action_dit
        self.architecture = architecture

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
