"""Abstract base class for video backbone networks (VACE, TI2V, etc.)."""

from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class BaseVideoBackbone(ABC):
    """Video backbone abstraction supporting VACE / TI2V / future models.

    Concrete implementations wrap the underlying DiT model and expose a
    uniform interface for the training, inference, and evaluation layers.
    """

    @abstractmethod
    def encode_video(self, video: Tensor) -> Tensor:
        """Encode raw video frames into latent space."""
        ...

    @abstractmethod
    def get_bridge_features(self, layer_indices: List[int]) -> List[Tensor]:
        """Extract intermediate hidden states for the ActionDiT bridge."""
        ...

    @abstractmethod
    def denoise_step(self, x_t: Tensor, t: Tensor, conditions: dict) -> Tensor:
        """Execute a single denoising step."""
        ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Hidden dimension of the video DiT (used for ActionDiT's video_dim)."""
        ...
