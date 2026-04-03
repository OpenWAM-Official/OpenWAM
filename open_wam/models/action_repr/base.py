"""Abstract base class for action representations.

Action representation defines how raw actions from the dataset are
transformed into the latent space that the diffusion model operates on.
This is orthogonal to the architecture choice (dual_system/moe_expert/
shared_backbone) — any representation can be used with any architecture.

Two representations are supported:
1. **Continuous** (default): raw action vectors, z-score normalized.
   ActionDiT operates directly on the continuous EEF/joint delta space.
2. **FAST**: actions are first discretized via a learned tokenizer,
   then embedded into a continuous latent for diffusion.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class BaseActionRepresentation(ABC, nn.Module):
    """Base class for action representation transforms.

    Subclasses define encode (dataset → diffusion space) and decode
    (diffusion output → raw actions).  The diffusion model's action_dim
    should be set to :attr:`latent_dim`.
    """

    @abstractmethod
    def encode(self, actions: Tensor) -> Tensor:
        """Encode raw actions from the dataset into diffusion latent space.

        Args:
            actions: (B, T, native_dim) raw action vectors.

        Returns:
            (B, T, latent_dim) encoded latents for diffusion.
        """
        ...

    @abstractmethod
    def decode(self, latents: Tensor) -> Tensor:
        """Decode diffusion output back to raw action space.

        Args:
            latents: (B, T, latent_dim) denoised latents from diffusion.

        Returns:
            (B, T, native_dim) reconstructed action vectors.
        """
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Dimensionality of the diffusion operating space."""
        ...

    @property
    @abstractmethod
    def native_dim(self) -> int:
        """Dimensionality of the original dataset action space."""
        ...
