"""Abstract base class for WAM (World-Action Model) architectures.

Defines how the action prediction stream integrates with the video DiT backbone.
Three paradigms are supported (see assets/arch.png):

1. **Shared Backbone**: Action tokens are part of the video DiT sequence.
   The same transformer processes both video and action tokens.

2. **MoE Action Expert**: Action tokens route to specialized expert FFN
   layers within the video DiT blocks (Mixture-of-Experts style).

3. **Dual-System**: A separate lightweight ActionDiT receives bridge
   features from the video DiT via cross-attention or joint self-attention.
   This is the architecture currently implemented in the codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class ActionState:
    """Mutable state container threaded through the video DiT block loop.

    Each WAM architecture populates this differently:
    - DualSystem: holds ActionDiTState (x_action, bridge features, etc.)
    - MoE: holds routing masks and expert outputs
    - SharedBackbone: holds indices into the video token sequence

    The ``extra`` dict allows architecture-specific state without subclassing.
    """

    action_latents: Optional[Tensor] = None  # (B, T_action, dim) or None
    action_prediction: Optional[Tensor] = None  # filled after extract_action_prediction
    timestep: Optional[Tensor] = None  # diffusion timestep for action stream
    extra: dict = field(default_factory=dict)  # architecture-specific state


class BaseWAMArchitecture(ABC, nn.Module):
    """Base class for WAM architecture variants.

    Concrete subclasses define how the action stream interacts with the
    video DiT during the denoising loop. The three hook points correspond
    to three phases of the forward pass:

    1. **prepare** — Before the DiT block loop: initialize action tokens/latents
    2. **on_dit_block** — Inside the loop: called after each DiT block with
       the current video hidden state. This is where bridge attention,
       MoE routing, or sequence manipulation happens.
    3. **extract** — After the loop: produce the final action noise prediction

    Args:
        cfg: Architecture-specific configuration (OmegaConf DictConfig or dict).
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def prepare_action_tokens(self, noisy_actions: Tensor, timestep: Tensor, **kwargs) -> ActionState:
        """Initialize action state before the video DiT block loop.

        Args:
            noisy_actions: (B, T_action, action_dim) noisy action latents.
            timestep: (B,) diffusion timestep for the action stream.
            **kwargs: Extra conditioning (e.g., action_mask).

        Returns:
            ActionState with initialized latents and any architecture-specific
            state in ``extra``.
        """
        ...

    @abstractmethod
    def on_dit_block(
        self,
        block_id: int,
        video_hidden: Tensor,
        action_state: ActionState,
    ) -> Tuple[Tensor, ActionState]:
        """Hook called after each video DiT block.

        This is the core extension point where architectures differ:
        - DualSystem: run ActionDiT block with bridge attention to video_hidden
        - MoE: route action tokens to expert FFN within this block
        - SharedBackbone: no-op (action tokens already in video_hidden)

        Args:
            block_id: Index of the current video DiT block (0-based).
            video_hidden: (B, T_video, dim) video hidden state after this block.
            action_state: Mutable action state from previous block.

        Returns:
            (video_hidden, action_state) — potentially modified video hidden
            state (for joint_self_attn or MoE) and updated action state.
        """
        ...

    @abstractmethod
    def extract_action_prediction(self, action_state: ActionState) -> Tensor:
        """Extract the final action noise prediction after the DiT block loop.

        Args:
            action_state: Final action state after all blocks.

        Returns:
            (B, T_action, action_dim) action noise prediction.
        """
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Raw action vector dimensionality."""
        ...

    @property
    @abstractmethod
    def bridge_layers(self) -> tuple:
        """Video DiT layer indices where on_dit_block should be active.

        For SharedBackbone this may return all layers or an empty tuple.
        For DualSystem this returns the configured bridge layer indices.
        """
        ...

    @property
    def is_interleaved(self) -> bool:
        """Whether the architecture participates in the video DiT forward pass.

        When True, action processing happens inside the video DiT block loop
        (e.g. joint_self_attn, MoE). When False, bridge features are collected
        first and action processing happens separately (e.g. cross_attn).
        """
        return False

    @property
    def uses_proprioception(self) -> bool:
        """Whether the architecture consumes a ``proprio_state`` input.

        Subclasses that support proprioceptive conditioning should override.
        """
        return False

    @property
    def action_mean(self) -> Tensor:
        """Per-dimension action mean for denormalization."""
        return torch.zeros(self.action_dim)

    @property
    def action_std(self) -> Tensor:
        """Per-dimension action std for denormalization."""
        return torch.ones(self.action_dim)
