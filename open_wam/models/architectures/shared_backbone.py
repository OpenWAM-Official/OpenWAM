"""Shared Backbone WAM Architecture.

Action tokens are directly concatenated to the video token sequence and
processed by the same video DiT transformer. No separate action model exists.
The DiT learns to jointly attend to video and action tokens.

Corresponds to the "Shared Backbone" diagram in assets/arch.png.

Status: Stub implementation — the token concatenation, positional encoding
adaptation, and action extraction logic need to be implemented.
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from open_wam.models.architectures.base import ActionState, BaseWAMArchitecture
from open_wam.models.architectures.registry import register_architecture


@register_architecture("shared_backbone")
class SharedBackboneArchitecture(BaseWAMArchitecture):
    """Shared Backbone: video DiT processes both video and action tokens.

    In this architecture, action tokens are appended to the video latent
    sequence before the DiT block loop. The transformer jointly attends
    to all tokens. After the loop, action tokens are extracted from the
    output sequence and projected to action predictions.

    This fully reuses the video generation weights for action prediction,
    maximizing knowledge transfer from the pretrained video model.

    Args:
        cfg: Configuration with keys:
            action_dim: Action vector dimension.
            video_dim: Hidden dimension of the video DiT.
            num_action_tokens: Number of action tokens to append.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        raise NotImplementedError(
            "SharedBackboneArchitecture is not yet implemented. "
            "Use 'dual_system' or 'moe_expert' instead. "
            "See assets/arch.png for architecture diagrams."
        )

    def prepare_action_tokens(
        self, noisy_actions: Tensor, timestep: Tensor, **kwargs
    ) -> ActionState:
        # TODO: Project noisy_actions to video_dim, create positional embeddings,
        # and prepare them for concatenation with video tokens
        return ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
            extra={"num_action_tokens": self._num_action_tokens},
        )

    def on_dit_block(
        self,
        block_id: int,
        video_hidden: Tensor,
        action_state: ActionState,
    ) -> Tuple[Tensor, ActionState]:
        # No-op: action tokens are already part of video_hidden sequence.
        # The video DiT block processes them together with video tokens.
        return video_hidden, action_state

    def extract_action_prediction(self, action_state: ActionState) -> Tensor:
        # TODO: Slice action tokens from the DiT output and project to action_dim
        raise NotImplementedError(
            "Shared Backbone extraction not yet implemented. "
            "Requires slicing action tokens from DiT output sequence."
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def bridge_layers(self) -> tuple:
        # SharedBackbone doesn't use bridge layers — action tokens are in the sequence
        return ()
