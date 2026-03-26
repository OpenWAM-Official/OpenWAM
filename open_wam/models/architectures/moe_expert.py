"""MoE Action Expert WAM Architecture.

Action tokens are processed within the video DiT blocks using Mixture-of-Experts
routing: at designated layers, action tokens route to specialized expert FFN
modules while video tokens use the standard FFN.

Corresponds to the "MoE Action Expert" diagram in assets/arch.png.

Status: Stub implementation — the MoE routing logic and expert layers need to
be implemented based on the specific MoE design chosen.
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from open_wam.models.architectures.base import ActionState, BaseWAMArchitecture
from open_wam.models.architectures.registry import register_architecture


@register_architecture("moe_expert")
class MoEActionExpertArchitecture(BaseWAMArchitecture):
    """MoE Action Expert: action tokens route to expert FFN within DiT blocks.

    In this architecture, action tokens are part of the input sequence to
    the video DiT, but at certain layers they are routed to specialized
    action expert FFN modules instead of the standard video FFN.

    This enables the action stream to share the self-attention computation
    with video tokens (learning joint representations) while having
    dedicated capacity for action prediction through expert FFN layers.

    Args:
        cfg: Configuration with keys:
            action_dim: Action vector dimension.
            num_action_tokens: Number of action tokens in the sequence.
            expert_layers: DiT layer indices where MoE routing activates.
            expert_ffn_dim: Hidden dimension of the action expert FFN.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._action_dim = int(cfg.get("action_dim", 14)) if cfg else 14
        self._num_action_tokens = int(cfg.get("num_action_tokens", 49)) if cfg else 49
        self._expert_layers = tuple(cfg.get("expert_layers", ())) if cfg else ()

        # TODO: Implement action expert FFN modules
        # self.action_experts = nn.ModuleList([...])
        # self.router = nn.Linear(dim, 2)  # video vs action routing

    def prepare_action_tokens(
        self, noisy_actions: Tensor, timestep: Tensor, **kwargs
    ) -> ActionState:
        return ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
            extra={"block_counter": 0},
        )

    def on_dit_block(
        self,
        block_id: int,
        video_hidden: Tensor,
        action_state: ActionState,
    ) -> Tuple[Tensor, ActionState]:
        if block_id not in self._expert_layers:
            return video_hidden, action_state

        # TODO: Implement MoE routing logic
        # 1. Split video_hidden into video tokens and action tokens
        # 2. Route action tokens through expert FFN
        # 3. Route video tokens through standard FFN
        # 4. Recombine and return
        raise NotImplementedError(
            "MoE Action Expert routing not yet implemented. "
            "This requires integrating expert FFN modules into the video DiT blocks."
        )

    def extract_action_prediction(self, action_state: ActionState) -> Tensor:
        # TODO: Extract action tokens from the combined sequence
        raise NotImplementedError(
            "MoE Action Expert extraction not yet implemented."
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def bridge_layers(self) -> tuple:
        return self._expert_layers
