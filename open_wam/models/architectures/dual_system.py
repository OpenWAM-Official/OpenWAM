"""Dual-System WAM Architecture.

The current (and most tested) architecture: a separate lightweight ActionDiT
receives bridge features from the video DiT at configurable layer indices.

Corresponds to the "Dual-System" diagram in assets/arch.png and all three
bridge types in assets/modal_merging.png:
- cross_attn: unidirectional video → action
- cross_attn_detach: same with gradient detachment
- joint_self_attn: bidirectional MMDiT-style
"""

from typing import Tuple

import torch
from torch import Tensor

from open_wam.models.action_dit import ActionDiT
from open_wam.models.architectures.base import ActionState, BaseWAMArchitecture
from open_wam.models.architectures.registry import register_architecture


@register_architecture(
    "dual_system",
    status="supported",
    note="Primary production architecture for the current OpenWAM stack.",
)
class DualSystemArchitecture(BaseWAMArchitecture):
    """Dual-System: independent ActionDiT with bridge attention to video DiT.

    This wraps the existing ActionDiT implementation, exposing it through
    the BaseWAMArchitecture interface. The ActionDiT runs its own transformer
    blocks in lockstep with the video DiT, receiving features at bridge layers.

    Args:
        cfg: Dict or DictConfig with ActionDiT parameters:
            action_dim, dim, ffn_dim, num_heads, num_layers,
            video_dim, bridge_layers, bridge_type, etc.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is not None:
            bl = cfg.get("bridge_layers", (3, 7, 11, 15, 19, 23, 26, 29))
            if isinstance(bl, str):
                bl = tuple(int(x) for x in bl.split(","))
            elif not isinstance(bl, tuple):
                bl = tuple(bl)

            self.action_dit = ActionDiT(
                action_dim=int(cfg.get("action_dim", 14)),
                dim=int(cfg.get("dim", 768)),
                ffn_dim=int(cfg.get("ffn_dim", 3072)),
                num_heads=int(cfg.get("num_heads", 12)),
                num_layers=int(cfg.get("num_layers", 8)),
                video_dim=int(cfg.get("video_dim", 1536)),
                bridge_layers=bl,
                bridge_type=cfg.get("bridge_type", "cross_attn_detach"),
            )
        else:
            self.action_dit = None

    def prepare_action_tokens(self, noisy_actions: Tensor, timestep: Tensor, **kwargs) -> ActionState:
        state = ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
        )
        if self.action_dit is not None and self.action_dit.bridge_type == "joint_self_attn":
            dit_state = self.action_dit.prepare_action_state(
                noisy_actions,
                timestep,
                use_gradient_checkpointing=kwargs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=kwargs.get("use_gradient_checkpointing_offload", False),
            )
            state.extra["dit_state"] = dit_state
        else:
            state.extra["bridge_features"] = []
            state.extra["bridge_block_counter"] = 0
        return state

    def on_dit_block(
        self,
        block_id: int,
        video_hidden: Tensor,
        action_state: ActionState,
    ) -> Tuple[Tensor, ActionState]:
        if self.action_dit is None:
            return video_hidden, action_state

        if block_id not in self.action_dit.bridge_layers_set:
            return video_hidden, action_state

        if "dit_state" in action_state.extra:
            # joint_self_attn path — interleaved execution
            dit_state = action_state.extra["dit_state"]
            i = dit_state.bridge_block_counter
            x_video_proj = self.action_dit.video_projs[i](video_hidden)
            if dit_state.x_video_proj is None:
                dit_state.x_video_proj = x_video_proj
            else:
                dit_state.x_video_proj = dit_state.x_video_proj + x_video_proj

            block = self.action_dit.blocks[i]
            dit_state.x_action, dit_state.x_video_proj = block(
                dit_state.x_action, dit_state.x_video_proj, dit_state.t_mod
            )

            # Back-project video residual
            x_video_new = self.action_dit.video_back_projs[i](dit_state.x_video_proj)
            video_hidden = video_hidden + x_video_new

            dit_state.bridge_block_counter = i + 1
        else:
            # cross_attn / cross_attn_detach path — collect bridge features
            action_state.extra["bridge_features"].append(video_hidden)

        return video_hidden, action_state

    def extract_action_prediction(self, action_state: ActionState) -> Tensor:
        if self.action_dit is None:
            raise RuntimeError("ActionDiT not initialized")

        if "dit_state" in action_state.extra:
            dit_state = action_state.extra["dit_state"]
            return self.action_dit.finalize_action_output(dit_state)
        else:
            bridge_features = action_state.extra["bridge_features"]
            return self.action_dit(
                action_tokens=action_state.action_latents,
                video_features=bridge_features,
                timestep=action_state.timestep,
            )

    @property
    def action_dim(self) -> int:
        return self.action_dit.action_dim if self.action_dit else 0

    @property
    def bridge_layers(self) -> tuple:
        return self.action_dit.bridge_layers if self.action_dit else ()

    @property
    def is_interleaved(self) -> bool:
        return self.action_dit is not None and self.action_dit.bridge_type == "joint_self_attn"

    @property
    def action_mean(self) -> Tensor:
        if self.action_dit is not None:
            return self.action_dit.action_mean
        return torch.zeros(self.action_dim)

    @property
    def action_std(self) -> Tensor:
        if self.action_dit is not None:
            return self.action_dit.action_std
        return torch.ones(self.action_dim)
