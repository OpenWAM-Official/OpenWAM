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

from openwam.model.action_model.action_dit import ActionDiT
from openwam.model.base import ActionState, BaseWAMArchitecture
from openwam.model.registry import register_architecture


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
            # bridge_layers: explicit list/tuple/comma-string. When null/missing,
            # fall back to `bridge_interval` + `num_dit_layers` (range(0, N, step)),
            # or the legacy 8-layer default when neither is provided.
            bl_raw = cfg.get("bridge_layers", None)
            if bl_raw is None:
                if "bridge_interval" in cfg:
                    num_dit_layers = int(cfg.get("num_dit_layers", 30))
                    interval = int(cfg["bridge_interval"])
                    assert interval >= 1, f"bridge_interval must be >= 1, got {interval}"
                    bl = tuple(range(0, num_dit_layers, interval))
                else:
                    bl = (3, 7, 11, 15, 19, 23, 26, 29)
            elif isinstance(bl_raw, str):
                bl = tuple(int(x) for x in bl_raw.split(","))
            elif isinstance(bl_raw, tuple):
                bl = bl_raw
            else:
                bl = tuple(bl_raw)

            # Enforce ascending order for bridge_layers to match model_fn_wan_video iteration order.
            # The video DiT iterates blocks 0..N-1 and appends to bridge_features; sorting here
            # guarantees bridge_features[i] matches the i-th sorted layer regardless of config order.
            bl = tuple(sorted(bl))
            assert len(set(bl)) == len(bl), f"bridge_layers must be unique, got {bl}"

            self.action_dit = ActionDiT(
                action_dim=int(cfg.get("action_dim", 14)),
                dim=int(cfg.get("dim", 768)),
                ffn_dim=int(cfg.get("ffn_dim", 3072)),
                num_heads=int(cfg.get("num_heads", 12)),
                num_layers=len(bl),  # auto-derived from bridge_layers
                video_dim=int(cfg.get("video_dim", 1536)),
                bridge_layers=bl,
                bridge_type=cfg.get("bridge_type", "cross_attn_detach"),
                use_proprioception=bool(cfg.get("use_proprioception", False)),
                state_dim=int(cfg.get("state_dim") or 0),  # null/0/missing → auto = action_dim
                proprio_fusion=cfg.get("proprio_fusion", "channel_concat"),
                num_state_tokens=int(cfg.get("num_state_tokens", 4)),
            )
        else:
            self.action_dit = None

    def prepare_action_tokens(self, noisy_actions: Tensor, timestep: Tensor, **kwargs) -> ActionState:
        proprio_state = kwargs.get("proprio_state", None)
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
                proprio_state=proprio_state,
            )
            state.extra["dit_state"] = dit_state
        else:
            state.extra["bridge_features"] = []
            state.extra["bridge_block_counter"] = 0
            # Stash for extract_action_prediction (cross_attn path runs the
            # ActionDiT forward only after the video DiT loop finishes).
            if proprio_state is not None:
                state.extra["proprio_state"] = proprio_state
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
            if dit_state.use_joint_rope:
                freqs_action = self.action_dit._get_rope_freqs(dit_state.x_action.shape[1], label="Action")
                freqs_video = self.action_dit._get_rope_freqs(dit_state.x_video_proj.shape[1], label="Video")
                dit_state.x_action, dit_state.x_video_proj = block(
                    dit_state.x_action,
                    dit_state.x_video_proj,
                    dit_state.t_mod,
                    freqs_action=freqs_action,
                    freqs_video=freqs_video,
                )
            else:
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
                proprio_state=action_state.extra.get("proprio_state", None),
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
    def uses_proprioception(self) -> bool:
        """Whether the architecture expects a ``proprio_state`` input."""
        return self.action_dit is not None and self.action_dit.proprio_encoder is not None

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
