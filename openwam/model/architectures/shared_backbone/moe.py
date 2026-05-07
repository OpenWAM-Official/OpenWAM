"""SharedBackbone MoE architecture.

Action tokens are concatenated to the video token sequence and ride
through the shared video DiT blocks. At configured ``expert_layers``
the action tokens receive an extra expert FFN correction for
modality-specific capacity.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.shared_moe import SharedMoEActionBackbone
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.shared_backbone.mask import (
    attach_shared_attention_mask,
    set_video_attention_mask_mode,
    validate_shared_attention_mask_mode,
)
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward


def _cfg_get(cfg, key: str, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def _cfg_has(cfg, key: str) -> bool:
    return key in cfg if isinstance(cfg, dict) else hasattr(cfg, key)


def resolve_expert_layers(cfg, *, num_layers: Optional[int]) -> tuple[int, ...]:
    """Resolve MoE expert layer ids from ``expert_layers`` or ``expert_interval``."""
    if _cfg_has(cfg, "bridge_layers") and not _cfg_has(cfg, "expert_layers"):
        raise ValueError(
            "SharedBackbone MoE renamed 'bridge_layers' to 'expert_layers'. "
            "Update the config key to avoid silently changing the expert topology."
        )
    if _cfg_has(cfg, "bridge_interval") and not _cfg_has(cfg, "expert_interval"):
        raise ValueError(
            "SharedBackbone MoE renamed 'bridge_interval' to 'expert_interval'. "
            "Update the config key to avoid silently changing the expert topology."
        )

    layers_raw = _cfg_get(cfg, "expert_layers", None)
    if layers_raw is None:
        interval_raw = _cfg_get(cfg, "expert_interval", None)
        if interval_raw is None:
            raise ValueError("expert_layers is null but expert_interval is not set")
        if num_layers is None:
            raise ValueError(
                "expert_layers is null but video_backbone.num_layers is unavailable. "
                "Build SharedBackbone MoE with a video_backbone so expert_interval can be resolved "
                "from the actual backbone depth."
            )
        interval = int(interval_raw)
        if interval < 1:
            raise ValueError(f"expert_interval must be >= 1, got {interval}")
        layers = tuple(range(0, int(num_layers), interval))
    elif isinstance(layers_raw, str):
        layers = tuple(int(x) for x in layers_raw.split(",") if x)
    else:
        layers = tuple(int(x) for x in layers_raw)

    layers = tuple(sorted(layers))
    if len(set(layers)) != len(layers):
        raise ValueError(f"expert_layers must be unique, got {layers}")
    invalid = [layer for layer in layers if layer < 0]
    if num_layers is not None:
        invalid.extend(layer for layer in layers if layer >= int(num_layers))
    if invalid:
        if num_layers is None:
            raise ValueError(f"expert_layers must be non-negative, got invalid layers {invalid}")
        raise ValueError(f"expert_layers must be in [0, {int(num_layers) - 1}], got invalid layers {invalid}")
    return layers


@register_architecture(
    "shared_backbone_moe",
    status="supported",
    note="SharedBackbone MoE: action tokens share the video DiT with expert FFN at configured layers.",
    framework="shared_backbone",
    variant="moe",
)
class SharedBackboneMoEArchitecture(BaseWAMArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        if bool(cfg.get("use_proprioception", False)):
            raise NotImplementedError(
                "SharedBackbone (MoE) does not support proprioception. "
                "Use framework=dual_system or set use_proprioception=False."
            )
        vb = self.video_backbone
        video_dim = self._resolve_video_dim(cfg)
        num_layers = vb.num_layers if vb is not None else None
        action_decoder_hidden_dim = cfg.get("action_decoder_hidden_dim")
        self.attention_mask_mode = validate_shared_attention_mask_mode(str(cfg.get("attention_mask_mode", "joint")))
        self.video_attention_mask_mode = str(cfg.get("video_attention_mask_mode", "first_frame_causal"))
        if vb is not None:
            set_video_attention_mask_mode(vb, self.video_attention_mask_mode)

        expert_layers = resolve_expert_layers(cfg, num_layers=num_layers)

        self.action_backbone = SharedMoEActionBackbone(
            action_dim=int(cfg.get("action_dim", 20)),
            video_dim=video_dim,
            expert_ffn_dim=int(cfg.get("expert_ffn_dim", 4096)),
            expert_layers=expert_layers,
            action_decoder_hidden_dim=action_decoder_hidden_dim,
        )

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,  # noqa: ARG002 — MoE doesn't consume proprio
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )
        invalid_expert_layers = [layer for layer in ab.expert_layers if layer >= vb.num_layers]
        if invalid_expert_layers:
            raise ValueError(
                f"expert_layers {invalid_expert_layers} exceed video_backbone.num_layers={vb.num_layers}. "
                "SharedBackbone MoE expert layers must match the actual video backbone depth."
            )
        set_video_attention_mask_mode(vb, getattr(self, "video_attention_mask_mode", None))

        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None

        # SharedBackbone relies on per-token t_mod so action tokens get their own
        # AdaLN (action timestep + modality_tmod_bias) at every video DiT block.
        # Wan2.2-TI2V-5B with fuse_vae_embedding_in_latents=True satisfies this.
        if vstate.t_mod.dim() != 4:
            raise RuntimeError(
                "SharedBackbone requires the video backbone to run in per-token t_mod mode "
                "(e.g. dit.seperated_timestep=True with fuse_vae_embedding_in_latents=True). "
                f"Got vstate.t_mod with dim={vstate.t_mod.dim()}; modality_tmod_bias and "
                "action timestep would be silently ignored otherwise."
            )

        action_tokens, t_mod = ab.encode(noisy_actions, action_timestep)
        n_action = action_tokens.shape[1]
        vstate = vb.inject_action_tokens(
            vstate,
            action_tokens,
            n_action,
            timestep=action_timestep,
            t_mod_bias=ab.modality_tmod_bias,
        )
        attach_shared_attention_mask(
            vb,
            vstate,
            n_action,
            attention_mask_mode=getattr(self, "attention_mask_mode", "joint"),
        )

        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
            if block_id in ab.expert_layers_set:
                n_video = vstate.x.shape[1] - n_action
                x_action = gradient_checkpoint_forward(
                    lambda x, t, _bid=block_id: ab.apply_expert(_bid, x, t),
                    use_gradient_checkpointing and self.training,
                    use_gradient_checkpointing_offload,
                    vstate.x[:, n_video:, :],
                    t_mod,
                )
                vstate.x = torch.cat([vstate.x[:, :n_video, :], x_action], dim=1)

        vstate, action_tail = vb.extract_action_tokens(vstate, n_action)
        return vb.finalize(vstate), ab.decode(action_tail)


__all__ = ["SharedBackboneMoEArchitecture", "resolve_expert_layers"]
