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
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward
from openwam.utils import resolve_bridge_layers


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
        num_layers = vb.num_layers if vb is not None else int(cfg.get("num_dit_layers", 30))

        # Accept ``expert_layers`` as the canonical cfg key for MoE, with
        # ``bridge_layers`` kept as an alias for back-compat. resolve_bridge_layers
        # only knows the latter, so route the alias through it.
        cfg_for_resolve = cfg
        expert_layers_raw = cfg.get("expert_layers") if isinstance(cfg, dict) else getattr(cfg, "expert_layers", None)
        if expert_layers_raw is not None:
            cfg_for_resolve = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg_for_resolve["bridge_layers"] = expert_layers_raw
        bl = resolve_bridge_layers(cfg_for_resolve, num_layers=num_layers)

        self.action_backbone = SharedMoEActionBackbone(
            action_dim=int(cfg.get("action_dim", 20)),
            video_dim=video_dim,
            expert_ffn_dim=int(cfg.get("expert_ffn_dim", 14336)),
            expert_layers=tuple(int(i) for i in bl),
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


__all__ = ["SharedBackboneMoEArchitecture"]
