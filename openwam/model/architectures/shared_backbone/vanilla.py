"""SharedBackbone Vanilla architecture.

Action tokens are concatenated to the video token sequence and ride
through every video DiT block as part of the shared sequence. Unlike
the MoE variant there are no expert FFN corrections — the raw shared
backbone learns the modality boundary itself. Output is taken from
the trailing action segment of the final hidden state via a small MLP.
"""

from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor

from openwam.model.action_backbone.shared_vanilla import SharedVanillaActionBackbone
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture


@register_architecture(
    "shared_backbone_vanilla",
    status="supported",
    note="SharedBackbone vanilla: action tokens share the video DiT with no expert FFN.",
    framework="shared_backbone",
    variant="vanilla",  # Also serves as default for framework="shared_backbone" when variant is unset
)
class SharedBackboneVanillaArchitecture(BaseWAMArchitecture):
    """SharedBackbone vanilla: video DiT processes both video and action tokens."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        if bool(cfg.get("use_proprioception", False)):
            raise NotImplementedError(
                "SharedBackbone (vanilla) does not support proprioception. "
                "Use framework=dual_system or set use_proprioception=False."
            )
        action_dim = int(cfg.get("action_dim", 20))
        video_dim = self._resolve_video_dim(cfg)
        max_action_len = int(cfg.get("max_action_len", 512))
        self.action_backbone = SharedVanillaActionBackbone(
            action_dim=action_dim,
            video_dim=video_dim,
            max_action_len=max_action_len,
        )

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,  # noqa: ARG002 — vanilla doesn't consume proprio
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

        action_tokens = ab.encode(noisy_actions, action_timestep)
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

        vstate, action_tail = vb.extract_action_tokens(vstate, n_action)
        return vb.finalize(vstate), ab.decode(action_tail)


__all__ = ["SharedBackboneVanillaArchitecture"]
