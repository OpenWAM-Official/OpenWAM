"""SingleSystem MoE architecture.

Action tokens are concatenated to the video token sequence and ride
through the shared video DiT blocks. At the configured ``bridge_layers``
the action tokens receive an extra expert FFN correction for
modality-specific capacity.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.shared_action_backbone import SharedMoEActionBackbone
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.single_system.state import (
    align_state_tokens_to_action_batch,
    attach_shared_attention_mask,
)
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    set_video_attention_mask_mode,
    validate_attention_mask_mode,
)
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward


@register_architecture(
    "single_system_moe",
    status="supported",
    note="SingleSystem MoE: action tokens share the video DiT with expert FFN at configured layers.",
    framework="single_system",
    variant="moe",
)
class SingleSystemMoEArchitecture(BaseWAMArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        vb = self.video_backbone
        video_dim = self._resolve_video_dim(cfg)
        num_layers = vb.num_layers if vb is not None else None
        action_decoder_hidden_dim = cfg.get("action_decoder_hidden_dim")
        use_proprioception = bool(cfg.get("use_proprioception", False))
        state_dim = int(cfg.get("state_dim", 0) or 0)
        self.attention_mask_mode = validate_attention_mask_mode(str(cfg.get("attention_mask_mode", ACTION_SEES_VIDEO)))
        self.video_attention_mask_mode = str(cfg.get("video_attention_mask_mode", "first_frame_causal"))
        if vb is not None:
            set_video_attention_mask_mode(vb, self.video_attention_mask_mode)

        bridge_layers = resolve_bridge_layers(cfg, num_layers=num_layers)

        self.action_backbone = SharedMoEActionBackbone(
            action_dim=int(cfg.get("action_dim", 20)),
            video_dim=video_dim,
            expert_ffn_dim=int(cfg.get("expert_ffn_dim", 4096)),
            bridge_layers=bridge_layers,
            action_decoder_hidden_dim=action_decoder_hidden_dim,
            use_proprioception=use_proprioception,
            state_dim=state_dim,
        )

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
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
        invalid_bridge_layers = [layer for layer in ab.bridge_layers if layer >= vb.num_layers]
        if invalid_bridge_layers:
            raise ValueError(
                f"bridge_layers {invalid_bridge_layers} exceed video_backbone.num_layers={vb.num_layers}. "
                "SingleSystem MoE expert layers must match the actual video backbone depth."
            )
        set_video_attention_mask_mode(vb, getattr(self, "video_attention_mask_mode", None))

        # SingleSystem needs per-token (4D) t_mod so action/state tokens can
        # extend it cleanly via inject_shared_tokens. TI2V-5B produces 4D
        # natively (seperated_timestep + fuse_vae_embedding_in_latents); other
        # Wan backbones broadcast a global timestep when this flag is set.
        # ``zero_clean_prefix_t_mod`` is load-bearing only for TI2V — VACE
        # and I2V do not emit ``first_frame_latents`` (VACE routes via
        # ``vace_context``; I2V via the ``y`` channel) so the flag is
        # structurally inert there. Kept on for symmetry with the rest of
        # the architecture suite.
        # setdefault so callers may still pass False explicitly.
        pipeline_inputs = dict(pipeline_inputs)
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_actions is None or ab is None:
            action_tokens = None
            t_mod = None
        else:
            action_tokens, t_mod = ab.encode(noisy_actions, action_timestep)
        state_tokens = None if ab is None else ab.encode_state(proprio)
        if action_tokens is not None:
            state_tokens = align_state_tokens_to_action_batch(state_tokens, action_tokens.shape[0])
        elif state_tokens is not None and state_tokens.shape[0] == 1 and vstate.hidden_states.shape[0] > 1:
            state_tokens = state_tokens.expand(vstate.hidden_states.shape[0], -1, -1)

        n_action = 0 if action_tokens is None else action_tokens.shape[1]
        n_state = 0 if state_tokens is None else state_tokens.shape[1]
        has_shared_tokens = n_action + n_state > 0
        if has_shared_tokens:
            vb.assert_ready_for_shared_tokens(vstate)
            shared_timestep = action_timestep if action_timestep is not None else pipeline_inputs.get("timestep")
            if shared_timestep is None:
                raise ValueError(
                    "SingleSystem state-token conditioning requires `action_timestep` or video `timestep`."
                )
            vstate = vb.inject_shared_tokens(
                vstate,
                action_tokens,
                n_action,
                state_tokens=state_tokens,
                n_state=n_state,
                timestep=shared_timestep,
            )
            attach_shared_attention_mask(
                vb,
                vstate,
                n_action,
                n_state=n_state,
                attention_mask_mode=getattr(self, "attention_mask_mode", ACTION_SEES_VIDEO),
            )

        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
            if n_action and block_id in ab.bridge_layers:
                n_video = vstate.hidden_states.shape[1] - n_action - n_state
                x_action = gradient_checkpoint_forward(
                    lambda x, t, _bid=block_id: ab.apply_expert(_bid, x, t),
                    use_gradient_checkpointing and self.training,
                    use_gradient_checkpointing_offload,
                    vstate.hidden_states[:, n_video : n_video + n_action, :],
                    t_mod,
                )
                if n_state:
                    vstate.hidden_states = torch.cat(
                        [
                            vstate.hidden_states[:, :n_video, :],
                            x_action,
                            vstate.hidden_states[:, n_video + n_action :, :],
                        ],
                        dim=1,
                    )
                else:
                    vstate.hidden_states = torch.cat([vstate.hidden_states[:, :n_video, :], x_action], dim=1)

        if n_action == 0:
            if n_state:
                vstate, _ = vb.extract_shared_tokens(vstate, n_action, n_state=n_state)
            return vb.finalize(vstate), None

        vstate, action_tail = vb.extract_shared_tokens(vstate, n_action, n_state=n_state)
        return vb.finalize(vstate), ab.decode(action_tail)


__all__ = ["SingleSystemMoEArchitecture"]
