"""MoE Action Expert WAM Architecture.

Inspired by BAGEL's Mixture-of-Transformer-Experts (MoT) pattern:
action tokens are concatenated to the video token sequence and processed
by the same video DiT blocks (shared self-attention), with expert FFN
layers providing modality-specific capacity at designated layers.

Architecture comparison:
- DualSystem (cross_attn): Separate ActionDiT with cross-attention bridge
- DualSystem (joint_self_attn): Separate ActionDiT with MMDiT-style bridge
- MoEExpert (this): Action tokens IN the video sequence, expert FFN correction

Key references:
- BAGEL (ByteDance Seed, arXiv:2505.14683): Shared attention + modality-
  specific expert FFN with deterministic routing by token type
- DreamZero: Shared backbone WAM (action+video in same DiT forward pass)

Status: Fully implemented. Requires MoE-aware pipeline support in
model_fn_wan_video (moe_expert_state parameter).
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor

from open_wam.models.architectures.base import ActionState, BaseWAMArchitecture
from open_wam.models.architectures.registry import register_architecture

# Import MoEExpertDiT from third_party
_THIRD_PARTY = str(Path(__file__).resolve().parent.parent.parent.parent / "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from diffsynth.models.moe_action_expert import MoEExpertDiT, MoEExpertState  # noqa: E402


@register_architecture("moe_expert")
class MoEActionExpertArchitecture(BaseWAMArchitecture):
    """MoE Action Expert: shared attention + expert FFN within video DiT.

    Action tokens are projected to the video DiT's hidden dimension and
    concatenated to the video token sequence. The video DiT processes
    the combined sequence with its standard self-attention and FFN blocks.
    At designated expert layers, an additional expert FFN applies a
    correction specifically to the action tokens.

    This design achieves:
    - Cross-modal grounding via shared self-attention (action tokens
      directly attend to all video tokens using the video DiT's
      learned attention weights)
    - Modality-specific capacity via expert FFN (action tokens get
      specialized processing beyond the standard video FFN)
    - Minimal new parameters (only input/output projections + expert FFNs)
    - Non-destructive initialization (expert FFN output is zero-initialized)

    Integration with model_fn_wan_video:
    - Before block loop: action tokens concatenated to video sequence,
      RoPE freqs extended with identity rotation for action positions
    - During block loop: at expert layers, action portion extracted,
      expert FFN applied, put back
    - After block loop: action tokens extracted, output head applied

    Args:
        cfg: Dict or DictConfig with MoEExpertDiT parameters:
            action_dim, video_dim, expert_ffn_dim, num_experts,
            expert_layers, etc.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is not None:
            el = cfg.get("expert_layers", (3, 7, 11, 15, 19, 23, 26, 29))
            if isinstance(el, str):
                el = tuple(int(x) for x in el.split(","))
            elif not isinstance(el, tuple):
                el = tuple(el)

            self.moe_dit = MoEExpertDiT(
                action_dim=int(cfg.get("action_dim", 14)),
                video_dim=int(cfg.get("video_dim", 1536)),
                expert_ffn_dim=int(cfg.get("expert_ffn_dim", 4096)),
                num_experts=int(cfg.get("num_experts", len(el))),
                expert_layers=el,
            )
        else:
            self.moe_dit = None

    def prepare_action_tokens(
        self, noisy_actions: Tensor, timestep: Tensor, **kwargs
    ) -> ActionState:
        state = ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
        )
        if self.moe_dit is not None:
            moe_state = self.moe_dit.prepare_state(
                noisy_actions, timestep,
                use_gradient_checkpointing=kwargs.get(
                    "use_gradient_checkpointing", False
                ),
                use_gradient_checkpointing_offload=kwargs.get(
                    "use_gradient_checkpointing_offload", False
                ),
            )
            state.extra["moe_state"] = moe_state
        return state

    def on_dit_block(
        self,
        block_id: int,
        video_hidden: Tensor,
        action_state: ActionState,
    ) -> Tuple[Tensor, ActionState]:
        """Apply expert FFN at designated layers.

        Note: In the MoE architecture, the primary interaction (shared
        attention) happens inside the video DiT block because action
        tokens are concatenated to the video sequence. This hook handles
        the expert FFN correction AFTER the block has processed the
        combined sequence.

        The actual sequence concatenation/extraction is managed by
        model_fn_wan_video when it detects moe_expert_state.
        """
        if self.moe_dit is None:
            return video_hidden, action_state

        if block_id not in self.moe_dit.expert_layers_set:
            return video_hidden, action_state

        moe_state = action_state.extra.get("moe_state")
        if moe_state is None:
            return video_hidden, action_state

        # Extract action tokens from the combined sequence
        n_action = moe_state.n_action_tokens
        n_video = video_hidden.shape[1] - n_action
        x_action = video_hidden[:, n_video:, :]

        # Apply expert FFN correction
        x_action = self.moe_dit.apply_expert(moe_state, x_action)

        # Put corrected action tokens back
        video_hidden = torch.cat(
            [video_hidden[:, :n_video, :], x_action], dim=1
        )

        # Update action tokens in state (for finalize)
        moe_state.action_tokens = x_action

        return video_hidden, action_state

    def extract_action_prediction(self, action_state: ActionState) -> Tensor:
        if self.moe_dit is None:
            raise RuntimeError("MoEExpertDiT not initialized")

        moe_state = action_state.extra.get("moe_state")
        if moe_state is None:
            raise RuntimeError("MoE state not found — was prepare_action_tokens called?")

        return self.moe_dit.finalize_output(moe_state)

    @property
    def action_dim(self) -> int:
        return self.moe_dit.action_dim if self.moe_dit else 0

    @property
    def bridge_layers(self) -> tuple:
        return self.moe_dit.expert_layers if self.moe_dit else ()
