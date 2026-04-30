"""Shared-backbone vanilla action backbone.

Used by ``SharedBackboneVanillaArchitecture``: action tokens are projected
into video_dim, concatenated to the video token sequence, ride through every
video DiT block, then sliced off and projected back to ``action_dim`` via a
small MLP. There is no separate action transformer — the shared video DiT
itself learns the modality boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from openwam.model.action_backbone.backbone import ActionBackbone
from openwam.model.action_backbone.components import ActionEncoder, ActionOutputMLP, LearnedPositionalEncoding

if TYPE_CHECKING:
    from openwam.model.base import ActionState, ExecutionPlan
    from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone


@dataclass
class SharedVanillaState:
    """Mutable per-forward state for SharedVanillaActionBackbone.

    Tracks the action tokens projected into video_dim and the raw timestep
    at loss-time granularity (preserved for the video DiT's per-token
    AdaLN, ``_build_action_t_mod`` in ``wan_adapter.py``).
    """

    action_tokens: torch.Tensor
    n_action_tokens: int = 0
    timestep: Optional[torch.Tensor] = None
    action_noise_pred: Optional[torch.Tensor] = field(default=None)


class SharedVanillaActionBackbone(ActionBackbone):
    """Action stream for SharedBackbone vanilla.

    Owns:
      - ``input_proj``: action_dim -> video_dim (fuses timestep)
      - ``pos_encoding``: learned positional encoding
      - ``action_output_head``: video_dim -> 64 -> action_dim
      - ``modality_tmod_bias``: per-modality bias added to the video DiT's
        AdaLN modulation signal for action tokens
      - ``action_mean`` / ``action_std`` persistent buffers for denormalization
    """

    def __init__(self, action_dim: int, video_dim: int, max_action_len: int = 512):
        super().__init__()
        self._action_dim = int(action_dim)
        self._video_dim = int(video_dim)

        self.input_proj = ActionEncoder(self._action_dim, self._video_dim)
        self.pos_encoding = LearnedPositionalEncoding(max_action_len, self._video_dim)
        self.action_output_head = ActionOutputMLP(self._video_dim, 64, self._action_dim)
        self.modality_tmod_bias = nn.Parameter(torch.zeros(1, 1, 6, self._video_dim))
        self.register_buffer("action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(self._action_dim), persistent=True)

    # === ActionBackbone interface ===

    @property
    def execution_plan(self) -> "ExecutionPlan":
        from openwam.model.base import ExecutionPlan

        return ExecutionPlan.INTERLEAVED_WHOLE_BLOCK

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def bridge_layers(self) -> Tuple[int, ...]:
        return ()

    def prepare_state(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        proprio_state: Optional[torch.Tensor] = None,  # noqa: ARG002 — vanilla doesn't consume proprio
        use_gradient_checkpointing: bool = False,  # noqa: ARG002
        use_gradient_checkpointing_offload: bool = False,  # noqa: ARG002
    ) -> "ActionState":
        from openwam.model.base import ActionState, ExecutionPlan, RuntimeState

        B, T, _ = noisy_actions.shape
        assert T <= self.pos_encoding.embedding.shape[1], (
            f"Action sequence length {T} exceeds max_action_len {self.pos_encoding.embedding.shape[1]}"
        )

        x = self.input_proj(noisy_actions, timestep)
        x = self.pos_encoding(x)

        timestep_flat = timestep.flatten()
        if timestep.numel() == 1:
            timestep_flat = timestep_flat.expand(B)
        elif timestep_flat.shape[0] != B:
            timestep_flat = timestep.view(B, -1)[:, 0]

        astate = ActionState(
            action_latents=x,
            timestep=timestep_flat,
            num_action_tokens=T,
        )
        astate.runtime_state = RuntimeState(
            framework="shared_backbone",
            variant="vanilla",
            execution_plan=ExecutionPlan.INTERLEAVED_WHOLE_BLOCK,
            payload=SharedVanillaState(
                action_tokens=x,
                n_action_tokens=T,
                timestep=timestep,
            ),
        )
        return astate

    def before_loop(
        self,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        payload: SharedVanillaState = astate.runtime_state.payload
        vstate = vb.inject_action_tokens(
            vstate,
            payload.action_tokens,
            astate.num_action_tokens,
            timestep=payload.timestep,
            t_mod_bias=self.modality_tmod_bias,
        )
        return vstate, astate

    def run_block(
        self,
        block_id: int,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        # Action tokens travel inside the shared video sequence — no per-block
        # action-side work; just run the video block.
        vstate = vb.run_block(block_id, vstate)
        return vstate, astate

    def after_loop(
        self,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        # Mirror the original vanilla forward: capture full (video+action) hidden
        # before extracting, then strip the action tail off the video sequence.
        astate.final_hidden = vstate.x
        vstate, _ = vb.extract_action_tokens(vstate, astate.num_action_tokens)
        return vstate, astate

    def extract_prediction(self, astate: "ActionState") -> torch.Tensor:
        n = astate.num_action_tokens
        if n is None:
            raise RuntimeError("SharedVanillaActionBackbone: num_action_tokens not set")
        x = astate.final_hidden[:, -n:, :] if astate.final_hidden is not None else astate.action_latents
        return self.action_output_head(x)


__all__ = ["SharedVanillaActionBackbone"]
