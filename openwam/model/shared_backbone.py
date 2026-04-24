"""Shared Backbone WAM Architecture.

Action tokens are directly concatenated to the video token sequence and
processed by the same video DiT transformer. No separate action model exists.
The DiT learns to jointly attend to video and action tokens.

Corresponds to the "Shared Backbone" diagram in assets/arch.png.

Design:
    1. ``prepare_action_tokens``: project noisy actions from ``action_dim``
       to ``video_dim``, add learned positional embeddings, and store the
       projected tokens in :class:`ActionState`.
    2. ``on_dit_block``: no-op — action tokens are part of the video
       sequence and processed by the same DiT blocks.
    3. ``extract_action_prediction``: slice action tokens from the end of
       the combined sequence, apply AdaLN modulation, and project back to
       ``action_dim``.

References:
    - DreamZero: shared backbone WAM with action+video in same DiT
    - Cosmos Policy: action tokens in video diffusion sequence
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.action_model.components import (
    ActionEncoder,
    ActionOutputMLP,
    LearnedPositionalEncoding,
    TimestepEmbedding,
    TimestepModulation,
)
from openwam.model.base import ActionState, BaseWAMArchitecture
from openwam.model.registry import register_architecture


@dataclass
class SharedBackboneState:
    """State passed to model_fn_wan_video for SharedBackbone token concat/extract.

    Fields:
        action_tokens: (B, T_action, video_dim) projected action tokens.
        n_action_tokens: Number of action tokens appended to the video sequence.
        timestep: Raw action diffusion timestep at loss-time granularity —
            (1,) broadcast, (B,) per-sample, or (B, T_action) per-token.
            Consumed by model_fn_wan_video._build_action_t_mod (which
            dispatches on ndim) when the video DiT uses per-token t_mod
            (Wan2.2-TI2V-5B fuse_vae_embedding_in_latents path).
        action_noise_pred: Filled after finalization.
    """

    action_tokens: torch.Tensor  # (B, T_action, video_dim) projected
    n_action_tokens: int = 0
    timestep: Optional[torch.Tensor] = None
    action_noise_pred: Optional[torch.Tensor] = None  # filled after finalization
    # Back-reference to architecture for finalize
    _architecture: Optional["SharedBackboneArchitecture"] = field(default=None, repr=False)
    _action_state: Optional[ActionState] = field(default=None, repr=False)


@register_architecture(
    "shared_backbone",
    status="supported",
    note="Shared backbone — action tokens processed by the video DiT directly.",
)
class SharedBackboneArchitecture(BaseWAMArchitecture):
    """Shared Backbone: video DiT processes both video and action tokens.

    Action tokens are appended to the video latent sequence before the DiT
    block loop.  The transformer jointly attends to all tokens.  After the
    loop, action tokens are extracted and projected to action predictions.

    This fully reuses the video generation weights for action prediction,
    maximizing knowledge transfer from the pretrained video model.

    Args:
        cfg: Configuration with keys:
            action_dim: Action vector dimension (default 14).
            video_dim: Hidden dimension of the video DiT (default 1536).
            num_action_tokens: Number of action tokens to append (default 49).
            freq_dim: Sinusoidal embedding dimension (default 256).
            max_action_len: Maximum supported action sequence length (default 512).
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            cfg = {}
        self._action_dim = int(cfg.get("action_dim", 14))
        self._video_dim = int(cfg.get("video_dim", 1536))
        self._freq_dim = int(cfg.get("freq_dim", 256))
        max_action_len = int(cfg.get("max_action_len", 512))

        # Input projection fuses the diffusion timestep into the action
        # embedding inside the encoder (see ActionEncoder).
        self.input_proj = ActionEncoder(self._action_dim, self._video_dim)

        # Learned positional encoding for action sequence positions.
        self.pos_encoding = LearnedPositionalEncoding(max_action_len, self._video_dim)

        # Dead code (kept per project convention: do not delete dead code).
        # After the output head switch to ActionOutputMLP below, t_embed /
        # t_mod are no longer consumed by extract_action_prediction; the
        # modules remain so prior state_dicts still load and so DualSystem
        # parity is preserved.
        self.time_embedding = TimestepEmbedding(self._freq_dim, self._video_dim)
        self.time_projection = TimestepModulation(self._video_dim, 2)

        # Output head: 2-layer MLP video_dim -> 64 -> action_dim.
        self.action_output_head = ActionOutputMLP(self._video_dim, 64, self._action_dim)

        # Per-modality bias on the video DiT's AdaLN modulation signal (6 params
        # per dim: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        # Zero-initialized so action tokens start by using video's time_projection
        # output verbatim at the action timestep; training pulls the modalities
        # apart as needed. Excluded from weight decay via NO_WD_PARAM_SUFFIXES
        # in openwam/train/utils/optimizer_groups.py (AdamW wd would otherwise
        # pull it back to zero).
        self.modality_tmod_bias = nn.Parameter(torch.zeros(1, 1, 6, self._video_dim))

        # Action normalization stats (persistent buffers)
        # Use _norm_ prefix to avoid conflict with base class properties
        self.register_buffer("_norm_action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("_norm_action_std", torch.ones(self._action_dim), persistent=True)

    def prepare_action_tokens(self, noisy_actions: Tensor, timestep: Tensor, **kwargs) -> ActionState:
        """Project noisy actions to video_dim and prepare for concatenation.

        Args:
            noisy_actions: (B, T_action, action_dim) noisy action trajectory.
            timestep: (1,) scalar broadcast, (B,) per-sample or (B, T_action)
                per-token diffusion timestep (ActionEncoder accepts all three).

        Returns:
            ActionState with projected action tokens stored in
            ``action_latents`` and metadata in ``extra``.
        """
        B, T, _ = noisy_actions.shape
        assert T <= self.pos_encoding.embedding.shape[1], (
            f"Action sequence length {T} exceeds max_action_len {self.pos_encoding.embedding.shape[1]}"
        )

        # Project to video_dim; ActionEncoder fuses timestep internally.
        x = self.input_proj(noisy_actions, timestep)
        x = self.pos_encoding(x)

        # Collapse to per-sample timestep (for per-token inputs, take the
        # first column). Supports (1,), (B,), and (B, T) inputs.
        timestep_flat = timestep.flatten()
        if timestep.numel() == 1:
            timestep_flat = timestep_flat.expand(B)
        elif timestep_flat.shape[0] != B:
            timestep_flat = timestep.view(B, -1)[:, 0]
        t = self.time_embedding(timestep_flat)
        t_mod = self.time_projection(t)

        action_state = ActionState(
            action_latents=x,  # (B, T_action, video_dim)
            timestep=timestep_flat,
            extra={
                "num_action_tokens": T,
                "t_embed": t,    # no consumer after output head swap; kept for state-dict stability
                "t_mod": t_mod,  # no consumer after output head swap
            },
        )
        # State for model_fn_wan_video to handle concatenation/extraction.
        # Preserve the raw timestep shape — model_fn_wan_video's
        # _build_action_t_mod dispatches on ndim to build per-sample
        # (B,) or per-token (B, T) t_mod rows for action positions.
        sb_state = SharedBackboneState(
            action_tokens=x,
            n_action_tokens=T,
            timestep=timestep,
            _architecture=self,
            _action_state=action_state,
        )
        action_state.extra["shared_backbone_state"] = sb_state
        return action_state

    def on_dit_block(
        self,
        block_id: int,
        video_hidden: Tensor,
        action_state: ActionState,
    ) -> Tuple[Tensor, ActionState]:
        """No-op: action tokens are part of the video sequence.

        The caller is responsible for concatenating action tokens to the
        video sequence before the DiT loop and passing the combined
        hidden state here.  We just propagate unchanged.
        """
        return video_hidden, action_state

    def extract_action_prediction(self, action_state: ActionState) -> Tensor:
        """Slice action tokens from the combined output and project to action_dim.

        Expects ``action_state.extra["final_hidden"]`` to contain the
        combined video+action hidden state after all DiT blocks, OR
        ``action_state.action_latents`` to already hold the sliced
        action tokens (if the caller pre-sliced them).

        Returns:
            (B, T_action, action_dim) predicted action noise.
        """
        n = action_state.extra["num_action_tokens"]

        # Get action tokens: prefer final_hidden (full sequence) if available
        if "final_hidden" in action_state.extra:
            x = action_state.extra["final_hidden"][:, -n:, :]
        else:
            x = action_state.action_latents

        return self.action_output_head(x)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def bridge_layers(self) -> tuple:
        # SharedBackbone doesn't use bridge layers — action tokens are in the sequence
        return ()

    @property
    def is_interleaved(self) -> bool:
        return True  # Action tokens ARE part of the video sequence

    @property
    def action_mean(self) -> Tensor:
        return self._norm_action_mean

    @property
    def action_std(self) -> Tensor:
        return self._norm_action_std
