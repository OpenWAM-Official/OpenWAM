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
from torch import Tensor, nn

from openwam.model.base import ActionState, BaseWAMArchitecture
from openwam.model.registry import register_architecture


@dataclass
class SharedBackboneState:
    """State passed to model_fn_wan_video for SharedBackbone token concat/extract."""

    action_tokens: torch.Tensor  # (B, T_action, video_dim) projected
    n_action_tokens: int = 0
    action_noise_pred: Optional[torch.Tensor] = None  # filled after finalization
    # Back-reference to architecture for finalize
    _architecture: Optional["SharedBackboneArchitecture"] = field(default=None, repr=False)
    _action_state: Optional[ActionState] = field(default=None, repr=False)


def _sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    """Sinusoidal timestep embedding matching ActionDiT/MoEExpertDiT."""
    half = dim // 2
    freq = torch.exp(
        -torch.arange(half, device=position.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half)
    )
    args = position.float().unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


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
        self._num_action_tokens = int(cfg.get("num_action_tokens", 49))
        self._freq_dim = int(cfg.get("freq_dim", 256))
        max_action_len = int(cfg.get("max_action_len", 512))

        # Input projection: action_dim -> video_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self._action_dim, self._video_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self._video_dim, self._video_dim),
        )

        # Learned positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_action_len, self._video_dim) * 0.02)

        # Timestep embedding (independent from video timestep)
        self.time_embedding = nn.Sequential(
            nn.Linear(self._freq_dim, self._video_dim),
            nn.SiLU(),
            nn.Linear(self._video_dim, self._video_dim),
        )

        # Output modulation: shift + scale (2 params)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self._video_dim, self._video_dim * 2),
        )

        # Output head: video_dim -> action_dim
        self.output_norm = nn.LayerNorm(self._video_dim, eps=1e-6, elementwise_affine=False)
        self.output_head = nn.Linear(self._video_dim, self._action_dim)

        # Zero-initialize output for stable training start
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # Action normalization stats (persistent buffers)
        # Use _norm_ prefix to avoid conflict with base class properties
        self.register_buffer("_norm_action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("_norm_action_std", torch.ones(self._action_dim), persistent=True)

    def prepare_action_tokens(self, noisy_actions: Tensor, timestep: Tensor, **kwargs) -> ActionState:
        """Project noisy actions to video_dim and prepare for concatenation.

        Args:
            noisy_actions: (B, T_action, action_dim) noisy action trajectory.
            timestep: (B,) diffusion timestep for the action stream.

        Returns:
            ActionState with projected action tokens stored in
            ``action_latents`` and metadata in ``extra``.
        """
        B, T, _ = noisy_actions.shape
        assert T <= self.pos_embedding.shape[1], (
            f"Action sequence length {T} exceeds max_action_len {self.pos_embedding.shape[1]}"
        )

        # Project to video_dim and add positional encoding
        x = self.input_proj(noisy_actions)
        x = x + self.pos_embedding[:, :T, :]

        # Timestep embedding for output modulation
        timestep = timestep.flatten()
        t = self.time_embedding(_sinusoidal_embedding_1d(self._freq_dim, timestep))
        t_mod = self.time_projection(t)  # (B, video_dim * 2)

        action_state = ActionState(
            action_latents=x,  # (B, T_action, video_dim)
            timestep=timestep,
            extra={
                "num_action_tokens": T,
                "t_embed": t,
                "t_mod": t_mod,
            },
        )
        # State for model_fn_wan_video to handle concatenation/extraction
        sb_state = SharedBackboneState(
            action_tokens=x,
            n_action_tokens=T,
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

        # AdaLN modulation
        t_mod = action_state.extra["t_mod"]  # (B, video_dim * 2)
        shift, scale = t_mod.chunk(2, dim=-1)  # each (B, video_dim)
        shift = shift.unsqueeze(1)  # (B, 1, video_dim)
        scale = scale.unsqueeze(1)

        x = self.output_norm(x) * (1 + scale) + shift
        return self.output_head(x)

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
