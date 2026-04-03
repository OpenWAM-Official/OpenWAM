"""MLP Action Head WAM Architecture.

A lightweight baseline architecture that collects video bridge features
from designated DiT layers and predicts actions via a simple MLP.  No
separate transformer or diffusion model is used for actions — the MLP
directly regresses action noise from pooled video features.

This serves as a minimal baseline for ablation studies and quick
prototyping. It follows the same bridge-collection pattern as the
DualSystem cross_attn path but replaces the ActionDiT with a two-layer
MLP.

Design:
    1. ``prepare_action_tokens``: store noisy actions and initialize an
       empty bridge feature buffer.
    2. ``on_dit_block``: at designated bridge layers, collect the video
       hidden state.
    3. ``extract_action_prediction``: concatenate bridge features, pool
       across the spatial/temporal dimension, concatenate with noisy
       actions, and run through the MLP to produce the action noise
       prediction.
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from open_wam.models.architectures.base import ActionState, BaseWAMArchitecture
from open_wam.models.architectures.registry import register_architecture


def _sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freq = torch.exp(-torch.arange(half, device=position.device, dtype=torch.float32)
                     * (torch.log(torch.tensor(10000.0)) / half))
    args = position.float().unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


@register_architecture(
    "mlp_action_head",
    status="supported",
    note="Lightweight MLP baseline for action prediction from video bridge features.",
)
class MLPActionHeadArchitecture(BaseWAMArchitecture):
    """MLP Action Head: bridge features from video DiT → MLP → actions.

    Collects video hidden states at designated bridge layers, pools them,
    and predicts action noise via a two-layer MLP.  This is the simplest
    possible architecture for video-to-action mapping.

    Args:
        cfg: Configuration with keys:
            action_dim: Action vector dimension (default 14).
            video_dim: Hidden dimension of the video DiT (default 1536).
            hidden_dim: MLP hidden dimension (default 1024).
            bridge_layers: Video DiT layers to collect features from.
            freq_dim: Sinusoidal embedding dimension (default 256).
            max_action_len: Maximum supported action sequence length (default 512).
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            cfg = {}

        self._action_dim = int(cfg.get("action_dim", 14))
        self._video_dim = int(cfg.get("video_dim", 1536))
        hidden_dim = int(cfg.get("hidden_dim", 1024))
        self._freq_dim = int(cfg.get("freq_dim", 256))
        max_action_len = int(cfg.get("max_action_len", 512))

        bl = cfg.get("bridge_layers", (3, 7, 11, 15, 19, 23, 26, 29))
        if isinstance(bl, str):
            bl = tuple(int(x.strip()) for x in bl.split(","))
        self._bridge_layers = tuple(bl)
        self._bridge_layers_set = set(self._bridge_layers)
        self._num_bridge_layers = len(self._bridge_layers)

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(self._freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Bridge feature projection: one per bridge layer → shared dim
        self.bridge_proj = nn.Linear(self._video_dim, hidden_dim)

        # Action input projection
        self.action_proj = nn.Linear(self._action_dim, hidden_dim)

        # Learned positional encoding for action tokens
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_action_len, hidden_dim) * 0.02
        )

        # MLP head: (bridge_pool + action + timestep) → action_dim
        # Input: hidden_dim * 3 (bridge pool + action + timestep)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, self._action_dim),
        )

        # Zero-initialize output for stable training start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Action normalization stats
        self.register_buffer("_norm_action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("_norm_action_std", torch.ones(self._action_dim), persistent=True)

    def prepare_action_tokens(
        self, noisy_actions: Tensor, timestep: Tensor, **kwargs
    ) -> ActionState:
        """Store noisy actions and initialize bridge feature buffer.

        Args:
            noisy_actions: (B, T_action, action_dim) noisy actions.
            timestep: (B,) diffusion timestep.

        Returns:
            ActionState with bridge_features list in extra.
        """
        return ActionState(
            action_latents=noisy_actions,
            timestep=timestep,
            extra={"bridge_features": []},
        )

    def on_dit_block(
        self,
        block_id: int,
        video_hidden: Tensor,
        action_state: ActionState,
    ) -> Tuple[Tensor, ActionState]:
        """Collect video hidden state at bridge layers.

        Args:
            block_id: Current DiT block index.
            video_hidden: (B, T_video, video_dim) hidden state.
            action_state: Current state with bridge_features list.

        Returns:
            Unchanged (video_hidden, action_state) with features appended.
        """
        if block_id in self._bridge_layers_set:
            action_state.extra["bridge_features"].append(video_hidden)
        return video_hidden, action_state

    def extract_action_prediction(self, action_state: ActionState) -> Tensor:
        """Pool bridge features and predict actions via MLP.

        Returns:
            (B, T_action, action_dim) predicted action noise.
        """
        bridge_features = action_state.extra["bridge_features"]
        noisy_actions = action_state.action_latents  # (B, T_action, action_dim)
        timestep = action_state.timestep  # (B,)
        B, T_action, _ = noisy_actions.shape

        # Pool bridge features: stack layers, mean over layers and spatial dim
        # Each feature: (B, T_video, video_dim)
        stacked = torch.stack(bridge_features, dim=1)  # (B, num_layers, T_video, video_dim)
        pooled = stacked.mean(dim=(1, 2))  # (B, video_dim)
        bridge_h = self.bridge_proj(pooled)  # (B, hidden_dim)

        # Project noisy actions
        action_h = self.action_proj(noisy_actions)  # (B, T_action, hidden_dim)
        action_h = action_h + self.pos_embedding[:, :T_action, :]

        # Timestep embedding
        timestep = timestep.flatten()
        t_h = self.time_embedding(
            _sinusoidal_embedding_1d(self._freq_dim, timestep)
        )  # (B, hidden_dim)

        # Expand bridge and timestep to match action sequence length
        bridge_expanded = bridge_h.unsqueeze(1).expand(-1, T_action, -1)  # (B, T, hidden_dim)
        t_expanded = t_h.unsqueeze(1).expand(-1, T_action, -1)  # (B, T, hidden_dim)

        # Concatenate and predict
        combined = torch.cat([bridge_expanded, action_h, t_expanded], dim=-1)  # (B, T, hidden_dim*3)
        return self.mlp(combined)  # (B, T_action, action_dim)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def bridge_layers(self) -> tuple:
        return self._bridge_layers

    @property
    def is_interleaved(self) -> bool:
        return False  # Bridge-collection pattern, not interleaved

    @property
    def action_mean(self) -> Tensor:
        return self._norm_action_mean

    @property
    def action_std(self) -> Tensor:
        return self._norm_action_std
