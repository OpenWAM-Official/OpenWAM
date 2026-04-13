"""Proprioceptive state conditioning for action prediction.

Encodes robot proprioceptive state (joint angles, EE pose, gripper state)
into conditioning tokens that are injected into the ActionDiT's input
sequence. This provides the model with current robot state information,
which is critical for accurate action prediction — especially in
closed-loop deployment where the model must react to actual robot state
rather than relying solely on visual observations.

Two injection modes:
1. **concat**: Append state tokens to the action sequence (adds T_state tokens)
2. **add**: Project state to same dim as action embedding and add as a bias

Usage:
    encoder = ProprioceptiveEncoder(
        state_dim=14,      # e.g., 7 joint angles + 7 joint velocities
        hidden_dim=768,    # must match ActionDiT.dim
        mode="concat",
        num_state_tokens=4,
    )
    # In prepare_action_tokens:
    action_embeds = action_dit.action_embedding(noisy_actions)
    action_embeds = encoder(action_embeds, state_vector)
"""

import torch
import torch.nn as nn


class ProprioceptiveEncoder(nn.Module):
    """Encodes proprioceptive state into ActionDiT-compatible conditioning.

    Args:
        state_dim: Dimension of the raw proprioceptive state vector.
            Common values: 7 (joint positions), 14 (+ velocities),
            13 (EE pos + quat + gripper).
        hidden_dim: ActionDiT hidden dimension (must match ``ActionDiT.dim``).
        mode: Injection mode — "concat" appends tokens, "add" adds a bias.
        num_state_tokens: Number of tokens to generate in "concat" mode.
            More tokens = more capacity but longer sequence.
        dropout: Dropout rate on state features (regularization).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        mode: str = "add",
        num_state_tokens: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if mode not in ("concat", "add"):
            raise ValueError(f"mode must be 'concat' or 'add', got '{mode}'")

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.num_state_tokens = num_state_tokens

        # State encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if mode == "concat":
            # Project single state vector to num_state_tokens token embeddings
            self.token_proj = nn.Linear(hidden_dim, hidden_dim * num_state_tokens)
            # Learned positional embeddings for state tokens (zero-init)
            self.state_pos_embedding = nn.Parameter(torch.zeros(1, num_state_tokens, hidden_dim))
        # In "add" mode, encoder output is directly added as a global bias

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights to preserve pretrained behavior."""
        # Zero-init last linear in encoder so state injection starts at zero
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        if self.mode == "concat":
            nn.init.zeros_(self.token_proj.weight)
            nn.init.zeros_(self.token_proj.bias)

    def forward(
        self,
        action_embeds: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Inject proprioceptive state into action token embeddings.

        Args:
            action_embeds: (B, T_action, hidden_dim) — embedded action tokens
                (output of ActionDiT.action_embedding + pos_embedding).
            state: (B, state_dim) — proprioceptive state vector.

        Returns:
            (B, T_out, hidden_dim) where T_out = T_action (add mode)
            or T_action + num_state_tokens (concat mode).
        """
        B = action_embeds.shape[0]

        # Encode state
        state_feat = self.encoder(state)  # (B, hidden_dim)

        if self.mode == "add":
            # Global bias: add state feature to every action token
            return action_embeds + state_feat.unsqueeze(1)
        else:
            # Concat: generate state tokens and prepend to action sequence
            state_tokens = self.token_proj(state_feat)  # (B, hidden_dim * num_tokens)
            state_tokens = state_tokens.view(B, self.num_state_tokens, self.hidden_dim)
            state_tokens = state_tokens + self.state_pos_embedding
            return torch.cat([state_tokens, action_embeds], dim=1)

    @property
    def extra_tokens(self) -> int:
        """Number of extra tokens added to the sequence (0 for 'add' mode)."""
        return self.num_state_tokens if self.mode == "concat" else 0
