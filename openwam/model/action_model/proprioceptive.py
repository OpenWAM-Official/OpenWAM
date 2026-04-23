"""Proprioceptive state conditioning for action prediction.

Encodes robot proprioceptive state (joint angles, EE pose, gripper state)
into conditioning tokens that are injected into the ActionDiT's input
sequence. This provides the model with current robot state information,
which is critical for accurate action prediction — especially in
closed-loop deployment where the model must react to actual robot state
rather than relying solely on visual observations.

Three injection modes:
1. **concat** / **sequence_concat**: Append state tokens to the action
   sequence (adds num_state_tokens tokens — sequence-wise concat).
2. **channel_concat**: Broadcast state features to every timestep and
   concatenate along the channel dim, then project back to hidden_dim
   (preserves sequence length — channel-wise concat).
3. **add**: Project state to same dim as action embedding and add as a bias.

Usage:
    encoder = ProprioceptiveEncoder(
        state_dim=14,      # e.g., 7 joint angles + 7 joint velocities
        hidden_dim=768,    # must match ActionDiT.dim
        mode="channel_concat",
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
        mode: Injection mode. One of:
            - ``"concat"`` / ``"sequence_concat"``: prepend ``num_state_tokens``
              tokens to the action sequence (sequence-wise concat).
            - ``"channel_concat"``: broadcast state along time and concat along
              channel dim, then project back to ``hidden_dim`` (channel-wise
              concat, preserves sequence length).
            - ``"add"``: project state to ``hidden_dim`` and add as a bias to
              every action token.
        num_state_tokens: Number of tokens to generate in sequence_concat mode.
            More tokens = more capacity but longer sequence.
        dropout: Dropout rate on state features (regularization).
    """

    # Accepted mode aliases. "concat" is kept for backward compatibility
    # and is treated identically to "sequence_concat".
    _SEQUENCE_CONCAT_ALIASES = ("concat", "sequence_concat")

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        mode: str = "add",
        num_state_tokens: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        valid = self._SEQUENCE_CONCAT_ALIASES + ("add", "channel_concat")
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}, got '{mode}'")

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.num_state_tokens = num_state_tokens

        # State encoder MLP — last linear is zero-initialized so the encoded
        # state starts at zero, preserving pretrained behavior at init.
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self._is_sequence_concat():
            # Project single state vector to num_state_tokens token embeddings
            self.token_proj = nn.Linear(hidden_dim, hidden_dim * num_state_tokens)
            # Learned positional embeddings for state tokens (zero-init).
            # Rationale: keeping this zero preserves the bit-identical-to-baseline
            # guarantee at step 0 (state tokens = 0 → attention over them is a
            # no-op). The N tokens ARE identical at init, but because each
            # index is a separate parameter, gradients from downstream layers
            # differentiate them after the first backward pass.
            self.state_pos_embedding = nn.Parameter(torch.zeros(1, num_state_tokens, hidden_dim))
        elif mode == "channel_concat":
            # Project [action_embeds; state_feat] (2*hidden_dim) -> hidden_dim.
            # Initialized so the action half is identity and the state half is
            # zero, meaning at init the encoder is a no-op on action_embeds.
            self.channel_merge = nn.Linear(2 * hidden_dim, hidden_dim)
        # In "add" mode, encoder output is directly added as a global bias

        self._init_weights()

    def _is_sequence_concat(self) -> bool:
        return self.mode in self._SEQUENCE_CONCAT_ALIASES

    def _init_weights(self):
        """Initialize with weights that preserve pretrained behavior at t=0."""
        # Zero-init last linear in encoder so state injection starts at zero
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        if self._is_sequence_concat():
            nn.init.zeros_(self.token_proj.weight)
            nn.init.zeros_(self.token_proj.bias)
        elif self.mode == "channel_concat":
            # Weight shape is (hidden_dim, 2*hidden_dim) = [W_action | W_state].
            # Set W_action = I, W_state = 0, bias = 0 → at init the layer
            # outputs action_embeds unchanged regardless of state value.
            with torch.no_grad():
                self.channel_merge.weight.zero_()
                self.channel_merge.weight[:, : self.hidden_dim] = torch.eye(self.hidden_dim)
                self.channel_merge.bias.zero_()

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
            (B, T_out, hidden_dim) where T_out = T_action (add /
            channel_concat) or T_action + num_state_tokens (sequence_concat).
        """
        B, T, _ = action_embeds.shape

        # Encode state
        state_feat = self.encoder(state)  # (B, hidden_dim)

        if self.mode == "add":
            # Global bias: add state feature to every action token
            return action_embeds + state_feat.unsqueeze(1)
        elif self.mode == "channel_concat":
            # Broadcast state across time, concat along channel dim, project back.
            state_broadcast = state_feat.unsqueeze(1).expand(B, T, self.hidden_dim)
            fused = torch.cat([action_embeds, state_broadcast], dim=-1)
            return self.channel_merge(fused)
        else:
            # sequence_concat: generate state tokens and prepend to action sequence
            state_tokens = self.token_proj(state_feat)  # (B, hidden_dim * num_tokens)
            state_tokens = state_tokens.view(B, self.num_state_tokens, self.hidden_dim)
            state_tokens = state_tokens + self.state_pos_embedding
            return torch.cat([state_tokens, action_embeds], dim=1)

    @property
    def extra_tokens(self) -> int:
        """Number of extra tokens prepended to the sequence.

        Non-zero only for sequence_concat; the output head must slice these
        off before producing per-timestep action noise predictions.
        """
        return self.num_state_tokens if self._is_sequence_concat() else 0
