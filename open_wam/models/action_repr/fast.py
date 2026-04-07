"""FAST discrete-token action representation.

FAST (Physical Intelligence) encodes continuous robot actions into
discrete token sequences via a learned tokenizer, similar to how
language models tokenize text.  This representation wraps the FAST
tokenizer and provides encode/decode for the diffusion pipeline.

In encode:
    continuous actions (B, T, native_dim)
    → tokenize → token_ids (B, T, num_tokens_per_step)
    → embed → latent (B, T, num_tokens_per_step * embed_dim)

In decode:
    latent (B, T, num_tokens_per_step * embed_dim)
    → reshape → (B, T, num_tokens_per_step, embed_dim)
    → project to logits → argmax → token_ids
    → detokenize → continuous actions (B, T, native_dim)

The diffusion model operates on the continuous embedding space, but the
underlying representation is discrete — this gives the model a structured
action vocabulary rather than raw continuous coordinates.

Installation:
    pip install fast-tokenizer   # or: from physical_intelligence
    See: https://github.com/physical-intelligence/fast
"""

import torch
from torch import Tensor, nn

from open_wam.models.action_repr.base import BaseActionRepresentation


class FASTActionRepresentation(BaseActionRepresentation):
    """FAST discrete-token representation for robot actions.

    Wraps a FAST-compatible action tokenizer. The diffusion model sees
    continuous embeddings of discrete action tokens.

    Args:
        native_dim: Original continuous action dimension (e.g. 14).
        vocab_size: Number of discrete action tokens in vocabulary.
        num_tokens_per_step: Number of tokens each timestep is encoded into.
        embed_dim: Embedding dimension per token.
        tokenizer_name: HuggingFace model name for the FAST tokenizer.
            Default: ``"physical-intelligence/fast"``.
    """

    def __init__(
        self,
        native_dim: int = 14,
        vocab_size: int = 1024,
        num_tokens_per_step: int = 1,
        embed_dim: int = 64,
        tokenizer_name: str = "physical-intelligence/fast",
    ):
        super().__init__()
        self._native_dim = native_dim
        self._vocab_size = vocab_size
        self._num_tokens_per_step = num_tokens_per_step
        self._embed_dim = embed_dim
        self._tokenizer_name = tokenizer_name

        # Learned embedding table: token_id → continuous latent
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Output projection: embed_dim → vocab_size logits (for decode)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tokenizer loaded lazily
        self._tokenizer = None

    def _lazy_load_tokenizer(self):
        """Lazily load the FAST tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            from fast_tokenizer import FastTokenizer

            self._tokenizer = FastTokenizer.from_pretrained(self._tokenizer_name)
        except ImportError:
            raise ImportError(
                "FAST tokenizer is required for FASTActionRepresentation. "
                "Install with: pip install fast-tokenizer\n"
                "See: https://github.com/physical-intelligence/fast"
            )

    def encode(self, actions: Tensor) -> Tensor:
        """Encode continuous actions to embedded discrete tokens.

        Args:
            actions: (B, T, native_dim) raw continuous actions.

        Returns:
            (B, T, latent_dim) continuous embeddings of action tokens.
        """
        self._lazy_load_tokenizer()
        B, T, D = actions.shape
        device = actions.device

        # Tokenize: continuous → discrete token IDs
        # Shape: (B, T, num_tokens_per_step)
        actions_np = actions.detach().cpu().numpy()
        token_ids = self._tokenizer_encode(actions_np)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)

        # Embed tokens
        embedded = self.token_embedding(token_ids)  # (B, T, ntps, embed_dim)
        # Flatten token dimension: (B, T, ntps * embed_dim) = (B, T, latent_dim)
        return embedded.reshape(B, T, -1)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode embedded latents back to continuous actions.

        Args:
            latents: (B, T, latent_dim) denoised embeddings.

        Returns:
            (B, T, native_dim) reconstructed continuous actions.
        """
        self._lazy_load_tokenizer()
        B, T, _ = latents.shape

        # Reshape to per-token embeddings
        embedded = latents.reshape(B, T, self._num_tokens_per_step, self._embed_dim)

        # Project to logits and argmax
        logits = self.output_proj(embedded)  # (B, T, ntps, vocab_size)
        token_ids = logits.argmax(dim=-1)  # (B, T, ntps)

        # Detokenize: discrete token IDs → continuous actions
        token_ids_np = token_ids.detach().cpu().numpy()
        actions = self._tokenizer_decode(token_ids_np)
        return torch.tensor(actions, dtype=latents.dtype, device=latents.device)

    def _tokenizer_encode(self, actions_np):
        """Encode numpy actions to token IDs via the FAST tokenizer.

        Args:
            actions_np: (B, T, native_dim) numpy array.

        Returns:
            (B, T, num_tokens_per_step) numpy array of token IDs.
        """
        import numpy as np

        B, T, D = actions_np.shape

        if hasattr(self._tokenizer, "encoder_action2fastoken"):
            # physical-intelligence/fast API
            token_ids = []
            for b in range(B):
                ids = self._tokenizer.encoder_action2fastoken(actions_np[b])
                token_ids.append(ids)
            return np.stack(token_ids)
        elif hasattr(self._tokenizer, "encode"):
            # Generic tokenizer API
            token_ids = []
            for b in range(B):
                ids = self._tokenizer.encode(actions_np[b])
                token_ids.append(ids)
            return np.stack(token_ids)
        else:
            raise RuntimeError(
                f"Tokenizer {type(self._tokenizer).__name__} has no "
                f"recognized encode method. Expected 'encoder_action2fastoken' "
                f"or 'encode'."
            )

    def _tokenizer_decode(self, token_ids_np):
        """Decode token IDs to continuous actions via the FAST tokenizer.

        Args:
            token_ids_np: (B, T, num_tokens_per_step) numpy array.

        Returns:
            (B, T, native_dim) numpy array of continuous actions.
        """
        import numpy as np

        B, T, K = token_ids_np.shape

        if hasattr(self._tokenizer, "decoder_action"):
            actions = []
            for b in range(B):
                act = self._tokenizer.decoder_action(token_ids_np[b])
                actions.append(act)
            return np.stack(actions)
        elif hasattr(self._tokenizer, "decode"):
            actions = []
            for b in range(B):
                act = self._tokenizer.decode(token_ids_np[b])
                actions.append(act)
            return np.stack(actions)
        else:
            raise RuntimeError(
                f"Tokenizer {type(self._tokenizer).__name__} has no "
                f"recognized decode method. Expected 'decoder_action' "
                f"or 'decode'."
            )

    @property
    def latent_dim(self) -> int:
        return self._num_tokens_per_step * self._embed_dim

    @property
    def native_dim(self) -> int:
        return self._native_dim
