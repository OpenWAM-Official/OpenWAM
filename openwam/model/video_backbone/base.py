"""Contract between the WAM architecture and a concrete video backbone.

Layering: only the architecture talks to the backbone through this contract;
train/deploy go through the architecture, never the backbone object directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class BlockLoopState:
    """Mutable state flowing through prepare → run_block → finalize.

    Architecture may read/write ``hidden_states`` / ``time_mod`` / ``rope_freqs``
    / ``context``; ``grid_*`` and ``extras`` are backbone-owned (read-only to it).
    """

    # Architecture may read/write
    hidden_states: Tensor  # (B, L, dim)
    time_mod: Tensor  # (B, 6, dim) or (B, L, 6, dim) per-token
    rope_freqs: Tensor
    context: Tensor  # text cross-attention embedding
    context_mask: Optional[Tensor] = None  # (B, L_context) bool, True = attend

    # Patch grid for unpatchify
    grid_frames: int = 0
    grid_height: int = 0
    grid_width: int = 0

    # Per-block VACE hints: dual_system IDM merges across branches; tri_system rejects.
    vace_hints: Optional[list] = None

    # Prefix K/V tokens the backbone prepends to its per-layer keys/values but
    # NOT to its queries (e.g. Cosmos3's cached und text stream). The MoT driver
    # widens the joint mask by this many leading key columns — visible to every
    # query row, gated per-sample by ``prefix_kv_mask`` (True = attend). Zero
    # for backbones whose K and Q sequences coincide (Wan, CosmosPredict25).
    prefix_kv_len: int = 0
    prefix_kv_mask: Optional[Tensor] = None  # (B, prefix_kv_len) bool

    # Loop config threaded from prepare() into run_block()
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False

    # Backbone-private escape hatch (Wan also stashes dit/vace/time_embed here)
    extras: dict = field(default_factory=dict)


class VideoBackbone(ABC, nn.Module):
    """Architecture ↔ video backbone contract — the architecture's private helper.

    Inherits ``nn.Module`` so named children (``self.dit`` / ``self.vae`` / ...)
    are moved by :meth:`set_dtype_device` and serialized into the state_dict.
    """

    # ================================================================
    # Required: structural metadata
    # ================================================================

    @property
    @abstractmethod
    def dim(self) -> int:
        """Hidden dim of the video DiT."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of DiT blocks."""

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Attention heads per block."""

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Per-head attention dim."""

    @property
    @abstractmethod
    def scheduler(self):
        """Flow-matching scheduler; must support set_timesteps / timesteps / sigmas."""

    @property
    def dit_patch_size(self) -> Tuple[int, int, int]:
        """DiT ``(T, H, W)`` patch size; set ``self._dit_patch_size`` in ``__init__``."""
        return self._dit_patch_size

    @property
    def temporal_compression(self) -> int:
        """``T_pixel / T_lat``; set ``self._temporal_compression`` in ``__init__``."""
        return self._temporal_compression

    # ================================================================
    # Required: construction + training preprocessing
    # ================================================================

    @classmethod
    @abstractmethod
    def from_pretrained(cls, source, **kw) -> "VideoBackbone":
        """Build from pretrained weights. ``source``: path / config / specs / pipe object."""

    @abstractmethod
    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Raw training data → tensor dict with at least ``input_latents`` /
        ``context`` / ``seq_lens``. Unconsumed kwargs are dropped via ``**kw``."""

    # ================================================================
    # Required: three-step execution
    # ================================================================

    @abstractmethod
    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        """Pre-loop: patchify / freqs / time_mod / VACE / SP. Returns a BlockLoopState."""

    @abstractmethod
    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """Run a single DiT block. Gradient checkpointing is transparent to the architecture."""

    @abstractmethod
    def finalize(self, state: BlockLoopState) -> Tensor:
        """Post-loop: head + SP gather + unpatchify. Returns ``(B, C, T, H, W)``."""

    # ================================================================
    # Optional metadata (defaults)
    # ================================================================

    @property
    def device(self) -> torch.device:
        return getattr(self, "_device", torch.device("cuda"))

    @property
    def dtype(self) -> torch.dtype:
        return getattr(self, "_dtype", torch.bfloat16)

    @property
    def shift_video(self) -> Optional[float]:
        """α-shift for the video scheduler (single source of truth for train +
        inference). ``None`` falls back to the scheduler default (Wan = 5.0)."""
        return getattr(self, "_shift_video", None)

    @property
    def external_encoder(self):
        """The swapped-in external :class:`VideoEncoder`, or ``None`` for the
        native VAE path."""
        return getattr(self, "video_encoder", None)

    def reinit_for_from_scratch(self, *, external_encoder=None, source=None) -> None:
        """Re-init the DiT for a ``from_scratch`` run, owning the dit/patch-size
        details internally so the architecture stays backbone-agnostic.

        Two paths, distinguished by ``source``:
          - ``source is None`` (training): random-reinit the DiT weights, after
            reshaping I/O to ``external_encoder``'s latent dim when one is swapped in.
          - ``source is not None`` (deploy): reshape I/O to ``external_encoder``'s
            latent dim WITHOUT resetting, so the subsequent strict checkpoint load
            populates the reshaped tensors. No-op when ``external_encoder is None``.

        Default raises — only backbones with a re-initializable DiT (Wan) support it."""
        raise NotImplementedError(f"{type(self).__name__} does not support from_scratch DiT re-initialization.")

    @property
    def text_dim(self) -> Optional[int]:
        """Per-token raw text/context embedding dim. ``None`` keeps the 4096 fallback."""
        return None

    @property
    def causal_temporal(self) -> bool:
        """Whether the first frame is encoded into its own standalone latent token."""
        return getattr(self, "_causal_temporal", True)

    @property
    def needs_first_frame_skip(self) -> bool:
        """Whether ``latent[0]`` is unconditionally a conditioning frame (Wan I2V)."""
        return False

    @property
    def video_attention_mask_mode(self) -> str:
        """v↔v mask mode for joint MoT. Default bidirectional; causal backbones override."""
        return "bidirectional"

    def build_video_to_video_mask(
        self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ) -> Tensor:
        """Build the v↔v attention-mask block; mode from :attr:`video_attention_mask_mode`."""
        if self.video_attention_mask_mode != "bidirectional":
            raise NotImplementedError(
                f"{type(self).__name__} does not implement build_video_to_video_mask "
                f"for mode '{self.video_attention_mask_mode}'."
            )
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    # ================================================================
    # Optional: deploy preprocessing + decode (default raise)
    # ================================================================

    def preprocess_input_for_inference(self, **kw) -> dict:
        """Deploy-time input prep → dict ready for the inference denoising loop.

        Producers pass explicit kwargs (prompt / vace_video / first_frame_image /
        num_frames / height / width / seed / num_inference_steps / shift / tiled /
        vace_cache / prompt_embed_cache); backbones declare what they consume and
        let ``**kw`` swallow the rest."""
        raise NotImplementedError(f"{type(self).__name__} does not support deploy inference.")

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """Latent ``(B, C, T, H, W)`` → PIL frames. Irreversible encoders omit it."""
        raise NotImplementedError(f"{type(self).__name__} does not support decode_video.")

    # ================================================================
    # Optional: joint self-attention split (default raise)
    # ================================================================

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Block first half (norm + modulate + Q/K/V + RoPE, no attention). Returns
        ``(q, k, v, post_state)``; ``post_state`` feeds :meth:`post_attn_at_layer`."""
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Block second half from :meth:`pre_attn_at_layer`: gate → cross-attn → FFN."""
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

    # ================================================================
    # Optional: IDM teacher-forcing branch merge/split (default raise)
    # ================================================================

    def merge_idm_video_branches(self, noisy: BlockLoopState, cond: BlockLoopState) -> Tuple[BlockLoopState, int, int]:
        """Concatenate the IDM noisy + cond video branches into one state along
        the frame/sequence axis for a single MoT pass.

        Returns ``(merged, s_noisy_tokens, s_cond_tokens)`` where the two seq
        lengths are **token counts** (``T·H·W``) — the granularity the
        teacher-forcing attention mask is built at. The driver must not inspect
        ``hidden_states.shape[1]`` (it is ``T`` for 5D-grid backbones), so the
        merge implementation — which owns its own layout — returns them here.
        Pair with :meth:`split_idm_video_branches`."""
        raise NotImplementedError(f"{type(self).__name__} does not support IDM teacher-forcing.")

    def split_idm_video_branches(
        self, merged: BlockLoopState, noisy: BlockLoopState, cond: BlockLoopState
    ) -> Tuple[BlockLoopState, BlockLoopState]:
        """Inverse of :meth:`merge_idm_video_branches`: write the post-loop merged
        ``hidden_states`` (and any per-branch fields) back onto the ``noisy`` and
        ``cond`` states. Returns ``(noisy, cond)``."""
        raise NotImplementedError(f"{type(self).__name__} does not support IDM teacher-forcing.")

    # ================================================================
    # Optional: shared-token injection (default raise)
    # ================================================================

    def inject_shared_tokens(
        self,
        state: BlockLoopState,
        action_tokens: Tensor,
        n_action: int,
        *,
        state_tokens: Optional[Tensor] = None,
        n_state: int = 0,
        timestep: Optional[Tensor] = None,
    ) -> BlockLoopState:
        """Append action (+ optional state) tokens as ``[video][action][state]``,
        extending RoPE / time_mod to match. Pair with :meth:`extract_shared_tokens`."""
        raise NotImplementedError(f"{type(self).__name__} does not support single-system.")

    def extract_shared_tokens(
        self, state: BlockLoopState, n_action: int, *, n_state: int = 0
    ) -> Tuple[BlockLoopState, Tensor]:
        """Slice action/state tokens off the sequence tail. Returns ``(state, action_tokens)``."""
        raise NotImplementedError(f"{type(self).__name__} does not support single-system.")

    def assert_ready_for_shared_tokens(self, state: BlockLoopState) -> None:
        """Validate the prepared state can carry action/state shared tokens.

        Default (Wan): the per-token ``time_mod`` must be 4D so injected
        action/state tokens get their own timestep instead of being silently
        modulated by a global one. Backbones that carry per-token modulation
        elsewhere (CosmosPredict25 builds it in ``inject_shared_tokens`` from
        ``extras``) override this to a no-op."""
        if state.time_mod.dim() != 4:
            raise RuntimeError(
                f"{type(self).__name__} single-system requires the video backbone to run in per-token "
                "t_mod mode (e.g. dit.seperated_timestep=True with fuse_vae_embedding_in_latents=True). "
                f"Got vstate.time_mod with dim={state.time_mod.dim()}; action/state timestep would be "
                "silently ignored otherwise."
            )

    # ================================================================
    # Lifecycle: device/dtype (default moves all registered children)
    # ================================================================

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move everything to ``(dtype, device)``. Overrides call ``super()`` first.
        Must NOT ``.eval()`` — trainable submodules stay in train mode."""
        self._dtype = dtype
        self._device = device
        self.to(dtype=dtype, device=device)

    # ================================================================
    # Optional deploy-asset hook (orchestrated by the architecture)
    # ================================================================

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's slice of the checkpoint self-contained.

        Both halves of self-containment in one place: (1) write the backbone's
        component/tokenizer reconstruction specs into ``cfg`` (so deploy rebuilds
        the module skeletons from ``config.yaml`` without the training-time
        ``model_path``), and (2) copy its non-weight artifact files (tokenizer /
        processor / external-encoder side files) into ``output_dir``. Runs before
        the architecture writes ``config.yaml``. Default no-op."""


__all__ = [
    "BlockLoopState",
    "VideoBackbone",
]
