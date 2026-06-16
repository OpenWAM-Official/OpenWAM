"""Contract between the WAM architecture and a concrete video backbone.

The video DiT block loop is backbone-specific (patchify, RoPE, VACE, SP); the
action-injection logic is architecture-specific. This module defines the seam:

- :class:`VideoBackbone` exposes the block loop via ``prepare`` / ``run_block``
  / ``finalize`` (+ optional joint-attention and shared-token hooks). The
  architecture writes its own for-loop, inserting its logic between calls.
- :class:`BlockLoopState` is the mutable state flowing through the three steps.
  Architecture code may mutate ``hidden_states`` / ``time_mod`` / ``rope_freqs``
  / ``context`` (e.g. append/slice shared tokens, extend RoPE / per-token
  time_mod). ``grid_*`` and ``extras`` are backbone-internal.

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

    The architecture may mutate ``hidden_states`` / ``time_mod`` / ``rope_freqs``
    / ``context`` between ``prepare`` and ``run_block`` (append action/state
    tokens, extend RoPE / per-token time_mod). ``inject_shared_tokens`` and
    ``extract_shared_tokens`` keep these aligned and must be called in pairs.
    ``grid_*`` and ``extras`` are backbone-owned; architecture code reads but
    does not write them.
    """

    # Architecture may read/write
    hidden_states: Tensor  # (B, L, dim)
    time_mod: Tensor  # (B, 6, dim) or (B, L, 6, dim) per-token
    rope_freqs: Tensor
    context: Tensor  # text cross-attention embedding
    context_mask: Optional[Tensor] = None  # (B, L_context) bool, True = attend

    # Patch grid for unpatchify (backbone-internal, architecture reads only)
    grid_frames: int = 0
    grid_height: int = 0
    grid_width: int = 0
    tokens_per_frame_patch: int = 0  # grid_height * grid_width; cached for video-slice arithmetic

    # Backbone-internal (architecture does not touch)
    time_embed: Optional[Tensor] = None  # head time embedding
    vace_hints: Optional[list] = None
    vace_scale: float = 1.0
    sp_pad_shape: int = 0

    # Loop config threaded from prepare() into run_block()
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False

    # Backbone-private escape hatch
    extras: dict = field(default_factory=dict)


class VideoBackbone(ABC, nn.Module):
    """Architecture ↔ video backbone contract — the architecture's private helper.

    Minimal implementation = the ``@abstractmethod`` members below. Optional
    hooks ship working defaults so a backbone that only runs the basic block
    loop need not implement them. Inherits ``nn.Module`` so components
    registered as named children (``self.dit`` / ``self.vae`` / ...) are moved
    by the default :meth:`set_dtype_device`, found by ``get_submodule``, and
    serialized into the architecture state_dict.
    """

    # ================================================================
    # Required: structural metadata
    # ================================================================

    @property
    @abstractmethod
    def dim(self) -> int:
        """Hidden dim of the video DiT. Action tokens project to this to concat."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of DiT blocks; the architecture's block loop iterates over this."""

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Attention heads per block. Joint-attention sizes Q/K/V reshapes from this."""

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Per-head attention dim. Used by joint-attention structural validation."""

    @property
    @abstractmethod
    def scheduler(self):
        """Flow-matching scheduler; must support set_timesteps / timesteps / sigmas."""

    @property
    def dit_patch_size(self) -> Tuple[int, int, int]:
        """DiT ``(T, H, W)`` patch size on the latent grid. Store the resolved
        tuple into ``self._dit_patch_size`` during ``__init__``."""
        return self._dit_patch_size

    @property
    def temporal_compression(self) -> int:
        """``T_pixel / T_lat`` of this backbone's latent path. Store the resolved
        value into ``self._temporal_compression`` during ``__init__``."""
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
        """Raw training data (PIL frames / text / VACE / first frame) → tensor dict.

        Returns at least ``input_latents`` / ``context`` / ``seq_lens``.
        Unconsumed kwargs are dropped via ``**kw``.
        """

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
        """Optional Esser-et-al. α-shift for the video scheduler (single source of
        truth, read by the architecture for both training and inference). ``None``
        falls back to the scheduler template default (Wan = 5.0)."""
        return getattr(self, "_shift_video", None)

    @property
    def external_encoder(self):
        """The swapped-in external :class:`VideoEncoder`, or ``None`` for the native
        VAE path. Exposed so the architecture can surface encoder state to the
        trainer without train code reaching into backbone privates."""
        return getattr(self, "_encoder", None)

    @property
    def context_dim(self) -> Optional[int]:
        """Per-token dim of the text/context embedding, when it differs from Wan's
        4096. ``None`` keeps the legacy 4096 fallback."""
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
    def submodule_names(self) -> list[str]:
        """Names of manageable sub-modules (dit, vae, text_encoder, ...) for the
        trainer's freeze/device bookkeeping. Default: registered child names."""
        return [name for name, _ in self.named_children()]

    @property
    def video_attention_mask_mode(self) -> str:
        """v↔v mask mode for joint MoT. Default bidirectional; causal backbones override."""
        return "bidirectional"

    def build_video_to_video_mask(
        self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ) -> Tensor:
        """Build the v↔v attention-mask block. Default bidirectional; mode is read
        from :attr:`video_attention_mask_mode`."""
        if self.video_attention_mask_mode != "bidirectional":
            raise NotImplementedError(
                f"{type(self).__name__} does not implement build_video_to_video_mask "
                f"for mode '{self.video_attention_mask_mode}'."
            )
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    # ================================================================
    # Optional: deploy preprocessing + decode (default raise)
    # ================================================================

    def preprocess_input_for_inference(self, inputs) -> dict:
        """Deploy-time input prep (prompt/image/VACE encode, noise init). Returns a
        dict ready for the inference denoising loop. Training-only backbones omit it."""
        raise NotImplementedError(f"{type(self).__name__} does not support deploy inference.")

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """Latent ``(B, C, T, H, W)`` → list of PIL frames. Irreversible encoders omit it."""
        raise NotImplementedError(f"{type(self).__name__} does not support decode_video.")

    # ================================================================
    # Optional: joint self-attention split (default raise)
    # ================================================================

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Block first half: norm + modulate + Q/K/V + RoPE, excluding attention itself.
        Returns ``(q, k, v, post_state)``; ``post_state`` carries residual/gate for
        :meth:`post_attn_at_layer`."""
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Block second half from :meth:`pre_attn_at_layer`: gate → cross-attn → FFN."""
        raise NotImplementedError(f"{type(self).__name__} does not support joint self-attention.")

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
        """Append action (+ optional state) tokens to the video sequence:
        ``[video][action][state]``. Extends RoPE / per-token time_mod to match.
        ``n_state=0`` degenerates to pure action injection."""
        raise NotImplementedError(f"{type(self).__name__} does not support shared-backbone.")

    def extract_shared_tokens(
        self, state: BlockLoopState, n_action: int, *, n_state: int = 0
    ) -> Tuple[BlockLoopState, Tensor]:
        """Slice action/state tokens off the sequence tail. Returns ``(state, action_tokens)``."""
        raise NotImplementedError(f"{type(self).__name__} does not support shared-backbone.")

    # ================================================================
    # Lifecycle: device/dtype (default moves all registered children)
    # ================================================================

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move everything to ``(dtype, device)``. Default covers registered
        children; backbones with out-of-tree state override (call ``super()``
        first). Must NOT ``.eval()`` — trainable submodules stay in train mode."""
        self._dtype = dtype
        self._device = device
        self.to(dtype=dtype, device=device)

    # ================================================================
    # Optional deploy-artifact hooks (orchestrated by the architecture)
    # ================================================================

    def copy_deploy_artifacts(self, output_dir: str, cfg) -> None:
        """Copy backbone-side deploy artifacts (tokenizer/processor) into ``output_dir``.
        Default no-op; backbones with external side files override."""

    # ================================================================
    # External encoder spec validation (helper for from_pretrained)
    # ================================================================

    _ENCODER_SPEC_REQUIRED_FIELDS: Tuple[str, ...] = (
        "z_dim",
        "spatial_compression",
        "temporal_compression",
        "causal_temporal",
    )

    @classmethod
    def validate_encoder_spec(cls, got, want) -> None:
        """Fail-fast when an external encoder's spec disagrees with the host
        backbone's expected spec. ``want=None`` skips the check."""
        if want is None:
            return
        mismatched = [f for f in cls._ENCODER_SPEC_REQUIRED_FIELDS if getattr(got, f) != getattr(want, f)]
        if mismatched:
            details = ", ".join(f"{f}: got={getattr(got, f)!r}, want={getattr(want, f)!r}" for f in mismatched)
            raise ValueError(f"encoder spec mismatch on {mismatched}: {details}")


__all__ = [
    "BlockLoopState",
    "VideoBackbone",
]
