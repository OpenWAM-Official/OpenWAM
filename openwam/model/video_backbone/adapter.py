"""Backbone-agnostic interface for the WAM architecture/backbone split.

The video DiT block loop is backbone-specific (Wan-specific patchify, RoPE,
VACE, animate, TeaCache, SP, ...). The action injection logic is
architecture-specific (DualSystem cross/self attention, SharedBackbone
vanilla/MoE). This module defines the contract between the two:

- :class:`VideoBackbone` exposes the block loop to architectures via three
  methods: ``prepare()`` / ``run_block()`` / ``finalize()``.  Architectures
  write their own for-loop, calling ``run_block`` per block and inserting
  architecture-specific logic between calls.

- :class:`BlockLoopState` is the mutable state object that flows between
  ``prepare`` → ``run_block`` → ``finalize``.  Architectures may modify
  ``state.x``, ``state.t_mod``, ``state.freqs`` between calls (e.g. to
  concat/slice action tokens, extend RoPE, extend per-token t_mod).

Concrete implementations live alongside their backbone code
(e.g. ``wan_adapter.py`` parallel to the ``wan/`` folder).
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

    Architecture code may freely mutate ``x``, ``t_mod``, ``freqs`` between
    ``prepare()`` and the first ``run_block()`` call (e.g. to append action
    tokens, extend RoPE freqs, extend per-token t_mod).

    Fields marked "backbone-internal" are managed by the backbone and should
    not be modified by architecture code.
    """

    # --- Core tensors (architecture may read/write) ---
    x: Tensor  # (B, L, dim) current hidden state
    t_mod: Tensor  # timestep modulation — (B, 6, dim) or (B, L, 6, dim)
    freqs: Tensor  # RoPE freqs
    context: Tensor  # text embedding

    # --- Spatial dims for unpatchify (backbone-internal) ---
    f: int = 0
    h: int = 0
    w: int = 0

    # --- Time embedding for head (backbone-internal) ---
    t: Optional[Tensor] = None

    # --- Optional fields ---
    reference_prefix_len: int = 0
    vace_hints: Optional[list] = None
    vace_scale: float = 1.0
    tea_cache_update: bool = False
    sp_pad_shape: int = 0
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False

    # --- Backbone-specific extras (Animate, VAP, TeaCache, etc.) ---
    extras: dict = field(default_factory=dict)


class VideoBackbone(ABC, nn.Module):
    """Architecture ↔ video backbone interface.

    Inherits both ``ABC`` and ``nn.Module`` so that sub-modules registered
    on concrete subclasses (e.g. ``WanVideoBackbone._pipe``) are
    automatically collected by ``architecture.state_dict()`` /
    ``architecture.load_state_dict()``.

    All external code (trainer, deploy, tests) accesses the backbone
    through ``architecture.video_backbone``, calling only the methods
    defined here. Backbone-specific internals live as private methods
    (``_xxx``) on concrete subclasses.

    Architectures call ``prepare()`` once, then loop over ``run_block()``
    for each DiT block (inserting their own logic between calls), and
    finally call ``finalize()`` to get the video noise prediction.
    """

    # ================================================================
    # Properties (6)
    # ================================================================

    @property
    def device(self) -> torch.device:
        """Current device. Set via ``set_dtype_device``."""
        return getattr(self, "_device", torch.device("cuda"))

    @property
    def dtype(self) -> torch.dtype:
        """Current dtype. Set via ``set_dtype_device``."""
        return getattr(self, "_dtype", torch.bfloat16)

    @property
    @abstractmethod
    def dim(self) -> int:
        """Hidden dimensionality of the video DiT."""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of DiT transformer blocks."""
        ...

    @property
    @abstractmethod
    def scheduler(self):
        """Flow matching scheduler object.

        Must support ``.set_timesteps()``, ``.timesteps``, ``.sigmas``.
        """
        ...

    @property
    @abstractmethod
    def submodule_names(self) -> list[str]:
        """Names of all manageable sub-modules (e.g. dit, vae, text_encoder).

        Used by trainer to compute trainable vs frozen sets via config.
        """
        ...

    # ================================================================
    # Construction (1)
    # ================================================================

    @classmethod
    @abstractmethod
    def from_pretrained(cls, source, **kw) -> "VideoBackbone":
        """Build a backbone instance from pretrained weights.

        Args:
            source: Model path (str), Hydra config (DictConfig), manifest
                path, or an already-built pipeline object.
            **kw: Backend-specific options (device, dtype, etc.).

        Returns:
            Fully initialized VideoBackbone with pretrained weights loaded.
        """
        ...

    # ================================================================
    # Three-step execution (3)
    # ================================================================

    @abstractmethod
    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        """Pre-block-loop setup: patchify, freqs, t_mod, VACE, TeaCache, SP.

        Returns a ``BlockLoopState`` that the architecture may modify before
        calling ``run_block``.  When ``state.tea_cache_update`` is True the
        architecture should skip the block loop and call ``finalize`` directly.
        """
        ...

    @abstractmethod
    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """Execute a single DiT block (+ VACE hint + Animate).

        Gradient checkpointing is handled internally — transparent to the
        architecture.  The architecture may inspect/modify ``state.x`` after
        this call returns (e.g. bridge capture, expert FFN, joint attention).
        """
        ...

    @abstractmethod
    def finalize(self, state: BlockLoopState) -> Tensor:
        """Post-block-loop: head + SP gather + reference removal + unpatchify.

        Returns ``(B, C, T, H, W)`` video noise prediction.
        """
        ...

    # ================================================================
    # Action token injection — SharedBackbone path (2)
    # ================================================================

    @abstractmethod
    def inject_action_tokens(
        self,
        state: BlockLoopState,
        action_tokens: Tensor,
        n_action: int,
        *,
        timestep: Optional[Tensor] = None,
        t_mod_bias: Optional[Tensor] = None,
    ) -> BlockLoopState:
        """Append action tokens to the video sequence in ``state``.

        Handles backbone-specific concerns (RoPE extension, per-token t_mod)
        so architectures don't need to know about them.

        Args:
            state: Current block loop state (modified in-place and returned).
            action_tokens: (B, n_action, dim) projected action tokens.
            n_action: Number of action tokens.
            timestep: Action diffusion timestep (for per-token t_mod backbones).
            t_mod_bias: Learnable modality bias (for per-token t_mod backbones).

        Returns:
            Updated state with action tokens appended to ``state.x``.
        """
        ...

    @abstractmethod
    def extract_action_tokens(
        self,
        state: BlockLoopState,
        n_action: int,
    ) -> Tuple[BlockLoopState, Tensor]:
        """Slice action tokens off the end of the video sequence.

        Args:
            state: Current block loop state (modified in-place and returned).
            n_action: Number of trailing action tokens to extract.

        Returns:
            (updated_state, action_tokens) where action_tokens is (B, n_action, dim).
        """
        ...

    # ================================================================
    # Unified preprocessing (1)
    # ================================================================

    @abstractmethod
    def preprocess_input(self, *, frames=None, text=None, **kw) -> dict:
        """Unified preprocessing entry point.

        Takes raw data (PIL frames, text strings, optional VACE video,
        first-frame image, etc.) and returns a dict of tensors ready for
        the denoising loop. Internally handles: video preprocessing, VAE
        encoding, text encoding, VACE context assembly, TI2V first-frame
        handling — all backbone-specific details are encapsulated here.

        Args:
            frames: List of video clips, each a list of PIL Images.
            text: List of text prompts.
            **kw: Backbone-specific inputs (vace_video, first_frame_image,
                ref_images, etc.).

        Returns:
            Dict with at least: ``input_latents``, ``context``, ``seq_lens``.
            May also include ``vace_context``, ``ref_latents``,
            ``first_frame_latents``, ``height``, ``width``, ``num_frames``, etc.
        """
        ...

    # ================================================================
    # Sub-module access (2)
    # ================================================================

    @abstractmethod
    def get_submodule(self, name: str) -> nn.Module | None:
        """Get a named sub-module (dit, vae, text_encoder, vace, ...).

        Used by trainer for freeze/to(device) and DeepSpeed wrapping.
        Returns None if the named module does not exist.
        """
        ...

    @abstractmethod
    def set_submodule(self, name: str, module: nn.Module) -> None:
        """Replace a named sub-module.

        Used after DeepSpeed ``prepare()`` to sync the wrapped version
        back into the backbone so that ``prepare/run_block/finalize``
        use the DeepSpeed-managed module.
        """
        ...

    # ================================================================
    # Decoding (1)
    # ================================================================

    @abstractmethod
    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """Decode latent tensor to PIL frames.

        Args:
            latents: (B, C, T, H, W) latent tensor.
            tiled: Whether to use tiled decoding (saves VRAM).

        Returns:
            List of PIL images.
        """
        ...

    # ================================================================
    # Device management (1)
    # ================================================================

    @abstractmethod
    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Set dtype and device for all sub-modules.

        Called during deploy initialization to move everything to the
        target dtype/device in one shot.
        """
        ...

    # ================================================================
    # Compilation (1)
    # ================================================================

    def apply_compile(self, compile_cfg) -> None:
        """Apply torch.compile to backbone sub-modules.

        Default is a no-op. Concrete backbones override to compile
        their heavy sub-modules (DiT blocks, VAE, etc.).

        Args:
            compile_cfg: Config object with backend-specific bool flags.
        """


__all__ = [
    "BlockLoopState",
    "VideoBackbone",
]
