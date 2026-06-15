"""Backbone-agnostic interface for the WAM architecture/backbone split.

The video DiT block loop is backbone-specific (Wan-specific patchify, RoPE,
VACE, SP, ...). The action injection logic is
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

    Inject / extract contract (SharedBackbone path):
        ``inject_action_tokens`` extends ``x`` (sequence dim) and, in
        per-token t_mod mode, also extends ``freqs`` and ``t_mod`` to match.
        ``extract_action_tokens`` removes the shared-token tail from ``x`` and
        keeps sequence-shaped ``freqs`` / per-token ``t_mod`` aligned with the
        remaining video tokens. Architectures must call ``inject`` and
        ``extract`` in matched pairs.

    Fields marked "backbone-internal" are managed by the backbone and should
    not be modified by architecture code.
    """

    # --- Core tensors (architecture may read/write) ---
    x: Tensor  # (B, L, dim) current hidden state
    t_mod: Tensor  # timestep modulation — (B, 6, dim) or (B, L, 6, dim)
    freqs: Tensor  # RoPE freqs
    context: Tensor  # text/context embedding
    context_mask: Optional[Tensor] = None  # (B, L_context) bool, True = attend

    # --- Spatial dims for unpatchify (backbone-internal) ---
    f: int = 0
    h: int = 0
    w: int = 0

    # --- Per-frame token layout (backbone-internal) ---
    # ``tokens_per_frame_patch == h*w``. Stored explicitly so video-slice
    # arithmetic in ``inject_shared_tokens`` / ``finalize`` / mask construction
    # does not have to re-infer it from the latent tensor shape.
    tokens_per_frame_patch: int = 0

    # --- Time embedding for head (backbone-internal) ---
    t: Optional[Tensor] = None

    # --- Optional fields ---
    reference_prefix_len: int = 0  # Deprecated: reference_latents path removed (23246ba). Kept for API compat.
    vace_hints: Optional[list] = None
    vace_scale: float = 1.0
    sp_pad_shape: int = 0
    use_gradient_checkpointing: bool = False
    use_gradient_checkpointing_offload: bool = False

    # --- Backbone-specific extras ---
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

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Number of attention heads per DiT block. Used by joint-attention
        validation to check structural compatibility with an action backbone."""
        ...

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Per-head attention dimension. Used by joint-attention validation."""
        ...

    @property
    def context_dim(self) -> Optional[int]:
        """Per-token dim of the text/context embedding produced by ``preprocess_input``.

        Returned by backbones whose text encoder output dim differs from Wan's
        4096 (e.g. Cosmos). Architectures that need to size cross-attention
        projections may consult this when ``cfg.text_dim`` is unset. Default
        ``None`` keeps existing Wan configs unaffected — they still fall back
        to the legacy 4096 default.
        """
        return None

    @property
    def needs_first_frame_skip(self) -> bool:
        """Whether ``latent[0]`` is unconditionally a conditioning frame for this backbone.

        When ``True``, :meth:`BaseWAMArchitecture.preprocess` passes
        ``skip_first=True`` to ``downsample_video_mask_to_latent`` so the
        ``video_is_pad`` mask is sized to ``T_lat - 1`` (the loss-side
        shape-detect fallback in :meth:`_compute_video_loss` then trims
        ``noise_pred`` to match).

        Override this when the backbone's *configuration* (not the input
        batch) guarantees ``latent[0]`` is conditioning. Wan I2V is the
        canonical example: its image conditioning rides on the ``y`` channel
        rather than the ``first_frame_latents`` input key, so the per-batch
        signal ``inputs.get("first_frame_latents") is not None`` would miss it.

        Backbones that condition on ``latent[0]`` only on *some* batches
        (TI2V / VACE / cosmos25 TI2V — driven by ``first_frame_latents`` in
        the inputs dict) should leave this at the default ``False``; the
        per-batch signal in ``preprocess`` already covers them.
        """
        return False

    def copy_deploy_artifacts(self, output_dir: str, cfg) -> None:
        """Copy backbone-specific deploy artifacts (tokenizer, processor, ...) into ``output_dir``.

        Called by ``BaseWAMArchitecture.copy_deploy_artifacts`` after the
        checkpoint config has been written, so deploy-time loaders can be
        fully self-contained. Default no-op; backbones with external tokenizer
        or processor files override this.
        """

    @property
    def video_attention_mask_mode(self) -> str:
        """Video self-attention mask mode used by joint MoT mask construction.

        Default ``bidirectional`` (full v↔v coupling). Concrete backbones
        override with ``per_frame_causal`` / ``first_frame_causal`` and
        provide :meth:`build_video_to_video_mask` matching the chosen mode.
        """
        return "bidirectional"

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the v↔v block of the joint attention mask.

        Default implementation honors :attr:`video_attention_mask_mode` set
        to ``bidirectional`` only. Concrete backbones override to support
        causal modes.
        """
        if self.video_attention_mask_mode != "bidirectional":
            raise NotImplementedError(
                f"{type(self).__name__} does not implement build_video_to_video_mask "
                f"for mode '{self.video_attention_mask_mode}'."
            )
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    # ================================================================
    # DiT patch geometry (1 property + 1 classmethod)
    # ================================================================

    @property
    def dit_patch_size(self) -> Tuple[int, int, int]:
        """Spatio-temporal patch size the host DiT applies on its first layer.

        Source of truth is :attr:`VideoEncoderSpec.dit_patch_size` when an
        external encoder is plugged in; otherwise the subclass's
        :meth:`get_native_dit_patch_size` value. Concrete backbones must store
        the resolved tuple into ``self._dit_patch_size`` during ``__init__`` so
        that all callers (token-count arithmetic, mask construction, DiT
        rebuild) consult a single backbone-owned attribute instead of branching
        on ``self._encoder is None``.
        """
        return self._dit_patch_size

    @classmethod
    @abstractmethod
    def get_native_dit_patch_size(cls, pipe) -> Tuple[int, int, int]:
        """Native ``(T, H, W)`` patch size when no external encoder is present.

        Wan family returns ``(1, 2, 2)``. Subclasses that wrap a different
        pretrained DiT family override here; the value must match what the
        loaded ``pipe.dit`` actually expects at its first layer.
        """
        ...

    # ================================================================
    # Temporal contract (2 properties + 1 classmethod)
    # ================================================================

    @property
    def temporal_compression(self) -> int:
        """``T_pixel / T_lat`` exposed by this backbone's latent path.

        Source of truth is :attr:`VideoEncoderSpec.temporal_compression` when
        an external encoder is plugged in; otherwise the subclass's
        :meth:`get_native_temporal_contract` value. Concrete backbones must
        store the resolved value into ``self._temporal_compression`` during
        ``__init__`` so downstream consumers (mask downsampling, dataloader
        divisibility) all read a single backbone-owned attribute instead of
        branching on ``self._encoder is None``.
        """
        return self._temporal_compression

    @property
    def causal_temporal(self) -> bool:
        """Whether the first input frame is encoded into its own standalone
        latent token (Wan-style causal VAE) vs uniform tubelet schedules.

        Same source-of-truth contract as :attr:`temporal_compression`: read
        from encoder spec on the external path, otherwise from
        :meth:`get_native_temporal_contract`.
        """
        return self._causal_temporal

    @classmethod
    @abstractmethod
    def get_native_temporal_contract(cls, pipe) -> Tuple[int, bool]:
        """Native ``(temporal_compression, causal_temporal)`` when no external
        encoder is present.

        Wan family returns ``(4, True)`` (the causal Wan2pt1 VAE). Subclasses
        wrapping a non-Wan native VAE override here; the values must match
        what the loaded ``pipe.vae`` actually produces.
        """
        ...

    # ================================================================
    # Flow-matching α-shift (1 property)
    # ================================================================

    @property
    def shift_video(self) -> Optional[float]:
        """Optional Esser-et-al. α-shift applied to the video scheduler.

        Single source of truth for the video-side shift: both
        :meth:`BaseWAMArchitecture.init_training_schedulers` (training)
        and ``openwam/deploy/joint_engine.py::generate`` (inference)
        consult this property, so train/inference sigma grids cannot drift
        apart regardless of which yaml file is loaded.

        Returns ``None`` when the backbone has no explicit override (the
        scheduler then falls back to its template default — Wan = 5.0).
        Concrete backbones expose this by storing the resolved value into
        ``self._shift_video`` during ``__init__``; subclasses without the
        attribute inherit the ``None`` default.

        The action scheduler is intentionally NOT split here — it always
        consumes the global ``cfg.inference.shift``. This keeps the
        Reconstruction-or-Semantics paper recipe (arXiv:2605.06388,
        dim-dependent shift on non-VAE encoders only) bit-faithful without
        forcing action callers to learn about a knob they never set.
        """
        return getattr(self, "_shift_video", None)

    # ================================================================
    # Construction (1)
    # ================================================================

    @classmethod
    @abstractmethod
    def from_pretrained(cls, source, **kw) -> "VideoBackbone":
        """Build a backbone instance from pretrained weights.

        Args:
            source: Model path (str), Hydra config (DictConfig), component
                specs dict, or an already-built pipeline object.
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
        """Pre-block-loop setup: patchify, freqs, t_mod, VACE, SP.

        Returns a ``BlockLoopState`` that the architecture may modify before
        calling ``run_block``.
        """
        ...

    @abstractmethod
    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """Execute a single DiT block (+ VACE hint).

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
    # Joint self-attention path (2) — split a DiT block into pre/post halves
    # ================================================================

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Run norm1 + AdaLN modulate + Q/K/V proj + RMSNorm + RoPE — but **not**
        the attention itself.

        This is the "first half" of a DiT block. Used by joint-attention
        drivers (e.g. ``MoTJointDriver``) that need to concatenate Q/K/V
        across modalities and run a single mixed attention.

        Returns:
            ``(q, k, v, post_state)``. ``q/k/v`` are shaped ``[B, S, H*D]``.
            ``post_state`` carries ``residual_x``, ``gate_msa``, ``shift_mlp``,
            ``scale_mlp``, ``gate_mlp``, plus a reference to the block, so
            ``post_attn_at_layer`` can finish the block without re-computing
            the modulation.

        Default implementation raises ``NotImplementedError``; concrete
        backbones that participate in joint attention must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pre_attn_at_layer; "
            "joint self-attention is not supported on this backbone."
        )

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Continue from where ``pre_attn_at_layer`` left off:
        ``block.gate(residual_x, gate_msa, block.self_attn.o(attn_out))`` →
        cross-attn → FFN, then any backbone-specific post-block residuals
        (VACE)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement post_attn_at_layer; "
            "joint self-attention is not supported on this backbone."
        )

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
    ) -> BlockLoopState:
        """Append action tokens to the video sequence in ``state``.

        Handles backbone-specific concerns (RoPE extension, per-token t_mod)
        so architectures don't need to know about them.

        Args:
            state: Current block loop state (modified in-place and returned).
            action_tokens: (B, n_action, dim) projected action tokens.
            n_action: Number of action tokens.
            timestep: Action diffusion timestep (for per-token t_mod backbones).

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
        """Append action tokens plus optional state tokens.

        Backbones that support SharedBackbone must override this explicitly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement inject_shared_tokens.")

    def extract_shared_tokens(
        self,
        state: BlockLoopState,
        n_action: int,
        *,
        n_state: int = 0,
    ) -> Tuple[BlockLoopState, Tensor]:
        """Extract action tokens from an action/state tail.

        Backbones that support SharedBackbone must override this explicitly.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement extract_shared_tokens.")

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
            **kw: Backbone-specific inputs. Recognized optional kwargs include
                ``vace_video``, ``first_frame_image``, ``ref_images`` and
                ``pre_encoded_text`` (``(B, L, D)`` tensor of cached prompt
                embeddings, e.g. Reason1 for Cosmos25 — backbones that don't
                consume it drop it silently via ``**kw``).

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
    # External encoder spec validation (1)
    # ================================================================

    # Fields that must match bit-for-bit between an external encoder's spec
    # and the host backbone's expected spec. ``pixel_range`` is deliberately
    # excluded — it is an encoder-internal normalization detail that the
    # backbone never inspects.
    _ENCODER_SPEC_REQUIRED_FIELDS: Tuple[str, ...] = (
        "z_dim",
        "spatial_compression",
        "temporal_compression",
        "causal_temporal",
    )

    @classmethod
    def validate_encoder_spec(cls, got, want) -> None:
        """Fail-fast when an external encoder's spec disagrees with what the
        host backbone needs.

        Args:
            got: The encoder's actual :class:`VideoEncoderSpec` (from loaded weights).
            want: The backbone's expected spec, OR ``None`` when the backbone
                has no concrete contract to enforce (e.g. its native VAE has
                already been released).

        Raises:
            ValueError: With a list of mismatching fields. Callers handle
                ``is_reversible=False`` upstream by either skipping this call
                or constructing a relaxed ``want`` — this method has no
                special-case knowledge of reversibility.
        """
        if want is None:
            return
        mismatched = [f for f in cls._ENCODER_SPEC_REQUIRED_FIELDS if getattr(got, f) != getattr(want, f)]
        if mismatched:
            details = ", ".join(f"{f}: got={getattr(got, f)!r}, want={getattr(want, f)!r}" for f in mismatched)
            raise ValueError(f"encoder spec mismatch on {mismatched}: {details}")

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
    # Optional training-time flow-matching hooks (used by BaseWAMArchitecture)
    # ================================================================
    # Subclasses may override the next two methods when their flow-matching
    # convention differs from the Wan default (noisy = (1 - σ)·clean + σ·noise,
    # target = noise − clean). Detected via ``hasattr`` at the call site so
    # existing backbones do not need to opt in.

    # def add_training_noise(self, clean: Tensor, noise: Tensor, timestep_ids: Tensor) -> Tensor: ...
    # def training_target(self, clean: Tensor, noise: Tensor, timestep_ids: Tensor) -> Tensor: ...

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
