"""Shared Wan :class:`VideoBackbone` base plus the concrete Wan subclasses.

:class:`WanBase` holds every piece of behavior common to the Wan family (DiT
forward, conditioning, deploy). Concrete backbones add only their construction
+ encoder specifics:
  - :class:`Wan22Ti2v` — Wan2.2-TI2V-5B; supports swapping the native VAE for
    an external :class:`VideoEncoder`.
  - :class:`Wan21` — Wan2.1 I2V / VACE; native VAE only.

Lives outside ``wan/`` to keep that package Wan-internal. Owns the Wan
modules (DiT/VAE/text encoder/tokenizer/VACE) directly — modules as named
children, scheduler/tokenizer/division factors as plain attributes; external
code reaches them only through the ABC methods.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    torch_compile_kwargs,
    wan_blocks_compile_cfg,
)
from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.wan import action_tokens as wan_action_tokens
from openwam.model.video_backbone.wan import conditioning as wan_conditioning
from openwam.model.video_backbone.wan import dit_forward as wan_dit_forward
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan import loader
from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply
from openwam.model.video_backbone.wan.preprocess import (
    check_resize_height_width,
)
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward

logger = logging.getLogger(__name__)


class WanBase(VideoBackbone):
    """Exposes the Wan modules through the VideoBackbone interface.

    Modules registered as named children (clean ``dit.*`` / ``vae.*`` keys);
    scheduler / tokenizer / division factors are plain attributes. Concrete
    subclasses construct via their own ``from_pretrained(source)``.
    """

    # ================================================================
    # Construction
    # ================================================================

    def __init__(self, holder, *, external_encoder=None, shift_video=None, text_dim: Optional[int] = None):
        """Internal constructor. Use a subclass ``from_pretrained()`` instead.

        ``external_encoder`` is ``None`` on the native VAE path so ``state_dict()``
        carries only ``vae.*`` keys; the external-encoder subclass passes one to
        activate VAE-IO routing and register it as a named child.

        ``shift_video`` is the optional Esser α-shift on the video scheduler,
        stored as the single source of truth behind the ABC property. ``None``
        keeps the scheduler template default (Wan = 5.0).
        """
        super().__init__()
        # ``holder`` is a transient carrier: drain its sub-modules + non-Module
        # state into self, then let it go out of scope. Nothing reads it after.
        # Set after nn.Module.__init__ (super) so an nn.Module encoder registers
        # as a named child; shared methods reference ``self.video_encoder``.
        self.video_encoder = external_encoder
        # Optional sub-modules declared up front so the attribute always exists
        # (the loop below only setattr's the ones the holder actually carries).
        self.vae = None
        self.vace = None
        self.image_encoder = None
        self.motion_controller = None
        # Promote sub-modules to named children so state_dict uses clean prefixes.
        for _name in ("dit", "dit2", "vae", "vace", "vace2", "text_encoder", "image_encoder", "motion_controller"):
            _mod = getattr(holder, _name, None)
            if _mod is not None:
                # nn.Module → named child; non-Module (test mocks) → plain attr,
                # same ``self.<name>`` access resolves on both.
                setattr(self, _name, _mod)
        # Backbone-owned non-Module state. from_pretrained sets the external
        # division factors / latent_spec before ``cls(holder, ...)``.
        self._scheduler = getattr(holder, "scheduler", None)
        self._tokenizer = getattr(holder, "tokenizer", None)
        self._height_division_factor = getattr(holder, "height_division_factor", None)
        self._width_division_factor = getattr(holder, "width_division_factor", None)
        self._time_division_factor = getattr(holder, "time_division_factor", None)
        self._time_division_remainder = getattr(holder, "time_division_remainder", None)
        self._latent_spec = getattr(holder, "latent_spec", None)
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._shift_video = None if shift_video is None else float(shift_video)
        actual_text_dim = loader.infer_text_dim(getattr(holder, "dit", None))
        self._text_dim = actual_text_dim if text_dim is None else int(text_dim)
        # Resolve the Wan variant (I2V/TI2V/VACE/plain) ONCE; first-frame
        # conditioning delegates to it so hot paths carry no per-variant branch.
        from openwam.model.video_backbone.wan import variants as _variants

        self._variant = _variants.detect(getattr(holder, "dit", None), getattr(holder, "vace", None))
        # Wan native contract, invariant across all Wan2.x variants: DiT
        # patch (1,2,2); VAE 4× temporal compression + causal first-frame token.
        # The external-encoder subclass overrides these from its encoder spec.
        self._dit_patch_size = (1, 2, 2)
        self._temporal_compression, self._causal_temporal = 4, True
        self._wan_blocks_compile_enabled = False
        self._wan_blocks_compile_kwargs: dict | None = None
        self._compiled_wan_blocks: dict[int, Callable[..., Tensor]] = {}

    # ================================================================
    # Internal properties
    # ================================================================

    @property
    def _dit(self):
        return self.dit

    @property
    def _uses_external_encoder(self) -> bool:
        """True when routing VAE IO through an external encoder, not the native VAE."""
        return self.video_encoder is not None

    @property
    def _has_vace(self) -> bool:
        return self.vace is not None

    @property
    def _is_ti2v(self) -> bool:
        return bool(getattr(self._dit, "fuse_vae_embedding_in_latents", False))

    @property
    def needs_first_frame_skip(self) -> bool:
        """``True`` iff ``latent[0]`` is a clean conditioning frame excluded from
        the diffusion loss. Only TI2V (per-token t=0 on frame-0 tokens) skips.

        I2V and VACE do NOT skip: their first-frame condition rides a side
        channel (``y`` / ``vace_context``) while ``latent[0]`` stays fully noised
        and supervised. For I2V, skipping starves frame-0 of gradient and
        produces garbage there at inference (the cell-4 mock-loss divergence).
        """
        return self._variant.needs_first_frame_skip

    # ================================================================
    # ABC: Properties (6) — device/dtype inherited from VideoBackbone
    # ================================================================

    @property
    def dim(self) -> int:
        return int(self._dit.dim)

    @property
    def num_layers(self) -> int:
        return len(self._dit.blocks)

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def num_heads(self) -> int:
        return int(self._dit.blocks[0].num_heads)

    @property
    def head_dim(self) -> int:
        return int(self._dit.dim) // self.num_heads

    @property
    def text_dim(self) -> Optional[int]:
        return getattr(self, "_text_dim", None)

    @property
    def video_attention_mask_mode(self) -> str:
        """Video self-attention mask mode for joint MoT mask construction.

        Modes: ``bidirectional`` (full v↔v, default), ``per_frame_causal``
        (token-level causal block-diagonal), ``first_frame_causal`` (first frame
        sees only itself, later frames see all). Explicit override wins, else
        the DiT's value, else ``bidirectional``.
        """
        explicit = getattr(self, "_video_attention_mask_mode", None)
        if explicit is not None:
            return explicit
        return getattr(self._dit, "video_attention_mask_mode", "bidirectional")

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = mode

    def reinit_for_from_scratch(self, *, external_encoder=None, source=None) -> None:
        """from_scratch DiT re-init, owning the Wan dit/patch-size internally.

        Training (``source is None``): random-reinit the DiT, reshaping I/O to
        ``external_encoder`` first when one is swapped in. Deploy
        (``source is not None``): reshape-only (no reset) so the strict checkpoint
        load populates the reshaped tensors; skipped when no external encoder.
        Both transparently no-op when the backbone carries no dit (logged inside
        the reinit helpers)."""
        if source is None:
            from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

            reinit_dit_from_scratch(
                self,
                external_encoder=external_encoder,
                dit_patch_size=self.dit_patch_size,
            )
        elif external_encoder is not None:
            from openwam.model.video_backbone.wan.reinit import adapt_dit_to_external_encoder

            adapt_dit_to_external_encoder(self, external_encoder, self.dit_patch_size)

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the video↔video block of the joint MoT attention mask
        (``True`` = attend to). Layout matches FastWAM's equivalent.
        """
        mode = self.video_attention_mask_mode
        if mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "video_seq_len must be divisible by video_tokens_per_frame in 'per_frame_causal' mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device))
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            # First-frame rows attend only to first-frame keys; later rows stay True.
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(
            f"Unsupported video_attention_mask_mode '{mode}'. "
            "Choose from: bidirectional, per_frame_causal, first_frame_causal."
        )

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Enable lazy per-layer Wan ``DiTBlock.forward`` compile for deploy."""

        self._compiled_wan_blocks.clear()
        self._wan_blocks_compile_enabled = False
        self._wan_blocks_compile_kwargs = None

        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = wan_blocks_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("Wan block compile disabled by config; running eager.")
            return

        kwargs = torch_compile_kwargs(section, default_mode="default")
        self._wan_blocks_compile_kwargs = kwargs
        self._wan_blocks_compile_enabled = True
        logger.info("Enabled lazy Wan block compile with torch.compile kwargs=%s", kwargs)

    def _compiled_wan_block(self, block_id: int, block: nn.Module) -> Callable[..., Tensor]:
        compiled = self._compiled_wan_blocks.get(block_id)
        if compiled is None:
            compiled = torch.compile(block, **(self._wan_blocks_compile_kwargs or {}))
            self._compiled_wan_blocks[block_id] = compiled
        return compiled

    def _run_wan_block(
        self,
        block_id: int,
        block: nn.Module,
        state: BlockLoopState,
        block_context_mask: Optional[Tensor],
        attn_mask: Optional[Tensor],
    ) -> Tensor:
        compile_allowed = (
            self._wan_blocks_compile_enabled
            and not state.use_gradient_checkpointing
            and not state.use_gradient_checkpointing_offload
        )
        if compile_allowed:
            try:
                compiled = self._compiled_wan_block(block_id, block)
                if attn_mask is not None:
                    return compiled(
                        state.hidden_states,
                        state.context,
                        state.time_mod,
                        state.rope_freqs,
                        block_context_mask,
                        attn_mask,
                    )
                return compiled(
                    state.hidden_states,
                    state.context,
                    state.time_mod,
                    state.rope_freqs,
                    block_context_mask,
                )
            except Exception as exc:
                self._compiled_wan_blocks.clear()
                self._wan_blocks_compile_enabled = False
                logger.warning("Wan block torch.compile failed at block %s; falling back to eager: %s", block_id, exc)

        if attn_mask is not None:
            return gradient_checkpoint_forward(
                block,
                state.use_gradient_checkpointing,
                state.use_gradient_checkpointing_offload,
                state.hidden_states,
                state.context,
                state.time_mod,
                state.rope_freqs,
                block_context_mask,
                attn_mask,
            )

        return gradient_checkpoint_forward(
            block,
            state.use_gradient_checkpointing,
            state.use_gradient_checkpointing_offload,
            state.hidden_states,
            state.context,
            state.time_mod,
            state.rope_freqs,
            block_context_mask,
        )

    # ================================================================
    # ABC: Three-step execution (3)
    # ================================================================

    def prepare(self, **kw) -> BlockLoopState:
        dit = self.dit
        motion_controller = self.motion_controller
        vace = self.vace
        latents = kw["latents"]
        timestep = kw["timestep"]
        context = kw["context"]
        context_mask = kw.get("context_mask")
        seq_lens = kw.get("seq_lens")
        clip_feature = kw.get("clip_feature")
        image_cond_latents = kw.get("y")
        vace_context = kw.get("vace_context")
        motion_bucket_id = kw.get("motion_bucket_id")
        control_camera_latents_input = kw.get("control_camera_latents_input")
        fuse_vae_embedding_in_latents = kw.get("fuse_vae_embedding_in_latents", False)
        num_clean_prefix_frames = kw.get("num_clean_prefix_frames", 0)
        use_gradient_checkpointing = kw.get("use_gradient_checkpointing", False)
        use_gradient_checkpointing_offload = kw.get("use_gradient_checkpointing_offload", False)
        force_per_token_t_mod = bool(kw.get("force_per_token_t_mod", False))

        time_embed, time_modulation = wan_dit_forward.build_time_modulation(
            dit,
            timestep,
            latents,
            patch_size=self._dit_patch_size,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            force_per_token_t_mod=force_per_token_t_mod,
            num_clean_prefix_frames=num_clean_prefix_frames,
            zero_clean_prefix_t_mod=bool(kw.get("zero_clean_prefix_t_mod", False)),
            has_first_frame_latents=kw.get("first_frame_latents") is not None,
        )

        if motion_bucket_id is not None and motion_controller is not None:
            motion_term = motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))  # (B, 6, dim)
            if time_modulation.dim() == 4:
                # Broadcast (B, 6, dim) across L; without the unsqueeze the add
                # right-aligns and aliases B onto L (mis-broadcasts when B == L).
                motion_term = motion_term.unsqueeze(1)  # (B, 1, 6, dim)
            time_modulation = time_modulation + motion_term
        context = dit.text_embedding(context)
        if context_mask is None:
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
            if context_mask.ndim != 2:
                raise ValueError(f"context_mask must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"context_mask shape must match context [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        hidden_states = latents
        if hidden_states.shape[0] != context.shape[0]:
            hidden_states = torch.concat([hidden_states] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        if image_cond_latents is not None and dit.require_vae_embedding:
            hidden_states = torch.cat([hidden_states, image_cond_latents], dim=1)
        if clip_feature is not None and dit.require_clip_embedding:
            clip_embdding = dit.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
            if context_mask is not None:
                clip_mask = torch.ones(
                    (context_mask.shape[0], clip_embdding.shape[1]),
                    dtype=torch.bool,
                    device=context_mask.device,
                )
                context_mask = torch.cat([clip_mask, context_mask], dim=1)

        hidden_states = dit.patchify(hidden_states, control_camera_latents_input)

        grid_frames, grid_height, grid_width = hidden_states.shape[2:]
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c").contiguous()
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:grid_frames]
                    .view(grid_frames, 1, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[1][:grid_height]
                    .view(1, grid_height, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[2][:grid_width]
                    .view(1, 1, grid_width, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                ],
                dim=-1,
            )
            .reshape(grid_frames * grid_height * grid_width, 1, -1)
            .to(hidden_states.device)
        )

        extras = {
            "dit": dit,
            "vace": vace,
            "time_embed": time_embed,  # Wan head time embedding; consumed in finalize()
        }
        vace_hints = None
        if vace_context is not None:
            vace_hints = vace(
                hidden_states,
                vace_context,
                context,
                time_modulation,
                freqs,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        return BlockLoopState(
            hidden_states=hidden_states,
            time_mod=time_modulation,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=grid_frames,
            grid_height=grid_height,
            grid_width=grid_width,
            vace_hints=vace_hints,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            extras=extras,
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        dit = state.extras["dit"]
        block = dit.blocks[block_id]
        attn_mask = state.extras.get("shared_attention_mask")
        context_mask = state.context_mask

        block_context_mask = (
            context_mask.unsqueeze(1).expand(-1, state.hidden_states.shape[1], -1) if context_mask is not None else None
        )
        if attn_mask is not None:
            if block_context_mask is None:
                block_context_mask = (
                    torch.ones(
                        (state.context.shape[0], state.context.shape[1]),
                        dtype=torch.bool,
                        device=state.context.device,
                    )
                    .unsqueeze(1)
                    .expand(-1, state.hidden_states.shape[1], -1)
                )
            state.hidden_states = self._run_wan_block(block_id, block, state, block_context_mask, attn_mask)
            wan_dit_forward.apply_post_block_residuals(block_id, state)
            return state

        state.hidden_states = self._run_wan_block(block_id, block, state, block_context_mask, None)

        wan_dit_forward.apply_post_block_residuals(block_id, state)
        return state

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """First half of a Wan DiT block (norm1 + AdaLN + Q/K/V + RoPE), up to
        the attention call. Lets DualSystemMoTDriver pull video-side Q/K/V before the
        mixed attention; pairs with :meth:`post_attn_at_layer`.
        """
        q, k, v, post_tuple = self.pre_attn_at_layer_for_compile(layer_id, state)
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_tuple
        block = state.extras["dit"].blocks[layer_id]
        post_state = {
            "block": block,
            "residual_x": residual_x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q, k, v, post_state

    def pre_attn_at_layer_for_compile(
        self, layer_id: int, state: BlockLoopState
    ) -> Tuple[Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
        """Compile-friendly Wan pre-attention half using a tensor tuple post-state."""
        block = state.extras["dit"].blocks[layer_id]

        time_modulation = state.time_mod
        has_seq = time_modulation.dim() == 4
        chunk_dim = 2 if has_seq else 1
        chunks = (
            block.modulation.to(dtype=time_modulation.dtype, device=time_modulation.device) + time_modulation
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            chunks = tuple(c.squeeze(2) for c in chunks)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        residual_x = state.hidden_states
        attn_input = modulate(block.norm1(state.hidden_states), shift_msa, scale_msa)

        self_attn = block.self_attn
        q = self_attn.norm_q(self_attn.q(attn_input))
        k = self_attn.norm_k(self_attn.k(attn_input))
        v = self_attn.v(attn_input)
        q = rope_apply(q, state.rope_freqs, self_attn.num_heads)
        k = rope_apply(k, state.rope_freqs, self_attn.num_heads)

        post_state = (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        return q, k, v, post_state

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Second half of a Wan DiT block: gate → cross-attn → FFN → VACE
        residuals. ``attn_out`` is the unprojected (pre ``self_attn.o``) attention
        output for the video slice of the joint mixed attention.
        """
        if isinstance(post_state, dict):
            post_state = (
                post_state["residual_x"],
                post_state["gate_msa"],
                post_state["shift_mlp"],
                post_state["scale_mlp"],
                post_state["gate_mlp"],
            )
        return self.post_attn_at_layer_for_compile(layer_id, state, attn_out, post_state)

    def post_attn_at_layer_for_compile(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: tuple[Tensor, ...]
    ) -> BlockLoopState:
        """Compile-friendly Wan post-attention half consuming a tensor tuple."""
        block = state.extras["dit"].blocks[layer_id]
        self_attn = block.self_attn
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

        hidden_states = block.gate(residual_x, gate_msa, self_attn.o(attn_out))
        context_mask = None
        if state.context_mask is not None:
            context_mask = state.context_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1).unsqueeze(1)
        hidden_states = hidden_states + block.cross_attn(
            block.norm3(hidden_states), state.context, ctx_mask=context_mask
        )
        mlp_input = modulate(block.norm2(hidden_states), shift_mlp, scale_mlp)
        hidden_states = block.gate(hidden_states, gate_mlp, block.ffn(mlp_input))
        state.hidden_states = hidden_states

        wan_dit_forward.apply_post_block_residuals(layer_id, state)
        return state

    def finalize(self, state: BlockLoopState):
        """Wan DiT head + unpatchify. Returns ``(B, z_dim, F, H, W)``."""
        dit = state.extras["dit"]
        head = dit.head
        time_embed = state.extras["time_embed"]
        head_time_embed = time_embed if time_embed.dim() == 3 else time_embed.unsqueeze(1)

        hidden_states = head(state.hidden_states, head_time_embed)

        hidden_states = dit.unpatchify(hidden_states, (state.grid_frames, state.grid_height, state.grid_width))
        return hidden_states

    # ================================================================
    # ABC: Action token injection (2)
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
        """Append action then optional state tokens (layout ``[video][action]
        [state]``). State tokens use independent 1D RoPE positions; action/state
        AdaLN t_mod comes from timestep only (DreamZero).
        """
        n_state = int(n_state or 0)
        batch_size = state.hidden_states.shape[0]
        appended_pieces = []
        if n_action:
            if action_tokens is None:
                raise ValueError("n_action > 0 requires action_tokens.")
            if action_tokens.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch in inject_shared_tokens: video batch={batch_size}, "
                    f"action batch={action_tokens.shape[0]}."
                )
            if action_tokens.shape[1] != n_action:
                raise ValueError(f"action_tokens length {action_tokens.shape[1]} does not match n_action={n_action}")
            if action_tokens.shape[2] != state.hidden_states.shape[2]:
                raise ValueError(
                    f"action_tokens dim {action_tokens.shape[2]} does not match video dim {state.hidden_states.shape[2]}"
                )
            appended_pieces.append(action_tokens.to(state.hidden_states.dtype))
        if n_state:
            if state_tokens is None:
                raise ValueError("n_state > 0 requires state_tokens.")
            if state_tokens.shape[0] != batch_size:
                raise ValueError(
                    f"Batch mismatch in inject_shared_tokens: video batch={batch_size}, "
                    f"state batch={state_tokens.shape[0]}."
                )
            if state_tokens.shape[1] != n_state:
                raise ValueError(f"state_tokens length {state_tokens.shape[1]} does not match n_state={n_state}")
            if state_tokens.shape[2] != state.hidden_states.shape[2]:
                raise ValueError(
                    f"state_tokens dim {state_tokens.shape[2]} does not match video dim {state.hidden_states.shape[2]}"
                )
            appended_pieces.append(state_tokens.to(state.hidden_states.dtype))
        appended = torch.cat(appended_pieces, dim=1)

        if wan_action_tokens.is_per_token_t_mod_active(state.time_mod) and timestep is None:
            raise ValueError("inject_shared_tokens requires `timestep` when per-token t_mod is active.")

        state.hidden_states = torch.cat([state.hidden_states, appended], dim=1)
        state.rope_freqs = wan_action_tokens.extend_freqs_with_shared_tokens(state.rope_freqs, n_action, n_state)
        if wan_action_tokens.is_per_token_t_mod_active(state.time_mod):
            tmod_pieces = []
            if n_action:
                tmod_pieces.append(
                    wan_action_tokens.build_action_t_mod(timestep, n_action, dit=self._dit, batch_size=batch_size)
                )
            if n_state:
                tmod_pieces.append(
                    wan_action_tokens.build_sample_t_mod(timestep, n_state, dit=self._dit, batch_size=batch_size)
                )
            state.time_mod = torch.cat([state.time_mod, *[p.to(state.time_mod.dtype) for p in tmod_pieces]], dim=1)
        return state

    def extract_shared_tokens(
        self,
        state: BlockLoopState,
        n_action: int,
        *,
        n_state: int = 0,
    ) -> Tuple[BlockLoopState, Tensor]:
        n_state = int(n_state or 0)
        n_tail = int(n_action) + n_state
        if n_action < 0 or n_tail <= 0 or n_tail >= state.hidden_states.shape[1]:
            raise ValueError(
                f"extract_shared_tokens called with n_action={n_action}, n_state={n_state} but state.hidden_states has "
                f"shape[1]={state.hidden_states.shape[1]}; expected n_action >= 0 and 0 < n_action + n_state < state.hidden_states.shape[1] "
                "(was inject_shared_tokens called first with the same lengths?)."
            )
        n_video = state.hidden_states.shape[1] - n_tail
        action_tokens = state.hidden_states[:, n_video : n_video + n_action, :]
        state.hidden_states = state.hidden_states[:, :n_video, :]
        state.rope_freqs = state.rope_freqs[:n_video]
        if state.time_mod.dim() == 4:
            state.time_mod = state.time_mod[:, :n_video, :, :]
        return state, action_tokens

    # ================================================================
    # IDM teacher-forcing branch merge/split
    # ================================================================

    def merge_idm_video_branches(self, noisy: BlockLoopState, cond: BlockLoopState) -> Tuple[BlockLoopState, int, int]:
        """Concatenate the IDM noisy + cond branches along the sequence axis.

        Wan's ``hidden_states`` is flat ``(B, L, D)`` with ``L == f·h·w``, so the
        token seq lengths equal ``hidden_states.shape[1]``. IDM teacher-forcing
        needs two different video timesteps inside one video-expert sequence, so
        the backbone must expose token-wise (4D) ``t_mod``.
        """
        import copy

        if noisy.time_mod.ndim != 4 or cond.time_mod.ndim != 4:
            raise ValueError(
                "IDM teacher-forcing requires token-wise video t_mod for noisy and cond branches; "
                "ensure the video backbone is running in separated-timestep/fused-first-frame mode."
            )
        if (noisy.grid_height, noisy.grid_width) != (cond.grid_height, cond.grid_width):
            raise ValueError(
                "IDM teacher-forcing requires noisy and cond video branches to share spatial token layout, "
                f"got noisy h/w={(noisy.grid_height, noisy.grid_width)} "
                f"and cond h/w={(cond.grid_height, cond.grid_width)}."
            )
        s_noisy = int(noisy.hidden_states.shape[1])
        s_cond = int(cond.hidden_states.shape[1])

        merged = copy.copy(noisy)
        merged.hidden_states = torch.cat([noisy.hidden_states, cond.hidden_states], dim=1)
        merged.rope_freqs = torch.cat([noisy.rope_freqs, cond.rope_freqs], dim=0)
        merged.time_mod = torch.cat([noisy.time_mod, cond.time_mod], dim=1)
        if noisy.vace_hints is not None or cond.vace_hints is not None:
            if noisy.vace_hints is None or cond.vace_hints is None:
                raise ValueError("IDM teacher-forcing requires both video branches to have VACE hints or neither.")
            if len(noisy.vace_hints) != len(cond.vace_hints):
                raise ValueError("IDM teacher-forcing VACE hint count mismatch between noisy and cond branches.")
            merged.vace_hints = [
                torch.cat([hint_noisy, hint_cond], dim=1)
                for hint_noisy, hint_cond in zip(noisy.vace_hints, cond.vace_hints)
            ]
        return merged, s_noisy, s_cond

    def split_idm_video_branches(
        self, merged: BlockLoopState, noisy: BlockLoopState, cond: BlockLoopState
    ) -> Tuple[BlockLoopState, BlockLoopState]:
        """Inverse of :meth:`merge_idm_video_branches` for Wan's flat sequence."""
        s_noisy = int(noisy.hidden_states.shape[1])
        noisy.hidden_states = merged.hidden_states[:, :s_noisy]
        cond.hidden_states = merged.hidden_states[:, s_noisy:]
        noisy.time_mod = merged.time_mod[:, :s_noisy]
        cond.time_mod = merged.time_mod[:, s_noisy:]
        return noisy, cond

    # ================================================================
    # ABC: Unified preprocessing (1)
    # ================================================================

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        """Unified train preprocessing: raw data (``frames``/``text``, optional
        ``vace_videos``/``ref_images`` in kw) → the denoising-loop input dict.
        """
        device = self.device
        dtype = self.dtype

        height, width, num_frames = check_resize_height_width(
            frames[0][0].size[1],
            frames[0][0].size[0],
            len(frames[0]),
            height_division_factor=self._height_division_factor,
            width_division_factor=self._width_division_factor,
            time_division_factor=self._time_division_factor,
            time_division_remainder=self._time_division_remainder,
        )

        batch_size = len(frames)
        context, seq_lens = wan_encode.encode_text(
            text, tokenizer=self._tokenizer, text_encoder=self.text_encoder, device=self.device
        )

        all_input_videos = []
        for clip_frames in frames:
            all_input_videos.append(
                wan_encode.preprocess_video(
                    clip_frames, encoder=self.video_encoder, dtype=self.dtype, device=self.device
                )
            )
        stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = wan_encode.encode_video(stacked_inputs, vae=self.vae, encoder=self.video_encoder)
        input_latents = input_latents.to(dtype=dtype, device=device)

        # Variant-specific first-frame / control conditioning lives in
        # ``self._variant``, so this method carries no per-variant ``if``.
        cond = self._variant.build_train_conditioning(
            self,
            input_latents=input_latents,
            frames=frames,
            ref_images=kw.get("ref_images"),
            vace_videos=kw.get("vace_videos"),
            stacked_inputs=stacked_inputs,
            B=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
            first_frame_image=kw.get("first_frame_image"),
        )

        return {
            "input_latents": input_latents,
            "context": context,
            "seq_lens": seq_lens,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "vace_context": cond.get("vace_context"),
            "fuse_vae_embedding_in_latents": cond.get("fuse_vae_embedding_in_latents", False),
            "num_clean_prefix_frames": cond.get("num_clean_prefix_frames", 0),
            "first_frame_latents": cond.get("first_frame_latents"),
            "clip_feature": cond.get("clip_feature"),
            "y": cond.get("y"),
        }

    # ================================================================
    # ABC: Sub-module access (1)
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        # Non-Module names (tokenizer / scheduler) resolve to None.
        mod = getattr(self, name, None)
        return mod if isinstance(mod, nn.Module) else None

    # ================================================================
    # ABC: Decoding (1)
    # ================================================================

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        video_tensor = wan_encode.decode_latents(
            latents, vae=self.vae, encoder=self.video_encoder, device=self.device, tiled=tiled
        )
        return wan_encode.latents_to_frames(video_tensor, encoder=self.video_encoder)

    # ================================================================
    # Self-contained checkpoint: specs into config + artifacts into dir
    # ================================================================

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's checkpoint slice self-contained in one pass:
        merge the Wan component + tokenizer specs into ``cfg`` and copy the Wan
        tokenizer into ``output_dir`` (single tokenizer-layout source, no
        duplication). The external-encoder subclass forwards to the encoder's
        own deploy-artifact hook so its side files land alongside.
        """
        from openwam.model.video_backbone.wan.component_specs import save_video_backbone_deploy_assets

        save_video_backbone_deploy_assets(output_dir, cfg)

    # ================================================================
    # ABC: Deploy-input preprocessing (override)
    # ================================================================

    def preprocess_input_for_inference(
        self,
        *,
        prompt: str,
        vace_video=None,
        first_frame_image=None,
        num_frames: int = 49,
        height: int = 384,
        width: int = 320,
        seed: int = 42,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tiled: bool = True,
        vace_cache: Optional[dict] = None,
        prompt_embed_cache: Optional[dict] = None,
        **kw,
    ) -> dict:
        """Build the inference denoising-loop input dict from explicit kwargs,
        via explicit backbone helpers (no unit-runner).

        Only the text embedding is cached (prompt-keyed, seed/dim-independent);
        noise/clip/y/vace_context/first_frame_latents are rebuilt every call.
        ``**kw`` swallows fields other backbones consume (CFG) that Wan
        ignores — Wan does no CFG at inference.
        """
        # Wan native tiling grid (other backbones may differ).
        tile_size = (30, 52)
        tile_stride = (15, 26)

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, shift=shift)

        # ShapeChecker: snap to a model-valid grid; noise / clip / y use these.
        height, width, num_frames = check_resize_height_width(
            height,
            width,
            num_frames,
            height_division_factor=self._height_division_factor,
            width_division_factor=self._width_division_factor,
            time_division_factor=self._time_division_factor,
            time_division_remainder=self._time_division_remainder,
        )

        context, seq_lens = wan_encode.encode_text_for_inference(
            prompt,
            vace_cache=vace_cache,
            prompt_embed_cache=prompt_embed_cache,
            tokenizer=self._tokenizer,
            text_encoder=self.text_encoder,
            device=self.device,
        )

        _DEFAULT_CAMERA_ORIGIN = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        )
        inputs_shared = {
            "input_image": None,
            "end_image": None,
            "input_video": None,
            "denoising_strength": 1.0,
            "control_video": None,
            "reference_image": None,
            "camera_control_direction": None,
            "camera_control_speed": 1 / 54,
            "camera_control_origin": _DEFAULT_CAMERA_ORIGIN,
            # vace_* slots cleared: native VACE's ref-prepend breaks our
            # T_lat == video-latent contract. VACE first-frame flows through
            # ``_build_vace_context_for_deploy`` below instead.
            "vace_video": None,
            "vace_video_mask": None,
            "vace_reference_image": None,
            "seed": seed,
            "rand_device": "cpu",
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "sigma_shift": shift,
            "motion_bucket_id": None,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "input_audio": None,
        }
        inputs_shared["context"] = context
        inputs_shared["seq_lens"] = seq_lens
        inputs_shared["prompt"] = prompt
        inputs_shared["num_inference_steps"] = num_inference_steps

        # Deploy ``input_video`` is always None, so ``latents`` == ``noise``
        # (same tensor under both keys, as base.generate expects).
        noise = wan_conditioning.build_deploy_noise(
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            rand_device="cpu",
            latent_spec=self._latent_spec,
            vae=self.vae,
            dtype=self.dtype,
            device=self.device,
        )
        inputs_shared["noise"] = noise
        inputs_shared["latents"] = noise

        # I2V first-frame: CLIP + VAE ``y``, each gated on the DiT flags.
        # ``resolve_i2v_input_image`` is None for non-I2V backbones.
        i2v_img = wan_conditioning.resolve_i2v_input_image(
            first_frame_image, dit=self._dit, is_ti2v=self._is_ti2v, has_vace=self._has_vace
        )
        inputs_shared["input_image"] = i2v_img
        if i2v_img is not None:
            clip_feature = wan_conditioning.build_deploy_i2v_clip(
                i2v_img,
                height=height,
                width=width,
                dit=self.dit,
                image_encoder=self.image_encoder,
                dtype=self.dtype,
                device=self.device,
            )
            if clip_feature is not None:
                inputs_shared["clip_feature"] = clip_feature
            image_cond_latents = wan_conditioning.build_deploy_i2v_y(
                i2v_img,
                num_frames=num_frames,
                height=height,
                width=width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                dit=self.dit,
                vae=self.vae,
                dtype=self.dtype,
                device=self.device,
            )
            if image_cond_latents is not None:
                inputs_shared["y"] = image_cond_latents

        if vace_cache is not None:
            vace_cache["populated"] = True
            vace_cache["prompt_key"] = prompt
            vace_cache["context"] = context
            vace_cache["seq_lens"] = seq_lens

        wan_conditioning.build_vace_context_for_deploy(
            inputs_shared,
            first_frame_image,
            vace_video,
            has_vace=self._has_vace,
            vae=self.vae,
            encoder=self.video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        wan_conditioning.finalize_ti2v_first_frame_latents(
            inputs_shared,
            first_frame_image,
            is_ti2v=self._is_ti2v,
            encoder=self.video_encoder,
            vae=self.vae,
            dtype=self.dtype,
            device=self.device,
        )
        return inputs_shared


class Wan22Ti2v(WanBase):
    """Wan2.2-TI2V-5B backbone with optional external-encoder VAE-IO routing.

    ``external_encoder`` is ``None`` on the default path so ``state_dict()``
    carries only ``vae.*`` keys; setting it activates external-encoder VAE-IO
    routing and aliases the encoder under ``"vae"``.
    """

    def __init__(self, holder, *, external_encoder=None, shift_video=None, text_dim: Optional[int] = None):
        """Internal constructor. Use ``from_pretrained()`` instead."""
        # Base sets self.video_encoder after nn.Module.__init__ (an nn.Module
        # encoder cannot be assigned before that), activating VAE-IO routing.
        super().__init__(holder, external_encoder=external_encoder, shift_video=shift_video, text_dim=text_dim)
        if external_encoder is not None:
            # Override the native (1,2,2)/4×/causal contract with the encoder's;
            # callers consult these attrs and never branch on the encoder.
            self._dit_patch_size = external_encoder.properties.dit_patch_size
            self._temporal_compression = int(external_encoder.properties.temporal_compression)
            self._causal_temporal = bool(external_encoder.properties.causal_temporal)

    @classmethod
    def from_pretrained(cls, source, *, external_encoder=None, text_dim: Optional[int] = None, **kw) -> "Wan22Ti2v":
        """Build a Wan22Ti2v from a source.

        Sources: ``DictConfig`` (full Hydra cfg → loader), ``str`` dir path /
        ``dict`` with ``model_path`` (lightweight build), else an already-built
        component holder. Construction returns a transient holder that
        ``__init__`` drains into the backbone.

        With ``external_encoder``: derive division factors from the encoder
        spec, release the native VAE, expose latent-shape metadata. See the
        inline comments.
        """
        from omegaconf import DictConfig

        # Skip materializing the native VAE (avoid ~1.5GB waste / a duplicate
        # VAE slot deploy has no weights for) on training-with-irreversible and
        # on deploy-with-ANY external encoder. Reversible-on-training keeps it,
        # needed for the step-(2) spec cross-check against ``v.z_dim`` etc.
        is_deploy = not isinstance(source, DictConfig)
        skip_native_vae = bool(
            external_encoder is not None and (is_deploy or not external_encoder.properties.pixel_decode)
        )

        holder = loader.build_holder(source, skip_native_vae=skip_native_vae, **kw)

        if external_encoder is not None:
            # (3) Division factors from the encoder spec, not a hardcoded ``* 2`` / Wan-VAE grid, else
            # ``check_resize_height_width`` rounds encoder-legal sizes to Wan's grid. Remainder is 1 iff causal.
            patch_size = external_encoder.properties.dit_patch_size
            holder.height_division_factor = external_encoder.properties.spatial_compression * patch_size[1]
            holder.width_division_factor = external_encoder.properties.spatial_compression * patch_size[2]
            holder.time_division_factor = external_encoder.properties.temporal_compression * patch_size[0]
            holder.time_division_remainder = 1 if external_encoder.properties.causal_temporal else 0

            # (4) Release the native VAE so state_dict keys don't double-count with the external encoder. print (not
            # logger.info) because arch init runs before the logger is wired up; rank-0 gated.
            holder.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if int(os.environ.get("RANK", 0)) == 0:
                print(
                    f"[Wan22Ti2v] native VAE released; "
                    f"external_encoder={type(external_encoder).__name__} "
                    f"(z_dim={external_encoder.properties.z_dim}, "
                    f"pixel_decode={external_encoder.properties.pixel_decode}, "
                    f"dit_patch_size={external_encoder.properties.dit_patch_size})",
                    flush=True,
                )

            # (5) Expose latent-shape metadata so deploy noise init reads it without the native VAE (now None).
            holder.latent_spec = external_encoder.properties

        # Resolve optional cfg-side ``shift_video`` here (not in __init__)
        # because the cfg shape depends on the ``source`` type.
        shift_video_cfg = loader.resolve_cfg_shift_video(source)

        return cls(holder, external_encoder=external_encoder, shift_video=shift_video_cfg, text_dim=text_dim)

    # ================================================================
    # External-encoder-aware overrides
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        if name == "vae" and self._uses_external_encoder:
            return self.video_encoder
        return super().get_submodule(name)

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        if self._uses_external_encoder and not self.video_encoder.properties.pixel_decode:
            raise NotImplementedError(
                f"decode_video on irreversible encoder ({type(self.video_encoder).__name__}; "
                "properties.pixel_decode=False). Pass decode_video=False to generate() to "
                "retrieve raw latents, or train a separate pixel decoder."
            )
        return super().decode_video(latents, tiled=tiled)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Wan deploy assets, then forward to the external encoder's own
        deploy-artifact hook so its side files (e.g. V-JEPA ``manifest.json``)
        land alongside.
        """
        super().save_deploy_assets(output_dir, cfg)
        if self.video_encoder is not None:
            self.video_encoder.save_deploy_assets(output_dir, cfg)


class Wan21(WanBase):
    """Wan2.1 I2V / VACE backbone — native VAE only (no external encoder)."""

    @classmethod
    def from_pretrained(cls, source, *, text_dim: Optional[int] = None, **kw) -> "Wan21":
        """Build a Wan21 from a source (see :func:`wan.loader.build_holder`)."""
        holder = loader.build_holder(source, **kw)
        return cls(holder, shift_video=loader.resolve_cfg_shift_video(source), text_dim=text_dim)


__all__ = ["WanBase", "Wan22Ti2v", "Wan21"]
