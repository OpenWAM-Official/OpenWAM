"""Wan-specific implementation of :class:`VideoBackbone`.

Lives outside ``wan/`` to keep the ``wan/`` package focused on Wan-internal
implementation (DiT, VACE, SP, etc.). This module
sits at the boundary between architecture-driven action injection and
Wan's video forward.

The three-step interface (``prepare`` / ``run_block`` / ``finalize``) is a
faithful decomposition of ``model_fn_wan_video`` in ``wan/pipeline.py``.
The original function is left untouched — consistency tests verify that
both paths produce identical outputs.

``WanVideoBackbone`` owns the entire Wan pipeline (DiT, VAE, text encoder,
tokenizer, VACE, etc.) as a private ``_pipe`` attribute. External code
accesses pipeline capabilities through the :class:`VideoBackbone` ABC
methods — ``_pipe`` is never exposed.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from openwam.model.compile_options import cfg_get, torch_compile_kwargs
from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone

if TYPE_CHECKING:
    from openwam.model.inference_inputs import InferenceInputs
from openwam.model.video_backbone.wan.dit import modulate, rope_apply, sinusoidal_embedding_1d
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward

logger = logging.getLogger(__name__)


class WanVideoBackbone(VideoBackbone):
    """Wraps a Wan pipeline to expose the VideoBackbone interface.

    Owns the entire pipeline (DiT, VAE, text encoder, tokenizer, VACE)
    as a private ``_pipe``. External code never touches ``_pipe`` directly.

    Construction: use ``from_pretrained(source)`` for all paths.
    """

    # ================================================================
    # Construction
    # ================================================================

    def __init__(self, pipe):
        """Internal constructor. Use ``from_pretrained()`` instead."""
        super().__init__()
        self._pipe = pipe
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, source, **kw) -> WanVideoBackbone:
        """Build a WanVideoBackbone from various source types.

        Supported sources:
        - ``DictConfig``: full Hydra config → ``build_training_pipeline(cfg)``
        - ``str`` directory path: auto-discover model files → lightweight build
        - ``dict`` with ``video_backbone.model_path``: lightweight build from model dir
        - anything else: treated as an already-built pipe object
        """
        from omegaconf import DictConfig

        if isinstance(source, DictConfig):
            from openwam.model.video_backbone.wan.pipeline_builder import build_training_pipeline

            pipe = build_training_pipeline(source)
        elif isinstance(source, str):
            if os.path.isdir(source):
                pipe = cls._build_pipe_from_model_path(source, device=kw.get("device", "cpu"))
            else:
                raise ValueError(f"from_pretrained(str) expects a directory path, got: {source!r}.")
        elif isinstance(source, dict):
            vb_cfg = source.get("video_backbone", source)
            if isinstance(vb_cfg, dict) and "components" in vb_cfg:
                pipe = cls._build_pipe_from_components(
                    vb_cfg["components"],
                    tokenizer=vb_cfg.get("tokenizer"),
                    device=kw.get("device", "cpu"),
                    ckpt_dir=kw.get("ckpt_dir"),
                    model_path=vb_cfg.get("model_path"),
                )
            else:
                model_path = (
                    vb_cfg.get("model_path") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "model_path", None)
                )
                if model_path is None:
                    raise ValueError(
                        "dict source must contain 'video_backbone.components' or 'video_backbone.model_path'"
                    )
                pipe = cls._build_pipe_from_model_path(str(model_path), device=kw.get("device", "cpu"))
        else:
            pipe = source
        return cls(pipe)

    # ================================================================
    # Internal properties
    # ================================================================

    @property
    def _dit(self):
        return self._pipe.dit

    @property
    def _has_vace(self) -> bool:
        return getattr(self._pipe, "vace", None) is not None

    @property
    def _is_ti2v(self) -> bool:
        return bool(getattr(self._dit, "fuse_vae_embedding_in_latents", False))

    @property
    def needs_first_frame_skip(self) -> bool:
        """Wan TI2V / VACE / I2V all keep ``latent[0]`` as a clean conditioning frame.

        TI2V / VACE additionally surface that as ``first_frame_latents`` in
        the inputs dict, so the per-batch signal in
        :meth:`BaseWAMArchitecture.preprocess` would already cover them.
        I2V wires the image conditioning through the ``y`` channel and does
        NOT set ``first_frame_latents``, so the property is the only signal
        that lets the loss-side mask trim ``latent[0]`` for I2V. Future Wan
        T2V configs (no TI2V / VACE / image input) correctly fall through
        to ``False`` so ``latent[0]`` enters the loss as a predicted frame.
        """
        has_image_input = bool(getattr(self._dit, "has_image_input", False))
        return self._is_ti2v or self._has_vace or has_image_input

    @property
    def _freq_dim(self) -> int:
        return int(self._dit.freq_dim)

    @property
    def _use_unified_sequence_parallel(self) -> bool:
        return bool(getattr(self._pipe, "use_unified_sequence_parallel", False))

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
        return self._pipe.scheduler

    @property
    def submodule_names(self) -> list[str]:
        names = []
        for name in ("dit", "vace", "text_encoder", "vae", "image_encoder"):
            if getattr(self._pipe, name, None) is not None:
                names.append(name)
        return names

    @property
    def num_heads(self) -> int:
        return int(self._dit.blocks[0].num_heads)

    @property
    def head_dim(self) -> int:
        return int(self._dit.dim) // self.num_heads

    @property
    def video_attention_mask_mode(self) -> str:
        """Video self-attention mask mode used by joint MoT mask construction.

        Three modes mirror FastWAM's ``WanVideoDiT.video_attention_mask_mode``
        (see [wan_video_dit.py:473-507](references/FastWAM/src/fastwam/models/wan22/wan_video_dit.py#L473)):

        - ``bidirectional``: full v↔v coupling (default).
        - ``per_frame_causal``: each frame's tokens may only attend to its own
          frame and earlier frames (token-level causal block-diagonal).
        - ``first_frame_causal``: the first-frame tokens see only themselves;
          all later frames see the full video. FastWAM-Joint default.

        Sourced from the underlying Wan DiT when available, otherwise from the
        ``video_attention_mask_mode`` attribute set on this backbone (default
        ``bidirectional`` for back-compat).
        """
        explicit = getattr(self, "_video_attention_mask_mode", None)
        if explicit is not None:
            return explicit
        return getattr(self._dit, "video_attention_mask_mode", "bidirectional")

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = mode

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the video↔video block of the joint MoT attention mask.

        ``True`` means "attend to". Mirrors FastWAM's
        :meth:`WanVideoDiT.build_video_to_video_mask` so MoTJointDriver and
        FastWAM-Joint produce the same mask layout.
        """
        if video_seq_len <= 0:
            raise ValueError(f"video_seq_len must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"video_tokens_per_frame must be positive, got {video_tokens_per_frame}")

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
            # First-frame query rows can attend only to the first-frame keys
            # (they don't peek at the rest of the video). All later rows are
            # left at True (= see everything).
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(
            f"Unsupported video_attention_mask_mode '{mode}'. "
            "Choose from: bidirectional, per_frame_causal, first_frame_causal."
        )

    # ================================================================
    # ABC: Three-step execution (3)
    # ================================================================

    def prepare(self, **kw) -> BlockLoopState:
        dit = self._pipe.dit
        motion_controller = getattr(self._pipe, "motion_controller", None)
        vace = getattr(self._pipe, "vace", None)
        latents = kw["latents"]
        timestep = kw["timestep"]
        context = kw["context"]
        context_mask = kw.get("context_mask")
        seq_lens = kw.get("seq_lens")
        clip_feature = kw.get("clip_feature")
        y = kw.get("y")
        vace_context = kw.get("vace_context")
        vace_scale = kw.get("vace_scale", 1.0)
        use_usp = kw.get("use_unified_sequence_parallel", self._use_unified_sequence_parallel)
        motion_bucket_id = kw.get("motion_bucket_id")
        control_camera_latents_input = kw.get("control_camera_latents_input")
        fuse_vae_embedding_in_latents = kw.get("fuse_vae_embedding_in_latents", False)
        num_clean_prefix_frames = kw.get("num_clean_prefix_frames", 0)
        use_gradient_checkpointing = kw.get("use_gradient_checkpointing", False)
        use_gradient_checkpointing_offload = kw.get("use_gradient_checkpointing_offload", False)
        force_per_token_t_mod = bool(kw.get("force_per_token_t_mod", False))

        if use_usp:
            import torch.distributed as dist
            from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size

        if dit.seperated_timestep and fuse_vae_embedding_in_latents:
            batch_size = latents.shape[0]
            num_clean = max(num_clean_prefix_frames, 1)
            f_lat = latents.shape[2]
            tokens_per_frame = latents.shape[3] * latents.shape[4] // 4
            token_timesteps = torch.ones(
                batch_size, f_lat, tokens_per_frame, dtype=latents.dtype, device=latents.device
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, :num_clean, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            t_emb = sinusoidal_embedding_1d(dit.freq_dim, token_timesteps.reshape(-1))
            t = dit.time_embedding(t_emb.to(latents.dtype)).reshape(batch_size, -1, dit.dim)
            if use_usp and dist.is_initialized() and dist.get_world_size() > 1:
                t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
                t_chunks = [
                    torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0)
                    for chunk in t_chunks
                ]
                t = t_chunks[get_sequence_parallel_rank()]
            t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
        elif force_per_token_t_mod:
            # Non-TI2V backbones under joint-attention / shared_backbone need 4D
            # t_mod (one of: ``inject_shared_tokens`` extending the residual with
            # per-token action/state entries; ``IDMMoTDriver`` concatenating
            # noisy + cond video sequences with different timesteps). Compute the
            # time embedding once on (B,) and broadcast to (B, L, dim) — running
            # the MLP per token would repeat the same Linear/SiLU/Linear stack
            # L times (L ≈ 4680 for VACE-1.3B, ≈18720 for I2V-14B-480P).
            batch_size = latents.shape[0]
            f_lat = latents.shape[2]
            tokens_per_frame = latents.shape[3] * latents.shape[4] // 4
            L = f_lat * tokens_per_frame
            t_base = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))  # (B, dim)
            t = t_base.unsqueeze(1).expand(batch_size, L, -1).contiguous()

            # Optional clean-prefix alignment. The TI2V branch above pins the
            # first ``num_clean`` frames' timesteps to 0 so the model receives a
            # "this frame is the clean ref" signal that matches the latent-side
            # ``first_frame_latents`` replacement done in ``base.compute_loss``.
            # Other Wan backbones (VACE, I2V) historically lacked this in the
            # broadcast path — the residual data was clean but t_mod still
            # carried the sampled timestep. When the caller opts in via
            # ``zero_clean_prefix_t_mod=True`` AND a clean prefix is actually
            # present, we mirror TI2V's behavior by overwriting the first
            # ``num_clean`` frames' time embedding with ``time_embedding(0)``.
            # Mathematically equivalent to TI2V's per-token path (the latter
            # builds an (B*L,) timestep vector with prefix=0 before embedding);
            # we do the same overwrite at the embedding layer instead so the
            # per-token MLP stays a single (B, dim) call. ``num_clean`` mirrors
            # the TI2V branch's ``max(num_clean_prefix_frames, 1)`` fallback so
            # callers can rely on ``first_frame_latents`` alone (with
            # ``num_clean_prefix_frames=0``) to trigger the prefix.
            zero_clean_prefix = bool(kw.get("zero_clean_prefix_t_mod", False))
            has_clean_ref = num_clean_prefix_frames > 0 or kw.get("first_frame_latents") is not None
            if zero_clean_prefix and has_clean_ref:
                num_clean = max(num_clean_prefix_frames, 1)
                zero_ts = torch.zeros_like(timestep)
                t_zero_base = dit.time_embedding(
                    sinusoidal_embedding_1d(dit.freq_dim, zero_ts).to(latents.dtype)
                )  # (B, dim)
                t = t.view(batch_size, f_lat, tokens_per_frame, -1)
                t[:, :num_clean] = t_zero_base.view(batch_size, 1, 1, -1)
                t = t.reshape(batch_size, L, -1)

            if use_usp and dist.is_initialized() and dist.get_world_size() > 1:
                # NOTE: known pre-existing limitation — when ``vace_context``
                # is set (VACE backbone) and USP is enabled, the VACE hint
                # generator below receives full-sequence ``x``/``vace_context``
                # but the rank-local ``t_mod``, producing a shape mismatch.
                # OpenWAM training does not enable USP today
                # (``pipe.use_unified_sequence_parallel`` defaults to False),
                # so this codepath is dormant; out of scope for PR#56. Same
                # caveat applies to the TI2V branch above.
                t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
                t_chunks = [
                    torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0)
                    for chunk in t_chunks
                ]
                t = t_chunks[get_sequence_parallel_rank()]
            t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
        else:
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        if motion_bucket_id is not None and motion_controller is not None:
            motion_term = motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))  # (B, 6, dim)
            if t_mod.dim() == 4:
                # Broadcast the (B, 6, dim) motion term across the L tokens of a
                # 4D t_mod. Without the unsqueeze, ``(B, 6, dim) + (B, L, 6, dim)``
                # right-aligns and aliases B onto L, which silently mis-broadcasts
                # when B == L (per-batch motion id, per-token t_mod) and shape-errors
                # when B != L. Latent bug exposed once non-TI2V archs opt into 4D
                # t_mod via ``force_per_token_t_mod=True``.
                motion_term = motion_term.unsqueeze(1)  # (B, 1, 6, dim)
            t_mod = t_mod + motion_term
        context = dit.text_embedding(context)
        if context_mask is None:
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
            else:
                context_mask = None
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
            if context_mask.ndim != 2:
                raise ValueError(f"context_mask must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"context_mask shape must match context [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        x = latents
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)
        if timestep.shape[0] != context.shape[0]:
            timestep = torch.concat([timestep] * context.shape[0], dim=0)

        if y is not None and dit.require_vae_embedding:
            x = torch.cat([x, y], dim=1)
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

        x = dit.patchify(x, control_camera_latents_input)

        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

        extras = {}
        extras["dit"] = dit
        extras["vace"] = vace
        extras["use_usp"] = use_usp
        vace_hints = None
        if vace_context is not None:
            vace_hints = vace(
                x,
                vace_context,
                context,
                t_mod,
                freqs,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        sp_pad_shape = 0
        if use_usp:
            if dist.is_initialized() and dist.get_world_size() > 1:
                chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
                sp_pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
                chunks = [
                    torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0)
                    for chunk in chunks
                ]
                x = chunks[get_sequence_parallel_rank()]

        return BlockLoopState(
            x=x,
            t_mod=t_mod,
            freqs=freqs,
            context=context,
            context_mask=context_mask,
            f=f,
            h=h,
            w=w,
            t=t,
            vace_hints=vace_hints,
            vace_scale=vace_scale,
            sp_pad_shape=sp_pad_shape,
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
            context_mask.unsqueeze(1).expand(-1, state.x.shape[1], -1) if context_mask is not None else None
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
                    .expand(-1, state.x.shape[1], -1)
                )
            state.x = gradient_checkpoint_forward(
                block,
                state.use_gradient_checkpointing,
                state.use_gradient_checkpointing_offload,
                state.x,
                state.context,
                state.t_mod,
                state.freqs,
                block_context_mask,
                attn_mask,
            )
            self._apply_post_block_residuals(block_id, state)
            return state

        state.x = gradient_checkpoint_forward(
            block,
            state.use_gradient_checkpointing,
            state.use_gradient_checkpointing_offload,
            state.x,
            state.context,
            state.t_mod,
            state.freqs,
            block_context_mask,
        )

        self._apply_post_block_residuals(block_id, state)
        return state

    def _apply_post_block_residuals(self, block_id: int, state: BlockLoopState) -> None:
        """Apply post-block residuals (VACE hint) to ``state.x``.

        Shared by :meth:`run_block` and :meth:`post_attn_at_layer` so the joint
        self-attention path picks up VACE without duplicating the residual
        logic. Mutates ``state.x`` in place.
        """
        vace = state.extras.get("vace")
        use_usp = state.extras.get("use_usp", False)

        if state.vace_hints is not None and vace is not None and block_id in vace.vace_layers_mapping:
            current_vace_hint = state.vace_hints[vace.vace_layers_mapping[block_id]]
            if use_usp:
                import torch.distributed as dist
                from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size

                if dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[
                        get_sequence_parallel_rank()
                    ]
                    target_len = state.x.shape[1]
                    if current_vace_hint.shape[1] < target_len:
                        current_vace_hint = torch.nn.functional.pad(
                            current_vace_hint,
                            (0, 0, 0, target_len - current_vace_hint.shape[1]),
                            value=0,
                        )
            vace_len = current_vace_hint.shape[1]
            if state.x.shape[1] == vace_len:
                # dual_system / video-only path: hint length matches the full
                # token sequence, apply the residual to everything.
                state.x = state.x + current_vace_hint * state.vace_scale
            elif state.x.shape[1] > vace_len:
                # shared_backbone path: state.x has been extended with
                # action/state tokens. VACE residuals only apply to the leading
                # video slice; action/state tokens still get the VACE signal
                # indirectly through self-attention (both 'joint' and
                # 'bidirectional' modes route action→video).
                video_slice = state.x[:, :vace_len] + current_vace_hint * state.vace_scale
                state.x = torch.cat([video_slice, state.x[:, vace_len:]], dim=1)
            else:
                # Defensive: no legitimate path makes state.x shorter than the
                # VACE hint. If we ever hit this, something upstream broke the
                # video-token-count invariant — investigate before patching.
                raise ValueError(
                    f"_apply_post_block_residuals: state.x.shape[1]={state.x.shape[1]} "
                    f"< vace_hint.shape[1]={vace_len} at block {block_id}; "
                    "this is unreachable under dual_system or shared_backbone today—"
                    "investigate the upstream caller before patching this branch."
                )

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """First half of a Wan DiT block: norm1 + AdaLN modulate + Q/K/V + RoPE.

        Faithful split of :meth:`DiTBlock.forward` (in ``wan/dit.py``) up to the
        attention call. Used by :class:`MoTJointDriver` to pull video-side
        Q/K/V before the mixed attention. Pairs with :meth:`post_attn_at_layer`.
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

        t_mod = state.t_mod
        has_seq = t_mod.dim() == 4
        chunk_dim = 2 if has_seq else 1
        chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            chunks = tuple(c.squeeze(2) for c in chunks)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        residual_x = state.x
        attn_input = modulate(block.norm1(state.x), shift_msa, scale_msa)

        sa = block.self_attn
        q = sa.norm_q(sa.q(attn_input))
        k = sa.norm_k(sa.k(attn_input))
        v = sa.v(attn_input)
        q = rope_apply(q, state.freqs, sa.num_heads)
        k = rope_apply(k, state.freqs, sa.num_heads)

        post_state = (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        return q, k, v, post_state

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Second half of a Wan DiT block: gate(residual, self_attn.o(attn_out))
        → cross-attn (text context) → FFN → VACE residuals.

        ``attn_out`` is the unprojected attention output (pre ``self_attn.o``)
        for the *video* slice of the joint mixed attention; this method finishes
        applying the block and the standard post-block residuals.
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
        sa = block.self_attn
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

        x = block.gate(residual_x, gate_msa, sa.o(attn_out))
        context_mask = None
        if state.context_mask is not None:
            context_mask = state.context_mask.unsqueeze(1).expand(-1, x.shape[1], -1).unsqueeze(1)
        x = x + block.cross_attn(block.norm3(x), state.context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        state.x = x

        self._apply_post_block_residuals(layer_id, state)
        return state

    def finalize(self, state: BlockLoopState) -> Tensor:
        dit = state.extras["dit"]
        use_usp = state.extras.get("use_usp", False)

        t_head = state.t if state.t.dim() == 3 else state.t.unsqueeze(1)
        x = dit.head(state.x, t_head)

        if use_usp:
            import torch.distributed as dist
            from xfuser.core.distributed import get_sp_group

            if dist.is_initialized() and dist.get_world_size() > 1:
                x = get_sp_group().all_gather(x, dim=1)
                if state.sp_pad_shape > 0:
                    x = x[:, : -state.sp_pad_shape]

        x = dit.unpatchify(x, (state.f, state.h, state.w))
        return x

    # ================================================================
    # ABC: Action token injection (2)
    # ================================================================

    def inject_action_tokens(
        self,
        state: BlockLoopState,
        action_tokens: Tensor,
        n_action: int,
        *,
        timestep: Optional[Tensor] = None,
    ) -> BlockLoopState:
        return self.inject_shared_tokens(
            state,
            action_tokens,
            n_action,
            timestep=timestep,
        )

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
        """Append action tokens followed by optional state tokens.

        SharedBackbone layout is ``[video][action][state]``. State tokens use
        independent 1D RoPE positions and the same sample-level action timestep
        as the action tokens. Following DreamZero, action/state AdaLN t_mod is
        generated from timestep only.
        """
        n_state = int(n_state or 0)
        if n_state < 0:
            raise ValueError(f"n_state must be non-negative, got {n_state}")
        if n_action < 0:
            raise ValueError(f"n_action must be non-negative, got {n_action}")
        if n_action + n_state <= 0:
            raise ValueError("inject_shared_tokens requires at least one action or state token.")
        batch_size = state.x.shape[0]
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
            if action_tokens.shape[2] != state.x.shape[2]:
                raise ValueError(
                    f"action_tokens dim {action_tokens.shape[2]} does not match video dim {state.x.shape[2]}"
                )
            appended_pieces.append(action_tokens.to(state.x.dtype))
        elif action_tokens is not None and action_tokens.shape[1] != 0:
            raise ValueError("action_tokens were provided but n_action=0.")
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
            if state_tokens.shape[2] != state.x.shape[2]:
                raise ValueError(
                    f"state_tokens dim {state_tokens.shape[2]} does not match video dim {state.x.shape[2]}"
                )
            appended_pieces.append(state_tokens.to(state.x.dtype))
        else:
            if state_tokens is not None and state_tokens.shape[1] != 0:
                raise ValueError("state_tokens were provided but n_state=0.")
        appended = torch.cat(appended_pieces, dim=1)

        if self._is_per_token_t_mod_active(state) and timestep is None:
            raise ValueError("inject_shared_tokens requires `timestep` when per-token t_mod is active.")

        state.x = torch.cat([state.x, appended], dim=1)
        state.freqs = self._extend_freqs_with_shared_tokens(state.freqs, n_action, n_state)
        if self._is_per_token_t_mod_active(state):
            tmod_pieces = []
            if n_action:
                tmod_pieces.append(self._build_action_t_mod(timestep, n_action, batch_size=batch_size))
            if n_state:
                tmod_pieces.append(self._build_sample_t_mod(timestep, n_state, batch_size=batch_size))
            state.t_mod = torch.cat([state.t_mod, *[p.to(state.t_mod.dtype) for p in tmod_pieces]], dim=1)
        return state

    def extract_action_tokens(
        self,
        state: BlockLoopState,
        n_action: int,
    ) -> Tuple[BlockLoopState, Tensor]:
        return self.extract_shared_tokens(state, n_action, n_state=0)

    def extract_shared_tokens(
        self,
        state: BlockLoopState,
        n_action: int,
        *,
        n_state: int = 0,
    ) -> Tuple[BlockLoopState, Tensor]:
        n_state = int(n_state or 0)
        n_tail = int(n_action) + n_state
        if n_action < 0 or n_tail <= 0 or n_tail >= state.x.shape[1]:
            raise ValueError(
                f"extract_shared_tokens called with n_action={n_action}, n_state={n_state} but state.x has "
                f"shape[1]={state.x.shape[1]}; expected n_action >= 0 and 0 < n_action + n_state < state.x.shape[1] "
                "(was inject_shared_tokens called first with the same lengths?)."
            )
        n_video = state.x.shape[1] - n_tail
        action_tokens = state.x[:, n_video : n_video + n_action, :]
        state.x = state.x[:, :n_video, :]
        state.freqs = state.freqs[:n_video]
        if state.t_mod.dim() == 4:
            state.t_mod = state.t_mod[:, :n_video, :, :]
        return state, action_tokens

    # ================================================================
    # ABC: Unified preprocessing (1)
    # ================================================================

    def preprocess_input(self, *, frames=None, text=None, **kw) -> dict:
        """Unified preprocessing: raw data → tensors ready for denoising loop.

        Args:
            frames: List of video clips (each a list of PIL Images), one per batch sample.
            text: List of text prompts.
            vace_videos: List of VACE video clips (each a list of PIL Images or None).
            ref_images: List of reference image clips (each a list of PIL Images or None).

        Returns:
            Dict with: input_latents, context, seq_lens, height, width, num_frames,
            vace_context (optional), first_frame_latents (optional),
            fuse_vae_embedding_in_latents, num_clean_prefix_frames.
        """
        import torch.nn.functional as F

        device = self.device
        dtype = self.dtype

        height, width, num_frames = self._check_resize(frames[0][0].size[1], frames[0][0].size[0], len(frames[0]))

        B = len(frames)
        context, seq_lens = self._encode_text(text)

        all_input_videos = []
        for clip_frames in frames:
            all_input_videos.append(self._preprocess_video(clip_frames))
        stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = self._encode_video(stacked_inputs).to(dtype=dtype, device=device)

        vace_videos = kw.get("vace_videos")
        ref_images = kw.get("ref_images")

        has_ref = ref_images is not None and ref_images[0] is not None
        has_image_input = bool(getattr(self._dit, "has_image_input", False))

        # Three-way ref-image handling, keyed off backbone identity:
        #   - I2V  (has_image_input=True): first-frame rides on clip_feature + y
        #     (channel-axis concat in prepare()); never prepends to input_latents.
        #   - TI2V (_is_ti2v=True): first_frame_latents = input_latents[:, :, 0:1]
        #     (FastWAM-style; no prepend, no second VAE encode).
        #   - VACE: same shape contract as TI2V (no prepend, first_frame_latents =
        #     input_latents[:, :, 0:1]). vace_context is built at T_lat (matching
        #     input_latents) and its frame-0 carries the ref image: the inactive
        #     channel slot at t=0 is overwritten with input_latents[:, :, 0:1]
        #     and the mask at t=0 is zeroed (so the VACE module treats frame 0 as
        #     a known reference, not a generation target). See ``_inject_vace_ref_frame``.

        vace_context = None
        if self._has_vace:
            all_vace = []
            for i in range(B):
                vv = vace_videos[i] if vace_videos is not None else None
                if vv is not None:
                    all_vace.append(self._preprocess_video(vv))
                else:
                    all_vace.append(torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device))
            stacked_vace = torch.cat(all_vace, dim=0)
            reactive_latents = self._encode_video(stacked_vace).to(dtype=dtype, device=device)
            single_zero = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            inactive_latent = self._encode_video(single_zero).to(dtype=dtype, device=device)
            # expand returns a zero-stride view; the subsequent torch.cat
            # allocates new storage and copies, so no aliasing escapes here.
            inactive_latents = inactive_latent.expand(B, -1, -1, -1, -1)
            vace_video_latents = torch.cat([inactive_latents, reactive_latents], dim=1)

            vace_mask = torch.ones(B, 1, num_frames, height, width, dtype=dtype, device=device)
            vace_mask_latents = rearrange(vace_mask[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=8, Q=8)
            T_lat = (vace_mask_latents.shape[2] + 3) // 4
            vace_mask_latents = F.interpolate(
                vace_mask_latents,
                size=(T_lat, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
                mode="nearest-exact",
            )

            vace_context = torch.cat([vace_video_latents, vace_mask_latents], dim=1)
            if has_ref:
                # Inject ref (= GT frame 0) into vace_context's frame 0. This keeps
                # ``vace_context.shape[2] == input_latents.shape[2]`` while preserving
                # the "ref-image conditions VACE" semantic that PR#19 dropped.
                vace_context = self._inject_vace_ref_frame(vace_context, input_latents[:, :, 0:1])

        # ---------------- I2V (clip_feature + y) ----------------
        # Three-way mutually exclusive condition pipelines, keyed off
        # has_image_input so VACE's require_vae_embedding=True does not
        # accidentally trigger I2V y construction:
        #   - TI2V (wan22_ti2v_5b):       _is_ti2v=True,  has_image_input=False
        #   - VACE (wan21_vace_1_3b):     _has_vace=True, has_image_input=False
        #   - I2V  (wan21_i2v_14b_480p):  has_image_input=True
        needs_clip = (
            has_image_input
            and bool(getattr(self._dit, "require_clip_embedding", False))
            and self._pipe.image_encoder is not None
        )
        needs_y = has_image_input and bool(getattr(self._dit, "require_vae_embedding", False))

        clip_feature = None
        y = None
        if needs_clip or needs_y:
            # I2V conditioning image source priority:
            #   1. kw["first_frame_image"]: explicit override (deploy may pass).
            #   2. kw["ref_images"]: training-time path — base.py:530 always
            #      collects sample["first_frame_image"] into ref_images=...
            #   3. frames[i][0]: fallback to first frame of the GT video clip.
            first_frame_image = kw.get("first_frame_image")
            if first_frame_image is not None and not isinstance(first_frame_image, list):
                first_frame_image = [first_frame_image] * B
            if first_frame_image is None and ref_images is not None:
                first_frame_image = []
                for ref in ref_images:
                    if isinstance(ref, list):
                        first_frame_image.append(ref[0])
                    else:
                        first_frame_image.append(ref)
            if first_frame_image is None:
                first_frame_image = [clip[0] for clip in frames]
            if len(first_frame_image) != B:
                raise ValueError(
                    f"first_frame_image batch ({len(first_frame_image)}) != frames batch ({B})"
                )

            if needs_clip:
                clip_pieces = []
                for img in first_frame_image:
                    img_t = self._pipe.preprocess_image(img.resize((width, height))).to(device)
                    clip_pieces.append(self._pipe.image_encoder.encode_image([img_t]))
                clip_feature = torch.cat(clip_pieces, dim=0).to(dtype=dtype, device=device)

            if needs_y:
                y = self._build_i2v_y(
                    first_frame_image=first_frame_image,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    device=device,
                    dtype=dtype,
                )

        # TI2V/VACE first-frame conditioning: extract from position 0 of the
        # already-encoded video latents (no separate VAE call, no prepend).
        # Aligned with FastWAM / main PR#19. ``base._add_noise_and_pred`` will
        # clean-replace ``latents[:, :, 0:1]`` with this on every step so the
        # DiT sees [clean ref, noisy 1..T_lat-1] just like TI2V. ``n_skip`` in
        # ``_compute_video_loss`` is then ``num_clean_prefix(0) + 1 = 1``,
        # matching the ``T_lat - 1`` tail mask shape emitted by
        # ``downsample_video_mask_to_latent``.
        # ``fuse_vae_embedding_in_latents`` stays gated on ``_is_ti2v`` —
        # only TI2V's DiT has the ``seperated_timestep`` path that consumes it.
        first_frame_latents = None
        num_clean_prefix = 0
        if has_ref and not has_image_input and (self._is_ti2v or self._has_vace):
            first_frame_latents = input_latents[:, :, 0:1].clone()

        return {
            "input_latents": input_latents,
            "context": context,
            "seq_lens": seq_lens,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "vace_context": vace_context,
            "vace_scale": 1.0,
            "fuse_vae_embedding_in_latents": self._is_ti2v and has_ref,
            "num_clean_prefix_frames": num_clean_prefix,
            "first_frame_latents": first_frame_latents,
            "clip_feature": clip_feature,
            "y": y,
        }

    # ================================================================
    # ABC: Sub-module access (2)
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        return getattr(self._pipe, name, None)

    def set_submodule(self, name: str, module: nn.Module) -> None:
        setattr(self._pipe, name, module)

    # ================================================================
    # ABC: Decoding (1)
    # ================================================================

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        video_tensor = self._decode_latents(latents, tiled=tiled)
        return self._latents_to_frames(video_tensor)

    # ================================================================
    # ABC: Device management (1)
    # ================================================================

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move all owned submodules to (dtype, device).

        Also syncs ``self._pipe.device`` because ``BasePipeline.device`` is a
        plain attribute consumed by inference-time pipeline units (VAE encode,
        image preprocess, control tensors — see ``wan/pipeline.py``). The
        training path never touches those units, but deploy
        (``openwam/deploy/model_loader.py``) relies on this single call to
        keep the pipeline's device record in sync.

        This method must NOT call ``mod.eval()``: trainable submodules
        (dit, vace) need to stay in train mode; eval/train state of frozen
        modules is irrelevant since their forward runs under ``no_grad``.
        """
        self._dtype = dtype
        self._device = device
        for name in self.submodule_names:
            mod = self.get_submodule(name)
            if mod is not None:
                mod.to(dtype=dtype, device=device)
        self._pipe.device = device

    # ================================================================
    # Component specs for self-contained checkpoints
    # ================================================================

    def get_component_specs(self, model_path: str) -> dict:
        """Generate component specs from *model_path* for config persistence.

        Uses MODEL_CONFIGS hash matching to discover sub-module classes and
        kwargs. The returned dict is injected into ``cfg.model.video_backbone``
        before saving ``config.yaml``, so deploy can reconstruct the pipeline
        without needing the original *model_path*.

        Returns a dict with keys ``components`` (list) and optionally
        ``tokenizer`` (dict).
        """
        from openwam.model.video_backbone.wan.component_specs import generate_video_backbone_component_specs

        return generate_video_backbone_component_specs(model_path)

    def copy_deploy_artifacts(self, output_dir: str, cfg) -> None:
        """Copy the Wan tokenizer next to ``config.yaml`` so deploy is self-contained."""
        from openwam.model.video_backbone.wan.component_specs import copy_video_backbone_tokenizer

        copy_video_backbone_tokenizer(output_dir, cfg)

    # ================================================================
    # Deploy-facing public methods (not in ABC — Wan-specific)
    # ================================================================

    @property
    def is_ti2v(self) -> bool:
        return self._is_ti2v

    def _resolve_i2v_input_image(self, first_frame_image):
        """Normalize ``first_frame_image`` into a single PIL (or None) for I2V deploy.

        ``openwam/deploy/policy.py`` wraps a single PIL into ``[img]`` before
        calling this method. Upstream I2V CLIP/VAE units (``wan/pipeline.py``)
        call ``.resize`` on the value directly and would raise on a list,
        whereas the TI2V finalize path handles a list itself. Always route the
        I2V deploy path through this helper.
        """
        # ``_dit`` / ``_is_ti2v`` / ``_has_vace`` are properties on the real
        # WanVideoBackbone; the defensive ``getattr`` is for lightweight test
        # mocks (e.g. ``tests/test_cache_behavior.py::_MockWanVB``) that
        # bypass the full Wan pipeline and set these as plain instance fields
        # or omit them entirely.
        is_i2v = (
            bool(getattr(getattr(self, "_dit", None), "has_image_input", False))
            and not getattr(self, "_is_ti2v", False)
            and not getattr(self, "_has_vace", False)
        )
        if not is_i2v or first_frame_image is None:
            return None
        if isinstance(first_frame_image, (list, tuple)):
            if len(first_frame_image) != 1:
                raise ValueError(
                    f"I2V deploy expects a single first-frame image; got list of {len(first_frame_image)}."
                )
            return first_frame_image[0]
        return first_frame_image

    def prepare_inputs_for_inference(self, inputs: "InferenceInputs") -> dict:
        """Prepare all inputs for the inference denoising loop.

        Encapsulates: scheduler setup, unit runner (text/image/VACE encoding),
        TI2V first-frame handling, and caching.
        Returns a single dict ready for the denoising loop.

        Takes a typed :class:`openwam.model.inference_inputs.InferenceInputs`
        instead of a long kwargs list. CFG fields are ignored — Wan adapters
        do not implement classifier-free guidance at inference today; the
        validator in ``BaseWAMArchitecture.generate`` rejects ``cfg_scale > 1``
        before we get here, so any non-default CFG state is a caller bug.
        """
        import time

        # Unpack with sensible Wan defaults for tile dims (the dataclass keeps
        # them as ``None`` so Cosmos25 / other backbones can opt in to their
        # own defaults — Wan has long-standing concrete defaults we preserve).
        prompt = inputs.prompt
        vace_video = inputs.vace_video
        first_frame_image = inputs.first_frame_image
        num_frames = inputs.num_frames
        height = inputs.height
        width = inputs.width
        seed = inputs.seed
        tiled = inputs.tiled
        num_inference_steps = inputs.num_inference_steps
        shift = inputs.shift
        tile_size = inputs.tile_size if inputs.tile_size is not None else (30, 52)
        tile_stride = inputs.tile_stride if inputs.tile_stride is not None else (15, 26)
        vace_cache = inputs.vace_cache
        prompt_embed_cache = inputs.prompt_embed_cache

        pipe = self._pipe
        pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, shift=shift)

        prompt_key = prompt

        if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt_key:
            inputs_shared = vace_cache["inputs_shared"].copy()
            inputs_shared["seed"] = seed
            inputs_shared["vace_video"] = vace_video
            inputs_shared["height"] = height
            inputs_shared["width"] = width
            inputs_shared["num_frames"] = num_frames
            inputs_shared["sigma_shift"] = shift
            inputs_shared["tiled"] = tiled
            inputs_shared["tile_size"] = tile_size
            inputs_shared["tile_stride"] = tile_stride

            # vace_reference_image: cache-miss path always settles this to None
            # (I2V/VACE explicitly clear it; TI2V never has a vendored unit
            # that consumes it — only ``WanVideoUnit_VACE`` reads the slot and
            # it is not in the TI2V pipeline). The cached copy is therefore
            # always None for this call's backbone; no refresh needed. If a
            # future TI2V flow adds a consumer, add an explicit refresh here.

            # cache-hit copies a prior inputs_shared dict — overwrite the I2V
            # condition slot every call so a stale input_image / clip_feature /
            # y does not survive into a non-I2V or no-first-frame invocation.
            _i2v_img = self._resolve_i2v_input_image(first_frame_image)
            inputs_shared["input_image"] = _i2v_img
            if _i2v_img is not None:
                # I2V routes the first-frame condition through input_image
                # (CLIP) + y (channel-axis). vace_reference_image would make
                # WanVideoUnit_{NoiseInitializer, InputVideoEmbedder} prepend
                # an extra latent frame and break the channel-cat with y in
                # prepare().
                inputs_shared["vace_reference_image"] = None
            else:
                inputs_shared.pop("clip_feature", None)
                inputs_shared.pop("y", None)
                if self._has_vace:
                    # VACE: the vendored ``WanVideoUnit_VACE`` would otherwise
                    # encode this and prepend a ref frame onto vace_context.
                    # We inject the ref into vace_context[..., 0:1] manually in
                    # ``_finalize_ti2v_inputs`` after units run, matching training.
                    inputs_shared["vace_reference_image"] = None

            for unit in pipe.units:
                if self._is_text_unit(unit):
                    continue
                inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, {}, {})

            WanVideoBackbone._ensure_prompt_seq_lens(self, inputs_shared, prompt)
            self._finalize_ti2v_inputs(inputs_shared, first_frame_image)
            return inputs_shared

        _text_embed_hit = prompt_embed_cache is not None and prompt_key in prompt_embed_cache
        if _text_embed_hit:
            cached_posi = prompt_embed_cache[prompt_key]
            inputs_posi = dict(cached_posi)
            inputs_posi["num_inference_steps"] = num_inference_steps
        else:
            inputs_posi = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
            }

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
            "vace_video": vace_video,
            "vace_video_mask": None,
            "vace_reference_image": first_frame_image,
            "vace_scale": 1.0,
            "seed": seed,
            "rand_device": "cpu",
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "sigma_shift": shift,
            "motion_bucket_id": None,
            "longcat_video": None,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "input_audio": None,
            "audio_sample_rate": 16000,
            "s2v_pose_video": None,
            "audio_embeds": None,
            "s2v_pose_latents": None,
            "motion_video": None,
        }

        # I2V-only: unwrap the deploy-side single-PIL-list back to a single PIL
        # before any unit runs (the upstream CLIP/VAE units call .resize() on
        # this slot directly). For non-I2V backbones the helper returns None,
        # keeping the default None set above and clearing any stale clip/y.
        _i2v_img = self._resolve_i2v_input_image(first_frame_image)
        inputs_shared["input_image"] = _i2v_img
        if _i2v_img is not None:
            # I2V routes the first-frame condition through input_image (CLIP)
            # + y (channel-axis). vace_reference_image would make
            # WanVideoUnit_{NoiseInitializer, InputVideoEmbedder} prepend an
            # extra latent frame and break the channel-cat with y in prepare().
            inputs_shared["vace_reference_image"] = None
        else:
            inputs_shared.pop("clip_feature", None)
            inputs_shared.pop("y", None)
            if self._has_vace:
                # VACE: vendored ``WanVideoUnit_VACE`` would prepend a ref frame
                # to vace_context if vace_reference_image is set. We construct
                # the ref-frame injection ourselves in ``_finalize_ti2v_inputs``
                # to mirror the no-prepend training contract (vace_context has
                # T_lat frames; frame 0 carries the ref). Clear the slot so the
                # vendored unit does not extend vace_context to T_lat + 1.
                inputs_shared["vace_reference_image"] = None

        _t_text = time.time()
        inputs_nega = {}

        if _text_embed_hit:
            for unit in pipe.units:
                if self._is_text_unit(unit):
                    continue
                inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                    unit, pipe, inputs_shared, inputs_posi, inputs_nega
                )
        else:
            last_text_idx = max(
                (i for i, u in enumerate(pipe.units) if self._is_text_unit(u)),
                default=-1,
            )
            for i, unit in enumerate(pipe.units):
                inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                    unit, pipe, inputs_shared, inputs_posi, inputs_nega
                )
                if i == last_text_idx and prompt_embed_cache is not None:
                    prompt_embed_cache[prompt_key] = inputs_posi.copy()

            if last_text_idx < 0 and prompt_embed_cache is not None:
                prompt_embed_cache[prompt_key] = inputs_posi.copy()

        inputs_shared.update(inputs_posi)
        WanVideoBackbone._ensure_prompt_seq_lens(self, inputs_shared, prompt)

        if vace_cache is not None:
            vace_cache["inputs_shared"] = inputs_shared.copy()
            vace_cache["populated"] = True
            vace_cache["prompt_key"] = prompt_key

        self._finalize_ti2v_inputs(inputs_shared, first_frame_image)
        return inputs_shared

    def _finalize_ti2v_inputs(self, inputs_shared: dict, first_frame_image) -> None:
        """Handle first-frame latent encoding for TI2V and VACE.

        After PR#19 (no ref-prefix prepend), both TI2V and VACE share the same
        contract: ``first_frame_latents`` holds the encoded ref frame, and the
        in-loop clean-replace in ``base.generate`` keeps ``latents[:, :, 0:1]``
        clean across every denoising step (mirroring the training path where
        ``_add_noise_and_pred`` overwrites the noisy frame 0 with the clean ref).

        VACE additionally needs ``vace_context``'s frame 0 to carry the same
        ref so the VACE module sees a consistent reference; the deploy
        ``WanVideoUnit_VACE`` is told to skip its own prepend by clearing
        ``vace_reference_image`` upstream, and we inject the ref here.

        TI2V additionally enables ``fuse_vae_embedding_in_latents`` so its
        ``seperated_timestep`` DiT zeroes the timestep on frame 0's tokens.
        VACE keeps the flag False — its DiT does not have that path.
        """
        is_target = self._is_ti2v or self._has_vace
        if not is_target or first_frame_image is None:
            if first_frame_image is None:
                inputs_shared.pop("first_frame_latents", None)
                inputs_shared["fuse_vae_embedding_in_latents"] = False
                inputs_shared["num_clean_prefix_frames"] = 0
            return
        device = self.device
        dtype = self.dtype
        inputs_shared["fuse_vae_embedding_in_latents"] = bool(self._is_ti2v)
        inputs_shared["num_clean_prefix_frames"] = 0
        ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        ref_tensor = self._preprocess_video(ref_frames)
        ref_image_latents = self._encode_video(ref_tensor.to(device)).to(dtype=dtype, device=device)
        inputs_shared["first_frame_latents"] = ref_image_latents
        if self._has_vace:
            vace_context = inputs_shared.get("vace_context")
            if vace_context is not None:
                inputs_shared["vace_context"] = self._inject_vace_ref_frame(
                    vace_context, ref_image_latents
                )

    @staticmethod
    def _is_text_unit(unit) -> bool:
        _TEXT_UNIT_CLASS_NAMES = frozenset({"WanVideoUnit_PromptEmbedder"})
        flag = getattr(unit, "is_text_unit", None)
        if flag is not None:
            return bool(flag)
        cls_name = getattr(unit, "__class__", type(unit)).__name__
        return cls_name in _TEXT_UNIT_CLASS_NAMES

    def _ensure_prompt_seq_lens(self, inputs_shared: dict, prompt) -> None:
        """Attach text ``seq_lens`` so deploy cross-attention masks padding.

        The Wan pipeline prompt unit returns ``context`` but not the tokenizer
        mask. Training uses :meth:`_encode_text`, which supplies ``seq_lens``;
        without it, deploy treats all 512 padded text positions as attendable.
        """
        if inputs_shared.get("context") is None:
            return
        tokenizer = getattr(self._pipe, "tokenizer", None)
        if tokenizer is None:
            return
        if inputs_shared.get("seq_lens") is not None or inputs_shared.get("context_mask") is not None:
            return

        _, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        seq_lens = mask.gt(0).sum(dim=1).long().to(self.device)
        context = inputs_shared["context"]
        if seq_lens.shape[0] != context.shape[0]:
            if seq_lens.shape[0] == 1:
                seq_lens = seq_lens.expand(context.shape[0])
            elif context.shape[0] % seq_lens.shape[0] == 0:
                repeat = context.shape[0] // seq_lens.shape[0]
                seq_lens = seq_lens.repeat_interleave(repeat)
            else:
                logger.warning(
                    "Cannot align prompt seq_lens batch %d with context batch %d; deploy text padding remains unmasked.",
                    seq_lens.shape[0],
                    context.shape[0],
                )
                return
        inputs_shared["seq_lens"] = seq_lens

    def _encode_text(self, prompts: list) -> Tuple[Tensor, Tensor]:
        device = self.device
        ids, mask = self._pipe.tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self._pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            context[i, v:] = 0
        return context, seq_lens

    def _preprocess_video(self, frames) -> Tensor:
        return self._pipe.preprocess_video(frames)

    def _encode_video(self, video_tensor: Tensor, *, tiled: bool = False) -> Tensor:
        return self._pipe.vae.batch_encode(video_tensor, device=video_tensor.device)

    def _decode_latents(self, latents: Tensor, *, tiled: bool = True) -> Tensor:
        return self._pipe.vae.decode(latents.to(self.device), device=self.device, tiled=tiled)

    def _latents_to_frames(self, video_tensor: Tensor) -> list:
        return self._pipe.vae_output_to_video(video_tensor)

    def _check_resize(self, h, w, num_frames):
        return self._pipe.check_resize_height_width(h, w, num_frames)

    def _build_vace_context(self, vace_video, input_latents, device) -> Tensor:
        all_vace = []
        for clip_frames in vace_video:
            vt = self._preprocess_video(clip_frames)
            vl = self._encode_video(vt.to(device))
            all_vace.append(vl)
        return torch.cat(all_vace, dim=0) if all_vace else None

    @staticmethod
    def _inject_vace_ref_frame(vace_context: Tensor, ref_latent: Tensor) -> Tensor:
        """Overwrite frame 0 of ``vace_context`` with the reference latent.

        Channel layout (matching ``preprocess_input`` / vendored
        ``WanVideoUnit_VACE``):

            vace_context = concat([inactive (C_v), reactive (C_v), mask (P*Q)], dim=1)

        where ``C_v == ref_latent.shape[1]`` (Wan VAE z_dim, typically 16).

        Frame 0 is overwritten so VACE treats it as a known reference:
          * inactive[..., 0:1] = ref_latent  (the actual reference content)
          * reactive[..., 0:1] = 0           (no "reactive" signal at the ref)
          * mask[..., 0:1] = 0               (mask=0 -> frame is given, not generated)

        ``ref_latent`` is expected to be ``(B, C_v, 1, H, W)``.
        """
        if ref_latent.dim() != 5 or ref_latent.shape[2] != 1:
            raise ValueError(f"ref_latent must be (B, C, 1, H, W); got {tuple(ref_latent.shape)}")
        c_v = ref_latent.shape[1]
        if vace_context.shape[1] < 2 * c_v:
            raise ValueError(
                f"vace_context channels ({vace_context.shape[1]}) too small for ref_latent C={c_v}"
            )
        out = vace_context.clone()
        out[:, :c_v, 0:1] = ref_latent.to(dtype=out.dtype, device=out.device)
        out[:, c_v : 2 * c_v, 0:1] = 0
        out[:, 2 * c_v :, 0:1] = 0
        return out

    def _is_per_token_t_mod_active(self, state: BlockLoopState) -> bool:
        return state.t_mod.dim() == 4

    def _build_i2v_y(
        self,
        *,
        first_frame_image: list,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build the Wan2.1-I2V ``y`` conditioning tensor batch-wise.

        Output shape: ``(B, 20, T_lat, H_lat, W_lat) = concat([msk(4), vae_y(16)])``
        on the channel dim. Mirrors ``WanVideoUnit_ImageEmbedderVAE.process``
        in ``wan/pipeline.py`` (which assumes ``B=1`` at inference) but
        encodes the whole batch in a single VAE forward via ``batch_encode``,
        matching ``_encode_video`` (line ~1311). ``batch_encode`` supports
        non-tiled only; training already runs the VAE non-tiled.
        """
        pipe = self._pipe
        vae_inputs = []
        msks = []
        for img in first_frame_image:
            image = pipe.preprocess_image(img.resize((width, height))).to(device)  # (1, 3, H, W)
            vae_input = torch.cat(
                [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width, device=device)],
                dim=1,
            )  # (3, num_frames, H, W)
            vae_inputs.append(vae_input)

            msk = torch.ones(1, num_frames, height // 8, width // 8, device=device)
            msk[:, 1:] = 0
            msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]  # (4, T_lat, H_lat, W_lat)
            msks.append(msk)

        vae_inputs_b = torch.stack(vae_inputs, dim=0).to(dtype=dtype, device=device)  # (B, 3, T, H, W)
        msks_b = torch.stack(msks, dim=0).to(dtype=dtype, device=device)  # (B, 4, T_lat, H_lat, W_lat)
        y_lat = pipe.vae.batch_encode(vae_inputs_b, device=device).to(dtype=dtype, device=device)
        y = torch.cat([msks_b, y_lat], dim=1)  # (B, 20, T_lat, H_lat, W_lat)
        return y

    def _build_action_t_mod(
        self,
        action_timestep: Tensor,
        n_action_tokens: int,
        *,
        batch_size: int,
    ) -> Tensor:
        dit = self._dit
        if action_timestep.dim() == 2:
            if action_timestep.shape != (batch_size, n_action_tokens):
                raise ValueError(
                    f"action_timestep has shape {tuple(action_timestep.shape)}; expected "
                    f"(B={batch_size}, n_action_tokens={n_action_tokens})."
                )
            B_t = action_timestep.shape[0]
            flat = action_timestep.reshape(B_t * n_action_tokens)
            t_emb = sinusoidal_embedding_1d(dit.freq_dim, flat)
            dtype = next(dit.time_embedding.parameters()).dtype
            t = dit.time_embedding(t_emb.to(dtype=dtype, device=t_emb.device))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
            t_mod = t_mod.view(B_t, n_action_tokens, 6, dit.dim)
        else:
            timestep_flat = action_timestep.flatten()
            if timestep_flat.numel() == 1:
                timestep_flat = timestep_flat.expand(batch_size)
            elif timestep_flat.numel() != batch_size:
                raise ValueError(
                    f"action_timestep has shape {tuple(action_timestep.shape)}; expected scalar, "
                    f"(B={batch_size},), or (B={batch_size}, n_action_tokens={n_action_tokens})."
                )
            t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep_flat)
            dtype = next(dit.time_embedding.parameters()).dtype
            t = dit.time_embedding(t_emb.to(dtype=dtype, device=t_emb.device))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
            t_mod = t_mod.unsqueeze(1).expand(-1, n_action_tokens, -1, -1)
        return t_mod

    def _build_sample_t_mod(
        self,
        timestep: Tensor,
        n_tokens: int,
        *,
        batch_size: int,
    ) -> Tensor:
        timestep_flat = timestep.flatten()
        if timestep_flat.numel() == 1:
            timestep_flat = timestep_flat.expand(batch_size)
        elif timestep_flat.numel() == batch_size:
            pass
        elif timestep.dim() == 2 and timestep.shape[0] == batch_size:
            timestep_flat = timestep[:, 0]
        else:
            raise ValueError(
                f"timestep has shape {tuple(timestep.shape)}; expected scalar, (B={batch_size},), "
                f"or (B={batch_size}, T) for sample-level state t_mod."
            )
        dit = self._dit
        t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep_flat)
        dtype = next(dit.time_embedding.parameters()).dtype
        t = dit.time_embedding(t_emb.to(dtype=dtype, device=t_emb.device))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        t_mod = t_mod.unsqueeze(1).expand(-1, n_tokens, -1, -1)
        return t_mod

    def _extend_freqs_with_action_tokens(self, freqs: Tensor, n_action_tokens: int) -> Tensor:
        """Append SharedBackbone 1D action RoPE frequencies to ``freqs``.

        This mirrors DreamZero's separate action RoPE: action tokens receive a
        1D sequence position in action-horizon space instead of being treated as
        extra video tokens.

        Tied to Wan's complex-form ``view_as_complex`` RoPE in
        ``components.rope_apply_1d`` — switching the video DiT to a different
        RoPE representation (sincos table, polar pair, …) requires changing this
        helper accordingly.
        """
        if n_action_tokens <= 0:
            return freqs
        return torch.cat([freqs, self._build_1d_action_freqs(freqs, n_action_tokens)], dim=0)

    def _extend_freqs_with_shared_tokens(self, freqs: Tensor, n_action_tokens: int, n_state_tokens: int = 0) -> Tensor:
        pieces = [freqs]
        if n_action_tokens > 0:
            pieces.append(self._build_1d_action_freqs(freqs, n_action_tokens))
        if n_state_tokens > 0:
            pieces.append(self._build_1d_state_freqs(freqs, n_state_tokens))
        return torch.cat(pieces, dim=0)

    @staticmethod
    def _build_1d_action_freqs(freqs: Tensor, n_action_tokens: int, theta: float = 10000.0) -> Tensor:
        head_dim = int(freqs.shape[-1]) * 2
        positions = torch.arange(n_action_tokens, dtype=torch.float64, device=freqs.device)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device=freqs.device) / head_dim))
        angles = torch.outer(positions, inv_freq)
        action_freqs = torch.polar(torch.ones_like(angles), angles).view(n_action_tokens, 1, -1)
        return action_freqs.to(dtype=freqs.dtype)

    @staticmethod
    def _build_1d_state_freqs(freqs: Tensor, n_state_tokens: int, theta: float = 10000.0) -> Tensor:
        head_dim = int(freqs.shape[-1]) * 2
        positions = torch.arange(n_state_tokens, dtype=torch.float64, device=freqs.device)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device=freqs.device) / head_dim))
        angles = torch.outer(positions, inv_freq)
        state_freqs = torch.polar(torch.ones_like(angles), angles).view(n_state_tokens, 1, -1)
        return state_freqs.to(dtype=freqs.dtype)

    def apply_compile(self, compile_cfg) -> None:
        """Apply torch.compile to backbone sub-modules based on config flags.

        For each sub-module in ``submodule_names``, checks for a matching
        bool flag in *compile_cfg*. The DiT is special-cased: its blocks
        are compiled individually (per-block compile is more CUDA-graph
        friendly than compiling the whole DiT).

        Recognized flag names: ``video_dit`` (or ``dit``), ``vae``,
        ``vace``, ``text_encoder``, ``image_encoder``.
        """
        flag_to_submodule = {
            "video_dit": "dit",
            "dit": "dit",
            "vae": "vae",
            "vace": "vace",
            "text_encoder": "text_encoder",
            "image_encoder": "image_encoder",
        }
        for flag, submod_name in flag_to_submodule.items():
            if not cfg_get(compile_cfg, flag, False):
                continue
            mod = getattr(self._pipe, submod_name, None)
            if mod is None:
                continue
            if submod_name == "dit" and hasattr(mod, "blocks"):
                compile_kwargs = torch_compile_kwargs(compile_cfg, default_mode="reduce-overhead")
                for i, block in enumerate(mod.blocks):
                    mod.blocks[i] = torch.compile(block, **compile_kwargs)
                logger.info(
                    "torch.compile enabled for %s blocks (%d, %s)", submod_name, len(mod.blocks), compile_kwargs
                )
            else:
                compile_kwargs = torch_compile_kwargs(compile_cfg)
                setattr(self._pipe, submod_name, torch.compile(mod, **compile_kwargs))
                logger.info("torch.compile enabled for %s (%s)", submod_name, compile_kwargs)

    @staticmethod
    def _build_pipe_from_components(
        components: list,
        tokenizer: dict = None,
        device: str = "cpu",
        ckpt_dir: str = None,
        model_path: str = None,
    ):
        """Build an empty WanVideoPipeline from component specs (config-driven).

        Weights are NOT loaded here — ``architecture.load_checkpoint`` handles
        that separately.

        Tokenizer resolution order:
          1. ``ckpt_dir`` + ``tokenizer.subdir`` — checkpoint-local tokenizer
             copied during training save.
          2. ``model_path`` upstream layout — components-based persistence
             falls back to ``<model_path>/google/umt5-xxl/``.
        """
        from openwam.model.video_backbone.wan.pipeline import WanVideoPipeline
        from openwam.model.video_backbone.wan.pipeline_builder import _build_tokenizer, _import_class

        pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)

        for entry in components:
            cls = _import_class(entry["model_class"])
            kwargs = entry.get("extra_kwargs", {}) or {}
            logger.info(
                "Instantiating %s as pipe.%s (extra_kwargs keys=%s)",
                entry["model_class"],
                entry["attr"],
                list(kwargs.keys()),
            )
            with torch.device(device):
                model = cls(**kwargs)
            model.to(dtype=torch.bfloat16)
            setattr(pipe, entry["attr"], model)

        if getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "upsampling_factor"):
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        if tokenizer:
            tok = None
            subdir = tokenizer.get("subdir", "")
            if ckpt_dir and subdir and os.path.isdir(os.path.join(ckpt_dir, subdir)):
                tok = _build_tokenizer(tokenizer, ckpt_dir)
            elif model_path and os.path.isdir(model_path):
                # Checkpoint-local specs store tokenizer paths under
                # ``tokenizer/``; upstream Wan model dirs store
                # ``google/umt5-xxl/`` directly.
                fallback_subdir = subdir
                if fallback_subdir.startswith("tokenizer/"):
                    fallback_subdir = fallback_subdir[len("tokenizer/") :]
                if fallback_subdir and os.path.isdir(os.path.join(model_path, fallback_subdir)):
                    fallback_cfg = dict(tokenizer)
                    fallback_cfg["subdir"] = fallback_subdir
                    logger.info(
                        "Tokenizer not found under ckpt_dir; falling back to model_path/%s",
                        fallback_subdir,
                    )
                    tok = _build_tokenizer(fallback_cfg, model_path)
            if tok is None:
                raise FileNotFoundError(
                    f"Tokenizer subdir {subdir!r} not found under ckpt_dir={ckpt_dir!r} "
                    f"nor under model_path={model_path!r} (with 'tokenizer/' prefix stripped). "
                    "Either copy the tokenizer into the checkpoint dir, or ensure model_path is reachable."
                )
            setattr(pipe, tokenizer.get("attr", "tokenizer"), tok)

        return pipe

    @staticmethod
    def _build_pipe_from_model_path(model_path: str, device: str = "cpu"):
        """Build a WanVideoPipeline from a model directory without full Hydra config."""
        from openwam.model.video_backbone.wan.pipeline import WanVideoPipeline
        from openwam.model.video_backbone.wan.pipeline_builder import discover_model_files

        model_configs, tokenizer_config = discover_model_files(model_path)
        return WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )


def _probe_dit_stats(dit) -> dict:
    """Snapshot a few representative tensors for before/after verification.

    Picks tensors that exercise both code paths of ``reinit_dit_from_scratch``:
      - ``blocks[0].self_attn.q.weight`` — covered by stdlib ``reset_parameters``
      - ``blocks[0].modulation`` — directly-mounted nn.Parameter, hand-reset
      - ``head.modulation`` — same category, separate code branch
    """
    return {
        "q.weight_mean": float(dit.blocks[0].self_attn.q.weight.float().mean().item()),
        "q.weight_std": float(dit.blocks[0].self_attn.q.weight.float().std().item()),
        "blocks[0].modulation_mean": float(dit.blocks[0].modulation.float().mean().item()),
        "blocks[0].modulation_std": float(dit.blocks[0].modulation.float().std().item()),
        "head.modulation_mean": float(dit.head.modulation.float().mean().item()),
        "head.modulation_std": float(dit.head.modulation.float().std().item()),
    }


def reinit_dit_from_scratch(pipe, *, verbose: bool = True) -> None:
    """Re-initialize all learnable parameters in ``pipe.dit`` (and
    ``pipe.dit2`` if present) using PyTorch standard initialization. Does
    NOT touch ``pipe.vae`` / ``pipe.text_encoder`` / ``pipe.image_encoder`` /
    ``pipe.vace`` — only the DiT(s).

    Used by the ``video_backbone.from_scratch`` config switch to ablate
    "pretrained DiT vs from-scratch DiT" while keeping VAE and the text
    encoder loaded with their pretrained weights (these are typically
    frozen by the training_strategy yaml).

    Two-step strategy:

    1. ``modules().reset_parameters()`` for stdlib layers
       (``nn.Linear`` / ``Conv2d`` / ``Conv3d`` / ``Embedding`` /
       ``LayerNorm``) — covers the vast majority of params.
    2. Hand-reset four classes of directly-mounted ``nn.Parameter`` that
       ``module.modules()`` does not yield. Without this, ~30
       ``DiTBlock.modulation`` tensors and ~180 ``RMSNorm.weight`` tensors
       per 30-layer Wan DiT would silently retain the loaded pretrained
       values and the ablation would not be clean. See the audit in
       ``plans/a-vectorized-crystal.md``.

    Buffers and non-parameter tensors (e.g. ``WanModel.freqs`` RoPE cache)
    are deterministic functions of the model hyperparams and are left
    untouched.

    Args:
        pipe: A Wan pipeline with ``.dit`` (and optionally ``.dit2``) attached.
        verbose: When True (default) and we're on rank 0, prints a
            human-readable BEFORE/AFTER summary directly to stdout — this is
            independent of the ``logging`` configuration so users see the
            verification trace in the terminal regardless of whether their
            launcher routes ``logger.info`` to stderr/stdout/a file.
    """
    import os

    import torch.nn as nn

    from openwam.model.video_backbone.wan.dit import MLP, DiTBlock, Head, RMSNorm

    stdlib_resettable = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Embedding, nn.LayerNorm)

    dits = [m for m in (getattr(pipe, "dit", None), getattr(pipe, "dit2", None)) if m is not None]
    if not dits:
        logger.warning("reinit_dit_from_scratch: pipe has no dit/dit2 to re-init")
        return

    rank = int(os.environ.get("RANK", 0))
    is_main = rank == 0

    # Snapshot a few representative tensors BEFORE the reset so the user can
    # eyeball "yes, the loaded pretrained values were actually thrown away".
    before_stats = [_probe_dit_stats(root) for root in dits] if verbose and is_main else None

    for root in dits:
        for sub in root.modules():
            if isinstance(sub, stdlib_resettable):
                sub.reset_parameters()
        with torch.no_grad():
            for sub in root.modules():
                if isinstance(sub, RMSNorm):
                    sub.weight.fill_(1.0)
                elif isinstance(sub, DiTBlock):
                    dim = sub.modulation.shape[-1]
                    sub.modulation.normal_(mean=0.0, std=dim**-0.5)
                elif isinstance(sub, Head):
                    dim = sub.modulation.shape[-1]
                    sub.modulation.normal_(mean=0.0, std=dim**-0.5)
                elif isinstance(sub, MLP) and getattr(sub, "has_pos_emb", False):
                    sub.emb_pos.zero_()

    logger.info(
        "reinit_dit_from_scratch: re-initialized %d DiT module(s); VAE/T5 untouched",
        len(dits),
    )

    if verbose and is_main:
        after_stats = [_probe_dit_stats(root) for root in dits]
        # Use print(..., flush=True) so the verification line surfaces even
        # under non-INFO logging configurations (e.g. plain torchrun without
        # logging.basicConfig). Bounded output: 4 lines per DiT module.
        bar = "=" * 78
        print(bar, flush=True)
        print(
            "[reinit_dit_from_scratch] video DiT weights re-initialized from scratch. "
            f"VAE / text_encoder kept pretrained. ({len(dits)} DiT module(s), rank=0 summary)",
            flush=True,
        )
        for i, (before, after, root) in enumerate(zip(before_stats, after_stats, dits)):
            label = "dit" if i == 0 else f"dit{i + 1}"
            expected_mod_std = root.dim**-0.5
            print(
                f"  [{label}] q.weight                 BEFORE mean={before['q.weight_mean']:+.4e} std={before['q.weight_std']:.4e}  "
                f"-> AFTER mean={after['q.weight_mean']:+.4e} std={after['q.weight_std']:.4e}",
                flush=True,
            )
            print(
                f"  [{label}] blocks[0].modulation     BEFORE mean={before['blocks[0].modulation_mean']:+.4e} std={before['blocks[0].modulation_std']:.4e}  "
                f"-> AFTER mean={after['blocks[0].modulation_mean']:+.4e} std={after['blocks[0].modulation_std']:.4e}  "
                f"(expected std≈{expected_mod_std:.4e})",
                flush=True,
            )
            print(
                f"  [{label}] head.modulation          BEFORE mean={before['head.modulation_mean']:+.4e} std={before['head.modulation_std']:.4e}  "
                f"-> AFTER mean={after['head.modulation_mean']:+.4e} std={after['head.modulation_std']:.4e}  "
                f"(expected std≈{expected_mod_std:.4e})",
                flush=True,
            )
        print(
            "  Reproducibility: with the same cfg.project.seed, these AFTER numbers "
            "are bit-exact across runs (rank-0 broadcast covers other ranks).",
            flush=True,
        )
        print(bar, flush=True)


__all__ = ["WanVideoBackbone", "reinit_dit_from_scratch"]
