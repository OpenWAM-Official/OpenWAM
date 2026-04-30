"""Wan-specific implementation of :class:`VideoBackbone`.

Lives outside ``wan/`` to keep the ``wan/`` package focused on Wan-internal
implementation (DiT, VACE, VAP, Animate, TeaCache, SP, etc.). This module
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
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.wan.dit import sinusoidal_embedding_1d
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
        - ``str`` ending in ``.json``: manifest path → ``build_video_backbone_from_manifest``
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
            elif source.endswith(".json"):
                from openwam.model.video_backbone.wan.pipeline_builder import build_video_backbone_from_manifest

                pipe = build_video_backbone_from_manifest(source, device=kw.get("device", "cpu"))
            else:
                raise ValueError(
                    f"from_pretrained(str) expects a directory path or manifest .json path, got: {source!r}."
                )
        elif isinstance(source, dict):
            vb_cfg = source.get("video_backbone", source)
            if isinstance(vb_cfg, dict) and "components" in vb_cfg:
                pipe = cls._build_pipe_from_components(
                    vb_cfg["components"],
                    tokenizer=vb_cfg.get("tokenizer"),
                    device=kw.get("device", "cpu"),
                    ckpt_dir=kw.get("ckpt_dir"),
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

    # ================================================================
    # ABC: Three-step execution (3)
    # ================================================================

    def prepare(self, **kw) -> BlockLoopState:
        dit = self._pipe.dit
        motion_controller = getattr(self._pipe, "motion_controller", None)
        vace = getattr(self._pipe, "vace", None)
        vap = getattr(self._pipe, "vap", None)
        animate_adapter = getattr(self._pipe, "animate_adapter", None)
        latents = kw["latents"]
        timestep = kw["timestep"]
        context = kw["context"]
        clip_feature = kw.get("clip_feature")
        y = kw.get("y")
        reference_latents = kw.get("reference_latents")
        vace_context = kw.get("vace_context")
        vace_scale = kw.get("vace_scale", 1.0)
        tea_cache = kw.get("tea_cache")
        use_usp = kw.get("use_unified_sequence_parallel", self._use_unified_sequence_parallel)
        motion_bucket_id = kw.get("motion_bucket_id")
        pose_latents = kw.get("pose_latents")
        face_pixel_values = kw.get("face_pixel_values")
        control_camera_latents_input = kw.get("control_camera_latents_input")
        fuse_vae_embedding_in_latents = kw.get("fuse_vae_embedding_in_latents", False)
        num_clean_prefix_frames = kw.get("num_clean_prefix_frames", 0)
        use_gradient_checkpointing = kw.get("use_gradient_checkpointing", False)
        use_gradient_checkpointing_offload = kw.get("use_gradient_checkpointing_offload", False)
        vap_hidden_state = kw.get("vap_hidden_state")
        vap_clip_feature = kw.get("vap_clip_feature")
        context_vap = kw.get("context_vap")

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
        else:
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        if motion_bucket_id is not None and motion_controller is not None:
            t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
        context = dit.text_embedding(context)

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

        x = dit.patchify(x, control_camera_latents_input)

        motion_vec = None
        if pose_latents is not None and face_pixel_values is not None:
            x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)

        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        ref_prefix_len = 0
        if reference_latents is not None:
            if len(reference_latents.shape) == 5:
                reference_latents = reference_latents[:, :, 0]
            reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
            ref_prefix_len = reference_latents.shape[1]
            x = torch.concat([reference_latents, x], dim=1)
            f += 1

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
        extras["vap"] = vap
        extras["animate_adapter"] = animate_adapter
        extras["pose_latents"] = pose_latents
        extras["face_pixel_values"] = face_pixel_values
        extras["motion_vec"] = motion_vec
        extras["tea_cache"] = tea_cache
        extras["use_usp"] = use_usp
        extras["reference_latents_for_finalize"] = kw.get("reference_latents")

        if vap is not None:
            x_vap = vap_hidden_state
            x_vap = vap.patchify(x_vap)
            x_vap = rearrange(x_vap, "b c f h w -> b (f h w) c").contiguous()
            clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
            t_vap = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
            t_mod_vap = vap.time_projection(t_vap).unflatten(1, (6, vap.dim))
            freqs_vap = vap.compute_freqs_mot(f, h, w).to(x.device)
            vap_clip_embedding = vap.img_emb(vap_clip_feature)
            context_vap_emb = vap.text_embedding(context_vap)
            context_vap_emb = torch.cat([vap_clip_embedding, context_vap_emb], dim=1)
            extras["x_vap"] = x_vap
            extras["t_mod_vap"] = t_mod_vap
            extras["freqs_vap"] = freqs_vap
            extras["context_vap"] = context_vap_emb

        tea_cache_update = False
        if tea_cache is not None:
            tea_cache_update = tea_cache.check(dit, x, t_mod)

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
            f=f,
            h=h,
            w=w,
            t=t,
            reference_prefix_len=ref_prefix_len,
            vace_hints=vace_hints,
            vace_scale=vace_scale,
            tea_cache_update=tea_cache_update,
            sp_pad_shape=sp_pad_shape,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            extras=extras,
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        dit = state.extras["dit"]
        vap = state.extras.get("vap")
        vace = state.extras.get("vace")
        animate_adapter = state.extras.get("animate_adapter")
        pose_latents = state.extras.get("pose_latents")
        face_pixel_values = state.extras.get("face_pixel_values")
        motion_vec = state.extras.get("motion_vec")
        use_usp = state.extras.get("use_usp", False)

        block = dit.blocks[block_id]

        if vap is not None and block_id in vap.mot_layers_mapping:
            x_vap = state.extras["x_vap"]
            context_vap = state.extras["context_vap"]
            t_mod_vap = state.extras["t_mod_vap"]
            freqs_vap = state.extras["freqs_vap"]

            def _vap_fwd(*inputs):
                return vap(block, *inputs)

            if state.use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    state.x, x_vap = torch.utils.checkpoint.checkpoint(
                        _vap_fwd,
                        state.x,
                        state.context,
                        state.t_mod,
                        state.freqs,
                        x_vap,
                        context_vap,
                        t_mod_vap,
                        freqs_vap,
                        block_id,
                        use_reentrant=False,
                    )
            elif state.use_gradient_checkpointing:
                state.x, x_vap = torch.utils.checkpoint.checkpoint(
                    _vap_fwd,
                    state.x,
                    state.context,
                    state.t_mod,
                    state.freqs,
                    x_vap,
                    context_vap,
                    t_mod_vap,
                    freqs_vap,
                    block_id,
                    use_reentrant=False,
                )
            else:
                state.x, x_vap = vap(
                    block,
                    state.x,
                    state.context,
                    state.t_mod,
                    state.freqs,
                    x_vap,
                    context_vap,
                    t_mod_vap,
                    freqs_vap,
                    block_id,
                )
            state.extras["x_vap"] = x_vap
        else:
            state.x = gradient_checkpoint_forward(
                block,
                state.use_gradient_checkpointing,
                state.use_gradient_checkpointing_offload,
                state.x,
                state.context,
                state.t_mod,
                state.freqs,
            )

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
            state.x = state.x + current_vace_hint * state.vace_scale

        if pose_latents is not None and face_pixel_values is not None:
            state.x = animate_adapter.after_transformer_block(block_id, state.x, motion_vec)

        return state

    def finalize(self, state: BlockLoopState) -> Tensor:
        dit = state.extras["dit"]
        tea_cache = state.extras.get("tea_cache")
        use_usp = state.extras.get("use_usp", False)
        reference_latents = state.extras.get("reference_latents_for_finalize")

        if tea_cache is not None and not state.tea_cache_update:
            tea_cache.store(state.x)

        t_head = state.t if state.t.dim() == 3 else state.t.unsqueeze(1)
        x = dit.head(state.x, t_head)

        if use_usp:
            import torch.distributed as dist
            from xfuser.core.distributed import get_sp_group

            if dist.is_initialized() and dist.get_world_size() > 1:
                x = get_sp_group().all_gather(x, dim=1)
                if state.sp_pad_shape > 0:
                    x = x[:, : -state.sp_pad_shape]

        f = state.f
        if reference_latents is not None:
            if len(reference_latents.shape) == 5:
                ref_tokens = reference_latents.shape[3] * reference_latents.shape[4]
            elif len(reference_latents.shape) == 4:
                ref_tokens = reference_latents.shape[2] * reference_latents.shape[3]
            else:
                ref_tokens = state.reference_prefix_len
            x = x[:, ref_tokens:]
            f -= 1

        x = dit.unpatchify(x, (f, state.h, state.w))
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
        t_mod_bias: Optional[Tensor] = None,
    ) -> BlockLoopState:
        state.x = torch.cat([state.x, action_tokens.to(state.x.dtype)], dim=1)
        state.freqs = self._extend_freqs_with_action_tokens(state.freqs, n_action)
        if self._is_per_token_t_mod_active(state) and timestep is not None and t_mod_bias is not None:
            a_tmod = self._build_action_t_mod(timestep, t_mod_bias, n_action)
            state.t_mod = torch.cat([state.t_mod, a_tmod.to(state.t_mod.dtype)], dim=1)
        return state

    def extract_action_tokens(
        self,
        state: BlockLoopState,
        n_action: int,
    ) -> Tuple[BlockLoopState, Tensor]:
        n_video = state.x.shape[1] - n_action
        action_tokens = state.x[:, n_video:, :]
        state.x = state.x[:, :n_video, :]
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
        ref_latents = None
        if has_ref:
            all_refs = []
            for ref in ref_images:
                if not isinstance(ref, list):
                    ref = [ref]
                all_refs.append(self._preprocess_video(ref))
            stacked_refs = torch.cat(all_refs, dim=0)
            ref_latents = self._encode_video(stacked_refs).to(dtype=dtype, device=device)
            input_latents = torch.cat([ref_latents, input_latents], dim=2)

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

            if has_ref:
                ref_f = ref_latents.shape[2]
                vace_ref_latents = torch.cat([ref_latents, torch.zeros_like(ref_latents)], dim=1)
                vace_video_latents = torch.cat([vace_ref_latents, vace_video_latents], dim=2)
                vace_mask_latents = torch.cat(
                    [
                        torch.zeros(
                            B,
                            vace_mask_latents.shape[1],
                            ref_f,
                            vace_mask_latents.shape[3],
                            vace_mask_latents.shape[4],
                            dtype=dtype,
                            device=device,
                        ),
                        vace_mask_latents,
                    ],
                    dim=2,
                )

            vace_context = torch.cat([vace_video_latents, vace_mask_latents], dim=1)

        is_ti2v = self._is_ti2v
        first_frame_latents = None
        num_clean_prefix = 0
        if is_ti2v and has_ref:
            first_frame_latents = ref_latents[:, :, 0:1].clone()
            num_clean_prefix += ref_latents.shape[2]

        return {
            "input_latents": input_latents,
            "context": context,
            "seq_lens": seq_lens,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "vace_context": vace_context,
            "vace_scale": 1.0,
            "fuse_vae_embedding_in_latents": is_ti2v and has_ref,
            "num_clean_prefix_frames": num_clean_prefix,
            "first_frame_latents": first_frame_latents,
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
        self._dtype = dtype
        self._device = device
        for name in self.submodule_names:
            mod = self.get_submodule(name)
            if mod is not None:
                mod.to(dtype=dtype, device=device)
                mod.eval()
        self._pipe.device = device

    # ================================================================
    # Component specs for self-contained checkpoints
    # ================================================================

    def get_component_specs(self, model_path: str) -> dict:
        """Generate component specs from *model_path* for config persistence.

        Uses MODEL_CONFIGS hash matching (same as manifest generation) to
        discover sub-module classes and kwargs. The returned dict is meant
        to be injected into ``cfg.model.video_backbone`` before saving
        ``config.yaml``, so deploy can reconstruct the pipeline without
        needing the original *model_path* or a separate manifest file.

        Returns a dict with keys ``components`` (list) and optionally
        ``tokenizer`` (dict).
        """
        from openwam.model.video_backbone.wan.manifest import generate_video_backbone_manifest

        manifest = generate_video_backbone_manifest(model_path)
        result = {"components": manifest["models"]}
        if "tokenizer" in manifest:
            result["tokenizer"] = manifest["tokenizer"]
        return result

    # ================================================================
    # Deploy-facing public methods (not in ABC — Wan-specific)
    # ================================================================

    @property
    def is_ti2v(self) -> bool:
        return self._is_ti2v

    def prepare_inputs_for_inference(
        self,
        prompt: str,
        *,
        vace_video=None,
        first_frame_image=None,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        seed: int = 42,
        tiled: bool = True,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tile_size: tuple = (30, 52),
        tile_stride: tuple = (15, 26),
        vace_cache: Optional[dict] = None,
        prompt_embed_cache: Optional[dict] = None,
    ) -> dict:
        """Prepare all inputs for the inference denoising loop.

        Encapsulates: scheduler setup, unit runner (text/image/VACE encoding),
        TI2V first-frame handling, and caching.
        Returns a single dict ready for the denoising loop.
        """
        import time

        pipe = self._pipe
        pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, shift=shift)

        prompt_key = prompt

        if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt_key:
            inputs_shared = vace_cache["inputs_shared"].copy()
            inputs_shared["seed"] = seed
            inputs_shared["vace_video"] = vace_video
            inputs_shared["num_frames"] = num_frames

            for unit in pipe.units:
                if self._is_text_unit(unit):
                    continue
                inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, {}, {})

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
                "vap_prompt": " ",
                "tea_cache_l1_thresh": None,
                "tea_cache_model_id": "",
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
            "animate_pose_video": None,
            "animate_face_video": None,
            "animate_inpaint_video": None,
            "animate_mask_video": None,
            "vap_video": None,
        }

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

        if vace_cache is not None:
            vace_cache["inputs_shared"] = inputs_shared.copy()
            vace_cache["populated"] = True
            vace_cache["prompt_key"] = prompt_key

        self._finalize_ti2v_inputs(inputs_shared, first_frame_image)
        return inputs_shared

    def _finalize_ti2v_inputs(self, inputs_shared: dict, first_frame_image) -> None:
        """Handle TI2V first-frame latent encoding and inject into inputs_shared."""
        if not self._is_ti2v or first_frame_image is None:
            return
        device = self.device
        dtype = self.dtype
        inputs_shared["fuse_vae_embedding_in_latents"] = True
        ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        num_clean_prefix = len(ref_frames)
        inputs_shared["num_clean_prefix_frames"] = num_clean_prefix
        ref_tensor = self._preprocess_video(ref_frames)
        ref_image_latents = self._encode_video(ref_tensor.to(device)).to(dtype=dtype, device=device)
        inputs_shared["first_frame_latents"] = ref_image_latents

    @staticmethod
    def _is_text_unit(unit) -> bool:
        _TEXT_UNIT_CLASS_NAMES = frozenset({"WanVideoUnit_PromptEmbedder"})
        flag = getattr(unit, "is_text_unit", None)
        if flag is not None:
            return bool(flag)
        cls_name = getattr(unit, "__class__", type(unit)).__name__
        return cls_name in _TEXT_UNIT_CLASS_NAMES

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
        return self._pipe.vae.decode(latents.to(self.device))

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

    def _is_per_token_t_mod_active(self, state: BlockLoopState) -> bool:
        return state.t_mod.dim() == 4

    def _build_action_t_mod(self, action_timestep: Tensor, modality_bias: Tensor, n_action_tokens: int) -> Tensor:
        dit = self._dit
        if action_timestep.dim() == 2:
            B_t = action_timestep.shape[0]
            flat = action_timestep.reshape(B_t * n_action_tokens)
            t_emb = sinusoidal_embedding_1d(dit.freq_dim, flat)
            t = dit.time_embedding(t_emb.to(modality_bias.dtype))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
            t_mod = t_mod.view(B_t, n_action_tokens, 6, dit.dim)
        else:
            t_emb = sinusoidal_embedding_1d(dit.freq_dim, action_timestep.flatten())
            t = dit.time_embedding(t_emb.to(modality_bias.dtype))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
            t_mod = t_mod.unsqueeze(1).expand(-1, n_action_tokens, -1, -1)
        return t_mod + modality_bias.to(dtype=t_mod.dtype, device=t_mod.device)

    def _extend_freqs_with_action_tokens(self, freqs: Tensor, n_action_tokens: int) -> Tensor:
        if n_action_tokens <= 0:
            return freqs
        identity_freq = torch.polar(
            torch.ones(n_action_tokens, 1, freqs.shape[-1], device=freqs.device),
            torch.zeros(n_action_tokens, 1, freqs.shape[-1], device=freqs.device),
        )
        return torch.cat([freqs, identity_freq], dim=0)

    def _compute_reference_prefix_len(self, reference_latents: Optional[Tensor]) -> int:
        if reference_latents is None:
            return 0
        if reference_latents.dim() == 5:
            _, _, _, h, w = reference_latents.shape
        elif reference_latents.dim() == 4:
            _, _, h, w = reference_latents.shape
        else:
            raise ValueError(f"reference_latents must be 4D or 5D, got shape {tuple(reference_latents.shape)}")
        return int(h * w)

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
            if not getattr(compile_cfg, flag, False):
                continue
            mod = getattr(self._pipe, submod_name, None)
            if mod is None:
                continue
            if submod_name == "dit" and hasattr(mod, "blocks"):
                for i, block in enumerate(mod.blocks):
                    mod.blocks[i] = torch.compile(block, dynamic=True, mode="reduce-overhead")
                logger.info("torch.compile enabled for %s blocks (%d)", submod_name, len(mod.blocks))
            else:
                setattr(self._pipe, submod_name, torch.compile(mod, dynamic=True))
                logger.info("torch.compile enabled for %s", submod_name)

    @staticmethod
    def _build_pipe_from_components(
        components: list,
        tokenizer: dict = None,
        device: str = "cpu",
        ckpt_dir: str = None,
    ):
        """Build an empty WanVideoPipeline from component specs (config-driven).

        Same logic as ``build_video_backbone_from_manifest`` but reads from
        a config dict instead of a JSON file. Weights are NOT loaded here —
        ``architecture.load_checkpoint`` handles that separately.
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

        if tokenizer and ckpt_dir:
            tok = _build_tokenizer(tokenizer, ckpt_dir)
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


__all__ = ["WanVideoBackbone"]
