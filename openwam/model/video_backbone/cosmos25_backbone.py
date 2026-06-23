"""Cosmos-Predict2.5 video backbone.

Implements the :class:`VideoBackbone` contract (``base.py``) on top of a
Cosmos-Predict2.5 pipeline object. The pipeline is constructed lazily inside
:meth:`Cosmos25VideoBackbone.from_pretrained` so importing this module does not
require ``cosmos_predict2`` to be installed — CPU-only CI keeps passing and
tests that inject a fake pipeline through the constructor exercise the full
block-loop path without the upstream dependency.

Public surface: **only** the methods/properties already declared on
:class:`VideoBackbone`. Everything else in this module is an auxiliary helper
(``_``-prefixed) and must not be treated as an external interface. Helper "src"
modules live under ``openwam/model/video_backbone/cosmos25/``.

Scope:
- ``dual_system`` + ``joint_cross_attn`` (default) and ``joint_self_attn``
  (block split helpers in ``cosmos25/block_split.py``). Shared-backbone variants
  fall through to the ABC's ``inject_shared_tokens`` / ``extract_shared_tokens``
  defaults, which raise.
- Inference (:meth:`preprocess_input_for_inference`, incl. CFG / TI2V) and
  deploy (:meth:`save_deploy_assets`, self-contained component specs + Reason1
  artifacts) are wired. VACE conditioning is still rejected (parity with
  training). IDM remains T2V-only / no CFG.
- Freeze policy is owned by the training-strategy / model freeze list
  (Wan-aligned mechanism), reached through native ``nn.Module.get_submodule``
  dotted paths such as ``_pipe._vae_inner`` / ``_pipe._reason1_inner`` /
  ``_pipe.net``. The ``freeze`` kwarg here is retained for tests / direct
  programmatic use and defaults to ``False``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.cosmos25._vae_utils import (
    _move_cosmos_reason1,
    _move_cosmos_vae,
    _vae_inner_module,
)
from openwam.model.video_backbone.cosmos25.scheduler import CosmosFlowSchedulerAdapter

logger = logging.getLogger(__name__)

# Cosmos-Predict2.5 native geometry, invariant across the 2B/14B size family.
# Mirrors the MiniTrainDIT config in ``pipeline_builder._COSMOS25_2B_NET_KWARGS``
# (``patch_temporal=1`` / ``patch_spatial=2``) and the Wan2pt1 VAE temporal
# contract (causal first frame + 4-frame tail). External-encoder swap is not
# supported on Cosmos25, so these native values are the only ones ever exposed.
_COSMOS25_DIT_PATCH_SIZE: Tuple[int, int, int] = (1, 2, 2)
_COSMOS25_TEMPORAL_COMPRESSION: int = 4
_COSMOS25_CAUSAL_TEMPORAL: bool = True

# Sub-modules to move in the (rare) non-nn.Module fallback path of
# :meth:`set_dtype_device` — real pipelines are nn.Module and skip this loop.
_COSMOS25_SUBMODULE_NAMES: Tuple[str, ...] = ("dit", "vae", "text_encoder", "net")


class Cosmos25VideoBackbone(VideoBackbone):
    """Wrap a Cosmos-Predict2.5 pipeline behind the :class:`VideoBackbone` ABC."""

    def __init__(
        self,
        pipeline: Any,
        *,
        dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        scheduler: Optional[CosmosFlowSchedulerAdapter] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self._pipe = pipeline
        self._dim = int(dim)
        self._num_layers = int(num_layers)
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        # Cosmos25's per-token text/context embedding dim (1024 for 2B, vs Wan's
        # 4096). Exposed through the base ``text_dim`` property so the action
        # backbone can size its context embedding to match.
        self._context_dim = int(context_dim)
        self._scheduler = scheduler if scheduler is not None else CosmosFlowSchedulerAdapter()
        self._freeze = bool(freeze)

        # Native patch size + temporal contract feed the base
        # dit_patch_size / temporal_compression / causal_temporal properties.
        self._dit_patch_size = _COSMOS25_DIT_PATCH_SIZE
        self._temporal_compression = _COSMOS25_TEMPORAL_COMPRESSION
        self._causal_temporal = _COSMOS25_CAUSAL_TEMPORAL

        # NOTE: `nn.Module.__setattr__` already registers `self._pipe` in
        # `self._modules['_pipe']` when `pipeline` is an nn.Module — state_dict
        # picks it up under the `_pipe.` prefix automatically. Fake non-Module
        # pipelines used by CPU tests fall through to plain attribute storage.

        if self._freeze and isinstance(pipeline, nn.Module):
            for p in pipeline.parameters():
                p.requires_grad_(False)

        # The Cosmos VAE (`Wan2pt1VAEInterface`) is a plain object, not an
        # nn.Module, so the loop above does not reach it. Upstream
        # `WanVAE.__init__` already freezes it, but we set it again defensively.
        if self._freeze:
            inner = _vae_inner_module(getattr(pipeline, "vae", None))
            if inner is not None:
                for p in inner.parameters():
                    p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, source: Any, *, device=None, ckpt_dir=None, **kw) -> "Cosmos25VideoBackbone":
        """Build a backbone from a config / model dir.

        Defers to :func:`cosmos25.pipeline_builder.build_cosmos25_pipeline`,
        which lazily imports ``cosmos_predict2``. Probes the built pipeline for
        ``dim`` / ``num_layers`` / ``num_heads`` / ``head_dim`` / ``context_dim``
        and builds the scheduler with the Cosmos ``flow_shift``.
        """
        from openwam.model.video_backbone.cosmos25.pipeline_builder import build_cosmos25_pipeline

        cfg_for_loader = _video_backbone_cfg(source)
        flow_shift = float(_cfg_get(cfg_for_loader, "flow_shift", 5.0))

        pipeline = build_cosmos25_pipeline(source, device=device, ckpt_dir=ckpt_dir, **kw)
        dim, num_layers, num_heads, head_dim, context_dim = _probe_pipeline_geometry(pipeline)
        return cls(
            pipeline=pipeline,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            scheduler=CosmosFlowSchedulerAdapter(flow_shift=flow_shift),
        )

    # ------------------------------------------------------------------
    # Required VideoBackbone properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def scheduler(self) -> CosmosFlowSchedulerAdapter:
        return self._scheduler

    @property
    def text_dim(self) -> Optional[int]:
        """Per-token text/context embedding dim (1024 for Cosmos25-2B)."""
        return self._context_dim

    # ------------------------------------------------------------------
    # Joint self-attention mask plumbing (consumed by the joint MoT driver)
    # ------------------------------------------------------------------

    @property
    def video_attention_mask_mode(self) -> str:
        """Video↔video self-attention mask mode for the joint MoT mask.

        Default ``bidirectional`` keeps the ABC contract; ``per_frame_causal``
        and ``first_frame_causal`` mirror the FastWAM-Joint modes. The math only
        depends on ``video_tokens_per_frame``, identical to the Wan backbone.
        """
        return getattr(self, "_video_attention_mask_mode", "bidirectional")

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = mode

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the video↔video block of the joint MoT attention mask.

        Mirrors the Wan backbone — the math only depends on
        ``video_tokens_per_frame`` (= ``state.h * state.w`` for Cosmos25), so the
        5D vs 3D ``state`` difference between backbones is irrelevant here.
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
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(
            f"Unsupported video_attention_mask_mode '{mode}'. "
            "Choose from: bidirectional, per_frame_causal, first_frame_causal."
        )

    # ------------------------------------------------------------------
    # Three-step block loop — delegates to the wrapped pipeline
    # ------------------------------------------------------------------

    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        """Patchify, position-embed and pack a :class:`BlockLoopState`.

        Delegates to ``pipeline.prepare_block_loop``, keeping the heavy
        patchify / RoPE details on the Cosmos side.
        """
        prepare_fn = getattr(self._pipe, "prepare_block_loop", None)
        if prepare_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.prepare requires the wrapped pipeline to expose "
                "`prepare_block_loop(**inputs) -> BlockLoopState`."
            )
        return prepare_fn(**pipeline_inputs)

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        run_fn = getattr(self._pipe, "run_block", None)
        if run_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.run_block requires the wrapped pipeline to expose "
                "`run_block(block_id, state) -> BlockLoopState`."
            )
        return run_fn(block_id, state)

    def finalize(self, state: BlockLoopState) -> Tensor:
        finalize_fn = getattr(self._pipe, "finalize_block_loop", None)
        if finalize_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.finalize requires the wrapped pipeline to expose "
                "`finalize_block_loop(state) -> Tensor`."
            )
        return finalize_fn(state)

    # ------------------------------------------------------------------
    # Preprocessing & decoding
    # ------------------------------------------------------------------

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        vace_videos = kw.get("vace_videos")
        if vace_videos is not None and any(v is not None for v in vace_videos):
            raise NotImplementedError(
                "Cosmos25VideoBackbone does not support VACE conditioning yet. "
                "Drop `vace_video` from the dataset for MVP runs."
            )
        # `ref_images` (auto-injected by FirstFrameConditioningTransform) is
        # consumed by the wrapper to emit `first_frame_latents` + `condition_mask`
        # + `num_clean_prefix_frames` for the Cosmos TI2V path; when it is
        # None/list-of-None the wrapper returns the T2V-only dict.
        preprocess_fn = getattr(self._pipe, "preprocess_input", None)
        if preprocess_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.preprocess_input_for_train requires the wrapped pipeline "
                "to expose `preprocess_input(frames=..., text=..., **kw) -> dict`."
            )
        return preprocess_fn(frames=frames, text=text, **kw)

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        decode_fn = getattr(self._pipe, "decode_video", None)
        if decode_fn is None:
            raise NotImplementedError("Cosmos25VideoBackbone.decode_video requires `pipeline.decode_video`.")
        return decode_fn(latents, tiled=tiled)

    def preprocess_input_for_inference(self, **kw) -> dict:
        """Build the inference denoising-loop input dict from explicit kwargs.

        Routes the prompt through the wrapper's cache > live precedence:
        ``pre_encoded_text`` (offline cache hit) wins, else the configured live
        Reason1 encoder. For Classifier-Free Guidance (``cfg_scale > 1.0``) an
        ``uncond_context`` tensor is also materialised — same source precedence
        as the cond branch (caller-supplied ``uncond_pre_encoded_text`` for the
        offline ``empty.safetensors``, then live ``text_encoder("")``). When
        ``cfg_scale == 1.0`` the ``uncond_context`` slot is ``None`` and the
        denoise loop skips the CFG branch.

        ``base.py:generate`` forwards ``cfg_scale`` / ``cfg_merge`` /
        ``pre_encoded_text`` / ``uncond_pre_encoded_text``; Cosmos defaults for
        the generic geometry kwargs match the architecture's ``generate``
        signature.
        """
        prompt = kw.get("prompt")
        vace_video = kw.get("vace_video")
        first_frame_image = kw.get("first_frame_image")
        pre_encoded_text = kw.get("pre_encoded_text")
        uncond_pre_encoded_text = kw.get("uncond_pre_encoded_text")
        num_frames = kw.get("num_frames", 49)
        height = kw.get("height", 384)
        width = kw.get("width", 320)
        seed = kw.get("seed", 42)
        tiled = kw.get("tiled", True)
        num_inference_steps = kw.get("num_inference_steps", 50)
        shift = kw.get("shift", None)
        cfg_scale = kw.get("cfg_scale", 1.0)
        cfg_merge = kw.get("cfg_merge", False)

        if vace_video is not None:
            raise NotImplementedError("Cosmos25VideoBackbone does not support VACE conditioning at inference yet.")
        # The wrapper itself does cache > live precedence; this top-level gate
        # only enforces that *at least one* prompt source is viable so users see
        # a clear error rather than a deep wrapper traceback.
        has_cache = pre_encoded_text is not None
        has_live = getattr(self._pipe, "text_encoder", None) is not None
        if not (has_cache or has_live):
            raise ValueError(
                "Cosmos25VideoBackbone.preprocess_input_for_inference has no prompt source: pass "
                "`pre_encoded_text` (offline cache hit) or configure "
                "`video_backbone.text_encoder=reason1_live`."
            )

        cfg_scale_f = float(cfg_scale)
        if cfg_scale_f < 1.0:
            raise ValueError(f"cfg_scale must be >= 1.0; got {cfg_scale!r}.")

        param = next(self._pipe.net.parameters(), None) if hasattr(self._pipe, "net") else None
        dtype = param.dtype if param is not None else getattr(self, "_dtype", torch.float32)
        device = param.device if param is not None else getattr(self, "_device", torch.device("cpu"))

        # Inference-time `input_latents` placeholder. The wrapper needs a
        # shape-correct latent to populate the canonical preprocess dict, but
        # the content is immediately overwritten by `init_noise` below. We
        # fabricate it from the explicit num_frames / height / width at the
        # Wan2pt1 geometry used by Cosmos-Predict2.5-2B (stride 4×8×8, z_dim=16).
        T_lat = 1 + (int(num_frames) - 1) // 4
        H_lat = int(height) // 8
        W_lat = int(width) // 8
        placeholder_latents = torch.randn((1, 16, T_lat, H_lat, W_lat), dtype=dtype, device=device)

        cond_kwargs = self._build_preprocess_kwargs(
            prompt=prompt,
            pre_encoded_text=pre_encoded_text,
            input_latents=placeholder_latents,
        )
        preproc = self._pipe.preprocess_input(**cond_kwargs)

        # `latents` is the initial denoise state; seed it so two generate(...)
        # calls with the same seed produce reproducible video noise.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        init_noise = torch.randn(preproc["input_latents"].shape, generator=gen, dtype=torch.float32).to(
            device=device, dtype=dtype
        )

        inputs_shared: dict = dict(preproc)
        inputs_shared["latents"] = init_noise
        inputs_shared["first_frame_latents"] = None
        inputs_shared["num_clean_prefix_frames"] = 0
        inputs_shared["fuse_vae_embedding_in_latents"] = False
        inputs_shared["vace_context"] = None
        inputs_shared["vace_scale"] = 1.0
        inputs_shared["seed"] = int(seed)
        inputs_shared["tiled"] = bool(tiled)
        inputs_shared["sigma_shift"] = (
            float(shift) if shift is not None else float(getattr(self._pipe, "flow_shift", 5.0))
        )
        inputs_shared["num_inference_steps"] = int(num_inference_steps)
        inputs_shared["cfg_scale"] = cfg_scale_f
        inputs_shared["cfg_merge"] = bool(cfg_merge)

        if cfg_scale_f > 1.0:
            inputs_shared["uncond_context"] = self._build_uncond_context(
                uncond_pre_encoded_text=uncond_pre_encoded_text,
                context_template=inputs_shared["context"],
            )
        else:
            inputs_shared["uncond_context"] = None

        self._finalize_ti2v_inputs(inputs_shared, first_frame_image)
        return inputs_shared

    def _finalize_ti2v_inputs(self, inputs_shared: dict, first_frame_image) -> None:
        """Encode the inference-time first frame and write the TI2V keys.

        When the caller passes a PIL image (or a list of one per batch entry),
        the wrapper's VAE encodes it into a ``(B, 16, 1, H/8, W/8)`` latent; we
        build a matching LVG ``condition_mask`` and overwrite ``latents[:, :, 0:1]``
        so the first diffusion step starts from the right state. When
        ``first_frame_image is None`` this is a no-op (the T2V defaults set by
        :meth:`preprocess_input_for_inference` are already correct).
        """
        if first_frame_image is None:
            return
        encode_fn = getattr(self._pipe, "_encode_frames", None)
        if encode_fn is None:
            raise RuntimeError(
                "Cosmos25VideoBackbone.preprocess_input_for_inference received `first_frame_image` "
                "but the wrapped pipeline does not expose `_encode_frames`. Ensure "
                "`video_backbone.vae: wan2pt1` is configured."
            )
        ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        ref_clips = [[r] for r in ref_frames]
        latents = inputs_shared["latents"]
        first_frame_latents = encode_fn(ref_clips).to(device=latents.device, dtype=latents.dtype)
        T_lat = latents.shape[2]
        H_lat = latents.shape[3]
        W_lat = latents.shape[4]
        B = latents.shape[0]
        condition_mask = torch.zeros(
            (B, 1, T_lat, H_lat, W_lat),
            dtype=latents.dtype,
            device=latents.device,
        )
        condition_mask[:, :, 0] = 1.0
        latents[:, :, 0:1] = first_frame_latents
        inputs_shared["latents"] = latents
        inputs_shared["first_frame_latents"] = first_frame_latents
        inputs_shared["condition_mask"] = condition_mask
        inputs_shared["num_clean_prefix_frames"] = 1

    def _build_preprocess_kwargs(
        self,
        *,
        prompt,
        pre_encoded_text: Optional[Tensor],
        input_latents: Tensor,
    ) -> dict:
        """Pick the right kwargs for the wrapper's ``preprocess_input``.

        Mirrors the wrapper's cache > live precedence at the adapter layer so
        the wrapper stays unchanged. ``input_latents`` is always supplied so the
        wrapper never falls into its VAE-or-frames branch during inference.
        """
        base = {"frames": None, "text": None, "input_latents": input_latents}
        if pre_encoded_text is not None:
            return {**base, "pre_encoded_text": pre_encoded_text}
        if getattr(self._pipe, "text_encoder", None) is not None:
            return {**base, "text": prompt}
        raise RuntimeError(
            "Cosmos25VideoBackbone._build_preprocess_kwargs reached the no-source branch despite "
            "the preprocess_input_for_inference gate. This is a bug."
        )

    def _build_uncond_context(
        self,
        *,
        uncond_pre_encoded_text: Optional[Tensor],
        context_template: Tensor,
    ) -> Tensor:
        """Materialise the unconditional text context for CFG.

        Source precedence mirrors the cond branch:
        1. Caller-supplied ``uncond_pre_encoded_text`` (offline
           ``empty.safetensors``). Accepts ``(L, D)`` or ``(B, L, D)``;
           ``(L, D)`` is broadcast across batch.
        2. Live encoder ``text_encoder("")`` + ``crossattn_proj`` if the DiT
           has one (mirrors the wrapper's text branch for numerical
           consistency with cache-path embeddings).

        Returns a tensor with shape/device/dtype matching ``context_template``.
        Raises ``ValueError`` if neither source is available.
        """
        if uncond_pre_encoded_text is not None:
            t = uncond_pre_encoded_text.to(device=context_template.device, dtype=context_template.dtype)
            if t.ndim == 2:
                t = t.unsqueeze(0).expand(context_template.shape[0], -1, -1).contiguous()
            if t.shape != context_template.shape:
                raise ValueError(
                    f"uncond_pre_encoded_text shape {tuple(t.shape)} doesn't match cond "
                    f"context shape {tuple(context_template.shape)}."
                )
            return t

        text_encoder = getattr(self._pipe, "text_encoder", None)
        if text_encoder is not None:
            ctx = text_encoder("")
            ctx = ctx.to(device=context_template.device, dtype=context_template.dtype)
            net = getattr(self._pipe, "net", None)
            if net is not None and getattr(net, "use_crossattn_projection", False):
                proj_in = int(getattr(net, "crossattn_proj_in_channels", -1))
                if ctx.shape[-1] == proj_in:
                    ctx = net.crossattn_proj(ctx)
            if ctx.shape[0] == 1 and context_template.shape[0] > 1:
                ctx = ctx.expand(context_template.shape[0], -1, -1).contiguous()
            if ctx.shape != context_template.shape:
                raise ValueError(
                    f"live uncond context shape {tuple(ctx.shape)} doesn't match cond "
                    f"context shape {tuple(context_template.shape)}; check encoder output."
                )
            return ctx

        raise ValueError(
            "cfg_scale > 1.0 requires either `uncond_pre_encoded_text` (offline "
            "`empty.safetensors`) or a configured `video_backbone.text_encoder` "
            "(e.g. reason1_live); got neither."
        )

    # ------------------------------------------------------------------
    # Joint self-attention hooks — delegate to the wrapped pipeline
    # ------------------------------------------------------------------

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState):
        """Pre half of one block — see ``Cosmos25PipelineWrapper.pre_attn_at_layer``."""
        pre_fn = getattr(self._pipe, "pre_attn_at_layer", None)
        if pre_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.pre_attn_at_layer requires the wrapped pipeline to expose "
                "`pre_attn_at_layer(layer_id, state) -> (q, k, v, post_state)`."
            )
        return pre_fn(layer_id, state)

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Post half of one block — see ``Cosmos25PipelineWrapper.post_attn_at_layer``."""
        post_fn = getattr(self._pipe, "post_attn_at_layer", None)
        if post_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.post_attn_at_layer requires the wrapped pipeline to expose "
                "`post_attn_at_layer(layer_id, state, attn_out, post_state) -> BlockLoopState`."
            )
        return post_fn(layer_id, state, attn_out, post_state)

    # `inject_shared_tokens` / `extract_shared_tokens` fall through to the ABC
    # defaults, which already raise — shared-backbone variants are out of scope
    # for Cosmos25.

    # ------------------------------------------------------------------
    # Device / dtype
    # ------------------------------------------------------------------

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        self._dtype = dtype
        self._device = device
        if isinstance(self._pipe, nn.Module):
            self._pipe.to(dtype=dtype, device=device)
            # `nn.Module.to(...)` walks `_modules` / `_parameters` / `_buffers`,
            # but the Cosmos VAE (`Wan2pt1VAEInterface`) and Reason1 encoder are
            # assigned as plain attributes, so they need explicit moves.
            vae = getattr(self._pipe, "vae", None)
            if vae is not None:
                _move_cosmos_vae(vae, dtype=dtype, device=device)
            text_encoder = getattr(self._pipe, "text_encoder", None)
            if text_encoder is not None and not isinstance(text_encoder, nn.Module):
                _move_cosmos_reason1(text_encoder, dtype=dtype, device=device)
            return
        # Fallback for fake non-Module pipelines used by CPU tests.
        for name in _COSMOS25_SUBMODULE_NAMES:
            mod = getattr(self._pipe, name, None)
            if isinstance(mod, nn.Module):
                mod.to(dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's slice of the checkpoint self-contained.

        Both halves in one pass, mirroring
        ``wan/component_specs.save_video_backbone_deploy_assets``:

        1. Merge the component reconstruction specs into
           ``cfg.model.video_backbone.components`` (only when absent — never
           clobber an explicit config) so the deploy loader rebuilds empty
           DiT/VAE/Reason1 shells from ``config.yaml`` without the training-time
           ``model_path`` (gate at ``deploy/model_loader.py``).
        2. Copy the small Reason1 (Qwen2.5-VL) structural JSON files into
           ``<output_dir>/reason1/`` so ``Reason1LiveTextEncoder.from_empty``
           can rebuild the meta-device shell at deploy time. The weights ride in
           the unified safetensors via ``_vae_inner`` / ``_reason1_inner``.

        The Reason1 half (the ``text_encoder`` component marker AND the artifact
        copy) is emitted ONLY when a live Reason1 encoder is actually part of
        this checkpoint — i.e. ``self._pipe.text_encoder`` is set, so its weights
        are registered as ``_reason1_inner`` and ride in the safetensors. In the
        default cache-only config (``text_encoder=none``, no ``text_encoder_path``)
        there is no Reason1 in the state_dict, so claiming the marker / copying
        JSONs would make deploy hunt for weights and artifacts that don't exist
        (and the copy itself would raise). The VAE component is always emitted.

        No-op when ``model_path`` is unreadable (fake-pipeline tests / offline
        builds), so it never blocks a save it cannot make self-contained.
        """
        from omegaconf import DictConfig, OmegaConf, open_dict

        from openwam.model.video_backbone.cosmos25.component_specs import (
            copy_cosmos25_artifacts,
            generate_cosmos25_component_specs,
        )

        # Production always passes a DictConfig (the trainer's ``self.cfg``); a
        # plain dict (tests / programmatic callers) is supported defensively by
        # operating on an OmegaConf view and writing the merged ``components``
        # back into the original dict so the caller still sees it.
        is_plain = not isinstance(cfg, DictConfig)
        oc = OmegaConf.create(cfg) if is_plain else cfg

        model_path = OmegaConf.select(oc, "model.video_backbone.model_path", default=None)
        specs = generate_cosmos25_component_specs(str(model_path) if model_path is not None else "")
        if specs is None:
            logger.info(
                "[cosmos25] video_backbone.model_path not readable (%s); skipping deploy-asset save.",
                model_path,
            )
            return

        # Ground truth for "is Reason1 in this checkpoint": the wrapper sets
        # `text_encoder` to None on the cache-only path and registers
        # `_reason1_inner` only when it is non-None.
        has_reason1 = getattr(self._pipe, "text_encoder", None) is not None
        components = [c for c in specs["components"] if c.get("attr") != "text_encoder" or has_reason1]

        if "components" not in oc.model.video_backbone:
            with open_dict(oc):
                OmegaConf.update(oc, "model.video_backbone.components", components)
            if is_plain:
                cfg["model"]["video_backbone"]["components"] = components

        if has_reason1:
            copy_cosmos25_artifacts(output_dir, oc)


# ----------------------------------------------------------------------
# Helpers (private)
# ----------------------------------------------------------------------


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _video_backbone_cfg(source: Any) -> Any:
    """Extract the ``video_backbone`` sub-config from a Hydra cfg / path / dict.

    Kept in lockstep with ``pipeline_builder._video_backbone_cfg`` so a path
    ``source`` resolves the same way on both sides (otherwise ``from_pretrained``
    would read ``flow_shift`` off a raw string and silently fall back to the
    5.0 default while the builder used the real value).
    """
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        return {"model_path": str(source)}
    if isinstance(source, dict):
        model = source.get("model") if "model" in source else source
        vb = model.get("video_backbone") if isinstance(model, dict) else None
        return vb if vb is not None else source
    model = getattr(source, "model", source)
    vb = getattr(model, "video_backbone", None)
    return vb if vb is not None else source


def _probe_pipeline_geometry(pipeline: Any) -> Tuple[int, int, int, int, int]:
    """Inspect a Cosmos-Predict2.5 pipeline for (dim, num_layers, num_heads, head_dim, context_dim).

    The builder attaches these as plain attributes on the pipeline object; test
    fakes do the same.
    """
    try:
        return (
            int(pipeline.dim),
            int(pipeline.num_layers),
            int(pipeline.num_heads),
            int(pipeline.head_dim),
            int(pipeline.context_dim),
        )
    except AttributeError as exc:
        raise AttributeError(
            "Cosmos pipeline is missing one of {dim, num_layers, num_heads, head_dim, "
            "context_dim}. Attach these to the pipeline in "
            "`pipeline_builder.build_cosmos25_pipeline` so the adapter can expose them."
        ) from exc


__all__ = ["Cosmos25VideoBackbone"]
