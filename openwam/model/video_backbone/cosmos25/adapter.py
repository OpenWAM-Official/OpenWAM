"""Cosmos-Predict2.5 video backbone adapter.

Implements the :class:`VideoBackbone` contract on top of a Cosmos-Predict2.5
pipeline object. The pipeline is constructed lazily inside
:meth:`from_pretrained` so importing this module does not require
``cosmos_predict2`` to be installed. Tests that inject a fake pipeline
through the constructor exercise the full code path without the upstream
dependency.

Scope:
- ``dual_system`` + ``joint_cross_attn`` (default) and ``joint_self_attn``
  (§17 — uses :class:`MoTJointDriver` with the block split helpers in
  ``block_split.py``). Shared-backbone variants are not supported (the
  ``inject_*`` / ``extract_*`` hooks raise :class:`NotImplementedError`).
- Freeze policy is owned by ``configs/training_strategy/*.yaml`` via
  :func:`BaseWAMModel.freeze_modules` (Wan-aligned mechanism). The
  ``freeze`` kwarg here is retained for tests / direct programmatic use,
  defaults to ``False``, and is no longer driven by ``cfg.video_backbone``.
  NOTE: this kwarg path is a partial freeze — it only flips
  ``requires_grad_(False)`` on the wrapped pipeline (and its VAE inner),
  whereas :func:`BaseWAMModel.freeze_modules` additionally wraps the forward
  in ``no_grad`` and pins ``training=False``. The two paths are not
  strictly equivalent; production freezing must go through the training
  strategy yaml.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.cosmos25.scheduler import CosmosFlowSchedulerAdapter

if TYPE_CHECKING:
    from openwam.model.inference_inputs import InferenceInputs

logger = logging.getLogger(__name__)


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
        submodule_names: Optional[list[str]] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self._pipe = pipeline
        self._dim = int(dim)
        self._num_layers = int(num_layers)
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        self._context_dim = int(context_dim)
        self._scheduler = scheduler if scheduler is not None else CosmosFlowSchedulerAdapter()
        self._submodule_names = list(submodule_names) if submodule_names is not None else ["dit", "vae", "text_encoder"]
        self._freeze = bool(freeze)

        # NOTE: `nn.Module.__setattr__` already registers `self._pipe` in
        # `self._modules['_pipe']` when `pipeline` is itself an nn.Module —
        # state_dict picks it up under the `_pipe.` prefix automatically.
        # We previously also kept `self._pipe_module = pipeline` as a
        # belt-and-suspenders registration; that duplicated every leaf
        # tensor in state_dict and broke `safetensors.save_file` (which
        # refuses to write tensors that share memory). Fake non-Module
        # pipelines used by CPU tests fall through to plain attribute
        # storage, which is fine for those code paths.

        if self._freeze and isinstance(pipeline, nn.Module):
            for p in pipeline.parameters():
                p.requires_grad_(False)

        # The Cosmos VAE (`Wan2pt1VAEInterface`) is a plain object, not an
        # `nn.Module`, so the loop above does NOT reach it. Upstream
        # `WanVAE.__init__` already does `model.eval().requires_grad_(False)`
        # (`wan2pt1.py:788`), but we set it a second time defensively in case
        # a future call path re-enables grads on the inner module.
        if self._freeze:
            vae = getattr(pipeline, "vae", None)
            inner = _vae_inner_module(vae)
            if inner is not None:
                for p in inner.parameters():
                    p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, source: Any, *, device=None, ckpt_dir=None, **kw) -> "Cosmos25VideoBackbone":
        """Build a backbone from a config / model dir.

        Defers to :func:`pipeline_builder.build_cosmos25_pipeline` which lazily
        imports ``cosmos_predict2``. Once the upstream API is wired in, this
        method probes the pipeline for ``dim``, ``num_layers``, ``num_heads``,
        ``head_dim``, and ``context_dim``, builds the scheduler with the
        Cosmos ``flow_shift``, and returns the configured backbone.
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
    def context_dim(self) -> int:
        return self._context_dim

    @property
    def scheduler(self) -> CosmosFlowSchedulerAdapter:
        return self._scheduler

    @property
    def submodule_names(self) -> list[str]:
        return list(self._submodule_names)

    # ------------------------------------------------------------------
    # Joint self-attention mask plumbing (consumed by MoTJointDriver)
    # ------------------------------------------------------------------

    @property
    def video_attention_mask_mode(self) -> str:
        """Video self-attention mask mode for the FastWAM-style joint mask.

        Mirrors :meth:`WanVideoBackbone.video_attention_mask_mode` so the
        ``dual_system_self_attn`` architecture builds the same mask shape on
        either backbone. Default ``bidirectional`` keeps the ABC contract;
        ``per_frame_causal`` and ``first_frame_causal`` mirror the FastWAM-Joint
        modes ported directly from Wan (math is generic in
        ``video_tokens_per_frame``).
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

        Direct port of :meth:`WanVideoBackbone.build_video_to_video_mask`
        (``wan_adapter.py:184-229``). The math only depends on
        ``video_tokens_per_frame`` (= ``state.h * state.w`` for Cosmos25), so
        the 5D vs 3D ``state.x`` difference between backbones is irrelevant
        here — the driver passes the already-flattened ``video_seq_len``.
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

        Delegates to ``pipeline.prepare_block_loop`` (the upstream wrapper used
        by the cosmos_predict2 builder), keeping the heavy patchify / RoPE
        details on the Cosmos side.
        """
        prepare_fn = getattr(self._pipe, "prepare_block_loop", None)
        if prepare_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.prepare requires the wrapped pipeline to "
                "expose `prepare_block_loop(**inputs) -> BlockLoopState`. The "
                "upstream cosmos_predict2 adapter has not been wired up yet."
            )
        return prepare_fn(**pipeline_inputs)

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        run_fn = getattr(self._pipe, "run_block", None)
        if run_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.run_block requires the wrapped pipeline to "
                "expose `run_block(block_id, state) -> BlockLoopState`."
            )
        return run_fn(block_id, state)

    def finalize(self, state: BlockLoopState) -> Tensor:
        finalize_fn = getattr(self._pipe, "finalize_block_loop", None)
        if finalize_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.finalize requires the wrapped pipeline to "
                "expose `finalize_block_loop(state) -> Tensor`."
            )
        return finalize_fn(state)

    # ------------------------------------------------------------------
    # Preprocessing & decoding
    # ------------------------------------------------------------------

    def preprocess_input(self, *, frames=None, text=None, **kw) -> dict:
        vace_videos = kw.get("vace_videos")
        if vace_videos is not None and any(v is not None for v in vace_videos):
            raise NotImplementedError(
                "Cosmos25VideoBackbone does not support VACE conditioning yet. "
                "Drop `vace_video` from the dataset for MVP runs."
            )
        # `ref_images` is auto-injected by `FirstFrameConditioningTransform`.
        # The wrapper consumes it to emit `first_frame_latents` + `condition_mask`
        # + `num_clean_prefix_frames` so the base trainer (`base.py:626-628,
        # 717-719, 817-819`) overwrites `latents[:, :, 0:1]` and skips frame 0
        # from the video loss — aligning the Cosmos TI2V path with Wan TI2V
        # (`wan_adapter.py:815-833`). When `ref_images is None`/list-of-`None`s,
        # the wrapper returns the same T2V-only dict as before.
        preprocess_fn = getattr(self._pipe, "preprocess_input", None)
        if preprocess_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.preprocess_input requires the wrapped "
                "pipeline to expose `preprocess_input(frames=..., text=..., **kw) -> dict`."
            )
        return preprocess_fn(frames=frames, text=text, **kw)

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        decode_fn = getattr(self._pipe, "decode_video", None)
        if decode_fn is None:
            raise NotImplementedError("Cosmos25VideoBackbone.decode_video requires `pipeline.decode_video`.")
        return decode_fn(latents, tiled=tiled)

    def prepare_inputs_for_inference(self, inputs: "InferenceInputs") -> dict:
        """Build the inference inputs dict consumed by :meth:`BaseWAMArchitecture.generate`.

        Routes the prompt through the wrapper's cache > live precedence
        (see ``Cosmos25PipelineWrapper.preprocess_input``):

        - ``pre_encoded_text`` non-None → caller-supplied cache hit (offline
          precompute path, §10).
        - ``video_backbone.text_encoder=reason1_live`` configured → live
          Reason1 encoder (§14).

        Classifier-Free Guidance (§15): when ``cfg_scale > 1.0`` an
        ``uncond_context`` tensor is also materialised — same source precedence
        as the cond branch (caller-supplied ``uncond_pre_encoded_text`` for the
        offline ``empty.safetensors``, then live ``text_encoder("")``). When
        ``cfg_scale == 1.0`` the ``uncond_context`` slot is ``None`` and the
        denoising loop in ``BaseWAMArchitecture.generate`` skips the CFG
        branch entirely.
        """
        # Unpack with Cosmos25-specific defaults for fields the dataclass keeps
        # generic (``num_frames`` / ``shift``). Cosmos uses 13-frame LVG and a
        # backbone-owned ``flow_shift``; Wan defaults differ.
        prompt = inputs.prompt
        vace_video = inputs.vace_video
        first_frame_image = inputs.first_frame_image
        pre_encoded_text = inputs.pre_encoded_text
        uncond_pre_encoded_text = inputs.uncond_pre_encoded_text
        num_frames = inputs.num_frames
        height = inputs.height
        width = inputs.width
        seed = inputs.seed
        tiled = inputs.tiled
        num_inference_steps = inputs.num_inference_steps
        shift = inputs.shift
        cfg_scale = inputs.cfg_scale
        cfg_merge = inputs.cfg_merge

        if vace_video is not None:
            raise NotImplementedError("Cosmos25VideoBackbone does not support VACE conditioning at inference yet.")
        # The wrapper itself does cache > live precedence; the only thing
        # this top-level gate enforces is that *at least one* prompt source
        # is viable, so users see a clear error rather than a deep wrapper
        # traceback.
        has_cache = pre_encoded_text is not None
        has_live = getattr(self._pipe, "text_encoder", None) is not None
        if not (has_cache or has_live):
            raise ValueError(
                "Cosmos25VideoBackbone.prepare_inputs_for_inference has no prompt source: pass "
                "`pre_encoded_text` (offline cache hit via JointInferenceEngine) or configure "
                "`video_backbone.text_encoder=reason1_live`."
            )

        cfg_scale_f = float(cfg_scale)
        if cfg_scale_f < 1.0:
            raise ValueError(f"cfg_scale must be >= 1.0; got {cfg_scale!r}.")

        param = next(self._pipe.net.parameters(), None) if hasattr(self._pipe, "net") else None
        dtype = param.dtype if param is not None else getattr(self, "_dtype", torch.float32)
        device = param.device if param is not None else getattr(self, "_device", torch.device("cpu"))

        # Inference-time `input_latents` placeholder. The wrapper needs a
        # shape-correct latent tensor to populate the canonical preprocess
        # dict, but the **content** is immediately overwritten by `init_noise`
        # below (and again by `input_video_latents` in
        # ``BaseWAMArchitecture.generate`` if the caller passes one). So we
        # fabricate noise here from the explicit ``num_frames / height /
        # width`` kwargs the caller has already promised, at the Wan2pt1
        # geometry used by Cosmos-Predict2.5-2B (stride 4×8×8, ``z_dim=16``).
        T_lat = 1 + (int(num_frames) - 1) // 4
        H_lat = int(height) // 8
        W_lat = int(width) // 8
        placeholder_latents = torch.randn((1, 16, T_lat, H_lat, W_lat), dtype=dtype, device=device)

        # Cond preprocess: route through the wrapper's precedence ladder. The
        # adapter chooses which kwarg to pass to the wrapper.
        cond_kwargs = self._build_preprocess_kwargs(
            prompt=prompt,
            pre_encoded_text=pre_encoded_text,
            input_latents=placeholder_latents,
        )
        preproc = self._pipe.preprocess_input(**cond_kwargs)

        # `latents` is the initial denoise state (Wan convention); seed it from
        # the requested seed so two `generate(...)` calls with the same seed
        # produce reproducible video noise drift.
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

        Cosmos analogue of ``WanVideoBackbone._finalize_ti2v_inputs``
        (``wan_adapter.py:1071-1087``). When the caller passes a PIL image
        (or a list of one per batch entry), the wrapper's VAE encodes it into
        a ``(B, 16, 1, H/8, W/8)`` latent; we then build a matching LVG
        ``condition_mask`` and overwrite ``latents[:, :, 0:1]`` so the first
        diffusion step starts from the right state — mirroring Wan's
        ``wan/pipeline.py:387-388``. ``base.py:718-719`` also re-applies the
        overwrite at every diffusion step, so this is belt-and-braces.

        When ``first_frame_image is None`` the function is a no-op; the
        ``first_frame_latents=None`` / ``num_clean_prefix_frames=0`` defaults
        already set by ``prepare_inputs_for_inference`` are correct.
        """
        if first_frame_image is None:
            return
        encode_fn = getattr(self._pipe, "_encode_frames", None)
        if encode_fn is None:
            raise RuntimeError(
                "Cosmos25VideoBackbone.prepare_inputs_for_inference received `first_frame_image` "
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
        prompt: str,
        pre_encoded_text: Optional[Tensor],
        input_latents: Tensor,
    ) -> dict:
        """Pick the right kwargs for :meth:`Cosmos25PipelineWrapper.preprocess_input`.

        Mirrors the wrapper's cache > live precedence at the adapter layer,
        so the wrapper itself stays unchanged. The top-level gate in
        ``prepare_inputs_for_inference`` already rejects the no-source case,
        so this method assumes at least one of cache / live is configured.
        ``input_latents`` is always supplied (computed in the caller from the
        requested ``num_frames / height / width``) so the wrapper never falls
        into its VAE-or-frames branch during inference.
        """
        base = {"frames": None, "text": None, "input_latents": input_latents}
        if pre_encoded_text is not None:
            return {**base, "pre_encoded_text": pre_encoded_text}
        if getattr(self._pipe, "text_encoder", None) is not None:
            return {**base, "text": prompt}
        raise RuntimeError(
            "Cosmos25VideoBackbone._build_preprocess_kwargs reached the no-source "
            "branch despite the prepare_inputs_for_inference gate. This is a bug."
        )

    def _build_uncond_context(
        self,
        *,
        uncond_pre_encoded_text: Optional[Tensor],
        context_template: Tensor,
    ) -> Tensor:
        """Materialise the unconditional text context for §15 CFG.

        Source precedence mirrors the cond branch:
        1. Caller-supplied ``uncond_pre_encoded_text`` (offline
           ``empty.safetensors``). Accepts ``(L, D)`` or ``(B, L, D)``;
           ``(L, D)`` is broadcast across batch.
        2. Live encoder ``text_encoder("")`` + ``crossattn_proj`` if the DiT
           has one (mirrors ``pipeline_wrapper.py:333-343`` for numerical
           consistency with cache-path embeddings).

        Returns a tensor with shape/device/dtype matching ``context_template``.
        Raises ``ValueError`` if neither source is available.
        """
        if uncond_pre_encoded_text is not None:
            t = uncond_pre_encoded_text.to(device=context_template.device, dtype=context_template.dtype)
            if t.ndim == 2:
                # (L, D) → (B, L, D), broadcast across batch.
                t = t.unsqueeze(0).expand(context_template.shape[0], -1, -1).contiguous()
            if t.shape != context_template.shape:
                raise ValueError(
                    f"uncond_pre_encoded_text shape {tuple(t.shape)} doesn't match cond "
                    f"context shape {tuple(context_template.shape)}."
                )
            return t

        text_encoder = getattr(self._pipe, "text_encoder", None)
        if text_encoder is not None:
            # Live empty embedding. We can't reuse `wrapper.preprocess_input`
            # here (it would also require `input_latents`/`frames` to land),
            # so manually mirror the wrapper's text branch
            # (`pipeline_wrapper.py:325-343`): encode → coerce dtype/device →
            # optional `crossattn_proj`.
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
            "`empty.safetensors` via JointInferenceEngine) or a configured "
            "`video_backbone.text_encoder` (e.g. reason1_live); got neither."
        )

    # ------------------------------------------------------------------
    # Joint self-attention hooks — delegate to the wrapped pipeline (§17)
    # ------------------------------------------------------------------

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState):
        """Pre half of one block — see :class:`Cosmos25PipelineWrapper.pre_attn_at_layer`."""
        pre_fn = getattr(self._pipe, "pre_attn_at_layer", None)
        if pre_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.pre_attn_at_layer requires the wrapped pipeline "
                "to expose `pre_attn_at_layer(layer_id, state) -> (q, k, v, post_state)`."
            )
        return pre_fn(layer_id, state)

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Post half of one block — see :class:`Cosmos25PipelineWrapper.post_attn_at_layer`."""
        post_fn = getattr(self._pipe, "post_attn_at_layer", None)
        if post_fn is None:
            raise NotImplementedError(
                "Cosmos25VideoBackbone.post_attn_at_layer requires the wrapped pipeline "
                "to expose `post_attn_at_layer(layer_id, state, attn_out, post_state) -> BlockLoopState`."
            )
        return post_fn(layer_id, state, attn_out, post_state)

    # ------------------------------------------------------------------
    # Unsupported hooks — clear NotImplementedError in MVP
    # ------------------------------------------------------------------

    def inject_action_tokens(self, state, action_tokens, n_action, *, timestep=None):
        raise NotImplementedError(
            "Cosmos25VideoBackbone does not support `inject_action_tokens` (shared-backbone path)."
        )

    def extract_action_tokens(self, state, n_action):
        raise NotImplementedError(
            "Cosmos25VideoBackbone does not support `extract_action_tokens` (shared-backbone path)."
        )

    # `inject_shared_tokens` / `extract_shared_tokens` still fall through to
    # the ABC defaults, which already raise — shared-backbone variants are
    # out of scope for Cosmos25 (see docs/cosmos25_backbone.md §6).

    # ------------------------------------------------------------------
    # Sub-module access (DeepSpeed wrap-back, freeze plumbing)
    # ------------------------------------------------------------------

    def get_submodule(self, name: str) -> Optional[nn.Module]:
        if "." in name:
            try:
                return super().get_submodule(name)
            except AttributeError:
                return None
        attr = getattr(self._pipe, name, None)
        # ABC contract: return only nn.Module (or None). The Cosmos VAE
        # (`Wan2pt1VAEInterface`) is a plain Python object — it must NOT
        # be returned here, or BaseWAMArchitecture.move_frozen_to_device
        # (which assumes the result has `.to(...)`) crashes. VAE device
        # placement is handled separately by `set_dtype_device` →
        # `_move_cosmos_vae`.
        return attr if isinstance(attr, nn.Module) else None

    def set_submodule(self, name: str, module: nn.Module) -> None:
        if not hasattr(self._pipe, name):
            raise AttributeError(
                f"Cosmos pipeline has no sub-module '{name}'. Known submodules: {self._submodule_names}"
            )
        setattr(self._pipe, name, module)

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        self._dtype = dtype
        self._device = device
        if isinstance(self._pipe, nn.Module):
            self._pipe.to(dtype=dtype, device=device)
            # `nn.Module.to(...)` walks `_modules` / `_parameters` / `_buffers`
            # but the Cosmos VAE (`Wan2pt1VAEInterface`) is assigned as a plain
            # attribute, so its inner nn.Module + 6 mean/std tensors need an
            # explicit move.
            vae = getattr(self._pipe, "vae", None)
            if vae is not None:
                _move_cosmos_vae(vae, dtype=dtype, device=device)
            # The Reason1 live text encoder (`Reason1LiveTextEncoder`) is a
            # plain Python class for the same `state_dict()` invariant the VAE
            # follows — see `text_encoder.py` module docstring. It's invisible
            # to `nn.Module.to(...)`, so move it explicitly here.
            text_encoder = getattr(self._pipe, "text_encoder", None)
            if text_encoder is not None and not isinstance(text_encoder, nn.Module):
                _move_cosmos_reason1(text_encoder, dtype=dtype, device=device)
            return
        for name in self._submodule_names:
            mod = getattr(self._pipe, name, None)
            if isinstance(mod, nn.Module):
                mod.to(dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Deploy artifact packaging
    # ------------------------------------------------------------------

    def get_component_specs(self, model_path: str) -> Optional[dict]:
        from openwam.model.video_backbone.cosmos25.component_specs import generate_cosmos25_component_specs

        return generate_cosmos25_component_specs(model_path)

    def copy_deploy_artifacts(self, output_dir: str, cfg) -> None:
        from openwam.model.video_backbone.cosmos25.component_specs import copy_cosmos25_artifacts

        copy_cosmos25_artifacts(output_dir, cfg)


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
    """Extract the ``video_backbone`` sub-config when *source* is a Hydra cfg."""
    if source is None:
        return None
    vb = None
    if isinstance(source, dict):
        model = source.get("model") if "model" in source else source
        vb = model.get("video_backbone") if isinstance(model, dict) else None
    else:
        model = getattr(source, "model", source)
        vb = getattr(model, "video_backbone", None)
    return vb if vb is not None else source


def _probe_pipeline_geometry(pipeline: Any) -> Tuple[int, int, int, int, int]:
    """Inspect a Cosmos-Predict2.5 pipeline for (dim, num_layers, num_heads, head_dim, context_dim).

    Until the upstream API is wired in, we expect the builder to attach these
    as plain attributes on the pipeline object. Test fakes do the same.
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
            "Cosmos pipeline is missing one of {dim, num_layers, num_heads, "
            "head_dim, context_dim}. Attach these to the pipeline in "
            "`pipeline_builder.build_cosmos25_pipeline` so the adapter can "
            "expose them through the VideoBackbone contract."
        ) from exc


# ----------------------------------------------------------------------
# VAE helpers (Phase 4)
# ----------------------------------------------------------------------
#
# `Wan2pt1VAEInterface` (the upstream Cosmos VAE wrapper) is a plain Python
# object, not an `nn.Module`. The actual nn.Module lives at
# `iface.model.model` (an instance of `WanVAE_`). Six mean/std tensors sit
# alongside it on `iface.model`: `mean`, `std`, `img_mean`, `img_std`,
# `video_mean`, `video_std`. Both the parameters and these tensors need
# explicit moves when set_dtype_device is called — `nn.Module.to(...)` on
# the surrounding Cosmos25PipelineWrapper does not walk plain attributes.
#
# Living here (rather than in pipeline_builder.py) so the helper is reachable
# without triggering the lazy upstream import.

_COSMOS_VAE_TENSOR_ATTRS: tuple = (
    "mean",
    "std",
    "img_mean",
    "img_std",
    "video_mean",
    "video_std",
)


def _vae_inner_module(vae: Any) -> Optional[nn.Module]:
    """Return the inner nn.Module of a Cosmos VAE wrapper, or None."""
    if vae is None:
        return None
    outer = getattr(vae, "model", None)
    inner = getattr(outer, "model", None) if outer is not None else None
    return inner if isinstance(inner, nn.Module) else None


def _move_cosmos_vae(vae: Any, *, dtype: torch.dtype, device: torch.device) -> None:
    """Move the Cosmos VAE inner nn.Module + mean/std tensors to (dtype, device)."""
    if vae is None:
        return
    outer = getattr(vae, "model", None)
    if outer is None:
        return
    inner = getattr(outer, "model", None)
    if isinstance(inner, nn.Module):
        inner.to(dtype=dtype, device=device)
    # Also update the cached `WanVAE.device` / `WanVAE.dtype` attrs so internal
    # encode/decode paths that read them stay consistent.
    if hasattr(outer, "device"):
        outer.device = device
    if hasattr(outer, "dtype"):
        outer.dtype = dtype
    for attr in _COSMOS_VAE_TENSOR_ATTRS:
        t = getattr(outer, attr, None)
        if isinstance(t, torch.Tensor):
            setattr(outer, attr, t.to(dtype=dtype, device=device))


def _move_cosmos_reason1(te: Any, *, dtype: torch.dtype, device: torch.device) -> None:
    """Move the plain-class :class:`Reason1LiveTextEncoder` to (dtype, device).

    Mirrors :func:`_move_cosmos_vae` — the encoder is intentionally not an
    ``nn.Module`` (so its 14 GB Qwen-VL weights stay out of ``state_dict()``),
    which means ``nn.Module.to(...)`` on the surrounding
    :class:`Cosmos25PipelineWrapper` cannot walk into it. Delegate to the
    encoder's own ``to`` shim instead.
    """
    if te is None:
        return
    mover = getattr(te, "to", None)
    if callable(mover):
        mover(dtype=dtype, device=device)


__all__ = ["Cosmos25VideoBackbone"]
