"""Cosmos3-Edge video backbone.

Implements the :class:`VideoBackbone` contract on NVIDIA's Cosmos3-Edge unified
dual-stream transformer (vendored in ``cosmos3/_vendor/``) — flat named children
(``self.dit`` / ``self.vae``) mirroring the other families. The heavy batched
block-loop lives in ``cosmos3/dit_forward.py``; prompt tokenization + mRoPE
packing in ``cosmos3/text_pack.py``; component construction in
``cosmos3/pipeline_builder.py`` (imported lazily inside :meth:`from_pretrained`
so this module stays CPU-CI safe).

Architecture recap (see docs/plans/cosmos3-edge-backbone.md): the und (text)
stream is causal, frozen, and computationally independent of the gen (video)
stream, so :meth:`preprocess_input_for_train` runs the whole und tower once
under ``no_grad`` and caches the per-layer gen-facing K/V; the und final hidden
(2048-wide) doubles as the ``context`` tensor the action stream consumes
(``text_dim == 2048``). There is no cross-attention and no external text
encoder; timestep conditioning is additive on noisy-frame tokens only.

Scope today: ``dual_system`` / ``joint_cross_attn`` training + deploy.
``joint_self_attn`` (MoT with und-prefix-KV + GQA KV-expand) lands in Phase 2;
IDM / shared-backbone raise via the ABC defaults.
"""

from __future__ import annotations

import logging
import random
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.cosmos3 import dit_forward, text_pack
from openwam.model.video_backbone.cosmos3.scheduler import Cosmos3FlowSchedulerAdapter

logger = logging.getLogger(__name__)

# Cosmos3-Edge native geometry (transformer/config.json; VAE = Wan2.2-TI2V).
_COSMOS3_DIT_PATCH_SIZE: Tuple[int, int, int] = (1, 2, 2)
_COSMOS3_TEMPORAL_COMPRESSION: int = 4
_COSMOS3_SPATIAL_COMPRESSION: int = 16
_COSMOS3_CAUSAL_TEMPORAL: bool = True
_COSMOS3_LATENT_CHANNELS: int = 48
_DEFAULT_FPS: float = 24.0


class Cosmos3EdgeVideoBackbone(VideoBackbone):
    """Wrap the Cosmos3-Edge generator + Wan2.2 VAE + tokenizer behind the ABC."""

    def __init__(
        self,
        *,
        net: Any,
        vae: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        latents_mean: Optional[Tensor] = None,
        latents_std: Optional[Tensor] = None,
        scheduler: Optional[Cosmos3FlowSchedulerAdapter] = None,
        shift_video: float = 5.0,
        use_system_prompt: bool = False,
        prompt_templates: bool = True,
        max_text_tokens: int = 512,
        text_dropout_p: float = 0.0,
        text_dropout_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # --- Flat named children (freeze list hits video_backbone.vae) ---
        self.dit = net
        if vae is not None:
            self.vae = vae
        self._tokenizer = tokenizer

        mean = latents_mean if latents_mean is not None else torch.zeros(_COSMOS3_LATENT_CHANNELS)
        std = latents_std if latents_std is not None else torch.ones(_COSMOS3_LATENT_CHANNELS)
        self._latents_mean = mean.float().view(1, -1, 1, 1, 1)
        self._latents_std = std.float().view(1, -1, 1, 1, 1)

        self._dim = int(dim)
        self._num_layers = int(num_layers)
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        self._context_dim = int(context_dim)
        self._scheduler = scheduler if scheduler is not None else Cosmos3FlowSchedulerAdapter(shift_video=shift_video)
        self._shift_video = float(shift_video)

        self._dit_patch_size = _COSMOS3_DIT_PATCH_SIZE
        self._temporal_compression = _COSMOS3_TEMPORAL_COMPRESSION
        self._causal_temporal = _COSMOS3_CAUSAL_TEMPORAL
        self._video_attention_mask_mode = "bidirectional"

        self._use_system_prompt = bool(use_system_prompt)
        self._prompt_templates = bool(prompt_templates)
        self._max_text_tokens = int(max_text_tokens)
        if not 0.0 <= float(text_dropout_p) <= 1.0:
            raise ValueError(f"text_dropout_p must be in [0, 1]; got {text_dropout_p!r}.")
        self._text_dropout_p = float(text_dropout_p)
        self._text_dropout_rng = random.Random(text_dropout_seed)

    # ================================================================
    # Structural metadata
    # ================================================================

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
    def scheduler(self):
        return self._scheduler

    @property
    def text_dim(self) -> Optional[int]:
        """Context width for the action stream = und hidden size (2048)."""
        return self._context_dim

    @property
    def video_attention_mask_mode(self) -> str:
        return self._video_attention_mask_mode

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode: str) -> None:
        self._video_attention_mask_mode = str(mode)

    def build_video_to_video_mask(
        self, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ) -> Tensor:
        """v↔v joint-mask block; same three modes as the predict2.5 backbone."""
        mode = self._video_attention_mask_mode
        if video_seq_len <= 0 or video_tokens_per_frame <= 0:
            raise ValueError(
                f"build_video_to_video_mask needs positive sizes; got seq={video_seq_len}, "
                f"tokens_per_frame={video_tokens_per_frame}."
            )
        if mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    f"per_frame_causal needs seq divisible by tokens_per_frame; got {video_seq_len} "
                    f"% {video_tokens_per_frame}."
                )
            frames = video_seq_len // video_tokens_per_frame
            frame_mask = torch.tril(torch.ones((frames, frames), dtype=torch.bool, device=device))
            return frame_mask.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )
        if mode == "first_frame_causal":
            mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            ff = min(video_tokens_per_frame, video_seq_len)
            mask[:ff, ff:] = False
            return mask
        raise NotImplementedError(f"{type(self).__name__} does not implement v-v mask mode '{mode}'.")

    # ================================================================
    # Construction
    # ================================================================

    @classmethod
    def from_pretrained(cls, source: Any, *, device=None, ckpt_dir=None, **kw) -> "Cosmos3EdgeVideoBackbone":
        from openwam.model.video_backbone.cosmos3.pipeline_builder import (
            _video_backbone_cfg,
            build_cosmos3_pipeline,
        )

        holder = build_cosmos3_pipeline(source, device=device, ckpt_dir=ckpt_dir, **kw)
        cfg_for_loader = _video_backbone_cfg(source)
        shift = float(getattr(holder, "shift_video", 5.0))
        backbone = cls(
            net=holder.net,
            vae=getattr(holder, "vae", None),
            tokenizer=getattr(holder, "tokenizer", None),
            dim=holder.dim,
            num_layers=holder.num_layers,
            num_heads=holder.num_heads,
            head_dim=holder.head_dim,
            context_dim=holder.context_dim,
            latents_mean=getattr(holder, "latents_mean", None),
            latents_std=getattr(holder, "latents_std", None),
            scheduler=Cosmos3FlowSchedulerAdapter(shift_video=shift),
            shift_video=shift,
            use_system_prompt=holder.use_system_prompt,
            prompt_templates=holder.prompt_templates,
            max_text_tokens=holder.max_text_tokens,
            text_dropout_p=holder.text_dropout_p,
            text_dropout_seed=holder.text_dropout_seed,
        )
        del cfg_for_loader
        return backbone

    # ================================================================
    # Internal helpers
    # ================================================================

    def _require_tokenizer(self):
        if self._tokenizer is None:
            raise ValueError("cosmos3_edge backbone has no tokenizer attached (bad build).")
        return self._tokenizer

    def _frames_to_tensor(self, frames: Any) -> Tensor:
        """Accept (B,3,T,H,W) in [-1,1], the dataloader's ``list[list[PIL]]``
        batch form (predict2.5 parity), or a single flat ``list[PIL]`` clip."""
        if torch.is_tensor(frames):
            if frames.ndim != 5:
                raise ValueError(f"cosmos3_edge expects (B, 3, T, H, W) frames; got {tuple(frames.shape)}.")
            return frames
        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _pil_video_to_tensor

        if isinstance(frames, (list, tuple)) and frames and not isinstance(frames[0], (list, tuple)):
            frames = [frames]  # single clip → batch of one
        return _pil_video_to_tensor(frames)

    def _encode_frames(self, frames: Tensor) -> Tensor:
        """Pixel video ``(B, 3, T, H, W)`` in [-1, 1] → normalized latents
        ``(B, 48, T_lat, H/16, W/16)``. Mode of the posterior (upstream
        ``sample_mode="argmax"``), then ``(μ − mean) / std`` in fp32, cast back."""
        vae = getattr(self, "vae", None)
        if vae is None:
            raise ValueError("cosmos3_edge backbone has no VAE attached.")
        vae_dtype = next(vae.parameters()).dtype
        posterior = vae.encode(frames.to(device=self.device, dtype=vae_dtype)).latent_dist
        raw = posterior.mode()
        mean = self._latents_mean.to(raw.device)
        std = self._latents_std.to(raw.device)
        return ((raw.float() - mean) / std).to(raw.dtype)

    def _unnormalize_latents(self, latents: Tensor) -> Tensor:
        mean = self._latents_mean.to(latents.device)
        std = self._latents_std.to(latents.device)
        return (latents.float() * std + mean).to(latents.dtype)

    def _encode_prompts(self, prompts: List[str], *, num_frames: int, height: int, width: int, fps: float) -> dict:
        """Tokenize + run the frozen und tower once. Returns context/masks/caches."""
        tokenizer = self._require_tokenizer()
        texts = prompts
        if self._prompt_templates:
            texts = [
                text_pack.apply_prompt_templates(p, num_frames=num_frames, height=height, width=width, fps=fps)
                for p in prompts
            ]
        ids = [
            text_pack.tokenize_prompt(
                tokenizer,
                t,
                use_system_prompt=self._use_system_prompt,
                is_image=num_frames == 1,
                max_length=self._max_text_tokens,
            )
            for t in texts
        ]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids, und_mask, seq_lens = text_pack.pad_und_batch(ids, pad_token_id=int(pad_id))
        input_ids = input_ids.to(self.device)
        und_mask = und_mask.to(self.device)

        net = self.dit
        float_pos = bool(net.config.enable_fps_modulation)
        text_pos = text_pack.text_mrope_positions(input_ids.shape[1], float_positions=float_pos)
        cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), self.device, self.dtype)
        context, und_kv = dit_forward.run_und_tower(net, input_ids, und_mask, cos_und, sin_und)
        return {
            "context": context,
            "context_mask": und_mask,
            "seq_lens": seq_lens.to(self.device),
            "und_kv": und_kv,
            "und_len_padded": int(input_ids.shape[1]),
        }

    def _vision_positions(self, und_len_padded: int, latent_grid: Tuple[int, int, int], fps: float) -> Tensor:
        net = self.dit
        grid = text_pack.patch_grid(*latent_grid, int(net.config.latent_patch_size))
        float_pos = bool(net.config.enable_fps_modulation)
        _, vision_pos = text_pack.build_joint_positions(
            und_len_padded,
            grid,
            modality_margin=int(net.config.unified_3d_mrope_temporal_modality_margin),
            fps=fps if float_pos else None,
            base_fps=float(net.config.base_fps),
            temporal_compression_factor=self._temporal_compression,
            float_positions=float_pos,
        )
        return vision_pos.unsqueeze(1)  # [3, 1, N] — broadcast over batch

    # ================================================================
    # Training preprocessing
    # ================================================================

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        if frames is None or text is None:
            raise ValueError("cosmos3_edge preprocess_input_for_train needs `frames` and `text`.")
        frames_t = self._frames_to_tensor(frames)
        b, _, t_pix, h_pix, w_pix = frames_t.shape

        prompts = [text] * b if isinstance(text, str) else list(text)
        if len(prompts) != b:
            raise ValueError(f"cosmos3_edge got {len(prompts)} prompts for batch of {b}.")
        if self.training and self._text_dropout_p > 0.0:
            prompts = [p if self._text_dropout_rng.random() >= self._text_dropout_p else "" for p in prompts]

        fps = float(kw.get("fps", _DEFAULT_FPS))
        with torch.no_grad():
            latents = self._encode_frames(frames_t)
            enc = self._encode_prompts(prompts, num_frames=t_pix, height=h_pix, width=w_pix, fps=fps)

        vision_pos = self._vision_positions(enc["und_len_padded"], tuple(latents.shape[2:]), fps)
        # NOTE: no `context_mask` key — the architecture appends a proprio token
        # to `context` and derives the action-side mask from `seq_lens`; a
        # provided mask would be one column short. The und padding mask rides
        # the private `und_mask` key for the gen-attention prefix instead.
        return {
            "input_latents": latents,
            "context": enc["context"],
            "und_mask": enc["context_mask"],
            "seq_lens": enc["seq_lens"],
            "und_kv": enc["und_kv"],
            "vision_positions": vision_pos,
            "first_frame_latents": latents[:, :, :1].clone(),
            "num_clean_prefix_frames": 1,
            "num_frames": t_pix,
            "height": h_pix,
            "width": w_pix,
        }

    # ================================================================
    # Three-step execution
    # ================================================================

    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        return dit_forward.prepare_block_loop(self.dit, **pipeline_inputs)

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        return dit_forward.run_block(self.dit, block_id, state)

    def finalize(self, state: BlockLoopState) -> Tensor:
        return dit_forward.finalize_block_loop(self.dit, state)

    # ================================================================
    # Joint self-attention split (MoT)
    # ================================================================

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        from openwam.model.video_backbone.cosmos3 import block_split

        return block_split.state_pre_attn(self.dit, layer_id, state)

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        _ = layer_id  # layer reference rides post_state
        from openwam.model.video_backbone.cosmos3 import block_split

        return block_split.state_post_attn(state, attn_out, post_state)

    # ================================================================
    # Deploy
    # ================================================================

    def preprocess_input_for_inference(self, **kw) -> dict:
        prompt = kw.get("prompt")
        if prompt is None:
            raise ValueError("cosmos3_edge preprocess_input_for_inference needs `prompt`.")
        first_frame_image = kw.get("first_frame_image")
        if first_frame_image is None:
            raise ValueError("cosmos3_edge deploy path is first-frame conditioned; pass `first_frame_image`.")
        num_frames = int(kw.get("num_frames", 29))
        height = int(kw.get("height", 480))
        width = int(kw.get("width", 832))
        fps = float(kw.get("fps", _DEFAULT_FPS))
        seed = kw.get("seed")
        cfg_scale = float(kw.get("cfg_scale", 1.0))
        cfg_merge = bool(kw.get("cfg_merge", False))
        if cfg_merge and cfg_scale > 1.0:
            raise NotImplementedError(
                "cosmos3_edge cannot run batched-CFG (cfg_merge): the cached und K/V lists are "
                "per-prompt and cannot be merged along the batch axis. Use separate CFG passes."
            )
        shift = kw.get("shift")
        prompt_embed_cache = kw.get("prompt_embed_cache")

        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _pil_video_to_tensor

        # Deploy may hand a single PIL image or a list (predict2.5 contract).
        ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        frame_t = _pil_video_to_tensor([[r] for r in ref_frames]).to(self.device)
        with torch.no_grad():
            first_frame_latents = self._encode_frames(frame_t)  # (1, 48, 1, h, w)

            def _cached_encode(p: str) -> dict:
                if prompt_embed_cache is not None and p in prompt_embed_cache:
                    return prompt_embed_cache[p]
                enc = self._encode_prompts([p], num_frames=num_frames, height=height, width=width, fps=fps)
                if prompt_embed_cache is not None:
                    prompt_embed_cache[p] = enc
                return enc

            enc = _cached_encode(str(prompt))
            uncond_enc = _cached_encode("") if cfg_scale > 1.0 else None

        t_lat = 1 + (num_frames - 1) // self._temporal_compression
        h_lat = first_frame_latents.shape[3]
        w_lat = first_frame_latents.shape[4]
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = torch.randn(
            (first_frame_latents.shape[0], _COSMOS3_LATENT_CHANNELS, t_lat, h_lat, w_lat),
            generator=generator,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        latents[:, :, :1] = first_frame_latents.to(latents.dtype)

        out = {
            "latents": latents,
            "context": enc["context"],
            "und_mask": enc["context_mask"],
            "seq_lens": enc["seq_lens"],
            "und_kv": enc["und_kv"],
            "vision_positions": self._vision_positions(enc["und_len_padded"], (t_lat, h_lat, w_lat), fps),
            "first_frame_latents": first_frame_latents,
            "num_clean_prefix_frames": 1,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "sigma_shift": float(shift) if shift is not None else self._shift_video,
            "num_inference_steps": int(kw.get("num_inference_steps", 10)),
            "cfg_scale": cfg_scale,
            "cfg_merge": False,
            "seed": int(seed) if seed is not None else 42,
            "tiled": bool(kw.get("tiled", True)),
            "uncond_context": None,
        }
        if uncond_enc is not None:
            out["uncond_context"] = uncond_enc["context"]
            out["uncond_und_mask"] = uncond_enc["context_mask"]
            out["uncond_und_kv"] = uncond_enc["und_kv"]
            out["uncond_vision_positions"] = self._vision_positions(
                uncond_enc["und_len_padded"], (t_lat, h_lat, w_lat), fps
            )
        return out

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        _ = tiled  # AutoencoderKLWan.decode has no tiling arg on this path
        vae = getattr(self, "vae", None)
        if vae is None:
            raise ValueError("cosmos3_edge backbone has no VAE attached.")
        vae_dtype = next(vae.parameters()).dtype
        z = self._unnormalize_latents(latents).to(dtype=vae_dtype)
        video = vae.decode(z).sample
        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _video_tensor_to_pil

        return _video_tensor_to_pil(video.float().clamp(-1, 1))

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        from omegaconf import OmegaConf

        from openwam.model.video_backbone.cosmos3.component_specs import (
            copy_cosmos3_artifacts,
            generate_cosmos3_component_specs,
        )

        oc = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
        model_path = OmegaConf.select(oc, "model.video_backbone.model_path")
        specs = generate_cosmos3_component_specs(model_path, has_vae=getattr(self, "vae", None) is not None)
        if specs is None:
            logger.info("cosmos3_edge: model_path unreadable; skipping deploy-asset embedding.")
            return
        if OmegaConf.select(oc, "model.video_backbone.components") is None:
            OmegaConf.update(oc, "model.video_backbone.components", specs["components"], force_add=True)
        copy_cosmos3_artifacts(output_dir, str(model_path))

    # ================================================================
    # Lifecycle
    # ================================================================

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        super().set_dtype_device(dtype, device)
        self._latents_mean = self._latents_mean.to(device=device)
        self._latents_std = self._latents_std.to(device=device)
