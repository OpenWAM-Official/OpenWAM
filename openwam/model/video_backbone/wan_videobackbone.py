"""Wan-specific :class:`VideoBackbone`: bridges architecture-driven action
injection and Wan's video forward.

Lives outside ``wan/`` to keep that package Wan-internal. Owns the Wan
modules (DiT/VAE/text encoder/tokenizer/VACE) directly — modules as named
children, scheduler/tokenizer/division factors as plain attributes; external
code reaches them only through the ABC methods.
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
from openwam.model.video_backbone.videobackbone_base import BlockLoopState, VideoBackbone

if TYPE_CHECKING:
    from openwam.model.inference_inputs import InferenceInputs
from openwam.model.video_backbone.wan import action_tokens as wan_action_tokens
from openwam.model.video_backbone.wan import conditioning as wan_conditioning
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply, sinusoidal_embedding_1d
from openwam.model.video_backbone.wan.preprocess import (
    check_resize_height_width,
)
from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import gradient_checkpoint_forward

logger = logging.getLogger(__name__)


class WanVideoBackbone(VideoBackbone):
    """Exposes the Wan modules through the VideoBackbone interface.

    Modules registered as named children (clean ``dit.*`` / ``vae.*`` keys);
    scheduler / tokenizer / division factors are plain attributes. Construct
    via ``from_pretrained(source)``.
    """

    # ================================================================
    # Construction
    # ================================================================

    def __init__(self, holder, *, external_encoder=None, shift_video=None):
        """Internal constructor. Use ``from_pretrained()`` instead.

        ``external_encoder`` is ``None`` on the default path so ``state_dict()``
        carries only ``vae.*`` keys; setting it activates external-encoder VAE-IO
        routing and aliases the encoder under ``"vae"``.

        ``shift_video`` is the optional Esser α-shift on the video scheduler,
        stored as the single source of truth behind the ABC property. ``None``
        keeps the scheduler template default (Wan = 5.0).
        """
        super().__init__()
        # ``holder`` is a transient carrier: drain its sub-modules + non-Module
        # state into self, then let it go out of scope. Nothing reads it after.
        self.video_encoder = external_encoder
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
        # Resolve the Wan variant (I2V/TI2V/VACE/plain) ONCE; first-frame
        # conditioning delegates to it so hot paths carry no per-variant branch.
        from openwam.model.video_backbone.wan import variants as _variants

        self._variant = _variants.detect(getattr(holder, "dit", None), getattr(holder, "vace", None))
        # Resolve patch size + temporal contract into instance attrs so the ABC
        # properties have a single value regardless of external encoder; callers
        # consult these attrs and never branch on ``self.video_encoder is None``.
        if external_encoder is not None:
            self._dit_patch_size = external_encoder.spec.dit_patch_size
            self._temporal_compression = int(external_encoder.spec.temporal_compression)
            self._causal_temporal = bool(external_encoder.spec.causal_temporal)
        else:
            # Wan native contract, invariant across all Wan2.x variants: DiT
            # patch (1,2,2); VAE 4× temporal compression + causal first-frame token.
            self._dit_patch_size = (1, 2, 2)
            self._temporal_compression, self._causal_temporal = 4, True

    @classmethod
    def from_pretrained(cls, source, *, external_encoder=None, **kw) -> WanVideoBackbone:
        """Build a WanVideoBackbone from a source.

        Sources: ``DictConfig`` (full Hydra cfg → loader), ``str`` dir path /
        ``dict`` with ``model_path`` (lightweight build), else an already-built
        component holder. Construction returns a transient holder that
        ``__init__`` drains into the backbone.

        With ``external_encoder``: (1) fail-fast for I2V/VACE backbones;
        (2) derive division factors from the encoder spec; (3) release the
        native VAE. See the inline numbered comments for the why.
        """
        from omegaconf import DictConfig

        # Skip materializing the native VAE (avoid ~1.5GB waste / a duplicate
        # VAE slot deploy has no weights for) on training-with-irreversible and
        # on deploy-with-ANY external encoder. Reversible-on-training keeps it,
        # needed for the step-(2) spec cross-check against ``v.z_dim`` etc.
        is_deploy = not isinstance(source, DictConfig)
        skip_native_vae = bool(external_encoder is not None and (is_deploy or not external_encoder.spec.is_reversible))

        if isinstance(source, DictConfig):
            from openwam.model.video_backbone.wan.pipeline_builder import build_training_pipeline

            holder = build_training_pipeline(source, skip_native_vae=skip_native_vae)
        elif isinstance(source, str):
            if os.path.isdir(source):
                holder = cls._build_holder_from_model_path(
                    source, device=kw.get("device", "cpu"), skip_native_vae=skip_native_vae
                )
            else:
                raise ValueError(f"from_pretrained(str) expects a directory path, got: {source!r}.")
        elif isinstance(source, dict):
            vb_cfg = source.get("video_backbone", source)
            if isinstance(vb_cfg, dict) and "components" in vb_cfg:
                holder = cls._build_holder_from_components(
                    vb_cfg["components"],
                    tokenizer=vb_cfg.get("tokenizer"),
                    device=kw.get("device", "cpu"),
                    ckpt_dir=kw.get("ckpt_dir"),
                    model_path=vb_cfg.get("model_path"),
                    skip_native_vae=skip_native_vae,
                )
            else:
                model_path = (
                    vb_cfg.get("model_path") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "model_path", None)
                )
                if model_path is None:
                    raise ValueError(
                        "dict source must contain 'video_backbone.components' or 'video_backbone.model_path'"
                    )
                holder = cls._build_holder_from_model_path(
                    str(model_path), device=kw.get("device", "cpu"), skip_native_vae=skip_native_vae
                )
        else:
            holder = source

        if external_encoder is not None:
            # (1) I2V fail-fast: the pretrained DiT first conv hardcodes
            # in_dim = 4 + z_dim, so an external encoder breaks the ``y``
            # channel-cat. Surface at construction, not at runtime.
            if bool(getattr(holder.dit, "has_image_input", False)):
                raise ValueError(
                    "I2V backbones cannot use external encoders: DiT first "
                    "conv in_dim = 4 + z_dim is hardcoded into the pretrained "
                    "weights. See docs/external_video_encoder.md §6."
                )

            # (1b) VACE fail-fast: two hard incompatibilities the adapter does
            # not rebuild — ``vace_patch_embedding`` hardcodes vace_in_dim=96
            # (native z_dim=16), and the deploy VACE path reads the native VAE
            # which is None here. Surface at construction.
            if getattr(holder, "vace", None) is not None:
                raise ValueError(
                    "VACE backbones cannot use external encoders: this PR's scope is "
                    "Wan2.2-TI2V-5B only. VaceWanModel.vace_patch_embedding's "
                    "vace_in_dim=96 is baked into the pretrained weights, and the "
                    "vendored WanVideoUnit_VACE inference path reads the native VAE which "
                    "is None on the external-encoder path. See "
                    "docs/external_video_encoder.md §6."
                )

            # (2) Spatial/time division factors derived from the encoder spec,
            # not a hardcoded ``* 2`` / Wan-VAE grid — otherwise
            # ``check_resize_height_width`` would round encoder-legal sizes to
            # Wan's grid. Remainder is 1 iff causal ("first frame separable,
            # then groups of temporal_compression").
            ps = external_encoder.spec.dit_patch_size
            holder.height_division_factor = external_encoder.spec.spatial_compression * ps[1]
            holder.width_division_factor = external_encoder.spec.spatial_compression * ps[2]
            holder.time_division_factor = external_encoder.spec.temporal_compression * ps[0]
            holder.time_division_remainder = 1 if external_encoder.spec.causal_temporal else 0

            # (3) Release the native VAE so state_dict keys don't double-count
            # with the external encoder. print (not logger.info) because arch
            # init runs before the logger is wired up; rank-0 gated.
            holder.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if int(os.environ.get("RANK", 0)) == 0:
                print(
                    f"[WanVideoBackbone] native VAE released; "
                    f"external_encoder={type(external_encoder).__name__} "
                    f"(z_dim={external_encoder.spec.z_dim}, "
                    f"is_reversible={external_encoder.spec.is_reversible}, "
                    f"dit_patch_size={external_encoder.spec.dit_patch_size})",
                    flush=True,
                )

            # (5) Expose latent-shape metadata so deploy noise init can read it
            # without falling back to the native VAE (now None). Plain attr — not
            # in state_dict.
            holder.latent_spec = external_encoder.spec

        # Resolve optional cfg-side ``shift_video`` here (not in __init__)
        # because the cfg shape depends on the ``source`` type.
        shift_video_cfg = cls._resolve_cfg_shift_video(source)

        return cls(holder, external_encoder=external_encoder, shift_video=shift_video_cfg)

    @staticmethod
    def _resolve_cfg_shift_video(source) -> Optional[float]:
        """Extract ``shift_video`` from the various ``from_pretrained`` source
        shapes; ``None`` when unset or the source has no cfg context.
        """
        from omegaconf import DictConfig

        vb_cfg = None
        if isinstance(source, DictConfig):
            vb_cfg = source.get("video_backbone") if "video_backbone" in source else None
        elif isinstance(source, dict):
            # model_loader hands a dict that either IS or contains video_backbone.
            vb_cfg = source.get("video_backbone", source) if "video_backbone" in source else source
        if vb_cfg is None:
            return None
        if isinstance(vb_cfg, dict):
            raw = vb_cfg.get("shift_video")
        else:
            raw = getattr(vb_cfg, "shift_video", None)
        return None if raw is None else float(raw)

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
        return getattr(self, "vace", None) is not None

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

    @property
    def _freq_dim(self) -> int:
        return int(self._dit.freq_dim)

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
    def submodule_names(self) -> list[str]:
        # ``vae`` is reported under either path so ``freeze_modules: [vae]``
        # works unchanged when an external encoder is swapped in (the alias
        # resolves to ``self.video_encoder`` in get_submodule).
        names = []
        for name in ("dit", "vace", "text_encoder", "vae", "image_encoder"):
            if name == "vae" and self._uses_external_encoder:
                names.append(name)
                continue
            if getattr(self, name, None) is not None:
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

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the video↔video block of the joint MoT attention mask
        (``True`` = attend to). Layout matches FastWAM's equivalent.
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
            # First-frame rows attend only to first-frame keys; later rows stay True.
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
        dit = self.dit
        motion_controller = getattr(self, "motion_controller", None)
        vace = getattr(self, "vace", None)
        latents = kw["latents"]
        timestep = kw["timestep"]
        context = kw["context"]
        context_mask = kw.get("context_mask")
        seq_lens = kw.get("seq_lens")
        clip_feature = kw.get("clip_feature")
        y = kw.get("y")
        vace_context = kw.get("vace_context")
        motion_bucket_id = kw.get("motion_bucket_id")
        control_camera_latents_input = kw.get("control_camera_latents_input")
        fuse_vae_embedding_in_latents = kw.get("fuse_vae_embedding_in_latents", False)
        num_clean_prefix_frames = kw.get("num_clean_prefix_frames", 0)
        use_gradient_checkpointing = kw.get("use_gradient_checkpointing", False)
        use_gradient_checkpointing_offload = kw.get("use_gradient_checkpointing_offload", False)
        force_per_token_t_mod = bool(kw.get("force_per_token_t_mod", False))

        if dit.seperated_timestep and fuse_vae_embedding_in_latents:
            batch_size = latents.shape[0]
            num_clean = max(num_clean_prefix_frames, 1)
            f_lat = latents.shape[2]
            ps = self._dit_patch_size
            tokens_per_frame_patch = latents.shape[3] * latents.shape[4] // (ps[1] * ps[2])
            tokens_per_frame = tokens_per_frame_patch
            token_timesteps = torch.ones(
                batch_size, f_lat, tokens_per_frame, dtype=latents.dtype, device=latents.device
            ) * timestep.view(batch_size, 1, 1)
            token_timesteps[:, :num_clean, :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            t_emb = sinusoidal_embedding_1d(dit.freq_dim, token_timesteps.reshape(-1))
            t = dit.time_embedding(t_emb.to(latents.dtype)).reshape(batch_size, -1, dit.dim)
            t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
        elif force_per_token_t_mod:
            # Non-TI2V backbones under joint-attention / shared_backbone need 4D
            # t_mod. Compute the time embedding once on (B,) and broadcast to
            # (B, L, dim) — a per-token MLP would repeat the stack L times.
            batch_size = latents.shape[0]
            f_lat = latents.shape[2]
            ps = self._dit_patch_size
            tokens_per_frame_patch = latents.shape[3] * latents.shape[4] // (ps[1] * ps[2])
            tokens_per_frame = tokens_per_frame_patch
            L = f_lat * tokens_per_frame
            t_base = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))  # (B, dim)
            t = t_base.unsqueeze(1).expand(batch_size, L, -1).contiguous()

            # Optional clean-prefix alignment: when opted in AND a clean prefix
            # is present, overwrite the first ``num_clean`` frames' time embedding
            # with ``time_embedding(0)`` — mirrors TI2V's per-token t=0 pin but
            # at the embedding layer, keeping the MLP a single (B, dim) call.
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

            t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
        else:
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        if motion_bucket_id is not None and motion_controller is not None:
            motion_term = motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))  # (B, 6, dim)
            if t_mod.dim() == 4:
                # Broadcast (B, 6, dim) across L; without the unsqueeze the add
                # right-aligns and aliases B onto L (mis-broadcasts when B == L).
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
        extras["time_embed"] = t  # Wan head time embedding; consumed in finalize()
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

        return BlockLoopState(
            hidden_states=x,
            time_mod=t_mod,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=f,
            grid_height=h,
            grid_width=w,
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
            state.hidden_states = gradient_checkpoint_forward(
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
            self._apply_post_block_residuals(block_id, state)
            return state

        state.hidden_states = gradient_checkpoint_forward(
            block,
            state.use_gradient_checkpointing,
            state.use_gradient_checkpointing_offload,
            state.hidden_states,
            state.context,
            state.time_mod,
            state.rope_freqs,
            block_context_mask,
        )

        self._apply_post_block_residuals(block_id, state)
        return state

    def _apply_post_block_residuals(self, block_id: int, state: BlockLoopState) -> None:
        """Apply the VACE hint residual to ``state.hidden_states`` in place.
        Shared by run_block and post_attn_at_layer.
        """
        vace = state.extras.get("vace")

        if state.vace_hints is not None and vace is not None and block_id in vace.vace_layers_mapping:
            current_vace_hint = state.vace_hints[vace.vace_layers_mapping[block_id]]
            vace_len = current_vace_hint.shape[1]
            if state.hidden_states.shape[1] == vace_len:
                # dual_system / video-only: hint spans the full sequence.
                state.hidden_states = state.hidden_states + current_vace_hint
            elif state.hidden_states.shape[1] > vace_len:
                # shared_backbone: residual applies only to the leading video
                # slice; action/state tokens get VACE via self-attention.
                video_slice = state.hidden_states[:, :vace_len] + current_vace_hint
                state.hidden_states = torch.cat([video_slice, state.hidden_states[:, vace_len:]], dim=1)
            else:
                # Defensive: hidden_states shorter than the hint breaks the
                # video-token-count invariant — investigate before patching.
                raise ValueError(
                    f"_apply_post_block_residuals: state.hidden_states.shape[1]={state.hidden_states.shape[1]} "
                    f"< vace_hint.shape[1]={vace_len} at block {block_id}; "
                    "this is unreachable under dual_system or shared_backbone today—"
                    "investigate the upstream caller before patching this branch."
                )

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """First half of a Wan DiT block (norm1 + AdaLN + Q/K/V + RoPE), up to
        the attention call. Lets MoTJointDriver pull video-side Q/K/V before the
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

        t_mod = state.time_mod
        has_seq = t_mod.dim() == 4
        chunk_dim = 2 if has_seq else 1
        chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            chunks = tuple(c.squeeze(2) for c in chunks)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        residual_x = state.hidden_states
        attn_input = modulate(block.norm1(state.hidden_states), shift_msa, scale_msa)

        sa = block.self_attn
        q = sa.norm_q(sa.q(attn_input))
        k = sa.norm_k(sa.k(attn_input))
        v = sa.v(attn_input)
        q = rope_apply(q, state.rope_freqs, sa.num_heads)
        k = rope_apply(k, state.rope_freqs, sa.num_heads)

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
        sa = block.self_attn
        residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

        x = block.gate(residual_x, gate_msa, sa.o(attn_out))
        context_mask = None
        if state.context_mask is not None:
            context_mask = state.context_mask.unsqueeze(1).expand(-1, x.shape[1], -1).unsqueeze(1)
        x = x + block.cross_attn(block.norm3(x), state.context, ctx_mask=context_mask)
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        state.hidden_states = x

        self._apply_post_block_residuals(layer_id, state)
        return state

    def finalize(self, state: BlockLoopState):
        """Wan DiT head + unpatchify. Returns ``(B, z_dim, F, H, W)``."""
        dit = state.extras["dit"]
        head = dit.head
        t_embed = state.extras["time_embed"]
        t_head = t_embed if t_embed.dim() == 3 else t_embed.unsqueeze(1)

        x = head(state.hidden_states, t_head)

        x = dit.unpatchify(x, (state.grid_frames, state.grid_height, state.grid_width))
        return x

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
        if n_state < 0:
            raise ValueError(f"n_state must be non-negative, got {n_state}")
        if n_action < 0:
            raise ValueError(f"n_action must be non-negative, got {n_action}")
        if n_action + n_state <= 0:
            raise ValueError("inject_shared_tokens requires at least one action or state token.")
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
            if state_tokens.shape[2] != state.hidden_states.shape[2]:
                raise ValueError(
                    f"state_tokens dim {state_tokens.shape[2]} does not match video dim {state.hidden_states.shape[2]}"
                )
            appended_pieces.append(state_tokens.to(state.hidden_states.dtype))
        else:
            if state_tokens is not None and state_tokens.shape[1] != 0:
                raise ValueError("state_tokens were provided but n_state=0.")
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

        B = len(frames)
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
        input_latents = wan_encode.encode_video(
            stacked_inputs, vae=getattr(self, "vae", None), encoder=self.video_encoder
        )
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
            B=B,
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
    # ABC: Sub-module access (2)
    # ================================================================

    def get_submodule(self, name: str) -> nn.Module | None:
        if name == "vae" and self._uses_external_encoder:
            return self.video_encoder
        # Non-Module names (tokenizer / scheduler) resolve to None.
        mod = getattr(self, name, None)
        return mod if isinstance(mod, nn.Module) else None

    def set_submodule(self, name: str, module: nn.Module) -> None:
        if name == "vae" and self._uses_external_encoder:
            self.video_encoder = module
            return
        setattr(self, name, module)

    # ================================================================
    # ABC: Decoding (1)
    # ================================================================

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        if self._uses_external_encoder and not self.video_encoder.spec.is_reversible:
            raise NotImplementedError(
                f"decode_video on irreversible encoder ({type(self.video_encoder).__name__}; "
                "spec.is_reversible=False). Pass decode_video=False to generate() to "
                "retrieve raw latents, or train a separate pixel decoder."
            )
        video_tensor = wan_encode.decode_latents(
            latents, vae=getattr(self, "vae", None), encoder=self.video_encoder, device=self.device, tiled=tiled
        )
        return wan_encode.latents_to_frames(video_tensor, encoder=self.video_encoder)

    # ================================================================
    # ABC: Device management (1)
    # ================================================================

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move all owned submodules to (dtype, device) and update the
        ``_dtype`` / ``_device`` records.

        Must NOT call ``mod.eval()``: trainable submodules (dit, vace) must stay
        in train mode; frozen ones run under ``no_grad`` so their mode is moot.
        """
        self._dtype = dtype
        self._device = device
        for name in self.submodule_names:
            mod = self.get_submodule(name)
            if mod is not None:
                mod.to(dtype=dtype, device=device)

    # ================================================================
    # Component specs for self-contained checkpoints
    # ================================================================

    def get_component_specs(self, model_path: str) -> dict:
        """Component specs from *model_path* (``components`` + optional
        ``tokenizer``), injected into the saved config so deploy can rebuild the
        pipeline without the original *model_path*.
        """
        from openwam.model.video_backbone.wan.component_specs import generate_video_backbone_component_specs

        return generate_video_backbone_component_specs(model_path)

    def copy_deploy_artifacts(self, output_dir: str, cfg) -> None:
        """Copy backbone-side deploy artifacts next to ``config.yaml``: the Wan
        tokenizer (always), plus the external encoder's own side files (forwarded
        to its ``copy_deploy_artifacts``; policy is encoder-specific).
        """
        from openwam.model.video_backbone.wan.component_specs import copy_video_backbone_tokenizer

        copy_video_backbone_tokenizer(output_dir, cfg)
        if self.video_encoder is not None:
            self.video_encoder.copy_deploy_artifacts(output_dir, cfg)

    # ================================================================
    # Deploy-facing public methods (not in ABC — Wan-specific)
    # ================================================================

    def preprocess_input_for_inference(self, inputs: "InferenceInputs") -> dict:
        """Build the inference denoising-loop input dict from a typed
        :class:`InferenceInputs`, via explicit backbone helpers (no unit-runner).

        Only the text embedding is cached (prompt-keyed, seed/dim-independent);
        noise/clip/y/vace_context/first_frame_latents are rebuilt every call.
        CFG fields are ignored — Wan does no CFG at inference.
        """
        # Wan tile defaults (dataclass keeps them None for other backbones).
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
            vae=getattr(self, "vae", None),
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
                image_encoder=getattr(self, "image_encoder", None),
                dtype=self.dtype,
                device=self.device,
            )
            if clip_feature is not None:
                inputs_shared["clip_feature"] = clip_feature
            y = wan_conditioning.build_deploy_i2v_y(
                i2v_img,
                num_frames=num_frames,
                height=height,
                width=width,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                dit=self.dit,
                vae=getattr(self, "vae", None),
                dtype=self.dtype,
                device=self.device,
            )
            if y is not None:
                inputs_shared["y"] = y

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
            vae=getattr(self, "vae", None),
            encoder=self.video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        wan_conditioning.finalize_ti2v_first_frame_latents(
            inputs_shared,
            first_frame_image,
            is_ti2v=self._is_ti2v,
            encoder=self.video_encoder,
            vae=getattr(self, "vae", None),
            dtype=self.dtype,
            device=self.device,
        )
        return inputs_shared

    def apply_compile(self, compile_cfg) -> None:
        """torch.compile backbone sub-modules per config bool flags
        (``video_dit``/``dit``, ``vae``, ``vace``, ``text_encoder``,
        ``image_encoder``). DiT blocks are compiled individually (more
        CUDA-graph friendly than compiling the whole DiT).
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
            mod = getattr(self, submod_name, None)
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
                setattr(self, submod_name, torch.compile(mod, **compile_kwargs))
                logger.info("torch.compile enabled for %s (%s)", submod_name, compile_kwargs)

    @staticmethod
    def _build_holder_from_components(
        components: list,
        tokenizer: dict = None,
        device: str = "cpu",
        ckpt_dir: str = None,
        model_path: str = None,
        *,
        skip_native_vae: bool = False,
    ):
        """Build an empty component holder from specs. Weights are NOT
        loaded here (``load_checkpoint`` does that). Tokenizer resolves
        ckpt-local first, then falls back to the ``model_path`` layout.

        ``skip_native_vae`` drops the ``vae`` entry so it never allocates CPU
        tensors (irreversible external-encoder path).
        """
        from openwam.model.video_backbone.wan.loader import new_components
        from openwam.model.video_backbone.wan.pipeline_builder import _build_tokenizer, _import_class

        holder = new_components(device=device, torch_dtype=torch.bfloat16)

        for entry in components:
            if skip_native_vae and entry.get("attr") == "vae":
                continue
            cls = _import_class(entry["model_class"])
            kwargs = entry.get("extra_kwargs", {}) or {}
            logger.info(
                "Instantiating %s as holder.%s (extra_kwargs keys=%s)",
                entry["model_class"],
                entry["attr"],
                list(kwargs.keys()),
            )
            with torch.device(device):
                model = cls(**kwargs)
            model.to(dtype=torch.bfloat16)
            setattr(holder, entry["attr"], model)

        if getattr(holder, "vae", None) is not None and hasattr(holder.vae, "upsampling_factor"):
            holder.height_division_factor = holder.vae.upsampling_factor * 2
            holder.width_division_factor = holder.vae.upsampling_factor * 2

        if tokenizer:
            tok = None
            subdir = tokenizer.get("subdir", "")
            if ckpt_dir and subdir and os.path.isdir(os.path.join(ckpt_dir, subdir)):
                tok = _build_tokenizer(tokenizer, ckpt_dir)
            elif model_path and os.path.isdir(model_path):
                # ckpt-local specs prefix ``tokenizer/``; upstream dirs don't.
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
            setattr(holder, tokenizer.get("attr", "tokenizer"), tok)

        return holder

    @staticmethod
    def _build_holder_from_model_path(model_path: str, device: str = "cpu", *, skip_native_vae: bool = False):
        """Build a component holder from a model dir without full Hydra config.
        ``skip_native_vae`` drops the native VAE weight file before it
        materializes (irreversible external-encoder path).
        """
        from openwam.model.video_backbone.wan.loader import load_wan_components
        from openwam.model.video_backbone.wan.pipeline_builder import (
            _filter_native_vae_configs,
            discover_model_files,
        )

        model_configs, tokenizer_config = discover_model_files(model_path)
        if skip_native_vae:
            model_configs = _filter_native_vae_configs(model_configs)
        return load_wan_components(
            model_configs,
            tokenizer_config,
            device=device,
            torch_dtype=torch.bfloat16,
        )


__all__ = ["WanVideoBackbone"]
