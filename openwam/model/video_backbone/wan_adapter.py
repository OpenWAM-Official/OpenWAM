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

    @classmethod
    def get_native_dit_patch_size(cls, pipe) -> Tuple[int, int, int]:
        """Wan family's native DiT first-layer patch size.

        All Wan2.x DiT variants (TI2V / I2V / VACE / Wan2.2) use ``(1, 2, 2)``
        — the value is baked into ``WanModel.patch_embedding``'s ``Conv3d``
        kernel and stride. We don't read it back from ``pipe.dit`` because
        the value is invariant across the family and querying ``pipe.dit``
        here would couple this classmethod to a non-trivial pipeline state.
        """
        return (1, 2, 2)

    @classmethod
    def get_native_temporal_contract(cls, pipe) -> Tuple[int, bool]:
        """Wan family's native VAE temporal contract.

        Hard-coded ``(4, True)`` across the Wan2.1 / Wan2.2 line — the causal
        first-frame token plus 4-frame tail grouping is invariant. Same
        rationale as :meth:`get_native_dit_patch_size`: probing ``pipe.vae``
        here would couple the classmethod to a non-trivial pipeline state for
        a value that is invariant by family.
        """
        return (4, True)

    def __init__(self, pipe, *, external_encoder=None, shift_video=None):
        """Internal constructor. Use ``from_pretrained()`` instead.

        ``external_encoder`` must be ``None`` on the default path so
        ``state_dict()`` carries only ``_pipe.vae.*`` keys (not also
        ``_encoder.*``). Setting it activates the external-encoder routing
        in :meth:`_preprocess_video` / :meth:`_encode_video` /
        :meth:`_decode_latents` / :meth:`_latents_to_frames` and aliases
        the encoder under ``"vae"`` in :attr:`submodule_names`.

        ``shift_video`` is the optional Esser-et-al. α-shift applied to
        the video scheduler. Stored as ``self._shift_video`` so the ABC
        :attr:`VideoBackbone.shift_video` property returns it — single
        source of truth consumed by both
        :meth:`BaseWAMArchitecture.init_training_schedulers` and
        ``openwam/deploy/joint_engine.py::generate``. ``None`` keeps the
        scheduler's template default (Wan = 5.0), which is bit-identical
        to pre-PR behavior.
        """
        super().__init__()
        self._pipe = pipe
        self._encoder = external_encoder
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        self._shift_video = None if shift_video is None else float(shift_video)
        # Resolve DiT patch size + temporal contract into backbone-owned
        # instance attributes so the properties defined on the ABC
        # (dit_patch_size / temporal_compression / causal_temporal) have
        # a single value to return regardless of whether an external
        # encoder is plugged in. Callers downstream (base.py mask
        # downsampling, dataloader divisibility) consult the backbone
        # attributes and never branch on ``self._encoder is None``.
        if external_encoder is not None:
            self._dit_patch_size = external_encoder.spec.dit_patch_size
            self._temporal_compression = int(external_encoder.spec.temporal_compression)
            self._causal_temporal = bool(external_encoder.spec.causal_temporal)
        else:
            self._dit_patch_size = self.get_native_dit_patch_size(pipe)
            self._temporal_compression, self._causal_temporal = self.get_native_temporal_contract(pipe)

    @classmethod
    def from_pretrained(cls, source, *, external_encoder=None, **kw) -> WanVideoBackbone:
        """Build a WanVideoBackbone from various source types.

        Supported sources:
        - ``DictConfig``: full Hydra config → ``build_training_pipeline(cfg)``
        - ``str`` directory path: auto-discover model files → lightweight build
        - ``dict`` with ``video_backbone.model_path``: lightweight build from model dir
        - anything else: treated as an already-built pipe object

        When ``external_encoder`` is provided:
          1. Fails fast for I2V backbones (their first conv hardcodes
             ``in_dim = 4 + z_dim`` — see ``_build_i2v_y``).
          2. Validates the encoder spec against the pipeline's native VAE.
             ``is_reversible=True`` enforces strict z_dim equality;
             ``is_reversible=False`` skips z_dim (patch_embedding will be
             rebuilt by :func:`reinit_dit_from_scratch`) but still checks
             spatial / temporal / causal — backbone-side code makes strong
             assumptions about these.
          3. Sets ``pipe.height/width_division_factor`` from
             ``encoder.spec.spatial_compression * encoder.spec.dit_patch_size[1or2]``.
          4. Releases ``pipe.vae`` so state_dict keys don't double-count
             VAE params with the external encoder.
        """
        from omegaconf import DictConfig

        # Skip native VAE materialization on:
        #   - training, irreversible external encoder: validation is already
        #     bypassed (encoder owns its latent geometry), so loading native
        #     VAE only to release it is pure waste (~1.5GB Wan2.2).
        #   - deploy with ANY external encoder: state_dict topology is
        #     ``_encoder._m.*`` (was saved that way during training);
        #     deploy must not also materialize ``_pipe.vae.*`` slot since
        #     (a) the slot has no checkpoint weights to fill it, (b)
        #     ``_build_pipe_from_components`` would otherwise duplicate the
        #     VAE inside the encoder, and (c) ``cls(pipe, external_encoder)``
        #     ends with ``pipe.vae = None`` anyway.
        # Reversible-on-training is the one case that keeps loading native
        # VAE — needed for the spec-equality cross-check at step (2) below
        # against ``v.z_dim`` / ``v.upsampling_factor``.
        is_deploy = not isinstance(source, DictConfig)
        skip_native_vae = bool(external_encoder is not None and (is_deploy or not external_encoder.spec.is_reversible))

        if isinstance(source, DictConfig):
            from openwam.model.video_backbone.wan.pipeline_builder import build_training_pipeline

            pipe = build_training_pipeline(source, skip_native_vae=skip_native_vae)
        elif isinstance(source, str):
            if os.path.isdir(source):
                pipe = cls._build_pipe_from_model_path(
                    source, device=kw.get("device", "cpu"), skip_native_vae=skip_native_vae
                )
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
                pipe = cls._build_pipe_from_model_path(
                    str(model_path), device=kw.get("device", "cpu"), skip_native_vae=skip_native_vae
                )
        else:
            pipe = source

        if external_encoder is not None:
            # (1) I2V fail-fast — the pretrained DiT's first conv is built with
            # in_dim = 4 + z_dim (mask channels + VAE z_dim); an external
            # encoder would silently break the channel-cat with ``y`` in
            # prepare(). Surface this at construction rather than wait for
            # the runtime AttributeError in ``_build_i2v_y``.
            if bool(getattr(pipe.dit, "has_image_input", False)):
                raise ValueError(
                    "I2V backbones cannot use external encoders: DiT first "
                    "conv in_dim = 4 + z_dim is hardcoded into the pretrained "
                    "weights. See docs/external_video_encoder.md §6."
                )

            # (1b) VACE fail-fast — current PR scope is Wan2.2-TI2V-5B
            # only. VACE backbones have two hard incompatibilities the
            # adapter does not yet rebuild:
            #
            #   - ``VaceWanModel.vace_patch_embedding`` hardcodes
            #     ``vace_in_dim = 2 * z_dim + 64 = 96`` (z_dim=16 of the
            #     native Wan VAE); ``reinit_dit_from_scratch`` only rebuilds
            #     ``dit.patch_embedding`` / ``dit.head.head``, not the VACE
            #     embedding — any encoder with a different z_dim makes
            #     ``_build_vace_context`` produce a tensor that does not
            #     match the embedding's in_channels.
            #
            #   - The vendored ``WanVideoUnit_VACE.process`` inference path
            #     calls ``pipe.vae.encode(...)`` directly (wan/pipeline.py).
            #     On the external-encoder path ``pipe.vae`` is None, so
            #     deploy raises AttributeError on the first ``vace_video``
            #     input.
            #
            # Surface at construction so users do not discover this only
            # at deploy time. See docs/external_video_encoder.md §6.
            if getattr(pipe, "vace", None) is not None:
                raise ValueError(
                    "VACE backbones cannot use external encoders: this PR's scope is "
                    "Wan2.2-TI2V-5B only. VaceWanModel.vace_patch_embedding's "
                    "vace_in_dim=96 is baked into the pretrained weights, and the "
                    "vendored WanVideoUnit_VACE inference path reads pipe.vae which "
                    "is None on the external-encoder path. See "
                    "docs/external_video_encoder.md §6."
                )

            from openwam.model.video_backbone.encoder.spec import VideoEncoderSpec

            v = getattr(pipe, "vae", None)
            if v is not None and external_encoder.spec.is_reversible:
                want = VideoEncoderSpec(
                    z_dim=int(v.z_dim),
                    spatial_compression=int(v.upsampling_factor),
                    temporal_compression=4,
                    causal_temporal=True,
                )
                VideoBackbone.validate_encoder_spec(external_encoder.spec, want)

            # (3) Spatial division factor derived from the encoder's declared
            # ``dit_patch_size`` rather than a hardcoded ``* 2``. For Wan VAE
            # (dit_patch_size=(1,2,2)) this is identical to the pre-existing
            # constant; for ViT-style encoders that pre-patchify at 16x and
            # declare dit_patch_size=(1,1,1), the DiT's first conv becomes a
            # pure channel projection and the total spatial division factor
            # equals the encoder's own spatial_compression.
            ps = external_encoder.spec.dit_patch_size
            pipe.height_division_factor = external_encoder.spec.spatial_compression * ps[1]
            pipe.width_division_factor = external_encoder.spec.spatial_compression * ps[2]
            # Time division must also follow the encoder spec, not the
            # pipeline default (which is hardcoded to ``time_division_factor=4,
            # time_division_remainder=1`` for native Wan VAE). Without this,
            # the pipeline's ``check_resize_height_width`` would silently
            # round V-JEPA-legal frame counts (e.g. 9, 11, 13 for
            # temporal_compression=2 causal) up to Wan VAE's grid. The
            # remainder is 1 iff the encoder is causal: that is the same
            # "first frame separable, then groups of ``temporal_compression``"
            # contract Wan VAE assumes and V-JEPA emulates.
            pipe.time_division_factor = external_encoder.spec.temporal_compression * ps[0]
            pipe.time_division_remainder = 1 if external_encoder.spec.causal_temporal else 0

            # (4) Release the native VAE so state_dict keys don't double-count
            # VAE params with the external encoder. Print rather than
            # logger.info because architecture init runs before the trainer's
            # logger is wired up and INFO would be swallowed; the user needs
            # this visible to confirm the external_encoder path is active.
            # Gated on rank=0 to avoid 4x duplicate lines under torchrun.
            pipe.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if int(os.environ.get("RANK", 0)) == 0:
                print(
                    f"[WanVideoBackbone] pipe.vae released; "
                    f"external_encoder={type(external_encoder).__name__} "
                    f"(z_dim={external_encoder.spec.z_dim}, "
                    f"is_reversible={external_encoder.spec.is_reversible}, "
                    f"dit_patch_size={external_encoder.spec.dit_patch_size})",
                    flush=True,
                )

            # (5) Expose latent-shape metadata on ``pipe`` so vendored
            # inference units (``WanVideoUnit_NoiseInitializer``) can read
            # ``z_dim`` / ``spatial_compression`` / ``temporal_compression`` /
            # ``causal_temporal`` without falling back to ``pipe.vae``
            # (which is now None). The unit's fallback branch still hits
            # ``pipe.vae`` on the native VAE path where ``pipe.latent_spec``
            # is absent. Plain attribute (not nn.Module) — does not enter
            # state_dict.
            pipe.latent_spec = external_encoder.spec

        # Resolve optional cfg-side ``shift_video`` (Esser SD3 α-shift on the
        # video scheduler) and pass it to the constructor. We read here
        # rather than in ``__init__`` because the cfg shape depends on the
        # ``source`` type (DictConfig from training, dict from deploy,
        # plain pipe with no cfg context). ``None`` keeps the scheduler's
        # template default (Wan = 5.0).
        shift_video_cfg = cls._resolve_cfg_shift_video(source)

        return cls(pipe, external_encoder=external_encoder, shift_video=shift_video_cfg)

    @staticmethod
    def _resolve_cfg_shift_video(source) -> Optional[float]:
        """Extract ``cfg.model.video_backbone.shift_video`` from various
        ``from_pretrained`` source shapes.

        Returns ``None`` when the field is unset, the source has no cfg
        context (plain pipe / model-path str), or the value is explicitly
        null. Caller stores the result on ``self._shift_video`` for
        downstream consumers (``init_training_schedulers`` /
        ``joint_engine.generate``).
        """
        from omegaconf import DictConfig

        vb_cfg = None
        if isinstance(source, DictConfig):
            # Training path: full Hydra cfg, video_backbone block lives under it.
            vb_cfg = source.get("video_backbone") if "video_backbone" in source else None
        elif isinstance(source, dict):
            # Deploy path: model_loader hands us a dict that either IS the
            # video_backbone block or contains it.
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
        return self._pipe.dit

    @property
    def _uses_external_encoder(self) -> bool:
        """True when this backbone is routing VAE IO through an external
        :class:`VideoEncoder` rather than its native ``pipe.vae``."""
        return self._encoder is not None

    @property
    def _has_vace(self) -> bool:
        return getattr(self._pipe, "vace", None) is not None

    @property
    def _is_ti2v(self) -> bool:
        return bool(getattr(self._dit, "fuse_vae_embedding_in_latents", False))

    @property
    def needs_first_frame_skip(self) -> bool:
        """``True`` iff this Wan variant unconditionally treats ``latent[0]`` as a
        clean conditioning frame that must be excluded from the diffusion loss.

        - TI2V (``fuse_vae_embedding_in_latents``): ``latent[0]`` is the encoded
          first-frame reference; the per-token timestep path pins t=0 on those
          tokens. Always skipped.
        - I2V (``has_image_input``): the first-frame condition rides on the
          ``y`` side channel and ``latent[0]`` itself is fully noised on both
          train and deploy. Deploy starts ``latent[0]`` from pure noise and
          the denoising loop must produce a meaningful frame-0 output, so
          training has to supervise ``latent[0]`` against that target.
          Skipping it here is exactly what drove the cell-4 mock loss
          divergence — model never gets a frame-0 gradient and produces
          garbage there at inference. So I2V is NOT in the skip list.
        - VACE: starting with the native-VACE PR, the first-frame condition is
          delivered exclusively through the ``vace_context`` bypass; ``video``
          itself is fully noised and fully supervised (matches native
          ``WanVideoUnit_VACE`` convention). So VACE is NOT in the skip list —
          ``latent[0]`` enters the loss as a predicted frame.
        - Future Wan T2V: none of the above → ``False``.
        """
        return self._is_ti2v

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
        # ``vae`` is reported under either path so training_strategy
        # ``freeze_modules: [vae, ...]`` works without yaml changes when an
        # external encoder is swapped in (pipe.vae is None on the external
        # path; the alias resolves to ``self._encoder`` in get_submodule).
        names = []
        for name in ("dit", "vace", "text_encoder", "vae", "image_encoder"):
            if name == "vae" and self._uses_external_encoder:
                names.append(name)
                continue
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
            ps = self._dit_patch_size
            tokens_per_frame = latents.shape[3] * latents.shape[4] // (ps[1] * ps[2])
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
            ps = self._dit_patch_size
            tokens_per_frame = latents.shape[3] * latents.shape[4] // (ps[1] * ps[2])
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

        # Three-way mutually exclusive backbone-condition pipelines, keyed off
        # backbone identity:
        #   - I2V  (has_image_input=True): first-frame rides on clip_feature + y
        #     (channel-axis concat in prepare()); native VACE bypass not present.
        #   - TI2V (_is_ti2v=True): first_frame_latents = input_latents[:, :, 0:1]
        #     plus the seperated_timestep path pins t=0 on frame-0 tokens.
        #   - VACE: native convention. Build pixel-space (vace_video, vace_mask,
        #     ref_image=None) inputs and feed the *batched* equivalent of
        #     ``WanVideoUnit_VACE.process``. Video latents stay fully noised and
        #     loss covers every frame; the first-frame signal flows solely via
        #     ``vace_context``. No ``first_frame_latents`` is emitted — that key
        #     is reserved for TI2V's clean-replacement contract.

        vace_context = None
        if self._has_vace:
            vace_video_pixels, vace_mask_pixels = self._build_vace_pixel_inputs(
                vace_videos=vace_videos,
                first_frame_image=ref_images,
                B=B,
                num_frames=num_frames,
                height=height,
                width=width,
                dtype=stacked_inputs.dtype,
                device=stacked_inputs.device,
                # Reuse the already-preprocessed input video so we don't
                # re-decode the first PIL frame from disk; ``stacked_inputs`` is
                # in the same [-1, 1] preprocessed space the native unit would
                # produce after ``pipe.preprocess_video``.
                preprocessed_video=stacked_inputs,
            )
            vace_context = self._build_vace_context_from_pixels(vace_video_pixels, vace_mask_pixels)

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
                raise ValueError(f"first_frame_image batch ({len(first_frame_image)}) != frames batch ({B})")

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

        # TI2V first-frame conditioning: extract latent[0] from the already-
        # encoded video latents (no second VAE call, no prepend). Aligned with
        # FastWAM / main PR#19. ``base.compute_loss`` clean-replaces
        # ``latents[:, :, 0:1]`` with this signal on every step so the DiT sees
        # [clean ref, noisy 1..T_lat-1]; ``_compute_video_loss`` then trims
        # frame 0 from pred/target via ``n_skip = max(num_clean_prefix, 1)``.
        # ``fuse_vae_embedding_in_latents`` stays gated on ``_is_ti2v`` — only
        # TI2V's DiT has the ``seperated_timestep`` consumer.
        #
        # VACE intentionally does NOT set ``first_frame_latents``: its
        # conditioning rides entirely on ``vace_context`` (built above via the
        # native pixel-space convention), and the video latent path stays fully
        # noised + fully supervised. See ``needs_first_frame_skip`` docstring
        # for why the VACE branch is absent from both the loss-side skip
        # signals.
        first_frame_latents = None
        num_clean_prefix = 0
        if has_ref and not has_image_input and self._is_ti2v:
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
        if name == "vae" and self._uses_external_encoder:
            return self._encoder
        return getattr(self._pipe, name, None)

    def set_submodule(self, name: str, module: nn.Module) -> None:
        if name == "vae" and self._uses_external_encoder:
            self._encoder = module
            return
        setattr(self._pipe, name, module)

    # ================================================================
    # ABC: Decoding (1)
    # ================================================================

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        if self._uses_external_encoder and not self._encoder.spec.is_reversible:
            raise NotImplementedError(
                f"decode_video on irreversible encoder ({type(self._encoder).__name__}; "
                "spec.is_reversible=False). Pass decode_video=False to generate() to "
                "retrieve raw latents, or train a separate pixel decoder."
            )
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

            # vace_reference_image is always None on the OpenWAM path: native
            # VACE's ref-prepend convention conflicts with our T_lat == video
            # length contract. The first-frame condition rides through
            # vace_context (built below by ``_build_vace_context_for_deploy``)
            # so the vendored ``WanVideoUnit_VACE`` has nothing to do for us.

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
                    inputs_shared["vace_reference_image"] = None

            for unit in pipe.units:
                if self._is_text_unit(unit):
                    continue
                if self._has_vace and self._is_vace_unit(unit):
                    # We supersede the vendored encode with the same batched
                    # helper used in training; skip the unit to avoid double
                    # work + the B=1 ref-prepend semantic that conflicts with
                    # our T_lat==video contract.
                    continue
                inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, {}, {})

            WanVideoBackbone._ensure_prompt_seq_lens(self, inputs_shared, prompt)
            self._build_vace_context_for_deploy(inputs_shared, first_frame_image, vace_video)
            self._finalize_ti2v_first_frame_latents(inputs_shared, first_frame_image)
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
            # vace_video / vace_video_mask / vace_reference_image are
            # intentionally cleared on the OpenWAM path: the vendored
            # ``WanVideoUnit_VACE`` would otherwise B=1-encode user PIL inputs
            # using a ref-prepend convention incompatible with our
            # T_lat == video-latent length contract. For VACE backbones we
            # supersede that unit via ``_build_vace_context_for_deploy``
            # below; for non-VACE backbones the slots are inert anyway. The
            # user-supplied ``vace_video`` flows through the deploy helper
            # rather than this dict.
            "vace_video": None,
            "vace_video_mask": None,
            "vace_reference_image": None,
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
        if _i2v_img is None:
            inputs_shared.pop("clip_feature", None)
            inputs_shared.pop("y", None)

        _t_text = time.time()
        inputs_nega = {}

        if _text_embed_hit:
            for unit in pipe.units:
                if self._is_text_unit(unit):
                    continue
                if self._has_vace and self._is_vace_unit(unit):
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
                if self._has_vace and self._is_vace_unit(unit):
                    # See cache-hit branch above: we replace the vendored VACE
                    # encode with the batched helper used in training.
                    if i == last_text_idx and prompt_embed_cache is not None:
                        prompt_embed_cache[prompt_key] = inputs_posi.copy()
                    continue
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

        self._build_vace_context_for_deploy(inputs_shared, first_frame_image, vace_video)
        self._finalize_ti2v_first_frame_latents(inputs_shared, first_frame_image)
        return inputs_shared

    def _finalize_ti2v_first_frame_latents(self, inputs_shared: dict, first_frame_image) -> None:
        """Emit ``first_frame_latents`` for TI2V deploy.

        TI2V's ``seperated_timestep`` DiT requires both
        ``fuse_vae_embedding_in_latents=True`` AND ``first_frame_latents`` so
        the per-token timestep path can zero the timestep on frame-0 tokens
        and ``base.generate`` can clean-replace ``latents[:, :, 0:1]`` on every
        denoising step.

        VACE intentionally has no branch here: its first-frame condition flows
        through ``vace_context`` (built by ``_build_vace_context_for_deploy``),
        and ``video`` itself stays fully noised — matching the native
        ``WanVideoUnit_VACE`` "predict everything via the bypass" semantic.

        I2V also skips this path: its conditioning rides on the ``y`` channel,
        ``first_frame_latents`` is never used.
        """
        if not self._is_ti2v or first_frame_image is None:
            if first_frame_image is None:
                inputs_shared.pop("first_frame_latents", None)
                inputs_shared["fuse_vae_embedding_in_latents"] = False
                inputs_shared["num_clean_prefix_frames"] = 0
            return
        device = self.device
        dtype = self.dtype
        inputs_shared["fuse_vae_embedding_in_latents"] = True
        inputs_shared["num_clean_prefix_frames"] = 0
        ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
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
        if self._uses_external_encoder:
            return self._encoder.preprocess_video(frames)
        return self._pipe.preprocess_video(frames)

    def _encode_video(self, video_tensor: Tensor, *, tiled: bool = False) -> Tensor:
        if self._uses_external_encoder:
            return self._encoder.batch_encode(video_tensor)
        return self._pipe.vae.batch_encode(video_tensor, device=video_tensor.device)

    def _encode_video_for_vace(
        self,
        pixels: Tensor,
        *,
        tiled: bool,
        tile_size: tuple,
        tile_stride: tuple,
    ) -> Tensor:
        """Tiled-aware VAE encode for the VACE pixel→latent helper.

        Two branches:

        - ``tiled=False`` (training default + non-tiled deploy): one batched
          ``vae.batch_encode(B, 3, T, H, W)`` call. Fast, but requires the
          full video to fit on one GPU.
        - ``tiled=True`` (deploy default at 480x832 / 720x1280): loop the
          batch and invoke ``vae.encode([video], tiled=True, ...)`` per
          sample. Slower but bounded peak memory. Mirrors what the vendored
          ``WanVideoUnit_VACE.process`` does at B=1 — without this the
          deploy VACE encode would silently switch to full-frame and OOM.

        Returns a tensor on ``pixels`` device/dtype with shape
        ``(B, z_dim, T_lat, H_lat, W_lat)``.
        """
        if not tiled:
            return self._encode_video(pixels).to(dtype=pixels.dtype, device=pixels.device)
        if self._uses_external_encoder:
            # External encoders do not expose a generic tiled-encode contract;
            # fall back to batch_encode. Currently unreachable since VACE +
            # external_encoder is fail-fast at construction
            # (``wan_adapter.py:_pipe.vace is not None`` branch).
            return self._encoder.batch_encode(pixels).to(dtype=pixels.dtype, device=pixels.device)
        # Native Wan VAE: loop B samples (deploy is B=1, training never sets
        # tiled=True) and call the per-sample tiled encode.
        outs = []
        for i in range(pixels.shape[0]):
            lat = self._pipe.vae.encode(
                [pixels[i]],
                device=self.device,
                tiled=True,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            outs.append(lat)
        return torch.cat(outs, dim=0).to(dtype=pixels.dtype, device=pixels.device)

    def _decode_latents(self, latents: Tensor, *, tiled: bool = True) -> Tensor:
        if self._uses_external_encoder:
            return self._encoder.decode(latents.to(self.device), tiled=tiled)
        return self._pipe.vae.decode(latents.to(self.device), device=self.device, tiled=tiled)

    def _latents_to_frames(self, video_tensor: Tensor) -> list:
        if self._uses_external_encoder:
            return self._encoder.to_frames(video_tensor)
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

    # ================================================================
    # Native-VACE input convention (training + deploy)
    # ================================================================
    #
    # These helpers replicate ``WanVideoUnit_VACE.process``
    # (``wan/pipeline.py:782-876``) verbatim except they:
    #
    #   - Accept a batch (B >= 1) where the vendored unit assumes B=1.
    #     Native unit is hardcoded for the inference path which never sees
    #     B > 1; OpenWAM training is batched. TI2V/I2V follow the same
    #     "manually batch around the unit" approach via ``batch_encode`` and
    #     ``_build_i2v_y``; this is the VACE counterpart.
    #
    #   - Skip the optional ``vace_reference_image`` prepend (lines 846-871 of
    #     the vendored unit). OpenWAM uses VACE for the "know first frame,
    #     predict the rest" use case; the canonical encoding is
    #     ``vace_video = [first_frame, black, ..., black]``,
    #     ``vace_mask = [0, 1, ..., 1]``, ``ref_image = None`` — keeping
    #     ``vace_context.shape[2] == video_latent.shape[2]`` and avoiding the
    #     extra latent frame that ref-prepend would inject.
    #
    # A parity test in ``tests/test_vace_native_input_path.py`` pins
    # ``_build_vace_context_from_pixels`` to be element-wise equal to the
    # vendored unit at B=1 (no ref_image case) to guarantee no semantic drift.

    @staticmethod
    def _is_vace_unit(unit) -> bool:
        """True iff ``unit`` is the vendored ``WanVideoUnit_VACE``.

        We supersede that unit's pixel→latent encode with
        :meth:`_build_vace_context_from_pixels` so train and deploy share the
        same batched path. Skipping the unit avoids both double work and the
        B=1 ref-prepend semantic that conflicts with our T_lat==video-latent
        length contract.
        """
        return unit.__class__.__name__ == "WanVideoUnit_VACE"

    def _build_vace_pixel_inputs(
        self,
        *,
        vace_videos,
        first_frame_image,
        B: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        preprocessed_video: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Construct the native pixel-space ``(vace_video, vace_video_mask)``
        pair fed to :meth:`_build_vace_context_from_pixels`.

        Output convention (matches what the vendored
        ``WanVideoUnit_VACE.process`` would expect to receive *after*
        ``pipe.preprocess_video`` had been applied to its PIL-list input):

            vace_video shape:   (B, 3, T_pix, H, W),   values in [-1, 1]
            vace_video_mask:    (B, 1, T_pix, H, W),   values in [0, 1]

        Three branches, in priority:

          1. ``vace_videos[i]`` is a user-provided PIL list / tensor: use it
             verbatim for sample i, with mask defaulting to all-ones (every
             frame reactive / to-be-predicted). Currently unreachable in
             training (the OpenWAM dataloader always sets vace_video=None)
             but kept for forward compatibility.

          2. ``first_frame_image`` is provided (training and deploy default):
             the canonical "know first frame, predict the rest" form.
             vace_video = [first_frame_pp, black, ..., black]
             vace_mask  = [0, 1, ..., 1]

          3. Neither: unconditional generation through the VACE bypass.
             vace_video stays all-black (preprocessed -1, NOT 0 — "0" in
             preprocessed space is *gray*, not black; the native unit derives
             this implicitly by passing PIL black PNGs through
             ``pipe.preprocess_video`` which maps RGB 0 → -1).
             vace_mask stays all-ones.

        ``preprocessed_video`` is the batched (B, 3, T, H, W) preprocessed
        video tensor returned by ``pipe.preprocess_video`` upstream. When
        provided AND branch (2) fires, we slice ``[:, :, 0:1]`` directly
        instead of re-preprocessing the first PIL frame, saving one CPU
        copy per training step.
        """
        # Padding init = preprocessed black (-1). This matches the native
        # unit's behavior when the user passes PIL black images through
        # ``pipe.preprocess_video`` (which maps RGB(0,0,0) → -1 via
        # ``image * (2/255) + (-1) = -1``). NOT torch.zeros — that would be
        # preprocessed-0 = "gray", which is the bug the old latent-space
        # construction effectively committed.
        vace_video_pixels = torch.full(
            (B, 3, num_frames, height, width), fill_value=-1.0, dtype=dtype, device=device
        )
        vace_mask_pixels = torch.ones(
            (B, 1, num_frames, height, width), dtype=dtype, device=device
        )

        has_ref = first_frame_image is not None
        user_provided_any = vace_videos is not None and any(vv is not None for vv in vace_videos)

        if user_provided_any:
            for i in range(B):
                vv = vace_videos[i] if vace_videos is not None else None
                if vv is not None:
                    vv_pp = self._preprocess_video(vv).to(dtype=dtype, device=device)
                    if vv_pp.shape[2] != num_frames:
                        raise ValueError(
                            f"User-provided vace_videos[{i}] has T={vv_pp.shape[2]} but "
                            f"num_frames={num_frames}; this branch does not auto pad/truncate."
                        )
                    vace_video_pixels[i] = vv_pp[0]
                    # User-supplied vace_video implies "predict every frame
                    # using this as reactive" — leave mask all-ones unless
                    # they also supplied a mask (future extension point).
                elif has_ref:
                    self._fill_first_frame_condition(
                        vace_video_pixels[i : i + 1],
                        vace_mask_pixels[i : i + 1],
                        ref_image=first_frame_image[i] if isinstance(first_frame_image, list) else first_frame_image,
                        height=height,
                        width=width,
                        dtype=dtype,
                        device=device,
                        preprocessed_first_frame=(
                            preprocessed_video[i : i + 1, :, 0:1] if preprocessed_video is not None else None
                        ),
                    )
        elif has_ref:
            # Batch-wide first-frame condition: zero the t=0 mask channel.
            vace_mask_pixels[:, :, 0:1] = 0.0
            if preprocessed_video is not None and preprocessed_video.shape[0] == B:
                # Fast path: reuse the already-preprocessed input video's
                # first frame. The dataloader always passes
                # first_frame_image = [video[0]] and video[0] is what got
                # preprocessed into ``preprocessed_video[:, :, 0:1]`` — same
                # bits, no PIL roundtrip.
                vace_video_pixels[:, :, 0:1] = preprocessed_video[:, :, 0:1].to(
                    dtype=dtype, device=device
                )
            else:
                for i in range(B):
                    ref = first_frame_image[i] if isinstance(first_frame_image, list) else first_frame_image
                    if isinstance(ref, list):
                        ref = ref[0]
                    pp = (
                        self._pipe.preprocess_image(ref.resize((width, height)))
                        .to(device=device, dtype=dtype)
                    )
                    if pp.dim() == 4 and pp.shape[0] == 1:
                        pp = pp[0]
                    vace_video_pixels[i, :, 0] = pp
        # else: unconditional — keep all-black, all-ones-mask.

        return vace_video_pixels, vace_mask_pixels

    def _fill_first_frame_condition(
        self,
        vace_video_slice: Tensor,
        vace_mask_slice: Tensor,
        *,
        ref_image,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        preprocessed_first_frame: Optional[Tensor] = None,
    ) -> None:
        """In-place fill of one sample's vace_video[t=0] + vace_mask[t=0].

        ``vace_video_slice`` / ``vace_mask_slice`` are (1, 3, T, H, W) and
        (1, 1, T, H, W) views into the per-sample slots, expected to start
        as all-black / all-ones (the defaults from
        :meth:`_build_vace_pixel_inputs`).
        """
        vace_mask_slice[:, :, 0:1] = 0.0
        if preprocessed_first_frame is not None:
            vace_video_slice[:, :, 0:1] = preprocessed_first_frame.to(dtype=dtype, device=device)
            return
        ref = ref_image[0] if isinstance(ref_image, list) else ref_image
        pp = self._pipe.preprocess_image(ref.resize((width, height))).to(device=device, dtype=dtype)
        if pp.dim() == 4 and pp.shape[0] == 1:
            pp = pp[0]
        vace_video_slice[0, :, 0] = pp

    def _build_vace_context_from_pixels(
        self,
        vace_video_pixels: Tensor,
        vace_mask_pixels: Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple = (34, 34),
        tile_stride: tuple = (18, 16),
    ) -> Tensor:
        """Batched reimplementation of ``WanVideoUnit_VACE.process``'s pixel
        → latent conversion (skipping the ``vace_reference_image`` prepend).

        Mirror of ``wan/pipeline.py:782-876`` — same formulas, only changes
        are (a) ``[:, 0]`` instead of ``[0, 0]`` to keep the batch dim, and
        (b) one extra ``B`` axis in the ``rearrange`` pattern. P=Q=8 and the
        ``(T_pix + 3) // 4`` temporal downsample come from the vendored unit
        verbatim; both are baked into the VACE module's pretrained weights
        (``vace_in_dim = 2*z_dim + P*Q = 96``) and the Wan VAE causal 4x
        temporal compression.

        ``tiled`` / ``tile_size`` / ``tile_stride`` mirror the vendored unit's
        tiled VAE encode path. Training keeps the default (``tiled=False``,
        batched ``vae.batch_encode``) since training resolutions fit native;
        deploy forwards its ``inputs_shared['tiled' / ...]`` values so the
        VACE pixel→latent encode does NOT silently regress from tiled (the
        vendored deploy default) to full-frame at 480x832 / 720x1280 and OOM
        on a single GPU.

        Output: ``(B, 96, T_lat, H_lat, W_lat) = concat([inactive(z=16),
        reactive(z=16), mask(P*Q=64)], dim=1)``.
        """
        import torch.nn.functional as F

        # Native (B=1):
        #   inactive = vace_video * (1 - mask) + 0 * mask
        #   reactive = vace_video * mask + 0 * (1 - mask)
        # The ``+ 0 * ...`` terms are redundant; we drop them. Mathematically
        # identical to the vendored unit at B=1.
        inactive = vace_video_pixels * (1 - vace_mask_pixels)
        reactive = vace_video_pixels * vace_mask_pixels
        inactive_lat = self._encode_video_for_vace(
            inactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        reactive_lat = self._encode_video_for_vace(
            reactive, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        vace_video_latents = torch.cat([inactive_lat, reactive_lat], dim=1)

        P, Q = 8, 8
        if vace_mask_pixels.shape[3] % P != 0 or vace_mask_pixels.shape[4] % Q != 0:
            raise ValueError(
                f"vace_mask_pixels spatial dims ({vace_mask_pixels.shape[3]}, "
                f"{vace_mask_pixels.shape[4]}) must be divisible by (P=8, Q=8) "
                f"for the native VACE rearrange (pretrained vace_in_dim=96 "
                f"requires this exact tile layout)."
            )
        # Native (B=1):
        #   rearrange(vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", ...)
        # Batched:
        vace_mask_latents = rearrange(
            vace_mask_pixels[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=P, Q=Q
        )
        T_pix = vace_mask_latents.shape[2]
        T_lat = (T_pix + 3) // 4
        vace_mask_latents = F.interpolate(
            vace_mask_latents,
            size=(T_lat, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
            mode="nearest-exact",
        )

        return torch.cat([vace_video_latents, vace_mask_latents], dim=1)

    def _build_vace_context_for_deploy(
        self,
        inputs_shared: dict,
        first_frame_image,
        vace_video,
    ) -> None:
        """Deploy-side wrapper around :meth:`_build_vace_context_from_pixels`.

        Mirrors the training path in :meth:`preprocess_input`: builds the
        same pixel-space (vace_video, vace_mask) pair from the user-facing
        ``first_frame_image`` / ``vace_video`` inputs and writes the
        resulting ``vace_context`` into ``inputs_shared``. Vendored
        ``WanVideoUnit_VACE`` is skipped (see :meth:`_is_vace_unit`) so the
        two paths produce bit-equivalent vace_context.

        ``tiled`` / ``tile_size`` / ``tile_stride`` are forwarded so the
        deploy default (``tiled=True``, set on the InferenceInputs dataclass)
        keeps using the tiled VAE encode path — without this, large-frame
        deploy (480x832 / 720x1280) would silently regress to full-frame
        VAE encode and OOM on a single GPU.
        """
        if not self._has_vace:
            return
        num_frames = inputs_shared["num_frames"]
        height = inputs_shared["height"]
        width = inputs_shared["width"]
        dtype = self.dtype if self.dtype is not None else torch.bfloat16
        device = self.device

        vace_videos = [vace_video] if vace_video is not None else None
        ff_list = first_frame_image if first_frame_image is not None else None
        if ff_list is not None and not isinstance(ff_list, list):
            ff_list = [ff_list]

        vace_video_pixels, vace_mask_pixels = self._build_vace_pixel_inputs(
            vace_videos=vace_videos,
            first_frame_image=ff_list,
            B=1,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
        )
        vace_context = self._build_vace_context_from_pixels(
            vace_video_pixels,
            vace_mask_pixels,
            tiled=bool(inputs_shared.get("tiled", False)),
            tile_size=tuple(inputs_shared.get("tile_size") or (34, 34)),
            tile_stride=tuple(inputs_shared.get("tile_stride") or (18, 16)),
        )
        inputs_shared["vace_context"] = vace_context
        inputs_shared["vace_scale"] = 1.0

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
        *,
        skip_native_vae: bool = False,
    ):
        """Build an empty WanVideoPipeline from component specs (config-driven).

        Weights are NOT loaded here — ``architecture.load_checkpoint`` handles
        that separately.

        Tokenizer resolution order:
          1. ``ckpt_dir`` + ``tokenizer.subdir`` — checkpoint-local tokenizer
             copied during training save.
          2. ``model_path`` upstream layout — components-based persistence
             falls back to ``<model_path>/google/umt5-xxl/``.

        Args:
            skip_native_vae: When True, drop ``attr == "vae"`` entries before
                instantiating, so the empty native VAE never allocates CPU
                tensors. Used by the irreversible external-encoder path.
        """
        from openwam.model.video_backbone.wan.pipeline import WanVideoPipeline
        from openwam.model.video_backbone.wan.pipeline_builder import _build_tokenizer, _import_class

        pipe = WanVideoPipeline(device=device, torch_dtype=torch.bfloat16)

        for entry in components:
            if skip_native_vae and entry.get("attr") == "vae":
                continue
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
    def _build_pipe_from_model_path(model_path: str, device: str = "cpu", *, skip_native_vae: bool = False):
        """Build a WanVideoPipeline from a model directory without full Hydra config.

        Args:
            skip_native_vae: When True, filter ``discover_model_files`` output
                to drop the native VAE weight file before
                :meth:`WanVideoPipeline.from_pretrained` materializes it.
                Used by the irreversible external-encoder path.
        """
        from openwam.model.video_backbone.wan.pipeline import WanVideoPipeline
        from openwam.model.video_backbone.wan.pipeline_builder import (
            _filter_native_vae_configs,
            discover_model_files,
        )

        model_configs, tokenizer_config = discover_model_files(model_path)
        if skip_native_vae:
            model_configs = _filter_native_vae_configs(model_configs)
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


def reinit_dit_from_scratch(
    pipe,
    *,
    external_encoder=None,
    dit_patch_size: Optional[Tuple[int, int, int]] = None,
    verbose: bool = True,
) -> None:
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
        external_encoder: Optional :class:`VideoEncoder`. When provided, the
            DiT's ``patch_embedding`` and (if present) ``head.head`` are
            rebuilt via the encoder's :meth:`build_dit_input_proj` /
            :meth:`build_dit_output_proj` hooks BEFORE the stdlib reset
            loop runs. For ``wan_vae`` this is a no-op shape-wise (defaults
            reproduce the original Wan layout); for non-VAE encoders this
            adapts the first conv's ``in_channels`` to the encoder's
            ``z_dim``. ``dit.in_dim`` metadata is synced afterwards so
            downstream consumers (I2V's ``_build_i2v_y`` etc.) see the
            updated value. None preserves the historical "reset weights
            only, do not touch shapes" behavior.
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

    # Rebuild patch_embedding + head.head BEFORE reset_parameters. The
    # encoder owns both hooks; defaults produce the Wan-original Conv3d/
    # Linear pair, so wan_vae lands here as a no-op shape-wise. Non-VAE
    # encoders adapt the first conv's in_channels and final Linear's
    # out_features to the encoder's z_dim and dit_patch_size. We also sync
    # the ``patch_size`` metadata on both ``WanModel`` and ``Head`` —
    # ``WanModel.unpatchify`` uses ``self.patch_size`` for its einops
    # rearrange (dit.py:372-381), so an encoder declaring
    # ``dit_patch_size=(1,1,1)`` would otherwise feed a Linear-out of width
    # ``z_dim`` into an unpatchify that still expects ``z_dim * 4`` and
    # shape-mismatch on the first forward. ``dit.in_dim`` metadata is
    # updated so downstream code that inspects it (e.g. I2V's _build_i2v_y)
    # sees the new value (I2V itself is blocked at construction by
    # WanVideoBackbone.from_pretrained's fail-fast). See
    # docs/external_video_encoder.md §6 for the contract. The subsequent
    # reset_parameters() loop re-initializes these new modules with the
    # standard distribution again — harmless, just a duplicate random init
    # in the same distribution.
    if external_encoder is not None:
        # ``dit_patch_size`` is sourced from the backbone (single source of
        # truth — see VideoBackbone.dit_patch_size); callers in
        # ``base.py._init_video_backbone`` forward
        # ``self.video_backbone.dit_patch_size``. Required when an
        # external_encoder is supplied — falling back to
        # ``external_encoder.spec.dit_patch_size`` would silently re-introduce
        # the encoder-spec read path the backbone abstraction was meant to
        # eliminate.
        if dit_patch_size is None:
            raise ValueError(
                "reinit_dit_from_scratch: dit_patch_size is required when "
                "external_encoder is provided. Source it from the backbone "
                "(e.g. self.video_backbone.dit_patch_size) — reading from "
                "external_encoder.spec.dit_patch_size directly would bypass "
                "the single-source-of-truth invariant on VideoBackbone."
            )
        ps = tuple(dit_patch_size)
        for dit in dits:
            dit.patch_embedding = external_encoder.build_dit_input_proj(dit.dim)
            dit.patch_size = ps
            head_mod = getattr(dit, "head", None)
            # MotWanModel-style DiTs may not own a head (they exit early into
            # a control adapter); guard the assignment so the rebuild path
            # remains generic across Wan variants.
            if head_mod is not None and hasattr(head_mod, "head"):
                head_mod.head = external_encoder.build_dit_output_proj(dit.dim)
                head_mod.patch_size = ps
            dit.in_dim = external_encoder.spec.z_dim

    # ZeRO-3 interaction: when the accelerator has ZeRO-3 enabled, each
    # nn.Parameter in the loaded ``pipe.dit`` is already partitioned into a
    # 1-D shard at this point (the ``_zero3_init_disabled`` scope in
    # ``OpenWAMTrainer.__init__`` is a no-op on deepspeed 0.18.5; partitioning
    # happens during ``deepspeed.zero.Init(enabled=True)`` which the
    # Accelerator activates globally). ``Linear.reset_parameters`` then trips
    # on ``_calculate_fan_in_and_fan_out`` because the weight is 1-D. Wrap
    # the reset + hand-init in ``deepspeed.zero.GatheredParameters`` so each
    # root's params are temporarily materialized to their full
    # 2-D/5-D shape, reset, and re-partitioned on exit.
    # ``modifier_rank=0`` broadcasts rank-0's values to the rest of the
    # group, so the random reset is bit-identical across ranks regardless
    # of pre-init torch RNG drift (cfg.project.seed already enforces this,
    # but the broadcast is the deterministic floor). Newly-built modules
    # from ``build_dit_input_proj`` / ``build_dit_output_proj`` lack
    # ``ds_id`` and pass through the gather unchanged. Non-ZeRO-3 paths
    # (DDP, ZeRO-2, single GPU) hit the ``nullcontext`` fast-path.
    from contextlib import nullcontext

    def _gather_zero3(root_mod):
        try:
            import deepspeed
        except ImportError:
            return nullcontext()
        params = [p for p in root_mod.parameters() if hasattr(p, "ds_id")]
        if not params:
            return nullcontext()
        return deepspeed.zero.GatheredParameters(params, modifier_rank=0)

    before_stats = [] if (verbose and is_main) else None
    after_stats = [] if (verbose and is_main) else None

    for root in dits:
        with _gather_zero3(root):
            if before_stats is not None:
                before_stats.append(_probe_dit_stats(root))
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
            if after_stats is not None:
                after_stats.append(_probe_dit_stats(root))

    logger.info(
        "reinit_dit_from_scratch: re-initialized %d DiT module(s); VAE/T5 untouched",
        len(dits),
    )

    if verbose and is_main:
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
