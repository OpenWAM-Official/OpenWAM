"""``SanaVideoBackbone`` — OpenWAM VideoBackbone wrapper around SANA-Video.

Owns a :class:`SanaPipe` (DiT + VAE + text encoder) and exposes the standard
``prepare → run_block × N → finalize`` lifecycle plus the
``pre_attn_at_layer`` / ``post_attn_at_layer`` split that a future
``SanaMoTJointDriver`` will consume.

Phase 0 scope:
- ``prepare``, ``run_block``, ``finalize``: bit-equivalent to upstream
  ``SanaMSVideo.forward`` for per-sample timesteps.
- ``pre_attn_at_layer``: returns the rotated and unrotated kernel-applied
  Q/K so a downstream driver can implement cumsum-linear-attn over a
  heterogeneous sequence. ``attn_kernel`` property advertises ``linear_relu``
  for driver dispatch.
- All other VideoBackbone abstract methods are stubbed with a clear
  ``NotImplementedError`` — they are not needed for the Phase 0 numerical
  equivalence tests and will be filled in alongside Phase 4 deploy work.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.sana.blocks_split import SanaMSVideoSplit
from openwam.model.video_backbone.sana.pipeline_builder import (
    SanaPipe,
    build_mini_sana_pipeline,
    build_sana_pipeline,
)


def _pil_video_to_tensor(frames: Any) -> Tensor:
    """``list[list[PIL.Image]]`` → ``(B, 3, T, H, W)`` float in ``[-1, 1]``.

    Mirrors ``cosmos25/pipeline_wrapper._pil_video_to_tensor`` — same convention
    used across OpenWAM (uint8 RGB → ``/127.5 - 1``). Kept private here so the
    SANA backbone doesn't depend on the Cosmos25 module.
    """
    import numpy as np

    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError(
            f"Expected `frames` as a non-empty list of clips (each a list of PIL frames); "
            f"got {type(frames).__name__}."
        )
    arrs = []
    for clip in frames:
        if not isinstance(clip, (list, tuple)) or not clip:
            raise ValueError("Each per-sample entry in `frames` must be a non-empty list of PIL.Image frames.")
        clip_arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in clip], axis=0)
        arrs.append(clip_arr)
    stack = np.stack(arrs, axis=0)  # (B, T, H, W, 3) uint8
    t = torch.from_numpy(stack).to(dtype=torch.float32)
    t = t / 127.5 - 1.0
    return t.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)

logger = logging.getLogger(__name__)


class SanaVideoBackbone(VideoBackbone):
    """SANA-Video backbone for OpenWAM.

    Owns a :class:`SanaPipe`; never exposes it. External code accesses
    capabilities through the VideoBackbone ABC.
    """

    supports_generic_mot_compile = False
    generic_mot_compile_skip_reason = "SANA uses linear-ReLU MoT attention and needs a dedicated compile helper"

    @classmethod
    def get_native_dit_patch_size(cls, pipe) -> Tuple[int, int, int]:
        """SANA-Video native DiT first-layer patch size.

        ``SanaMSVideo`` constructs ``x_embedder`` with ``patch_size=(1, 2, 2)``
        (see ``pipeline_builder._SANA_VIDEO_2B_480P_PRESET`` and
        ``build_mini_sana_pipeline``). Hard-coded — same rationale as Wan:
        invariant across the SANA-Video family and probing ``pipe.dit`` would
        couple this classmethod to a non-trivial pipeline state.
        """
        return (1, 2, 2)

    @classmethod
    def get_native_temporal_contract(cls, pipe) -> Tuple[int, bool]:
        """SANA-Video native VAE temporal contract.

        SANA-Video 2B 480p reuses the Wan2.1 causal VAE
        (``pipeline_builder._load_wan_vae`` loads ``Wan2.1_VAE.pth``), so the
        contract matches the Wan family: ``(4, True)`` — causal first-frame
        token plus 4-frame tail grouping.
        """
        return (4, True)

    def __init__(self, pipe: SanaPipe):
        super().__init__()
        self._pipe = pipe
        self._split = SanaMSVideoSplit(pipe.dit)

        # SanaPipe is @dataclass (not nn.Module), so `self._pipe = pipe`
        # doesn't auto-register its DiT/VAE/text_encoder as nn.Module
        # children — they stay invisible to self.parameters() /
        # named_children(), and the optimizer-groups builder finds 0
        # trainable params on the video side. Register them explicitly so
        # the standard parameter-discovery path works (the DiT becomes
        # trainable; the VAE / text encoder are already
        # requires_grad_(False) at construction time and stay frozen).
        # `_pipe.<name>` remains the canonical access — add_module just
        # files the same object reference under self._modules[name].
        self.add_module("dit", pipe.dit)
        if pipe.vae is not None:
            self.add_module("vae", pipe.vae)
        if pipe.text_encoder is not None:
            self.add_module("text_encoder", pipe.text_encoder)

        # Resolve DiT patch geometry + VAE temporal contract into backbone-owned
        # instance attributes so the ABC properties (dit_patch_size /
        # temporal_compression / causal_temporal) return without any branching.
        # SanaVideoBackbone doesn't accept an external encoder, so the native
        # values are the only values this backbone ever exposes.
        self._dit_patch_size = self.get_native_dit_patch_size(pipe)
        self._temporal_compression, self._causal_temporal = self.get_native_temporal_contract(pipe)

        first_param = next(pipe.dit.parameters(), None)
        self._dtype = first_param.dtype if first_param is not None else torch.bfloat16
        self._device = first_param.device if first_param is not None else torch.device("cuda")

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, source: Any, **kw: Any) -> "SanaVideoBackbone":
        """Build a SanaVideoBackbone from a config / dict / model directory.

        Delegates to :func:`build_sana_pipeline`; see that function's docstring
        for accepted shapes. For unit tests, use :meth:`from_mini_config`.
        """
        pipe = build_sana_pipeline(source, **kw)
        return cls(pipe)

    @classmethod
    def from_mini_config(cls, **mini_kwargs: Any) -> "SanaVideoBackbone":
        """Random-weight tiny backbone for unit tests. See
        :func:`build_mini_sana_pipeline` for available kwargs."""
        pipe = build_mini_sana_pipeline(**mini_kwargs)
        return cls(pipe)

    # ----------------------------------------------------------------
    # ABC: properties
    # ----------------------------------------------------------------

    @property
    def _dit(self) -> nn.Module:
        return self._pipe.dit

    @property
    def dim(self) -> int:
        return int(self._dit.hidden_size)

    @property
    def num_layers(self) -> int:
        return len(self._dit.blocks)

    @property
    def num_heads(self) -> int:
        """Number of SELF-attention heads.

        SANA's ``LiteLAReLURope`` exposes ``heads`` + ``dim`` (sana_blocks.py:326),
        while ``FlashAttention`` uses ``num_heads`` + ``head_dim``. Both are
        possible attn modules depending on the factory preset, so probe in order.
        Falls back to ``hidden_size // head_dim`` if neither attr is present.
        """
        attn = self._dit.blocks[0].attn
        for name in ("heads", "num_heads"):
            v = getattr(attn, name, None)
            if v is not None:
                return int(v)
        return int(self._dit.hidden_size // self.head_dim)

    @property
    def context_dim(self) -> int:
        """SANA-Video text context dim (Gemma-2-2B last_hidden_state = 2304).

        Read by :meth:`BaseWAMArchitecture._resolve_text_dim` to size the
        proprio encoder and any action-side context projections. SANA's DiT
        doesn't store ``caption_channels`` directly; we read it off the
        ``y_embedder.y_embedding`` buffer's last dim (the 480p preset pins
        this to 2304 via ``_SANA_VIDEO_2B_480P_PRESET``; the mini-config
        factory uses a tiny value for unit tests).
        """
        return int(self._dit.y_embedder.y_embedding.shape[-1])

    @property
    def head_dim(self) -> int:
        attn = self._dit.blocks[0].attn
        for name in ("dim", "head_dim"):
            v = getattr(attn, name, None)
            if v is not None:
                return int(v)
        raise AttributeError(
            f"SanaVideoBackbone.head_dim: attn module {type(attn).__name__!r} "
            "exposes neither 'dim' nor 'head_dim'."
        )

    @property
    def scheduler(self):
        return self._pipe.scheduler

    @property
    def submodule_names(self) -> list[str]:
        names = ["dit"]
        for n in ("vae", "text_encoder"):
            if getattr(self._pipe, n, None) is not None:
                names.append(n)
        return names

    @property
    def video_attention_mask_mode(self) -> str:
        """SANA-Video uses ``first_frame_causal`` semantics in OpenWAM joint
        MoT (matching the FastWAM default). Phase 3+ ``SanaMoTJointDriver``
        relies on this; Phase 0 doesn't read it but the value is set now so the
        contract is stable across phases."""
        return "first_frame_causal"

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        """Build the video↔video block of the joint MoT attention mask.

        Direct port of :meth:`WanVideoBackbone.build_video_to_video_mask`
        (``wan_adapter.py:203-248``); the math only depends on
        ``video_tokens_per_frame``, so the Wan↔SANA layout difference is
        transparent here.
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

    @property
    def attn_kernel(self) -> str:
        """Driver dispatch hook (read by :class:`DualSystemSelfAttnArchitecture`
        in Phase 3). ``"linear_relu"`` tells the architecture to construct
        ``SanaMoTJointDriver`` instead of the SDPA-based ``MoTJointDriver``.

        This property is intentionally NOT abstract on the base ABC — Wan /
        Cosmos25 backbones don't define it and ``getattr(..., "softmax")`` at
        the dispatch site lets them keep their existing path.
        """
        return "linear_relu"

    # ----------------------------------------------------------------
    # ABC: three-step lifecycle
    # ----------------------------------------------------------------

    def prepare(self, **pipeline_inputs: Any) -> BlockLoopState:
        """Patchify + RoPE + t_mod + y_embed. Mirrors upstream forward prelude.

        Accepts two equivalent input shapes:

        - SANA-native (Phase 0 / GPU smoke):
            ``x: (B, C, T, H, W)``, ``timestep: (B,)``,
            ``y: (B, 1, L, D)``, ``mask: (B, 1, 1, L)`` int/bool

        - OpenWAM architecture flow (``BaseWAMArchitecture.compute_loss``):
            ``latents: (B, C, T, H, W)``, ``timestep: (B,)``,
            ``context: (B, L, D)``, optionally ``context_mask: (B, L) bool``
            or ``seq_lens: (B,) long``.

        When called from the architecture, this method translates the standard
        keys → SANA shapes inline so the rest of ``SanaMSVideoSplit.prepare`` is
        unchanged. Unrecognized kwargs are silently dropped (e.g. ``force_per_token_t_mod``
        / ``zero_clean_prefix_t_mod`` from :class:`DualSystemSelfAttnArchitecture`
        — those are Wan-only knobs; SANA's time embedding is per-sample).
        """
        # --- 1. latent ---
        if "x" in pipeline_inputs:
            x = pipeline_inputs.pop("x")
        elif "latents" in pipeline_inputs:
            x = pipeline_inputs.pop("latents")
        else:
            raise KeyError(
                "SanaVideoBackbone.prepare requires either 'x' or 'latents' in pipeline_inputs."
            )

        timestep = pipeline_inputs.pop("timestep")

        # --- 2. caption embeddings → (B, 1, L, D) ---
        if "y" in pipeline_inputs:
            y = pipeline_inputs.pop("y")
        elif "context" in pipeline_inputs:
            context = pipeline_inputs.pop("context")
            if context.dim() == 3:
                # (B, L, D) → (B, 1, L, D) — SANA's MultiHeadCrossAttention expects
                # a leading singleton dimension for the per-sample caption track.
                y = context.unsqueeze(1)
            else:
                y = context
        else:
            raise KeyError(
                "SanaVideoBackbone.prepare requires 'y' or 'context' in pipeline_inputs."
            )

        # --- 3. caption mask → (B, 1, 1, L) int16 ---
        # SANA's DISABLE_XFORMERS=1 path requires a non-None caption mask. The
        # architecture passes ``seq_lens`` (B,) or ``context_mask`` (B, L) bool;
        # both are converted to the (B, 1, 1, L) int16 layout the split expects.
        if "mask" in pipeline_inputs:
            mask = pipeline_inputs.pop("mask")
        else:
            context_mask = pipeline_inputs.pop("context_mask", None)
            seq_lens = pipeline_inputs.pop("seq_lens", None)
            B, L = y.shape[0], y.shape[2]
            if context_mask is not None:
                # (B, L) bool → (B, 1, 1, L)
                mask = context_mask.to(dtype=torch.int16, device=y.device).view(B, 1, 1, L)
            elif seq_lens is not None:
                positions = torch.arange(L, device=y.device).view(1, 1, 1, L)
                lens = seq_lens.to(device=y.device).view(B, 1, 1, 1)
                mask = (positions < lens).to(torch.int16)
            else:
                # All-attend mask: every text token is valid.
                mask = torch.ones(B, 1, 1, L, dtype=torch.int16, device=y.device)

        # SANA's split.prepare doesn't consume the rest of the OpenWAM keys
        # (``input_latents``, ``num_clean_prefix_frames``, ``height``, ``width``,
        # ``num_frames``, ``vace_context``, ``first_frame_latents``,
        # ``actions``, ``proprio``, padding masks, etc.). Drop them so we don't
        # collide with the upstream ``**kwargs`` slot.
        for k in (
            "input_latents",
            "num_clean_prefix_frames",
            "height",
            "width",
            "num_frames",
            "vace_context",
            "vace_scale",
            "first_frame_latents",
            "fuse_vae_embedding_in_latents",
            "clip_feature",
            "force_per_token_t_mod",
            "zero_clean_prefix_t_mod",
            "actions",
            "proprio",
            "action_mask",
            "video_mask",
            "use_gradient_checkpointing",
            "use_gradient_checkpointing_offload",
        ):
            pipeline_inputs.pop(k, None)

        prep = self._split.prepare(x, timestep, y, mask, **pipeline_inputs)

        # Stash everything the per-block loop needs in ``extras`` to keep the
        # standard fields semantically pure (e.g. ``t_mod`` is the AdaLN
        # 6-scale modulation, not the raw time embedding).
        return BlockLoopState(
            x=prep["x"],
            t_mod=prep["t0"],
            freqs=prep["image_pos_embed"],
            context=prep["y"],
            context_mask=prep["y_lens"],
            f=prep["f"],
            h=prep["h"],
            w=prep["w"],
            t=prep["t"],
            extras={
                "split": self._split,
                "bs": prep["bs"],
                "block_kwargs": prep["kwargs"],
            },
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        """One upstream-block step — used by the video-only path.

        Joint-MoT drivers call :meth:`pre_attn_at_layer` and
        :meth:`post_attn_at_layer` separately and skip this method.
        """
        split: SanaMSVideoSplit = state.extras["split"]
        out = split.run_block(
            block_id,
            {
                "x": state.x,
                "y": state.context,
                "t0": state.t_mod,
                "y_lens": state.context_mask,
                "f": state.f,
                "h": state.h,
                "w": state.w,
                "image_pos_embed": state.freqs,
                "kwargs": state.extras.get("block_kwargs", {}),
            },
        )
        state.x = out["x"]
        return state

    def finalize(self, state: BlockLoopState) -> Tensor:
        split: SanaMSVideoSplit = state.extras["split"]
        return split.finalize(
            {
                "x": state.x,
                "t": state.t,
            }
        )

    # ----------------------------------------------------------------
    # ABC: joint self-attention split (Phase 3-ready, available in Phase 0)
    # ----------------------------------------------------------------

    def pre_attn_at_layer(
        self, layer_id: int, state: BlockLoopState
    ) -> Tuple[Tensor, Tensor, Tensor, dict]:
        return self._split.block_pre_attn(layer_id, state.x, state.t_mod, state.freqs)

    def post_attn_at_layer(
        self,
        layer_id: int,
        state: BlockLoopState,
        attn_out: Tensor,
        post_state: dict,
    ) -> BlockLoopState:
        state.x = self._split.block_post_attn(
            layer_id,
            attn_out,
            post_state,
            y=state.context,
            y_lens=state.context_mask,
            f=state.f,
            h=state.h,
            w=state.w,
        )
        return state

    # ----------------------------------------------------------------
    # ABC: action injection (Phase 0 scope: not supported)
    # ----------------------------------------------------------------

    def inject_action_tokens(
        self,
        state: BlockLoopState,
        action_tokens: Tensor,
        n_action: int,
        *,
        timestep: Optional[Tensor] = None,
    ) -> BlockLoopState:
        # SharedBackbone path. Not part of Phase 0 — SanaVideoBackbone only
        # supports DualSystem (joint cross/self-attn) for now.
        raise NotImplementedError(
            "SanaVideoBackbone.inject_action_tokens: SharedBackbone path is out "
            "of scope for Phase 0 of the SANA integration. See "
            "plans/sana_mot_integration_plan.md §8 for the deferred work."
        )

    def extract_action_tokens(
        self, state: BlockLoopState, n_action: int
    ) -> Tuple[BlockLoopState, Tensor]:
        raise NotImplementedError(
            "SanaVideoBackbone.extract_action_tokens: see inject_action_tokens."
        )

    # ----------------------------------------------------------------
    # ABC: preprocessing / decoding / device (deferred — Phase 4)
    # ----------------------------------------------------------------

    def preprocess_input(
        self,
        *,
        frames: Any = None,
        text: Any = None,
        pre_encoded_text: Optional[Tensor] = None,
        input_latents: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        proprio: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
        video_mask: Optional[Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        **kw: Any,
    ) -> dict:
        """Convert a raw sample → dict consumable by ``BaseWAMArchitecture.compute_loss``.

        SANA-Video is text-to-video; first-frame conditioning is out of scope
        here (see plan §8). ``FirstFrameConditioningTransform`` may inject
        ``first_frame_image`` / ``ref_images`` into the sample — those are
        silently dropped via ``**kw`` so the T2V semantics stay clean.

        Args:
            frames: ``list[list[PIL.Image]]`` per-sample video clips, or ``None``
              when ``input_latents`` is supplied (cache path).
            text: ``str`` or ``list[str]`` raw captions, or ``None`` when
              ``pre_encoded_text`` is supplied. Live encoding requires both
              ``self._pipe.text_encoder`` and ``self._pipe.tokenizer`` to be set.
            pre_encoded_text: ``(B, L, 2304)`` Gemma-2-2B last_hidden_state, or
              ``(L, 2304)`` (single sample, will be unsqueezed). Takes precedence
              over ``text`` when both are present (cache > live, matching the
              Cosmos25 contract).
            input_latents: ``(B, 16, T_lat, H_lat, W_lat)`` pre-encoded latents;
              when set, the VAE encode step is skipped.

        Returns:
            Dict with ``input_latents``, ``context``, ``seq_lens``,
            ``num_clean_prefix_frames=0``, ``height``, ``width``, ``num_frames``,
            and any of ``actions`` / ``proprio`` / ``action_mask`` / ``video_mask``
            that were supplied.
        """
        device = self._device
        dtype = self._dtype

        # --- video → latents (cache > VAE encode) ---
        if input_latents is None:
            if frames is None:
                raise ValueError(
                    "SanaVideoBackbone.preprocess_input requires either 'frames' or 'input_latents'."
                )
            if self._pipe.vae is None:
                raise RuntimeError(
                    "SanaVideoBackbone.preprocess_input(frames=...) requires a loaded "
                    "VAE. Build the pipeline with vae_path pointing to Wan2.1_VAE.pth."
                )
            video = _pil_video_to_tensor(frames).to(device=device, dtype=dtype)
            B = video.shape[0]
            videos_list = [video[i] for i in range(B)]  # list of (3, T, H, W)
            with torch.no_grad():
                # ``tiled=True``: encode in (T,H,W) tiles. At 81 frames × 480p,
                # ``tiled=False`` would peak at ~30 GB of intermediate activations
                # through the WanVAE downsampling stack and OOM 80 GB cards once
                # the DiT/ActionDiT are also resident.
                input_latents = self._pipe.vae.encode(videos_list, device=device, tiled=True)
            input_latents = input_latents.to(device=device, dtype=dtype)
            vid_T, vid_H, vid_W = video.shape[2], video.shape[3], video.shape[4]
        else:
            input_latents = input_latents.to(device=device, dtype=dtype)
            B = input_latents.shape[0]
            vid_T = num_frames if num_frames is not None else input_latents.shape[2]
            vid_H = height if height is not None else input_latents.shape[3] * 8
            vid_W = width if width is not None else input_latents.shape[4] * 8

        # --- text → context (pre-encoded > live Gemma) ---
        if pre_encoded_text is not None:
            context = pre_encoded_text
            if context.dim() == 2:
                context = context.unsqueeze(0)
            context = context.to(device=device, dtype=dtype)
        elif text is not None:
            if self._pipe.text_encoder is None or self._pipe.tokenizer is None:
                raise RuntimeError(
                    "Live text encoding requires a loaded Gemma-2-2B text encoder "
                    "and tokenizer. Pass `pre_encoded_text` or wire `text_encoder_name` "
                    "in the pipeline config."
                )
            context = self._encode_text(text).to(device=device, dtype=dtype)
        else:
            raise ValueError(
                "SanaVideoBackbone.preprocess_input requires either 'text' or 'pre_encoded_text'."
            )

        seq_lens = torch.full(
            (context.shape[0],), context.shape[1], dtype=torch.long, device=device
        )

        out: dict = {
            "input_latents": input_latents,
            "context": context,
            "seq_lens": seq_lens,
            "num_clean_prefix_frames": 0,  # SANA is pure T2V; no TI2V prefix injection.
            "height": vid_H,
            "width": vid_W,
            "num_frames": vid_T,
        }
        for name, value in (
            ("actions", actions),
            ("proprio", proprio),
            ("action_mask", action_mask),
            ("video_mask", video_mask),
        ):
            if value is not None:
                out[name] = value
        return out

    def _encode_text(self, text: Any) -> Tensor:
        """Tokenize + run Gemma-2-2B → ``last_hidden_state`` of shape (B, L, 2304).

        Live encoding path for deploy. Smoke tests bypass this by passing
        ``pre_encoded_text``.
        """
        if isinstance(text, str):
            text = [text]
        tokenizer = self._pipe.tokenizer
        encoder = self._pipe.text_encoder
        enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
            next(encoder.parameters()).device
        )
        with torch.no_grad():
            out = encoder(**enc, output_hidden_states=False)
        return out.last_hidden_state

    def get_submodule(self, name: str) -> nn.Module | None:
        return getattr(self._pipe, name, None)

    def set_submodule(self, name: str, module: nn.Module) -> None:
        if not hasattr(self._pipe, name):
            raise KeyError(f"SanaPipe has no submodule {name!r}")
        setattr(self._pipe, name, module)
        if name == "dit":
            # Re-target the split view at the new DiT instance.
            self._split = SanaMSVideoSplit(module)
        # Keep the nn.Module child registration in sync so parameter
        # discovery (self.parameters() / named_children()) sees the new
        # module. `add_module` overwrites self._modules[name] in place.
        if name in self._modules:
            self.add_module(name, module)

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        """Decode ``(B, 16, T_lat, H_lat, W_lat)`` latents → per-sample
        ``(3, T, H, W)`` videos in ``[-1, 1]``.

        Calls into the Wan2.1 VAE (``vae/Wan2.1_VAE.pth``) bundled with the
        SANA-Video 480p HF asset. ``tiled=True`` enables tile-by-tile decoding
        for inference paths where full-frame decode would OOM; training /
        smoke can pass ``tiled=False`` for speed.
        """
        if self._pipe.vae is None:
            raise RuntimeError(
                "SanaVideoBackbone.decode_video requires a loaded VAE. Build the "
                "pipeline with vae_path pointing to Wan2.1_VAE.pth."
            )
        B = latents.shape[0]
        latents_list = [latents[i] for i in range(B)]
        with torch.no_grad():
            videos = self._pipe.vae.decode(latents_list, device=self._device, tiled=tiled)
        return [videos[i] for i in range(videos.shape[0])]

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        self._dtype = dtype
        self._device = device
        self._pipe.dit.to(device=device, dtype=dtype)
        for name in ("vae", "text_encoder"):
            sub = getattr(self._pipe, name, None)
            if sub is not None:
                sub.to(device=device, dtype=dtype)
