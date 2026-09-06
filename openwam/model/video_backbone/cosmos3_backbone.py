"""Cosmos3-Edge video backbone.

Implements the :class:`VideoBackbone` contract on NVIDIA's Cosmos3-Edge unified
dual-stream transformer (vendored in ``cosmos3/_vendor/``) — flat named children
(``self.dit`` / ``self.vae``) mirroring the other families. The heavy batched
block-loop lives in ``cosmos3/dit_forward.py``; prompt tokenization + mRoPE
packing in ``cosmos3/text_pack.py``; component construction in
``cosmos3/pipeline_builder.py`` (imported lazily inside :meth:`from_pretrained`
so this module stays CPU-CI safe).

The und (text) stream is causal, frozen, and computationally independent of the gen (video)
stream, so :meth:`preprocess_input_for_train` runs the whole und tower once
under ``no_grad`` and caches the per-layer gen-facing K/V; the und final hidden
(2048-wide) doubles as the ``context`` tensor the action stream consumes
(``text_dim == 2048``). There is no cross-attention and no external text
encoder; timestep conditioning is additive on noisy-frame tokens only.

Supported architectures: ``dual_system`` / {``joint_cross_attn``,
``joint_self_attn``, ``idm``}, ``single_system`` / {``vanilla``, ``moe``},
and ``tri_system`` / {``joint_self_attn``}. ``joint_self_attn``, ``idm``, and
``tri_system`` ride MoT via the und-prefix-KV declaration + GQA KV-expand;
the tri-system driver widens the joint mask for the cached und prefix.
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
        duration_template: bool = True,
        clip_fps: float = _DEFAULT_FPS,
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
        self._duration_template = bool(duration_template)
        # Clip frame rate. Nothing in the dataloader stack reports one, so this
        # is a configured assumption rather than a measurement — see
        # ``_DEFAULT_CLIP_FPS`` in pipeline_builder. Callers may still override
        # per call via the ``fps`` kwarg.
        self._clip_fps = float(clip_fps)
        self._max_text_tokens = int(max_text_tokens)
        if not 0.0 <= float(text_dropout_p) <= 1.0:
            raise ValueError(f"text_dropout_p must be in [0, 1]; got {text_dropout_p!r}.")
        self._text_dropout_p = float(text_dropout_p)
        self._text_dropout_rng = random.Random(text_dropout_seed)
        self._pristine_inv_freq = self._capture_pristine_inv_freq()

    def _capture_pristine_inv_freq(self) -> Optional[Tensor]:
        """Keep an fp32 copy of the rotary table that no dtype cast can reach.

        Held as a plain attribute, deliberately: ``nn.Module._apply`` walks
        parameters and registered buffers, so a plain tensor survives every
        ``.to(dtype=)`` / ``.bfloat16()`` performed on the module — including
        DeepSpeed's ``self.module.bfloat16()`` inside ``_configure_distributed_model``,
        which casts floating-point *buffers* and runs between the trainer's two
        ``set_dtype_device`` calls. Snapshotting inside ``set_dtype_device``
        cannot survive that: by the second call the buffer is already bf16, and
        upcasting a rounded value recovers the dtype, not the bits.

        Falls back to recomputing from config when the live buffer is already
        non-fp32 at construction (a caller that pre-cast the net), since a
        rounded snapshot would be worse than the exact closed form.
        """
        rope = getattr(self.dit, "rotary_emb", None)
        inv_freq = getattr(rope, "inv_freq", None) if rope is not None else None
        if inv_freq is None:
            return None
        if inv_freq.dtype == torch.float32:
            return inv_freq.detach().to(device="cpu", copy=True)
        cfg = getattr(self.dit, "config", None)
        head_dim = getattr(cfg, "head_dim", None)
        rope_theta = getattr(cfg, "rope_theta", None)
        if head_dim is None or rope_theta is None:
            logger.warning("cosmos3_edge: rotary inv_freq is %s at build and cannot be recomputed.", inv_freq.dtype)
            return None
        logger.warning(
            "cosmos3_edge: rotary inv_freq was already %s at build; recomputing the fp32 table from config.",
            inv_freq.dtype,
        )
        return dit_forward.rotary_inv_freq(int(head_dim), float(rope_theta))

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
        from openwam.model.video_backbone.cosmos3.pipeline_builder import build_cosmos3_pipeline

        holder = build_cosmos3_pipeline(source, device=device, ckpt_dir=ckpt_dir, **kw)
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
            duration_template=getattr(holder, "duration_template", True),
            clip_fps=getattr(holder, "clip_fps", _DEFAULT_FPS),
            max_text_tokens=holder.max_text_tokens,
            text_dropout_p=holder.text_dropout_p,
            text_dropout_seed=holder.text_dropout_seed,
        )
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
                text_pack.apply_prompt_templates(
                    p,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=fps,
                    add_duration_template=self._duration_template,
                )
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
        # Whether any prompt is padded is host-side knowledge (Python list
        # lengths), so decide it here and hand ``None`` down when there is no
        # padding — that is what keeps SDPA on a fused kernel and keeps the
        # compiled MoT graph free of a ``.all()`` device sync. Every B=1 request
        # and any uniform-length batch takes this path.
        und_mask = und_mask.to(self.device) if len(set(len(i) for i in ids)) > 1 else None

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
            reset_spatial_indices=bool(net.config.unified_3d_mrope_reset_spatial_ids),
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

        fps = float(kw.get("fps", self._clip_fps))
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
    # IDM teacher-forcing branch merge/split
    # ================================================================

    def merge_idm_video_branches(self, noisy: BlockLoopState, cond: BlockLoopState):
        """Concatenate the IDM noisy + cond branches — delegates to ``idm_merge``."""
        from openwam.model.video_backbone.cosmos3 import idm_merge

        return idm_merge.merge_branches(noisy, cond)

    def split_idm_video_branches(self, merged: BlockLoopState, noisy: BlockLoopState, cond: BlockLoopState):
        """Inverse of :meth:`merge_idm_video_branches` — delegates to ``idm_merge``."""
        from openwam.model.video_backbone.cosmos3 import idm_merge

        return idm_merge.split_branches(merged, noisy, cond)

    # ================================================================
    # Single-system: action/state tokens ride the gen stream
    # ================================================================

    def assert_ready_for_shared_tokens(self, state: BlockLoopState) -> None:
        """Cosmos3 has no AdaLN — timestep conditioning is additive per token and
        is applied to injected tokens in :meth:`inject_shared_tokens`, so the Wan
        4D-``time_mod`` precondition does not apply. No-op."""
        return None

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
        """Append ``[action][state]`` tokens to the flat gen sequence.

        The gen stream is already ``(B, S, D)``, so injection is a concat plus an
        additive timestep embedding and identity rotary rows for the new tokens.
        Pair with :meth:`extract_shared_tokens`.
        """
        from openwam.model.video_backbone.cosmos3 import shared_block

        n_action = int(n_action or 0)
        n_state = int(n_state or 0)
        n_tail = n_action + n_state
        if n_tail <= 0:
            return state
        if timestep is None:
            raise ValueError("inject_shared_tokens requires `timestep` for action/state token conditioning.")

        gen = state.hidden_states
        b, _, dim = gen.shape
        pieces = [gen]
        for n_tok, tokens, name in ((n_action, action_tokens, "action"), (n_state, state_tokens, "state")):
            if not n_tok:
                continue
            tok = shared_block.validate_shared_tokens(tokens, n_tok, name, b, dim).to(gen.dtype)
            emb = shared_block.shared_token_timestep_embedding(self.dit, timestep, n_tok, b, gen.dtype)
            pieces.append(tok + emb)

        state.hidden_states = torch.cat(pieces, dim=1)
        cos, sin = shared_block.extend_rotary_with_shared_tokens(
            state.extras["cos_gen"], state.extras["sin_gen"], n_tail
        )
        new_extras = dict(state.extras)
        new_extras["cos_gen"] = cos
        new_extras["sin_gen"] = sin
        new_extras["shared_mode"] = True
        state.extras = new_extras
        return state

    def extract_shared_tokens(
        self, state: BlockLoopState, n_action: int, *, n_state: int = 0
    ) -> Tuple[BlockLoopState, Tensor]:
        """Slice the action tail off the gen sequence and leave shared mode."""
        n_action = int(n_action or 0)
        n_state = int(n_state or 0)
        n_tail = n_action + n_state
        total = int(state.hidden_states.shape[1])
        s_video = int(state.grid_frames) * int(state.grid_height) * int(state.grid_width)
        if n_tail <= 0 or n_tail >= total:
            raise ValueError(
                f"extract_shared_tokens: n_action={n_action}, n_state={n_state} but sequence length is "
                f"{total} (was inject_shared_tokens called first with the same lengths?)."
            )
        if total - n_tail != s_video:
            raise ValueError(
                f"extract_shared_tokens: video token count {total - n_tail} != grid T·H·W={s_video} "
                "(grid changed between inject and extract?)."
            )
        action_tokens = state.hidden_states[:, s_video : s_video + n_action, :]
        state.hidden_states = state.hidden_states[:, :s_video, :]
        new_extras = dict(state.extras)
        new_extras["cos_gen"] = new_extras["cos_gen"][:, :s_video]
        new_extras["sin_gen"] = new_extras["sin_gen"][:, :s_video]
        for key in ("shared_mode", "shared_attention_mask"):
            new_extras.pop(key, None)
        state.extras = new_extras
        return state, action_tokens

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
        if isinstance(first_frame_image, list):
            if len(first_frame_image) != 1:
                raise ValueError(
                    "cosmos3_edge deploy accepts exactly one conditioning image; got a list of "
                    f"{len(first_frame_image)} — the B=1 prompt encoding (und K/V cache) cannot "
                    "broadcast over an image batch."
                )
            first_frame_image = first_frame_image[0]
        num_frames = int(kw.get("num_frames", 29))
        if (num_frames - 1) % self._temporal_compression != 0:
            raise ValueError(
                f"cosmos3_edge num_frames must be 4k+1 (causal Wan2.2 VAE grid); got {num_frames}. "
                f"Nearest valid: {((num_frames - 1) // self._temporal_compression) * self._temporal_compression + 1}."
            )
        height = int(kw.get("height", 480))
        width = int(kw.get("width", 832))
        seed = kw.get("seed")
        cfg_scale = float(kw.get("cfg_scale", 1.0))
        if cfg_scale > 1.0:
            raise NotImplementedError(
                "cosmos3_edge deploy CFG (cfg_scale > 1.0) is not wired yet: the shared CFG pass "
                "swaps only `context`, which the cosmos3 gen blocks never read (text conditioning "
                "rides the cached und K/V), so guidance would silently be a no-op at 2x cost. Run "
                "cfg_scale=1.0, or implement an und-bundle-swapping CFG pass first."
            )
        shift = kw.get("shift")
        prompt_embed_cache = kw.get("prompt_embed_cache")
        # Read after the argument gates: every check above is pure-argument, so
        # a caller passing a bad request gets the specific error rather than an
        # AttributeError from touching instance state first.
        fps = float(kw.get("fps", self._clip_fps))

        from openwam.model.video_backbone.cosmos_predict25._vae_utils import _pil_video_to_tensor

        frame_t = _pil_video_to_tensor([[first_frame_image]]).to(self.device)
        with torch.no_grad():
            first_frame_latents = self._encode_frames(frame_t)  # (1, 48, 1, h, w)

            # The templated text (and the und rope offsets) bake in geometry, so
            # the cache key must carry it — a bare-prompt key would silently
            # reuse stale conditioning across resolutions/durations.
            cache_key = (str(prompt), num_frames, height, width, fps)
            if prompt_embed_cache is not None and cache_key in prompt_embed_cache:
                enc = prompt_embed_cache[cache_key]
            else:
                enc = self._encode_prompts([str(prompt)], num_frames=num_frames, height=height, width=width, fps=fps)
                if prompt_embed_cache is not None:
                    prompt_embed_cache[cache_key] = enc

        t_lat = 1 + (num_frames - 1) // self._temporal_compression
        h_lat = first_frame_latents.shape[3]
        w_lat = first_frame_latents.shape[4]
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = torch.randn(
            (1, _COSMOS3_LATENT_CHANNELS, t_lat, h_lat, w_lat),
            generator=generator,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.dtype)
        latents[:, :, :1] = first_frame_latents.to(latents.dtype)

        return {
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

        # Restore the rotary table from the build-time fp32 copy. It is NOT
        # snapshotted here: this method runs twice on the training path
        # (openwam_trainer.py:126 and :530) with ``accelerator.prepare()`` in
        # between, and DeepSpeed's ``_configure_distributed_model`` calls
        # ``self.module.bfloat16()``, which casts floating-point buffers. A
        # snapshot taken at the second call is therefore already rounded, and
        # upcasting it back recovers the dtype but not the bits.
        #
        # Why it matters: bf16 costs ~0.37% relative precision on inv_freq,
        # which the vision stream's temporal offset (und_len +
        # unified_3d_mrope_temporal_modality_margin = 15000) multiplies into a
        # phase error large enough to scramble cos/sin (cosine ~0.77 against the
        # fp32 phases, 6.6 rad max) while leaving the text positions intact.
        # Cosmos3VLTextRotaryEmbedding runs the position matmul in fp32 with
        # autocast disabled, but that protects the product, not the table.
        rope = getattr(self.dit, "rotary_emb", None)
        if rope is not None and self._pristine_inv_freq is not None:
            rope.register_buffer("inv_freq", self._pristine_inv_freq.to(device=device), persistent=False)

        # The timestep-embedding MLP. Upstream lists it in
        # ``_keep_in_fp32_modules``, and this pin holds wherever OpenWAM owns
        # the dtype: deploy, eager training, and any non-ZeRO path. It does NOT
        # survive DeepSpeed ZeRO — ``_update_model_bit16_weights`` rebinds each
        # ``p.data`` back into the flattened bf16 group after a step — so under
        # ZeRO the MLP runs bf16 from step 1. That is a real train/deploy
        # difference, bounded to this 256→2048 MLP: the sinusoid itself is
        # computed by the parameter-free ``time_proj`` on an fp32 input in both
        # cases (see dit_forward.prepare_block_loop), so the frequency content
        # is identical and only the projection's arithmetic precision differs.
        # Left pinned rather than dropped because the paths where it does hold
        # are the ones that serve the checkpoint.
        te = getattr(self.dit, "time_embedder", None)
        if te is not None:
            te.to(dtype=torch.float32)
