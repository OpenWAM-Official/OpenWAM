"""Cosmos-Predict2.5 video backbone.

Implements the :class:`VideoBackbone` contract (``base.py``) directly on the
upstream Cosmos DiT, VAE, and Reason1 text encoder — **flat named children**,
mirroring the Wan backbone (no wrapper indirection). The heavy DiT block-loop
orchestration lives as stateless helpers in ``cosmos_predict25/dit_forward.py`` (the
analogue of ``wan/dit_forward.py``); this class delegates ``prepare`` /
``run_block`` / ``finalize`` / ``pre_attn_at_layer`` / ``post_attn_at_layer``
to them, reading ``self.dit``.

The components are built lazily inside :meth:`from_pretrained` (which imports
``cosmos_predict2`` only there), so importing this module is CPU-only-CI safe.

Public surface: **only** the methods/properties already declared on
:class:`VideoBackbone`. Everything else is an auxiliary helper (``_``-prefixed).

Plain-object reality (upstream-imposed): the VAE (``Wan2pt1VAEInterface``) and
Reason1 encoder (``Reason1LiveTextEncoder``) are plain Python objects, not
``nn.Module``. The callable facade is kept as a plain attribute
(``self._vae_iface`` / ``self.text_encoder``) for encode/decode + dtype/device
tracking, while the inner ``nn.Module`` is registered under the clean child
name (``self.vae`` / ``self.reason1``) so its weights enter the unified
state_dict (``vae.*`` / ``reason1.*``). Identity is preserved, so the facade's
``iface.model.model`` still resolves to the same tensors. The inner modules are
moved explicitly in :meth:`set_dtype_device` via ``cosmos_predict25/_vae_utils.py``.

Scope: ``dual_system`` + ``joint_cross_attn`` / ``joint_self_attn``, plus IDM
teacher-forcing (``cosmos_predict25/idm_merge.py``) and the single-system mode
that rides action/state tokens on the video DiT (:meth:`inject_shared_tokens`).
VACE is rejected. Freeze policy is owned by the training-strategy /
model freeze list, reached via native ``nn.Module.get_submodule`` dotted paths
(``dit`` / ``vae`` / ``reason1``). The ``freeze`` kwarg here is retained for
tests / direct programmatic use and defaults to ``False``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState, VideoBackbone
from openwam.model.video_backbone.cosmos_predict25 import dit_forward
from openwam.model.video_backbone.cosmos_predict25._vae_utils import (
    _move_cosmos_reason1,
    _move_cosmos_vae,
    _pil_video_to_tensor,
    _vae_device,
    _vae_inner_module,
    _video_tensor_to_pil,
)
from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter

logger = logging.getLogger(__name__)

# Cosmos-Predict2.5 native geometry, invariant across the 2B/14B size family.
# Mirrors the MiniTrainDIT config (``patch_temporal=1`` / ``patch_spatial=2``)
# and the Wan2pt1 VAE temporal contract (causal first frame + 4-frame tail).
_COSMOS25_DIT_PATCH_SIZE: Tuple[int, int, int] = (1, 2, 2)
_COSMOS25_TEMPORAL_COMPRESSION: int = 4
_COSMOS25_CAUSAL_TEMPORAL: bool = True


class CosmosPredict25VideoBackbone(VideoBackbone):
    """Wrap a Cosmos-Predict2.5 DiT/VAE/text-encoder behind the VideoBackbone ABC."""

    def __init__(
        self,
        *,
        net: nn.Module,
        vae: Optional[Any] = None,
        text_encoder: Optional[Any] = None,
        dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        scheduler: Optional[CosmosFlowSchedulerAdapter] = None,
        shift_video: float = 5.0,
        text_dropout_p: float = 0.0,
        text_dropout_seed: Optional[int] = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        # --- Flat named children (Wan-shape) ---
        self.dit = net
        # VAE / Reason1 are plain objects: keep the facade as a plain attr (for
        # encode/decode + dtype/device tracking) but register the inner nn.Module
        # so its weights ride the unified state_dict. Identity is preserved, so
        # the facade's `iface.model.model` still resolves to the same tensors.
        self._vae_iface = vae
        inner_vae = _vae_inner_module(vae)
        if inner_vae is not None:
            self.vae = inner_vae
        self.text_encoder = text_encoder
        if text_encoder is not None:
            te_inner = getattr(text_encoder, "model", None)
            if isinstance(te_inner, nn.Module):
                self.reason1 = te_inner

        # --- Geometry + scheduler ---
        self._dim = int(dim)
        self._num_layers = int(num_layers)
        self._num_heads = int(num_heads)
        self._head_dim = int(head_dim)
        # CosmosPredict25's per-token text/context embedding dim (1024 for 2B, vs Wan's
        # 4096). Exposed through the base ``text_dim`` property.
        self._context_dim = int(context_dim)
        self._scheduler = scheduler if scheduler is not None else CosmosFlowSchedulerAdapter()
        self._shift_video = float(shift_video)

        # --- §14.7 CFG dropout (live-encoder path) ---
        if not 0.0 <= float(text_dropout_p) <= 1.0:
            raise ValueError(f"text_dropout_p must be in [0, 1]; got {text_dropout_p!r}.")
        self.text_dropout_p = float(text_dropout_p)
        self._text_dropout_rng = random.Random(text_dropout_seed)

        # Native patch size + temporal contract feed the base properties.
        self._dit_patch_size = _COSMOS25_DIT_PATCH_SIZE
        self._temporal_compression = _COSMOS25_TEMPORAL_COMPRESSION
        self._causal_temporal = _COSMOS25_CAUSAL_TEMPORAL

        self._freeze = bool(freeze)
        if self._freeze:
            for p in self.parameters():  # dit + vae + reason1
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, source: Any, *, device=None, ckpt_dir=None, **kw) -> "CosmosPredict25VideoBackbone":
        """Build a backbone from a config / model dir.

        Defers to :func:`cosmos_predict25.pipeline_builder.build_cosmos_predict25_pipeline`, which
        lazily imports ``cosmos_predict2`` and returns a lightweight holder
        (net + vae + text_encoder + geometry + shift_video). Drains it into flat
        children (Wan holder-drain parity).
        """
        from openwam.model.video_backbone.cosmos_predict25.pipeline_builder import build_cosmos_predict25_pipeline

        cfg_for_loader = _video_backbone_cfg(source)
        shift_video = float(_cfg_get(cfg_for_loader, "shift_video", 5.0))

        holder = build_cosmos_predict25_pipeline(source, device=device, ckpt_dir=ckpt_dir, **kw)
        dim, num_layers, num_heads, head_dim, context_dim = _probe_pipeline_geometry(holder)
        return cls(
            net=holder.net,
            vae=getattr(holder, "vae", None),
            text_encoder=getattr(holder, "text_encoder", None),
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            scheduler=CosmosFlowSchedulerAdapter(shift_video=shift_video),
            shift_video=float(getattr(holder, "shift_video", shift_video)),
            text_dropout_p=float(getattr(holder, "text_dropout_p", 0.0)),
            text_dropout_seed=getattr(holder, "text_dropout_seed", None),
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
        """Per-token text/context embedding dim (1024 for CosmosPredict25-2B)."""
        return self._context_dim

    # ------------------------------------------------------------------
    # Joint self-attention mask plumbing (consumed by the joint MoT driver)
    # ------------------------------------------------------------------

    @property
    def video_attention_mask_mode(self) -> str:
        """Video↔video self-attention mask mode for the joint MoT mask."""
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
        ``video_tokens_per_frame`` (= ``grid_height * grid_width`` for CosmosPredict25).
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
    # Three-step block loop — delegates to cosmos_predict25/dit_forward.py
    # ------------------------------------------------------------------

    def prepare(self, **pipeline_inputs) -> BlockLoopState:
        """Patchify / position-embed / pack a :class:`BlockLoopState`."""
        return dit_forward.prepare_block_loop(self.dit, **pipeline_inputs)

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        if state.extras.get("shared_mode"):
            return self._run_block_shared(block_id, state)
        return dit_forward.run_block(self.dit, block_id, state)

    def finalize(self, state: BlockLoopState) -> Tensor:
        return dit_forward.finalize_block_loop(self.dit, state)

    # ------------------------------------------------------------------
    # Joint self-attention hooks
    # ------------------------------------------------------------------

    def pre_attn_at_layer(self, layer_id: int, state: BlockLoopState):
        """Pre half of one block — delegates to ``dit_forward.pre_attn_at_layer``."""
        if not hasattr(getattr(self, "dit", None), "blocks"):
            raise NotImplementedError(
                "CosmosPredict25VideoBackbone.pre_attn_at_layer requires self.dit to expose `.blocks` "
                "(the upstream Cosmos DiT block-split interface)."
            )
        return dit_forward.pre_attn_at_layer(self.dit, layer_id, state)

    def post_attn_at_layer(
        self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: dict
    ) -> BlockLoopState:
        """Post half of one block — delegates to ``dit_forward.post_attn_at_layer``."""
        if not hasattr(getattr(self, "dit", None), "blocks"):
            raise NotImplementedError(
                "CosmosPredict25VideoBackbone.post_attn_at_layer requires self.dit to expose `.blocks` "
                "(the upstream Cosmos DiT block-split interface)."
            )
        return dit_forward.post_attn_at_layer(layer_id, state, attn_out, post_state)

    # ------------------------------------------------------------------
    # IDM teacher-forcing branch merge/split
    # ------------------------------------------------------------------

    def merge_idm_video_branches(self, noisy: BlockLoopState, cond: BlockLoopState):
        """Concatenate the IDM noisy + cond branches — delegates to ``idm_merge``."""
        from openwam.model.video_backbone.cosmos_predict25 import idm_merge

        return idm_merge.merge_branches(noisy, cond)

    def split_idm_video_branches(self, merged: BlockLoopState, noisy: BlockLoopState, cond: BlockLoopState):
        """Inverse of :meth:`merge_idm_video_branches` — delegates to ``idm_merge``."""
        from openwam.model.video_backbone.cosmos_predict25 import idm_merge

        return idm_merge.split_branches(merged, noisy, cond)

    # ------------------------------------------------------------------
    # Single-system: action/state tokens ride the video DiT (3D mode)
    # ------------------------------------------------------------------

    def assert_ready_for_shared_tokens(self, state: BlockLoopState) -> None:
        """Cosmos builds per-token modulation in :meth:`inject_shared_tokens` from
        ``extras`` (not ``state.time_mod``), so the Wan 4D-``time_mod`` check does
        not apply — no-op."""
        return None

    def _shared_token_emb(self, timestep: Tensor, n_tokens: int, batch_size: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Per-token AdaLN emb + lora for ``n_tokens`` non-grid tokens at ``timestep``.

        Accepts a scalar, a per-sample ``(B,)`` timestep, or a per-token
        ``(B, n_tokens)`` tensor (parity with Wan's ``build_action_t_mod``). ``lora``
        is ``None`` when the DiT was built with ``use_adaln_lora=False``.
        """
        if timestep.dim() == 2:
            if tuple(timestep.shape) != (batch_size, n_tokens):
                raise ValueError(
                    f"shared-token timestep has shape {tuple(timestep.shape)}; expected "
                    f"(B={batch_size}, n_tokens={n_tokens}) for the per-token form."
                )
            ts_tok = timestep
        else:
            ts = timestep.flatten()
            if ts.numel() == 1:
                ts = ts.expand(batch_size)
            elif ts.numel() != batch_size:
                raise ValueError(
                    f"shared-token timestep has {ts.numel()} elements; expected 1, "
                    f"batch_size={batch_size}, or a (B={batch_size}, n_tokens={n_tokens}) per-token tensor."
                )
            ts_tok = ts.view(batch_size, 1).expand(batch_size, n_tokens)
        # Match the video path (dit_forward.prepare_block_loop): MinimalV1LVGDiT
        # scales timesteps by ``timestep_scale`` before ``t_embedder`` so shared
        # action/state tokens land in the same time domain as the video grid.
        ts_tok = ts_tok * float(getattr(self.dit, "timestep_scale", 1.0))
        emb, lora = self.dit.t_embedder(ts_tok)
        emb = self.dit.t_embedding_norm(emb)
        return emb, lora

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
        """Flatten the 5D video grid to 3D and append ``[action][state]`` tokens.

        Builds per-token AdaLN emb/lora (video per-frame emb expanded over H·W,
        plus action/state emb from ``timestep``), extends the rope with identity
        rows, and flips the state into ``shared_mode`` so :meth:`run_block`
        dispatches to the 3D block forward. Pair with :meth:`extract_shared_tokens`.
        """
        from einops import rearrange

        from openwam.model.video_backbone.cosmos_predict25 import shared_block

        n_action = int(n_action or 0)
        n_state = int(n_state or 0)
        if state.extras.get("extra_per_block_pos_emb") is not None:
            raise NotImplementedError(
                "CosmosPredict25 single-system supports only the rope position variant "
                "(extra_per_block_pos_emb must be None)."
            )
        B = state.hidden_states.shape[0]
        T = int(state.grid_frames)
        video_3d = rearrange(state.hidden_states, "b t h w d -> b (t h w) d")
        dim = video_3d.shape[2]
        if (n_action + n_state) > 0 and timestep is None:
            raise ValueError("inject_shared_tokens requires `timestep` for action/state token modulation.")

        # Keep the AdaLN emb COMPACT — video per-frame (B, T, D) rather than
        # expanded over H·W. run_block_3d expands the modulation, not the emb, so
        # the AdaLN projections run at H·W× fewer FLOPs (see shared_block).
        def _to_per_frame(t: Tensor) -> Tensor:
            return t.expand(B, T, t.shape[2]) if t.shape[1] == 1 else t

        pieces_x = [video_3d]
        pieces_emb = [_to_per_frame(state.extras["t_embedding_B_T_D"])]
        # A DiT built with use_adaln_lora=False has no LoRA term; carry None
        # through so run_block_3d takes its no-LoRA branch instead of crashing on
        # a None tensor.
        video_lora = state.extras.get("adaln_lora_B_T_3D")
        use_lora = video_lora is not None
        pieces_lora = [_to_per_frame(video_lora)] if use_lora else None

        for n_tok, tokens, name in ((n_action, action_tokens, "action"), (n_state, state_tokens, "state")):
            if not n_tok:
                continue
            if tokens is None:
                raise ValueError(f"n_{name} > 0 requires {name}_tokens.")
            if tokens.shape[0] != B or tokens.shape[1] != n_tok or tokens.shape[2] != dim:
                raise ValueError(
                    f"{name}_tokens shape {tuple(tokens.shape)} does not match (B={B}, n_{name}={n_tok}, dim={dim})."
                )
            emb, lora = self._shared_token_emb(timestep, n_tok, B)
            pieces_x.append(tokens.to(video_3d.dtype))
            pieces_emb.append(emb)
            if use_lora:
                if lora is None:
                    raise ValueError("video branch uses AdaLN-LoRA but t_embedder returned no LoRA for shared tokens.")
                pieces_lora.append(lora)

        state.hidden_states = torch.cat(pieces_x, dim=1)
        new_extras = dict(state.extras)
        new_extras["shared_emb_B_C_D"] = torch.cat(pieces_emb, dim=1)
        new_extras["shared_adaln_lora_B_C_3D"] = torch.cat(pieces_lora, dim=1) if use_lora else None
        new_extras["shared_rope"] = shared_block.extend_rope_with_shared_tokens(
            state.extras["rope_emb_L_1_1_D"], n_action + n_state
        )
        new_extras["shared_mode"] = True
        state.extras = new_extras
        return state

    def _run_block_shared(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        from openwam.model.video_backbone.cosmos_predict25 import shared_block
        from openwam.model.video_backbone.wan.shared.core.gradient.gradient_checkpoint import (
            gradient_checkpoint_forward,
        )

        block = self.dit.blocks[block_id]
        state.hidden_states = gradient_checkpoint_forward(
            shared_block.run_block_3d,
            state.use_gradient_checkpointing,
            state.use_gradient_checkpointing_offload,
            block,
            state.hidden_states,
            state.extras["shared_emb_B_C_D"],
            state.extras["shared_adaln_lora_B_C_3D"],
            state.extras["shared_rope"],
            state.context,
            state.extras.get("shared_attention_mask"),
            grid_frames=int(state.grid_frames),
            tokens_per_frame=int(state.grid_height) * int(state.grid_width),
        )
        return state

    def extract_shared_tokens(
        self, state: BlockLoopState, n_action: int, *, n_state: int = 0
    ) -> Tuple[BlockLoopState, Tensor]:
        """Slice the action tail off, restore the 5D video grid, leave shared mode."""
        from einops import rearrange

        n_action = int(n_action or 0)
        n_state = int(n_state or 0)
        T, H, W = int(state.grid_frames), int(state.grid_height), int(state.grid_width)
        s_video = T * H * W
        n_tail = n_action + n_state
        total = state.hidden_states.shape[1]
        if n_tail <= 0 or n_tail >= total:
            raise ValueError(
                f"extract_shared_tokens: n_action={n_action}, n_state={n_state} but sequence length is {total} "
                "(was inject_shared_tokens called first with the same lengths?)."
            )
        if total - n_tail != s_video:
            raise ValueError(
                f"extract_shared_tokens: video token count {total - n_tail} != grid T·H·W={s_video} "
                "(grid changed between inject and extract?)."
            )
        action_tokens = state.hidden_states[:, s_video : s_video + n_action, :]
        video_3d = state.hidden_states[:, :s_video, :]
        state.hidden_states = rearrange(video_3d, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        new_extras = dict(state.extras)
        for key in (
            "shared_mode",
            "shared_emb_B_C_D",
            "shared_adaln_lora_B_C_3D",
            "shared_rope",
            "shared_attention_mask",
        ):
            new_extras.pop(key, None)
        state.extras = new_extras
        return state, action_tokens

    # ------------------------------------------------------------------
    # Preprocessing & decoding
    # ------------------------------------------------------------------

    def preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
        return self._preprocess_input(frames=frames, text=text, **kw)

    def _preprocess_input(
        self,
        *,
        frames: Any = None,
        text: Any = None,
        input_latents: Optional[Tensor] = None,
        vace_videos: Any = None,
        ref_images: Any = None,
        **kw: Any,
    ) -> dict:
        """VAE-encode frames (or accept latents) + encode text → training/inference dict.

        Reads ``self.dit`` (for ``crossattn_proj``), ``self._vae_iface``,
        ``self.text_encoder``, and the live-path CFG-dropout state
        (``self.training`` / ``self.text_dropout_p`` / ``self._text_dropout_rng``).
        Relocated verbatim from the former pipeline wrapper.
        """
        if vace_videos is not None and any(v is not None for v in vace_videos):
            raise NotImplementedError(
                "CosmosPredict25 MVP does not support VACE conditioning. Drop `vace_video` from the dataset "
                "when training with the CosmosPredict25 backbone."
            )

        if input_latents is None:
            if frames is None:
                raise ValueError(
                    "CosmosPredict25VideoBackbone._preprocess_input requires either `input_latents` or `frames`."
                )
            if self._vae_iface is None:
                raise RuntimeError(
                    "CosmosPredict25 VAE is not configured. Set `video_backbone.vae: wan2pt1` "
                    "(default; loads `<model_path>/tokenizer.pth`)."
                )
            input_latents = self._encode_frames(frames)

        if text is None:
            raise ValueError("CosmosPredict25VideoBackbone._preprocess_input requires `text`.")
        if self.text_encoder is None:
            raise ValueError("`text=` requires a configured text encoder (Reason1LiveTextEncoder).")
        # §14.7 — CFG dropout: substitute selected prompts with `""` so the
        # encoder produces the canonical empty embedding.
        if self.training and self.text_dropout_p > 0.0:
            text_list = [text] if isinstance(text, str) else list(text)
            text = [t if self._text_dropout_rng.random() >= self.text_dropout_p else "" for t in text_list]
        # Inference-only memoization: the deploy engine passes its bounded
        # server-lifetime `prompt_embed_cache`; training never does (CFG
        # dropout must re-encode per step).
        cache = kw.get("prompt_embed_cache")
        if cache is not None and isinstance(text, str) and text in cache:
            context = cache[text].to(device=input_latents.device, dtype=input_latents.dtype)
        else:
            context = self._encode_text_context(text, device=input_latents.device, dtype=input_latents.dtype)
            if cache is not None and isinstance(text, str):
                cache[text] = context

        B = input_latents.shape[0]
        seq_lens = torch.full((B,), context.shape[1], dtype=torch.long, device=context.device)
        out: dict = {
            "input_latents": input_latents,
            "context": context,
            "context_mask": kw.get("context_mask"),
            "seq_lens": seq_lens,
            "num_frames": input_latents.shape[2],
            "height": input_latents.shape[3],
            "width": input_latents.shape[4],
        }

        # TI2V first-frame conditioning (activated by `ref_images`): VAE-encode
        # one reference frame per sample → `first_frame_latents` + LVG mask.
        ref_active = (
            ref_images is not None
            and isinstance(ref_images, (list, tuple))
            and len(ref_images) > 0
            and all(r is not None for r in ref_images)
        )
        if ref_active:
            if self._vae_iface is None:
                raise RuntimeError(
                    "CosmosPredict25VideoBackbone._preprocess_input received `ref_images` but no VAE is "
                    "configured. Set `video_backbone.vae: wan2pt1` to enable TI2V."
                )
            ref_clips = [r if isinstance(r, (list, tuple)) else [r] for r in ref_images]
            first_frame_latents = self._encode_frames(ref_clips).to(
                device=input_latents.device, dtype=input_latents.dtype
            )
            T_lat = input_latents.shape[2]
            H_lat = input_latents.shape[3]
            W_lat = input_latents.shape[4]
            condition_mask = torch.zeros(
                (B, 1, T_lat, H_lat, W_lat), dtype=input_latents.dtype, device=input_latents.device
            )
            condition_mask[:, :, 0] = 1.0
            out["first_frame_latents"] = first_frame_latents
            out["condition_mask"] = condition_mask
            out["num_clean_prefix_frames"] = 1

        return out

    def _encode_text_context(self, text: Any, *, device, dtype) -> Tensor:
        """Encode text into the post-projection context shared by cond and uncond CFG paths.

        Live encoder returns pre-projection ``(B, 512, 100352) bf16``; the
        DiT-owned ``crossattn_proj`` is applied HERE (not in prepare) so the
        architecture's proprio-token concat sees 1024-d context.
        """
        context = self.text_encoder(text)
        context = context.to(device=device, dtype=dtype)
        net = self.dit
        if getattr(net, "use_crossattn_projection", False) and context.shape[-1] == int(
            getattr(net, "crossattn_proj_in_channels", -1)
        ):
            context = net.crossattn_proj(context)
        return context

    def _encode_frames(self, frames: Any) -> Tensor:
        """PIL frames → bf16 ``(B, 16, T_lat, H/8, W/8)`` Wan2pt1 latents."""
        if self._vae_iface is None:
            raise RuntimeError("CosmosPredict25VideoBackbone._encode_frames called without a configured VAE.")
        video = _pil_video_to_tensor(frames)
        video = video.to(device=_vae_device(self._vae_iface), dtype=torch.bfloat16)
        return self._vae_iface.encode(video)

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        if self._vae_iface is None:
            raise NotImplementedError(
                "CosmosPredict25VideoBackbone.decode_video requires a configured VAE. Set `video_backbone.vae: wan2pt1`."
            )
        _ = tiled  # Wan2pt1VAEInterface.decode has no `tiled`; internal temporal_window=4.
        video = self._vae_iface.decode(latents.to(device=_vae_device(self._vae_iface)))
        return _video_tensor_to_pil(video)

    def preprocess_input_for_inference(self, **kw) -> dict:
        """Build the inference denoising-loop input dict from explicit kwargs.

        Materialises ``uncond_context`` for CFG (``cfg_scale > 1.0``); TI2V
        first-frame via condition_mask. ``base.py:generate`` forwards
        ``cfg_scale`` / ``cfg_merge``.
        """
        prompt = kw.get("prompt")
        vace_video = kw.get("vace_video")
        first_frame_image = kw.get("first_frame_image")
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
            raise NotImplementedError(
                "CosmosPredict25VideoBackbone does not support VACE conditioning at inference yet."
            )
        if getattr(self, "text_encoder", None) is None:
            raise ValueError(
                "CosmosPredict25VideoBackbone.preprocess_input_for_inference requires a configured "
                "text encoder (Reason1LiveTextEncoder)."
            )

        cfg_scale_f = float(cfg_scale)
        if cfg_scale_f < 1.0:
            raise ValueError(f"cfg_scale must be >= 1.0; got {cfg_scale!r}.")

        dit = getattr(self, "dit", None)
        param = next(dit.parameters(), None) if dit is not None else None
        dtype = param.dtype if param is not None else getattr(self, "_dtype", torch.float32)
        device = param.device if param is not None else getattr(self, "_device", torch.device("cpu"))

        # Inference-time `input_latents` placeholder (overwritten by init_noise),
        # at Wan2pt1 geometry (stride 4×8×8, z_dim=16).
        T_lat = 1 + (int(num_frames) - 1) // 4
        H_lat = int(height) // 8
        W_lat = int(width) // 8
        placeholder_latents = torch.randn((1, 16, T_lat, H_lat, W_lat), dtype=dtype, device=device)

        prompt_embed_cache = kw.get("prompt_embed_cache")
        preproc = self._preprocess_input(
            frames=None, text=prompt, input_latents=placeholder_latents, prompt_embed_cache=prompt_embed_cache
        )

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
        inputs_shared["sigma_shift"] = float(shift) if shift is not None else float(self._shift_video)
        inputs_shared["num_inference_steps"] = int(num_inference_steps)
        inputs_shared["cfg_scale"] = cfg_scale_f
        inputs_shared["cfg_merge"] = bool(cfg_merge)

        if cfg_scale_f > 1.0:
            inputs_shared["uncond_context"] = self._build_uncond_context(
                context_template=inputs_shared["context"], prompt_embed_cache=prompt_embed_cache
            )
        else:
            inputs_shared["uncond_context"] = None

        self._finalize_ti2v_inputs(inputs_shared, first_frame_image)
        return inputs_shared

    def _finalize_ti2v_inputs(self, inputs_shared: dict, first_frame_image) -> None:
        """Encode the inference-time first frame and write the TI2V keys."""
        if first_frame_image is None:
            return
        if self._vae_iface is None:
            raise RuntimeError(
                "CosmosPredict25VideoBackbone.preprocess_input_for_inference received `first_frame_image` "
                "but no VAE is configured. Ensure `video_backbone.vae: wan2pt1`."
            )
        ref_frames = first_frame_image if isinstance(first_frame_image, list) else [first_frame_image]
        ref_clips = [[r] for r in ref_frames]
        latents = inputs_shared["latents"]
        first_frame_latents = self._encode_frames(ref_clips).to(device=latents.device, dtype=latents.dtype)
        T_lat = latents.shape[2]
        H_lat = latents.shape[3]
        W_lat = latents.shape[4]
        B = latents.shape[0]
        condition_mask = torch.zeros((B, 1, T_lat, H_lat, W_lat), dtype=latents.dtype, device=latents.device)
        condition_mask[:, :, 0] = 1.0
        latents[:, :, 0:1] = first_frame_latents
        inputs_shared["latents"] = latents
        inputs_shared["first_frame_latents"] = first_frame_latents
        inputs_shared["condition_mask"] = condition_mask
        inputs_shared["num_clean_prefix_frames"] = 1

    def _build_uncond_context(self, *, context_template: Tensor, prompt_embed_cache: Optional[dict] = None) -> Tensor:
        """Materialise the unconditional text context for CFG via ``text_encoder("")``."""
        if prompt_embed_cache is not None and "" in prompt_embed_cache:
            ctx = prompt_embed_cache[""].to(device=context_template.device, dtype=context_template.dtype)
        else:
            ctx = self._encode_text_context("", device=context_template.device, dtype=context_template.dtype)
            if prompt_embed_cache is not None:
                prompt_embed_cache[""] = ctx
        if ctx.shape[0] == 1 and context_template.shape[0] > 1:
            ctx = ctx.expand(context_template.shape[0], -1, -1).contiguous()
        if ctx.shape != context_template.shape:
            raise ValueError(
                f"live uncond context shape {tuple(ctx.shape)} doesn't match cond "
                f"context shape {tuple(context_template.shape)}; check encoder output."
            )
        return ctx

    # ------------------------------------------------------------------
    # Device / dtype
    # ------------------------------------------------------------------

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        self._dtype = dtype
        self._device = device
        # nn.Module.to(...) walks the flat children (dit / vae / reason1).
        self.to(dtype=dtype, device=device)
        # The plain-object VAE / Reason1 facades are not nn.Modules, so move them
        # explicitly (incl. the wan2pt1 mean/std + scale-list stale-device fix).
        if self._vae_iface is not None:
            _move_cosmos_vae(self._vae_iface, dtype=dtype, device=device)
        if self.text_encoder is not None and not isinstance(self.text_encoder, nn.Module):
            _move_cosmos_reason1(self.text_encoder, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Make this backbone's slice of the checkpoint self-contained.

        (1) Merge component reconstruction specs into
        ``cfg.model.video_backbone.components`` (only when absent). (2) Copy the
        Reason1 structural JSONs into ``<output_dir>/reason1/`` — ONLY when a live
        Reason1 encoder is part of this checkpoint (``self.text_encoder`` set, so
        its weights ride the safetensors via ``reason1``). The VAE
        component is emitted only when the ``vae`` child is registered (a VAE was
        configured); under ``vae: none`` no ``vae`` child exists, so emitting the
        spec would leave the saved config internally inconsistent. No-op when
        ``model_path`` is unreadable.
        """
        from omegaconf import DictConfig, OmegaConf, open_dict

        from openwam.model.video_backbone.cosmos_predict25.component_specs import (
            copy_cosmos_predict25_artifacts,
            generate_cosmos_predict25_component_specs,
        )

        is_plain = not isinstance(cfg, DictConfig)
        oc = OmegaConf.create(cfg) if is_plain else cfg

        model_path = OmegaConf.select(oc, "model.video_backbone.model_path", default=None)
        specs = generate_cosmos_predict25_component_specs(str(model_path) if model_path is not None else "")
        if specs is None:
            logger.info(
                "[cosmos_predict25] video_backbone.model_path not readable (%s); skipping deploy-asset save.",
                model_path,
            )
            return

        has_reason1 = getattr(self, "text_encoder", None) is not None
        has_vae = getattr(self, "vae", None) is not None

        def _keep_component(c) -> bool:
            attr = c.get("attr")
            if attr == "text_encoder":
                return has_reason1
            if attr == "vae":
                return has_vae
            return True

        components = [c for c in specs["components"] if _keep_component(c)]

        if "components" not in oc.model.video_backbone:
            with open_dict(oc):
                OmegaConf.update(oc, "model.video_backbone.components", components)
            if is_plain:
                cfg["model"]["video_backbone"]["components"] = components

        if has_reason1:
            copy_cosmos_predict25_artifacts(output_dir, oc)


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

    Kept in lockstep with ``pipeline_builder._video_backbone_cfg``.
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


def _probe_pipeline_geometry(holder: Any) -> Tuple[int, int, int, int, int]:
    """Read (dim, num_layers, num_heads, head_dim, context_dim) off the builder holder."""
    try:
        return (
            int(holder.dim),
            int(holder.num_layers),
            int(holder.num_heads),
            int(holder.head_dim),
            int(holder.context_dim),
        )
    except AttributeError as exc:
        raise AttributeError(
            "Cosmos holder is missing one of {dim, num_layers, num_heads, head_dim, context_dim}. "
            "Attach these in `pipeline_builder.build_cosmos_predict25_pipeline`."
        ) from exc


__all__ = ["CosmosPredict25VideoBackbone"]
