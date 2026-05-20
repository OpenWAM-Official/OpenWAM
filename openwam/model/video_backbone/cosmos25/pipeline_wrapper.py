"""Concrete pipeline that bridges the upstream cosmos_predict2 DiT to the
openwam ``VideoBackbone`` 3-step block-loop contract.

The wrapper owns the upstream ``net`` (``MinimalV1LVGDiT`` / ``MiniTrainDIT``),
plus optional ``vae`` and ``text_encoder`` slots, and re-exposes
``prepare_block_loop`` / ``run_block`` / ``finalize_block_loop`` /
``preprocess_input`` / ``decode_video`` — the duck-typed interface that
``Cosmos25VideoBackbone`` probes via ``getattr`` (see ``adapter.py:149,159,
168,187,196``).

Why a wrapper instead of building the adapter directly on ``MiniTrainDIT``?
- ``MiniTrainDIT.forward`` is monolithic; we need per-block stepping so the
  action backbone can tap intermediate features.
- ``MinimalV1LVGDiT`` prepends a condition-mask channel before
  ``prepare_embedded_sequence``; we keep that pre-processing here instead of
  leaking it into ``Cosmos25VideoBackbone``.

BlockLoopState contract (Cosmos-specific):
- ``state.x``        : ``(B, T, H, W, D)`` patchified hidden state. Load-bearing.
- ``state.context``  : ``(B, L, context_dim)`` cross-attn key/value source. Load-bearing.
- ``state.extras``   : everything Cosmos-specific lives here:
      ``t_embedding_B_T_D``       : ``(B, T, D)`` time embedding (post ``t_embedding_norm``).
      ``adaln_lora_B_T_3D``       : ``(B, T, 3·D)`` AdaLN-LoRA residual (``None`` if disabled).
      ``rope_emb_L_1_1_D``        : 3D-factored RoPE freqs from ``net.pos_embedder``.
      ``extra_per_block_pos_emb`` : optional learnable per-block pos emb (typically ``None``).
- ``state.t_mod`` / ``state.freqs`` : Wan-style dummies. Architectures using
  ``joint_cross_attn`` (the MVP) do **not** read them. Other architectures
  must NOT assume they carry Cosmos data.

VAE contract (Phase 4):
- ``self.vae`` (when non-None) is a ``Wan2pt1VAEInterface`` instance. It is a
  plain Python object, **not** an ``nn.Module``, so ``__setattr__`` does not
  register it in ``self._modules`` and ``self.state_dict()`` excludes its
  parameters — the ``tokenizer.pth`` lives outside the OpenWAM checkpoint.
- Stride is ``temporal=4`` / ``spatial=8`` / ``z_dim=16`` (matches DiT
  ``in_channels=16``). Latent num-frames is ``1 + (T_pixels - 1) // 4``
  (``wan2pt1.py:1028``).
- Both ``encode`` and ``decode`` are ``@torch.no_grad`` upstream
  (``wan2pt1.py:799,837``), so the VAE never builds an autograd graph even
  if ``freeze=False`` somewhere downstream.

The CPU smoke tests in ``tests/test_cosmos25_pipeline_wrapper.py`` exercise
this contract through a thin fake net that mimics the upstream interface
without actually loading any cosmos_predict2 module.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.adapter import BlockLoopState

logger = logging.getLogger(__name__)


class Cosmos25PipelineWrapper(nn.Module):
    """Thin nn.Module facade around an upstream Cosmos DiT."""

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
        flow_shift: float = 5.0,
        text_dropout_p: float = 0.0,
        text_dropout_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.net = net
        # ``vae`` is an upstream ``Wan2pt1VAEInterface`` — a plain Python class,
        # not an ``nn.Module`` (see module docstring). Assigning it to ``self.vae``
        # therefore does NOT register the inner WanVAE_ params in ``_modules``,
        # so its ~485 MB of weights would be silently absent from ``state_dict``.
        # Wan's VAE goes through ``WanVideoPipeline(BasePipeline=nn.Module)`` and
        # ``pipeline.vae = WanVideoVAE(nn.Module)``, so its params join the
        # unified safetensors automatically. To match that convention here, walk
        # to the inner ``vae.model.model`` (the real ``WanVAE_`` nn.Module) and
        # register it via a separate attribute. ``nn.Module.__setattr__`` adds
        # it to ``self._modules['_vae_inner']``; Python identity is preserved,
        # so the upstream call sites that read ``iface.model.model`` keep
        # working bit-for-bit. ``load_state_dict`` does in-place buffer/param
        # copies so reloading via either path yields the same loaded values.
        # The 6 mean/std tensors (``mean``/``std``/``img_*``/``video_*``) are
        # NOT registered: ``mean``/``std`` come from a hardcoded list in
        # ``WanVAE.__init__`` (constants, no need to roundtrip), and
        # ``load_mean_std=False`` makes ``img_mean``/``video_mean`` placeholder
        # zeros/ones (see ``_video_vae`` at wan2pt1.py:692-697).
        self.vae = vae
        if vae is not None:
            outer = getattr(vae, "model", None)
            inner = getattr(outer, "model", None) if outer is not None else None
            if isinstance(inner, nn.Module):
                self._vae_inner = inner
        self.text_encoder = text_encoder
        # ``text_encoder`` (``Reason1LiveTextEncoder``) is a plain Python class,
        # not an ``nn.Module`` — see ``text_encoder.py:13-28`` for why. Mirroring
        # the ``_vae_inner`` trick above, register its inner ``nn.Module``
        # (Qwen2.5-VL ~16 GB) here so it rides into the unified state_dict /
        # safetensors. The wrapper class stays a plain attribute at
        # ``self.text_encoder`` for dtype/device tracking + tokenizer access.
        # Single registration point per tensor — does not trigger the shared-
        # memory trap noted in ``adapter.py:62-70``.
        if text_encoder is not None:
            te_inner = getattr(text_encoder, "model", None)
            if isinstance(te_inner, nn.Module):
                self._reason1_inner = te_inner
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.context_dim = int(context_dim)
        self.flow_shift = float(flow_shift)
        # §14.7 — CFG dropout for the live encoder path. With probability
        # ``text_dropout_p`` (training mode only) the wrapper substitutes ``""``
        # for each prompt before calling ``self.text_encoder(text)``, so the
        # encoder produces the canonical empty-prompt embedding. Mirrors the
        # cache-path dropout in ``TextEmbeddingCacheTransform``; cache > live
        # precedence makes the two mutually exclusive per-sample. ``self.training``
        # is propagated via ``nn.Module`` (training default ``True``;
        # ``deploy/model_loader.py:161`` flips eval at load; ``base.py::generate``
        # also flips eval defensively).
        if not 0.0 <= float(text_dropout_p) <= 1.0:
            raise ValueError(f"text_dropout_p must be in [0, 1]; got {text_dropout_p!r}.")
        self.text_dropout_p = float(text_dropout_p)
        self._text_dropout_rng = random.Random(text_dropout_seed)

    # ------------------------------------------------------------------
    # 3-step block loop
    # ------------------------------------------------------------------

    def prepare_block_loop(
        self,
        *,
        latents: Optional[Tensor] = None,
        input_latents: Optional[Tensor] = None,
        context: Tensor,
        timestep: Tensor,
        context_mask: Optional[Tensor] = None,
        condition_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **_ignored: Any,
    ) -> BlockLoopState:
        """Replicate ``MiniTrainDIT.forward`` up to (but excluding) the block loop.

        Args:
            latents: ``(B, C_z, T, H, W)`` current DiT input. During training
                this is the noised version of ``input_latents`` produced by
                ``compute_loss``; during inference it is the current denoise
                state. Preferred entry point — matches Wan's convention
                (see ``wan_adapter.py:239``).
            input_latents: ``(B, C_z, T, H, W)`` fallback when ``latents`` is
                not provided (the legacy shape-smoke path in
                ``tests/test_cosmos25_real_load.py``). ``C_z = 16`` for 2B.
            context: ``(B, L, *)`` text/context embedding. If its last dim
                matches ``net.crossattn_proj_in_channels`` and the net was
                built with ``use_crossattn_projection=True``, the projection
                is applied here.
            timestep: ``(B,)`` or ``(B, T)`` flow-matching timestep.
            condition_mask: ``(B, 1, T, H, W)`` LVG condition mask. ``None``
                gets auto-filled with zeros when the net is LVG-flavoured.
            padding_mask: ``(B, 1, H, W)`` spatial padding mask consumed by
                ``concat_padding_mask=True`` nets. Auto-filled with zeros
                when missing.
        """
        x_in = latents if latents is not None else input_latents
        if x_in is None:
            raise ValueError(
                "Cosmos25PipelineWrapper.prepare_block_loop requires either "
                "`latents` (preferred, noised at training time) or `input_latents`."
            )
        net = self.net
        B, _C, T_lat, H_lat, W_lat = x_in.shape
        device = x_in.device
        dtype = x_in.dtype

        # MinimalV1LVGDiT prepends a condition-mask channel and scales timesteps
        # before delegating to MiniTrainDIT.prepare_embedded_sequence.
        if hasattr(net, "timestep_scale"):
            if condition_mask is None:
                condition_mask = torch.zeros(B, 1, T_lat, H_lat, W_lat, dtype=dtype, device=device)
            x_B_C_T_H_W = torch.cat([x_in, condition_mask.to(dtype=dtype)], dim=1)
            timesteps_eff = timestep * float(net.timestep_scale)
        else:
            x_B_C_T_H_W = x_in
            timesteps_eff = timestep

        if getattr(net, "concat_padding_mask", False) and padding_mask is None:
            padding_mask = torch.zeros(B, 1, H_lat, W_lat, dtype=dtype, device=device)

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_per_block_pos_emb = net.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )

        # Optional cross-attn projection (Stage-c 2B uses 100352 → 1024).
        if getattr(net, "use_crossattn_projection", False):
            proj_in = int(getattr(net, "crossattn_proj_in_channels", -1))
            if context.shape[-1] == proj_in:
                context = net.crossattn_proj(context)

        # TI2V per-token timestep: when ``condition_mask`` marks any frames as
        # clean prefix (``mask == 1``), broadcast the per-sample timestep to
        # ``(B, T_lat)`` and zero out the prefix positions. The DiT then sees
        # ``t=0`` on those frames (i.e. "this is already clean"), the LVG-native
        # equivalent of Wan TI2V's ``fuse_vae_embedding_in_latents`` per-token
        # AdaLN. T2V keeps the legacy ``(B, 1)`` broadcast shape so this branch
        # is a pure no-op for non-TI2V steps.
        ti2v_active = (
            timesteps_eff.ndim == 1
            and condition_mask is not None
            and bool(condition_mask.any())
        )
        if timesteps_eff.ndim == 1:
            if ti2v_active:
                timesteps_eff = timesteps_eff.unsqueeze(1).expand(-1, T_lat).clone()
                frame_mask = condition_mask[:, 0, :, 0, 0]  # (B, T_lat)
                timesteps_eff = torch.where(
                    frame_mask > 0,
                    torch.zeros_like(timesteps_eff),
                    timesteps_eff,
                )
            else:
                timesteps_eff = timesteps_eff.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = net.t_embedder(timesteps_eff)
        t_embedding_B_T_D = net.t_embedding_norm(t_embedding_B_T_D)

        f = x_B_T_H_W_D.shape[1]
        h = x_B_T_H_W_D.shape[2]
        w = x_B_T_H_W_D.shape[3]

        return BlockLoopState(
            x=x_B_T_H_W_D,
            t_mod=torch.zeros((), dtype=dtype, device=device),  # Wan-style placeholder; unused on joint_cross_attn
            freqs=torch.zeros((), dtype=torch.complex64, device=device),  # placeholder
            context=context,
            context_mask=context_mask,
            f=f,
            h=h,
            w=w,
            t=None,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            extras={
                "t_embedding_B_T_D": t_embedding_B_T_D,
                "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
                "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
                "extra_per_block_pos_emb": extra_per_block_pos_emb,
            },
        )

    def run_block(self, block_id: int, state: BlockLoopState) -> BlockLoopState:
        block = self.net.blocks[block_id]
        state.x = block(
            state.x,
            state.extras["t_embedding_B_T_D"],
            state.context,
            rope_emb_L_1_1_D=state.extras["rope_emb_L_1_1_D"],
            adaln_lora_B_T_3D=state.extras["adaln_lora_B_T_3D"],
            extra_per_block_pos_emb=state.extras["extra_per_block_pos_emb"],
        )
        return state

    # ------------------------------------------------------------------
    # Joint self-attention split — pre / post halves of one block (§17)
    # ------------------------------------------------------------------

    def pre_attn_at_layer(self, block_id: int, state: BlockLoopState):
        """Pre half of block ``block_id`` for the MoT joint-attention driver.

        Thin wrapper over :func:`block_split.pre_self_attn`. Returns
        ``(q, k, v, post_state)`` with Q/K/V at ``(B, T·H·W, H·D)`` and the
        dict-shaped ``post_state`` Wan's contract expects.
        """
        from openwam.model.video_backbone.cosmos25.block_split import pre_self_attn

        block = self.net.blocks[block_id]
        return pre_self_attn(
            block,
            state.x,
            state.extras["t_embedding_B_T_D"],
            state.extras["adaln_lora_B_T_3D"],
            state.extras["rope_emb_L_1_1_D"],
            state.extras["extra_per_block_pos_emb"],
        )

    def post_attn_at_layer(
        self,
        block_id: int,
        state: BlockLoopState,
        attn_out: Tensor,
        post_state: dict,
    ) -> BlockLoopState:
        """Post half of block ``block_id``: output_proj → residual → cross-attn → MLP.

        ``attn_out`` is the pre-output-projection joint attention result the
        :class:`MoTJointDriver` sliced for the video modality. The block-id
        argument is part of the Wan-compatible hook signature but unused here
        because ``post_state`` already carries a direct reference to the
        block — leaving the kwarg in place keeps the driver call shape
        identical across backbones.
        """
        from openwam.model.video_backbone.cosmos25.block_split import post_self_attn

        _ = block_id  # part of the hook signature; block ref lives in post_state
        state.x = post_self_attn(attn_out, state.context, post_state)
        return state

    def finalize_block_loop(self, state: BlockLoopState) -> Tensor:
        x_B_T_H_W_O = self.net.final_layer(
            state.x,
            state.extras["t_embedding_B_T_D"],
            adaln_lora_B_T_3D=state.extras["adaln_lora_B_T_3D"],
        )
        return self.net.unpatchify(x_B_T_H_W_O)

    # ------------------------------------------------------------------
    # Preprocessing & decoding
    # ------------------------------------------------------------------

    def preprocess_input(
        self,
        *,
        frames: Any = None,
        text: Any = None,
        pre_encoded_text: Optional[Tensor] = None,
        input_latents: Optional[Tensor] = None,
        vace_videos: Any = None,
        ref_images: Any = None,
        **kw: Any,
    ) -> dict:
        if vace_videos is not None and any(v is not None for v in vace_videos):
            raise NotImplementedError(
                "Cosmos25 MVP does not support VACE conditioning. "
                "Drop `vace_video` from the dataset for `dual_system_cosmos25` runs."
            )

        if input_latents is None:
            if frames is None:
                raise ValueError(
                    "Cosmos25PipelineWrapper.preprocess_input requires either `input_latents` or `frames`."
                )
            if self.vae is None:
                raise RuntimeError(
                    "Cosmos25 VAE is not configured. Set `video_backbone.vae: wan2pt1` "
                    "(default; loads `<model_path>/tokenizer.pth`)."
                )
            input_latents = self._encode_frames(frames)

        if pre_encoded_text is not None:
            # Cache > live precedence: when both a cache hit AND a live encoder
            # are available, the cache wins per-sample (the live encoder is
            # silently skipped). See docs/cosmos25_backbone.md §14.
            context = pre_encoded_text
        elif text is not None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text=` requires a configured text encoder. Pass `pre_encoded_text` "
                    "directly, or set `video_backbone.text_encoder` to a registered loader."
                )
            # §14.7 — CFG dropout for the live path. Substitutes selected
            # prompts with ``""`` so the encoder produces the canonical empty
            # embedding (same numerical target as the cache path's
            # ``empty.safetensors`` lookup, modulo the post-projection step
            # which still happens below).
            if self.training and self.text_dropout_p > 0.0:
                text_list = [text] if isinstance(text, str) else list(text)
                text = [t if self._text_dropout_rng.random() >= self.text_dropout_p else "" for t in text_list]
            # Live encoder returns pre-projection `(B, L=512, 100352) bf16`.
            # We MUST apply the DiT's owned `net.crossattn_proj` (Linear+GELU)
            # here, NOT in `prepare_block_loop` — the architecture's
            # `_append_proprio_context_token` (base.py) concats a proprio
            # token at action-backbone dim (1024) onto context BEFORE the
            # block loop, so leaving context at 100352 would crash the
            # concat. The auto-gate at `prepare_block_loop:174-177` stays in
            # place as a defensive no-op (1024 ≠ 100352, so it skips).
            context = self.text_encoder(text)
            # Coerce to `input_latents`' device/dtype so block-loop arithmetic
            # stays uniform; the encoder lives on a possibly different
            # device when device_map="auto" is used.
            context = context.to(device=input_latents.device, dtype=input_latents.dtype)
            net = self.net
            if getattr(net, "use_crossattn_projection", False) and context.shape[-1] == int(
                getattr(net, "crossattn_proj_in_channels", -1)
            ):
                context = net.crossattn_proj(context)
        else:
            raise ValueError(
                "Cosmos25PipelineWrapper.preprocess_input requires either `text` (with text_encoder) "
                "or `pre_encoded_text`."
            )

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

        # TI2V first-frame conditioning. Activated by `ref_images` arriving in
        # the batch (auto-injected by ``FirstFrameConditioningTransform`` when
        # ``use_first_frame_as_reference=True``). The wrapper VAE-encodes one
        # reference frame per sample → ``first_frame_latents`` of shape
        # ``(B, 16, 1, H/8, W/8)`` and a matching LVG ``condition_mask`` with
        # frame 0 set to 1. ``base.py`` then overwrites ``latents[:, :, 0:1]``
        # at every diffusion step and skips frame 0 from the loss; the DiT
        # learns the tail conditioned on the clean prefix. Mirrors Wan TI2V
        # (``wan_adapter.py:815-833``), with the only mechanism difference
        # being LVG ``condition_mask`` + per-token timestep here vs. Wan's
        # ``fuse_vae_embedding_in_latents`` flag.
        ref_active = (
            ref_images is not None
            and isinstance(ref_images, (list, tuple))
            and len(ref_images) > 0
            and all(r is not None for r in ref_images)
        )
        if ref_active:
            if self.vae is None:
                raise RuntimeError(
                    "Cosmos25PipelineWrapper.preprocess_input received `ref_images` but no VAE is "
                    "configured. Set `video_backbone.vae: wan2pt1` to enable TI2V."
                )
            # ``FirstFrameConditioningTransform`` emits ``[PIL]`` per sample so
            # ``ref_images`` is already ``list[list[PIL]]`` — exactly the shape
            # ``_encode_frames`` expects. Tolerate flat ``list[PIL]`` callers
            # (e.g. tests / adapter inference) by wrapping bare images.
            ref_clips = [r if isinstance(r, (list, tuple)) else [r] for r in ref_images]
            first_frame_latents = self._encode_frames(ref_clips).to(
                device=input_latents.device, dtype=input_latents.dtype
            )
            T_lat = input_latents.shape[2]
            H_lat = input_latents.shape[3]
            W_lat = input_latents.shape[4]
            condition_mask = torch.zeros(
                (B, 1, T_lat, H_lat, W_lat),
                dtype=input_latents.dtype,
                device=input_latents.device,
            )
            condition_mask[:, :, 0] = 1.0
            out["first_frame_latents"] = first_frame_latents
            out["condition_mask"] = condition_mask
            out["num_clean_prefix_frames"] = 1

        return out

    def _encode_frames(self, frames: Any) -> Tensor:
        """PIL frames → bf16 ``(B, 16, T_lat, H/8, W/8)`` Wan2pt1 latents."""
        if self.vae is None:
            raise RuntimeError("Cosmos25PipelineWrapper._encode_frames called without a configured VAE.")
        video = _pil_video_to_tensor(frames)
        target_device = _vae_device(self.vae)
        video = video.to(device=target_device, dtype=torch.bfloat16)
        latents = self.vae.encode(video)
        return latents

    def decode_video(self, latents: Tensor, *, tiled: bool = True) -> list:
        if self.vae is None:
            raise RuntimeError(
                "Cosmos25PipelineWrapper.decode_video requires a configured VAE. Set `video_backbone.vae: wan2pt1`."
            )
        # Wan2pt1VAEInterface.decode takes no `tiled` kwarg; accept for ABC
        # signature parity. Internal tiling happens via temporal_window=4.
        _ = tiled
        target_device = _vae_device(self.vae)
        video = self.vae.decode(latents.to(device=target_device))
        return _video_tensor_to_pil(video)


def _pil_video_to_tensor(frames: Any) -> Tensor:
    """Convert ``list[list[PIL.Image]]`` → ``(B, 3, T, H, W)`` float in ``[-1, 1]``.

    Mirrors Wan's pipeline preprocessing (``wan/shared/diffusion/base_pipeline.py``):
    uint8 RGB → float / 127.5 - 1. Stays self-contained (no Wan imports) so
    Cosmos25 can be used without the Wan backbone installed.
    """
    import numpy as np

    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError(
            f"Expected `frames` as a non-empty list of clips (each a list of PIL frames); got {type(frames).__name__}."
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
    t = t.permute(0, 4, 1, 2, 3).contiguous()  # (B, 3, T, H, W)
    return t


def _video_tensor_to_pil(video: Tensor) -> list:
    """Convert ``(B, 3, T, H, W)`` in ``[-1, 1]`` → ``list[PIL.Image]`` (B=1 only)."""
    from PIL import Image  # local import; not always present on minimal CI

    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"_video_tensor_to_pil expected (B, 3, T, H, W); got shape {tuple(video.shape)}.")
    if video.shape[0] != 1:
        raise NotImplementedError(
            f"Cosmos25 decode currently supports B=1 only; got B={video.shape[0]}. Deploy paths call decode per-sample."
        )
    frame_uint8 = (
        ((video[0].float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
    )  # (T, H, W, 3) uint8
    return [Image.fromarray(frame) for frame in frame_uint8]


def _vae_device(vae: Any) -> torch.device:
    """Return the device of a ``Wan2pt1VAEInterface``-like VAE."""
    inner = getattr(getattr(vae, "model", None), "model", None)
    if isinstance(inner, nn.Module):
        try:
            return next(inner.parameters()).device
        except StopIteration:
            pass
    # Fallback: WanVAE caches its constructor `device=` attribute (wan2pt1.py:718).
    cached = getattr(getattr(vae, "model", None), "device", None)
    if isinstance(cached, torch.device):
        return cached
    if isinstance(cached, str):
        return torch.device(cached)
    return torch.device("cpu")


__all__ = ["Cosmos25PipelineWrapper"]
