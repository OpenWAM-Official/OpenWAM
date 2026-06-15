"""View over upstream ``SanaMSVideo`` exposing prepare / per-block / finalize.

This is the *heart* of the Phase 0 scaffold. It re-decomposes
``SanaMSVideo.forward`` (`third_party/Sana/diffusion/model/nets/sana_multi_scale_video.py:590`)
into three pieces that fit the OpenWAM block-loop contract:

1. ``prepare(x, timestep, y, mask)`` — patchify, t_embed, t_block (6-scale
   AdaLN modulation), y_embed, RoPE freqs. Returns a dict that the adapter
   feeds into a :class:`BlockLoopState`.
2. ``run_block(i, state_dict)`` — one upstream ``SanaVideoMSBlock.forward``
   call; identical to the loop body at ``sana_multi_scale_video.py:653-664``.
3. ``finalize(x, t, f, h, w)`` — ``final_layer + unpatchify``.

For the joint-MoT path (Phase 3+), the per-block forward is further split
into ``block_pre_attn`` + ``native_attn`` + ``block_post_attn`` — matching
the layout in ``SanaVideoMSBlock.forward`` lines 256-333 and
``LiteLAReLURope.forward`` at ``sana_blocks.py:340-388``. ``block_pre_attn``
returns BOTH the rotated and the unrotated kernel-applied Q/K — the unrotated
versions populate the denominator ``z`` that the SANA cumsum needs, and they
are stashed in ``post_state`` so a future driver can consume them without
re-running the projection.

The wrapper holds **references** to upstream submodules (it is NOT an
``nn.Module``) — pin-bumps of ``third_party/Sana`` are transparent unless they
break the wrapped signatures, in which case the smoke test
``test_sana_split_eq_native`` fires.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

logger = logging.getLogger(__name__)

# Match upstream's xformers-disabled path before any SANA module loads — this
# ensures `_xformers_available` resolves to False inside both the upstream
# `SanaMSVideo.forward` and the helpers below, keeping the mask plumbing
# consistent.
os.environ.setdefault("DISABLE_XFORMERS", "1")


def _t2i_modulate(x, shift, scale):
    """Mirror upstream's ``t2i_modulate`` (sana_blocks.py).

    Inlined to avoid pulling extra upstream imports at module load. Equivalent
    to ``x * (1 + scale) + shift`` with the standard broadcast rules.
    """
    return x * (1 + scale) + shift


def _apply_rope_lite(hidden_states: Tensor, freqs: Tensor) -> Tensor:
    """Mirror the ``apply_rotary_emb`` closure inside ``LiteLAReLURope.forward``.

    Reproduced verbatim from ``sana_blocks.py:359-362`` so any pin-bump that
    changes the RoPE math triggers ``test_sana_split_eq_native`` rather than
    silently diverging.
    """
    x_rotated = torch.view_as_complex(hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
    return x_out.type_as(hidden_states)


class SanaMSVideoSplit:
    """Re-decomposition of ``SanaMSVideo.forward`` into prepare/loop/finalize.

    Not an ``nn.Module``: holds references to the wrapped model's children but
    does not re-register parameters. The owning ``SanaVideoBackbone`` is the
    ``nn.Module`` boundary.

    Phase 0 contract:
      - :meth:`prepare` + N× :meth:`run_block` + :meth:`finalize` produces
        bit-equivalent output to ``self._dit(x, timestep, y, mask)`` for the
        ``len(timestep.shape) <= 2`` (per-sample, non-frame-aware) path.
      - :meth:`block_pre_attn` + :meth:`native_attn` + :meth:`block_post_attn`
        is bit-equivalent to ``self._dit.blocks[i].forward(...)``.

    Future (Phase 3) hook: ``post_state["q_unrot"]``/``["k_unrot"]`` lets a
    derivative ``SanaMoTJointDriver`` skip ``native_attn`` and run its own
    cumsum-linear-attn over a heterogeneous sequence.
    """

    def __init__(self, sana_model: nn.Module):
        # NOTE: store ONLY references — no `Parameter` copies.
        self._dit = sana_model

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> dict:
        """Run the pre-block-loop portion of ``SanaMSVideo.forward``.

        Inputs match upstream:
          - ``x``: ``(B, C, T, H, W)`` latent tensor
          - ``timestep``: ``(B,)`` per-sample step
          - ``y``: ``(B, 1, L_text, C_text)`` caption embeds
          - ``mask``: optional caption mask
        """
        dit = self._dit
        bs = x.shape[0]
        x = x.to(dit.dtype)
        # Mirror upstream type discipline:
        if dit.timestep_norm_scale_factor != 1.0:
            timestep = (timestep.float() / dit.timestep_norm_scale_factor).to(torch.float32)
        else:
            timestep = timestep.long().to(torch.float32)
        y = y.to(dit.dtype)

        dit.f, dit.h, dit.w = (
            x.shape[-3] // dit.patch_size[0],
            x.shape[-2] // dit.patch_size[1],
            x.shape[-1] // dit.patch_size[2],
        )

        data_info = kwargs.get("data_info", {})
        if data_info.get("image_vae_embeds", None) is not None:
            x = torch.cat([x, data_info["image_vae_embeds"].to(dit.dtype)], dim=1)
        if data_info.get("image_embeds", None) is not None:
            image_embeds = data_info["image_embeds"].to(dit.dtype)
            image_embeds = dit.image_embedder(image_embeds)
            kwargs["image_embeds"] = image_embeds

        if getattr(dit, "pack_latents", False):
            x = dit._pack_latents(x, bs, dit.in_channels, dit.h, dit.w, dit.f)
            dit.h = dit.h // 2
            dit.w = dit.w // 2
        if dit.x_embedder.patch_size != dit.x_embedder.kernel_size and dit.x_embedder.kernel_size == (1, 2, 2):
            x = F.pad(x, (0, 1, 0, 1, 0, 0))

        x = dit.x_embedder(x)
        image_pos_embed: Optional[Tensor] = None
        if dit.use_pe:
            x, image_pos_embed = dit._apply_positional_embedding(x, bs)

        t = dit.t_embedder(timestep.flatten())  # (B, D)
        t0 = dit.t_block(t)  # (B, 6*D) — fed as AdaLN modulation
        t = t.unflatten(dim=0, sizes=timestep.shape)
        t0 = t0.unflatten(dim=0, sizes=timestep.shape)

        y = dit.y_embedder(y, dit.training, mask=mask)
        if dit.y_norm:
            y = dit.attention_y_norm(y)

        # Caption mask plumbing — mirror upstream's xformers-vs-mask branching.
        # We force the xformers-off path here (see DISABLE_XFORMERS at module
        # top); upstream then routes the int16 mask as ``y_lens`` straight to
        # ``MultiHeadCrossAttention``.
        if mask is not None:
            mask = mask.to(torch.int16)
            if mask.shape[0] != y.shape[0]:
                # Dim-agnostic: torch.repeat requires len(repeats) >= mask.dim(),
                # so 2D (B, L) and 4D (B, 1, 1, L) CFG-path masks both need a
                # trailing 1 per extra dim.
                repeats = (y.shape[0] // mask.shape[0],) + (1,) * (mask.dim() - 1)
                mask = mask.repeat(*repeats)
            mask = mask.squeeze(1).squeeze(1)
            y_lens = mask
        else:
            # No mask + no xformers ⇒ upstream raises. Cross-attention here is
            # used by the smoke tests that pass a mask explicitly. For action-
            # only / inference paths the adapter constructs an "all-attend"
            # mask before calling prepare.
            raise ValueError(
                "SanaMSVideoSplit.prepare requires a caption ``mask`` when DISABLE_XFORMERS=1 (the OpenWAM default)."
            )

        return {
            "x": x,
            "t": t,
            "t0": t0,
            "y": y,
            "y_lens": y_lens,
            "image_pos_embed": image_pos_embed,
            "f": dit.f,
            "h": dit.h,
            "w": dit.w,
            "bs": bs,
            "kwargs": kwargs,
        }

    # ------------------------------------------------------------------
    # run_block (whole-block reference path)
    # ------------------------------------------------------------------

    def run_block(self, block_id: int, state: dict) -> dict:
        """Run a single upstream block — equivalent to one iteration of the
        loop at ``sana_multi_scale_video.py:653-664``.

        This is the path used by the OpenWAM video-only loop. The
        joint-MoT path (Phase 3+) drives ``block_pre_attn`` + ``native_attn``
        + ``block_post_attn`` instead.
        """
        block = self._dit.blocks[block_id]
        x = block(
            state["x"],
            state["y"],
            state["t0"],
            state["y_lens"],
            (state["f"], state["h"], state["w"]),
            state["image_pos_embed"],
            **state.get("kwargs", {}),
        )
        state["x"] = x
        return state

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(self, state: dict) -> Tensor:
        """Run the post-block-loop portion of ``SanaMSVideo.forward``."""
        dit = self._dit
        x = dit.final_layer(state["x"], state["t"])  # (B, T, P²·C_out)
        x = dit.unpatchify(x)  # (B, C_out, T, H, W)
        if getattr(dit, "pack_latents", False):
            x = dit._unpack_latents(x, dit.h * 2, dit.w * 2, dit.f)
        return x

    # ------------------------------------------------------------------
    # Per-block pre / native_attn / post split (joint-MoT-ready)
    # ------------------------------------------------------------------

    def block_pre_attn(
        self,
        block_id: int,
        x: Tensor,
        t0: Tensor,
        rotary_emb: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, dict]:
        """Run norm1 + AdaLN modulate + QKV proj + qk_norm + ReLU + RoPE.

        Returns ``(tilde_q, tilde_k, v, post_state)`` with ``tilde_q/tilde_k/v``
        shaped ``(B, S, H*D)`` to match the OpenWAM MoT contract. The
        ``post_state`` dict has:

        - ``residual_x``, ``gate_msa``, ``shift_mlp``, ``scale_mlp``, ``gate_mlp``,
          ``block`` — needed to finish the block in :meth:`block_post_attn`
        - ``q_unrot``, ``k_unrot`` — ReLU(q)/ReLU(k) **before** RoPE, also shaped
          ``(B, S, H*D)``; consumed by future ``SanaMoTJointDriver`` to compute
          the linear-attn denominator
        - ``uses_linear_attn = True`` — dispatch flag for the driver
        """
        block = self._dit.blocks[block_id]
        attn = block.attn  # LiteLAReLURope instance
        residual_x = x
        B, N, C = x.shape

        # AdaLN modulation — mirror the non-frame-aware branch
        # (sana_multi_scale_video.py:276-279).
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.scale_shift_table[None] + t0.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_sa_in = _t2i_modulate(block.norm1(x), shift_msa, scale_msa)

        # LiteLAReLURope.forward up to RoPE (sana_blocks.py:340-365).
        qkv = attn.qkv(x_sa_in).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # each (B, N, C)
        q = attn.q_norm(q).transpose(-1, -2)  # (B, C, N)
        k = attn.k_norm(k).transpose(-1, -2)
        v_raw = v.transpose(-1, -2)  # (B, C, N)

        # Upstream stores heads on dim=-3: (B, h, h_d, N). The OpenWAM MoT
        # contract wants (B, S, H*D) — we keep BOTH representations so this
        # function is the one place where the layout pivot is done.
        h_count = C // attn.dim
        h_d = attn.dim
        q_heads = q.reshape(B, h_count, h_d, N)
        k_heads = k.reshape(B, h_count, h_d, N)
        v_heads = v_raw.reshape(B, h_count, h_d, N)

        q_relu = attn.kernel_func(q_heads)
        k_relu = attn.kernel_func(k_heads)

        q_rot = _apply_rope_lite(q_relu, rotary_emb) if rotary_emb is not None else q_relu
        k_rot = _apply_rope_lite(k_relu, rotary_emb) if rotary_emb is not None else k_relu

        # Reshape to MoT contract: (B, h, h_d, N) → (B, N, h*h_d) = (B, S, H*D).
        def to_mot(t: Tensor) -> Tensor:
            # einops keeps the operation legible against the upstream shape.
            return rearrange(t, "b h d n -> b n (h d)")

        post_state = {
            "block": block,
            "residual_x": residual_x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "q_unrot": to_mot(q_relu),
            "k_unrot": to_mot(k_relu),
            "uses_linear_attn": True,
            # Internal-only — re-used by `native_attn` to avoid re-pivoting.
            "_q_rot_heads": q_rot,
            "_k_rot_heads": k_rot,
            "_q_relu_heads": q_relu,
            "_k_relu_heads": k_relu,
            "_v_heads": v_heads,
        }
        return to_mot(q_rot), to_mot(k_rot), to_mot(v_heads), post_state

    def native_attn(self, post_state: dict, *, fp32: bool = False) -> Tensor:
        """Run SANA's fused LiteLAReLURope post-projection math.

        Reproduces ``sana_blocks.py:378-385`` — the matmuls and dual-track
        normalization. Returns the attention output in MoT layout
        ``(B, S, H*D)``, **before** the o_proj (which lives in
        :meth:`block_post_attn`). Phase 0 calls this from ``run_block`` to
        verify the split is numerically equivalent to running
        ``block.forward`` directly. Phase 3+ ``SanaMoTJointDriver`` bypasses
        this method and runs its own mixed-attention over a concatenated
        video+action sequence.
        """
        q_rot = post_state["_q_rot_heads"]
        k_rot = post_state["_k_rot_heads"]
        q_relu = post_state["_q_relu_heads"]
        k_relu = post_state["_k_relu_heads"]
        v = post_state["_v_heads"]
        attn = post_state["block"].attn

        dtype_out = v.dtype
        if fp32 or getattr(attn, "fp32_attention", False):
            q_rot = q_rot.float()
            k_rot = k_rot.float()
            v_local = v.float()
        else:
            v_local = v

        z = 1.0 / (k_relu.sum(dim=-1, keepdim=True).transpose(-2, -1) @ q_relu + attn.eps)
        vk = torch.matmul(v_local, k_rot.transpose(-1, -2))
        out = torch.matmul(vk, q_rot)
        out = (out * z).to(dtype_out)
        # (B, h, h_d, N) → (B, N, h*h_d)
        return rearrange(out, "b h d n -> b n (h d)")

    def block_post_attn(
        self,
        block_id: int,
        attn_out: Tensor,
        post_state: dict,
        *,
        y: Tensor,
        y_lens: Tensor,
        f: int,
        h: int,
        w: int,
        image_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply ``proj + gate + cross_attn + FFN`` to attention output.

        Mirrors ``SanaVideoMSBlock.forward`` lines 301-323. Returns the updated
        ``x``; ``post_attn_at_layer`` in the adapter wraps this and writes back
        into ``state.x``.
        """
        block = post_state["block"]
        attn = block.attn

        # LiteLAReLURope.proj — the trailing projection that lives inside
        # ``attn.forward`` upstream but which the MoT split moves to post.
        # ``attn_out`` is shaped (B, N, C); ``proj`` is a Linear(C, C).
        proj_out = attn.proj(attn_out)

        # Optional secondary flash-attention residual (sana_multi_scale_video.py:298-299).
        # Phase 0 assumes ``additional_flash_attn=None`` (the 2B config), so
        # ``flash_attn_additional`` is None.
        x = post_state["residual_x"] + block.drop_path(post_state["gate_msa"] * proj_out)

        # Cross-attention (text) — upstream signature variants for image_embeds.
        if getattr(block, "cross_attn_image_embeds", False):
            x = x + block.cross_attn(x, y, mask=y_lens, image_embeds=image_embeds)
        else:
            x = x + block.cross_attn(x, y, mask=y_lens)

        # FFN (GLUMBConvTemp) — needs spatial dims for the temporal conv reshape.
        mlp_in = _t2i_modulate(block.norm2(x), post_state["shift_mlp"], post_state["scale_mlp"])
        mlp_out = block.mlp(mlp_in, HW=(f, h, w))
        x = x + block.drop_path(post_state["gate_mlp"] * mlp_out)
        return x
