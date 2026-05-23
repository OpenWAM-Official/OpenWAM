"""SanaMoTJointDriver: MoT driver for SANA-style linear-attention backbones.

Replaces the SDPA-based :meth:`MoTJointDriver._mixed_attention` with a cumsum
expansion of SANA's ReLU-kernel linear attention, preserving the pretrained
dual-track normalization (rotated phi(q)/phi(k) in the numerator, *un*-rotated
phi(q)/phi(k) in the denominator — see plans/sana_mot_math_conflict.md §1)
while supporting OpenWAM's joint mask topology (plan §1 closing example).

Two extensions of the base contract:

1. Override :meth:`_step_impl` so the unrotated ReLU'd Q/K (sitting in each
   backbone's ``post_state`` dict under ``"q_unrot"`` / ``"k_unrot"``) can be
   threaded into the mixed-attention call. The base class only passes
   ``(q_cat, k_cat, v_cat, attn_mask)``, which is fine for SDPA but loses
   the second-track inputs the SANA math needs.
2. Override :meth:`_mixed_attention` with kw-only ``phi_q`` / ``phi_k`` /
   ``use_ckpt`` extensions. This stays Liskov-compatible with the base
   signature (callers passing only the four positional args still get a
   well-defined error path, see :meth:`_mixed_attention`).

Everything else — ``_build_joint_mask``, ``_build_attention_mask``,
``_video_tokens_per_frame``, ``run_joint_loop``, ``step``, ``_step_checkpointed``
— is inherited unchanged.

See plans/sana_mot_integration_plan.md §3 for the design discussion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from openwam.model.architectures.dual_system.mot_driver import MoTJointDriver
from openwam.model.architectures.dual_system.sana_linear_attn import (
    _chunked_linear_attn,
    _chunked_linear_attn_checkpointed,
    _expanded_linear_attn,
    _mask_to_chunk_index,
)

if TYPE_CHECKING:
    from openwam.model.action_backbone.backbone import ActionBackbone
    from openwam.model.base import ActionState
    from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone

logger = logging.getLogger(__name__)


class SanaMoTJointDriver(MoTJointDriver):
    """MoT driver for SANA-style ReLU-kernel linear attention.

    Strict kernel alignment is required at construction time: both
    ``vb.attn_kernel`` and ``ab.attn_kernel`` must be ``"linear_relu"``.
    A mixed-kernel pair (e.g. SANA video + softmax action) is rejected
    because the cross-modality inner products
    ``tilde_q_v · tilde_k_a`` / ``tilde_q_a · tilde_k_v`` have no meaning
    if one side was kernel-applied and the other wasn't.
    """

    def __init__(
        self,
        vb: "VideoBackbone",
        ab: "ActionBackbone",
        *,
        eps: float = 1e-15,
        **kw,
    ) -> None:
        super().__init__(vb, ab, **kw)
        self.eps = float(eps)
        # `run_joint_loop` builds attn_mask once and reuses across all N layers,
        # so the chunk-index translation (O(N) row-scan over an N×N bool mask;
        # N can be ~32k for 81f × 480p joint masks) is identical every layer.
        # Cache by id() to skip the rescan within one forward.
        #
        # Cache lifetime is bounded to a single ``run_joint_loop`` call:
        # ``run_joint_loop`` clears the cache on entry below, so Python's
        # id-recycling (a freed mask's id reassigned to a new mask of the
        # same N but different topology, e.g. switching
        # ``attention_mask_mode``) cannot deliver a stale ``chunk_index`` to
        # a subsequent forward.
        self._chunk_cache_key: Optional[int] = None
        self._chunk_cache_value: Optional[List[int]] = None

        v_kernel = getattr(vb, "attn_kernel", "softmax")
        a_kernel = getattr(ab, "attn_kernel", "softmax")
        if v_kernel != "linear_relu":
            raise ValueError(
                f"SanaMoTJointDriver requires video_backbone.attn_kernel='linear_relu', "
                f"got '{v_kernel}'. The SANA linear-attn path cannot be combined with "
                "a softmax video backbone — use MoTJointDriver instead."
            )
        if a_kernel != "linear_relu":
            raise ValueError(
                f"SanaMoTJointDriver requires action_backbone.attn_kernel='linear_relu', "
                f"got '{a_kernel}'. Cross-modality inner products require kernel-aligned "
                "Q/K on both sides; set ActionDiT(attn_kernel='linear_relu') in the config."
            )

    def run_joint_loop(self, *args, **kwargs):
        """Reset the chunk-index cache, then delegate to the base loop.

        The cache key is ``id(attn_mask)``; Python is free to reuse the id
        of a previously-freed mask for a new mask of the same shape but
        different topology (e.g. an ``attention_mask_mode`` change between
        forwards). Clearing the cache here bounds its lifetime to one
        ``run_joint_loop`` invocation, which is exactly the window over
        which the mask object is guaranteed live and reused by all layers.
        Within a single loop the cache is still a per-layer O(1) hit; the
        first-layer rescan cost is unavoidable (we have to materialize the
        chunk boundary somewhere) and unchanged.
        """
        self._chunk_cache_key = None
        self._chunk_cache_value = None
        return super().run_joint_loop(*args, **kwargs)

    def _step_impl(
        self,
        layer_id: int,
        vstate: "BlockLoopState",
        astate: "ActionState",
        attn_mask: Optional[Tensor] = None,
        *,
        suppress_inner_attn_ckpt: bool = False,
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Override of :meth:`MoTJointDriver._step_impl`.

        Mirrors the parent's structure (pre_attn → concat → mixed → split →
        post_attn), but additionally threads ``post_state["q_unrot"]`` /
        ``["k_unrot"]`` from each backbone into the SANA linear-attn call.
        The parent's INVARIANT (only ``vstate.x`` and ``astate.payload.x_action``
        may be mutated; layer-invariant fields stay shared by reference) is
        preserved.
        """
        vb = self.vb
        ab = self.ab

        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(layer_id, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer_id, astate)

        if q_v.dtype != q_a.dtype:
            raise RuntimeError(
                f"SanaMoTJointDriver: dtype mismatch at layer {layer_id} "
                f"(video={q_v.dtype}, action={q_a.dtype}). Both backbones "
                "must produce attention inputs in matching dtype."
            )
        if q_v.device != q_a.device:
            raise RuntimeError(
                f"SanaMoTJointDriver: device mismatch at layer {layer_id} "
                f"(video={q_v.device}, action={q_a.device})."
            )
        if not (vpost.get("uses_linear_attn") and apost.get("uses_linear_attn")):
            raise RuntimeError(
                "SanaMoTJointDriver expected both backbones to publish "
                "uses_linear_attn=True in post_state. Got "
                f"video={vpost.get('uses_linear_attn')}, action={apost.get('uses_linear_attn')}. "
                "Check that the backbones implement the linear-attn pre_attn_at_layer "
                "contract (see openwam/model/video_backbone/sana/blocks_split.py)."
            )

        s_video = q_v.shape[1]
        s_action = q_a.shape[1]
        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        phi_q = torch.cat([vpost["q_unrot"], apost["q_unrot"]], dim=1)
        phi_k = torch.cat([vpost["k_unrot"], apost["k_unrot"]], dim=1)
        # `run_joint_loop` pre-builds ``attn_mask`` and reuses it across layers;
        # SanaMoTJointDriver follows the same contract — _step_impl does NOT
        # rebuild the mask itself.

        use_ckpt = (
            self.mot_checkpoint_mixed_attn
            and ab.training
            and not suppress_inner_attn_ckpt
        )

        mixed = self._mixed_attention(
            q_cat,
            k_cat,
            v_cat,
            attn_mask,
            phi_q=phi_q,
            phi_k=phi_k,
            use_ckpt=use_ckpt,
        )

        attn_v, attn_a = mixed.split([s_video, s_action], dim=1)
        vstate = vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_a.contiguous(), apost)
        return vstate, astate

    def _mixed_attention(  # type: ignore[override]
        self,
        q_cat: Tensor,
        k_cat: Tensor,
        v_cat: Tensor,
        attn_mask: Optional[Tensor],
        *,
        phi_q: Optional[Tensor] = None,
        phi_k: Optional[Tensor] = None,
        use_ckpt: bool = False,
    ) -> Tensor:
        """SANA cumsum-linear-attention replacement for SDPA.

        Layout pivot ``(B, S, H*D) ↔ (B, H, S, d)`` happens here so the
        underlying primitives (:func:`_chunked_linear_attn`,
        :func:`_expanded_linear_attn`) can stay in the upstream-SANA layout
        without forcing every caller to rearrange.

        Dispatch rules:

        - If ``_mask_to_chunk_index(attn_mask)`` returns a non-``None`` chunk
          boundary list, use the cumsum fast path
          (:func:`_chunked_linear_attn` or, when ``use_ckpt`` is true,
          :func:`_chunked_linear_attn_checkpointed` for per-chunk gradient
          checkpointing).
        - Otherwise (mask doesn't factorize into monotonic block-causal
          chunks, or ``attn_mask is None``), fall back to
          :func:`_expanded_linear_attn`. When ``use_ckpt`` is true, wrap that
          single call in :func:`torch.utils.checkpoint.checkpoint` to match
          the memory profile of the base SDPA path's ``mot_checkpoint_mixed_attn``.

        ``phi_q`` / ``phi_k`` carry the unrotated ReLU'd Q/K (the second
        track required for SANA's dual-track denominator). If either is
        ``None``, we raise immediately rather than silently falling back to
        SDPA — that downgrade would produce mathematically meaningless output
        and mask a real bug in the caller.
        """
        if phi_q is None or phi_k is None:
            raise RuntimeError(
                "SanaMoTJointDriver._mixed_attention requires phi_q and phi_k "
                "(unrotated ReLU'd Q/K from post_state). Got "
                f"phi_q={'set' if phi_q is not None else 'None'}, "
                f"phi_k={'set' if phi_k is not None else 'None'}. "
                "This usually means _step_impl was bypassed by a direct SDPA-style "
                "call site that doesn't know about the dual-track contract."
            )

        n = self.num_heads
        tilde_q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
        tilde_k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
        v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
        pq = rearrange(phi_q, "b s (n d) -> b n s d", n=n)
        pk = rearrange(phi_k, "b s (n d) -> b n s d", n=n)

        if attn_mask is None:
            # Bidirectional: every row attends to every column. Equivalent to
            # a single chunk spanning [0, N] in the block-causal cumsum form
            # (state accumulates over all keys before any query reads it).
            # Routing through the chunked path keeps memory at O(N · d²)
            # instead of the O(N²) expanded fallback — critical for the long
            # 32k-token joint sequences where bidirectional was the original
            # motivation for SANA's linear attention.
            n_tokens = tilde_q.shape[-2]
            chunk_index: Optional[List[int]] = [0, int(n_tokens)]
        elif self._chunk_cache_key == id(attn_mask):
            chunk_index = self._chunk_cache_value
        else:
            chunk_index = _mask_to_chunk_index(attn_mask)
            self._chunk_cache_key = id(attn_mask)
            self._chunk_cache_value = chunk_index

        if chunk_index is not None:
            attn_fn = _chunked_linear_attn_checkpointed if use_ckpt else _chunked_linear_attn
            out = attn_fn(tilde_q, tilde_k, v, pq, pk, chunk_index, eps=self.eps)
        else:
            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    _expanded_linear_attn,
                    tilde_q,
                    tilde_k,
                    v,
                    pq,
                    pk,
                    attn_mask,
                    self.eps,
                    use_reentrant=False,
                )
            else:
                out = _expanded_linear_attn(
                    tilde_q, tilde_k, v, pq, pk, mask=attn_mask, eps=self.eps
                )

        return rearrange(out, "b n s d -> b s (n d)", n=n)


__all__ = ["SanaMoTJointDriver"]
