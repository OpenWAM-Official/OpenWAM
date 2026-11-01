"""torch.compile'd memory-bound chains of the MoT layer halves (``OPENWAM_COMPILE_BLOCKS=1``).

Each MoT layer runs, per backbone, long chains of small bf16/fp32 elementwise and
row-reduction kernels around its GEMMs: the AdaLN modulation sum and chunks,
LayerNorm + ``x * (1 + scale) + shift``, the fp32 RMSNorms of q/k, the gated
residuals, the tanh-GELU. The functions below are the same formulas as the
eager modules (``wan.models.dit`` / ``action_backbone.separate_action_dit``),
written over explicit parameter tensors so one compiled graph serves all layers
(no per-layer guards or recompiles); Inductor fuses the elementwise chains in the
forward and the generated backward. GEMMs stay cuBLAS (extern kernels). The
attentions (joint, cross), the RoPE kernel and the context K/V projections stay
eager between the compiled pieces. Same real-valued function; rounding can differ
from eager (fused fp32 intermediates).

Deterministic mode: the compiled chains are not reproducible from run to run under
``torch.use_deterministic_algorithms(True)`` (two runs of the same code can give
different gradients), so while deterministic algorithms are enabled the layers use
the eager modules instead.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("OPENWAM_COMPILE_BLOCKS", "0") == "1"
_COMPILED: dict = {}


def enabled() -> bool:
    return _ENABLED


def _compiled(name, fn):
    c = _COMPILED.get(name)
    if c is None:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 16)
        c = torch.compile(fn, dynamic=False)
        _COMPILED[name] = c
    return c


def _rms(x, w, eps: float):
    dtype = x.dtype
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype) * w


# ---------------------------------------------------------------- video (Wan DiT block)
def _wan_pre(x, tmod, modulation, q_w, q_b, k_w, k_b, v_w, v_b, nq_w, nk_w, eps_ln: float, eps_q: float, eps_k: float):
    chunks = (modulation.to(dtype=tmod.dtype) + tmod).chunk(6, dim=2)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [c.squeeze(2) for c in chunks]
    h = F.layer_norm(x, (x.shape[-1],), None, None, eps_ln) * (1 + scale_msa) + shift_msa
    q = _rms(F.linear(h, q_w, q_b), nq_w, eps_q)
    k = _rms(F.linear(h, k_w, k_b), nk_w, eps_k)
    v = F.linear(h, v_w, v_b)
    return q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _wan_post_a(residual_x, gate_msa, attn_out, o_w, o_b, n3_w, n3_b, cq_w, cq_b, cnq_w, eps3: float, eps_q: float):
    hs = residual_x + gate_msa * F.linear(attn_out, o_w, o_b)
    h3 = F.layer_norm(hs, (hs.shape[-1],), n3_w, n3_b, eps3)
    cq = _rms(F.linear(h3, cq_w, cq_b), cnq_w, eps_q)
    return hs, cq


def _post_b(hs, cx, co_w, co_b, shift_mlp, scale_mlp, gate_mlp, f1_w, f1_b, f2_w, f2_b, eps2: float, n2_w, n2_b):
    hs = hs + F.linear(cx, co_w, co_b)
    m = F.layer_norm(hs, (hs.shape[-1],), n2_w, n2_b, eps2) * (1 + scale_mlp) + shift_mlp
    return hs + gate_mlp * F.linear(F.gelu(F.linear(m, f1_w, f1_b), approximate="tanh"), f2_w, f2_b)


def wan_block_supported(block, time_mod) -> bool:
    ca = block.cross_attn
    return (
        _ENABLED
        and not torch.are_deterministic_algorithms_enabled()
        and time_mod.dim() == 4
        and not getattr(ca, "has_image_input", False)
        and block.norm1.weight is None
        and block.norm2.weight is None
        and block.norm3.weight is not None
        and isinstance(block.ffn[1], torch.nn.GELU)
        and block.ffn[1].approximate == "tanh"
    )


def wan_pre(block, x, time_mod):
    sa = block.self_attn
    return _compiled("wan_pre", _wan_pre)(
        x,
        time_mod,
        block.modulation,
        sa.q.weight,
        sa.q.bias,
        sa.k.weight,
        sa.k.bias,
        sa.v.weight,
        sa.v.bias,
        sa.norm_q.weight,
        sa.norm_k.weight,
        float(block.norm1.eps),
        float(sa.norm_q.eps),
        float(sa.norm_k.eps),
    )


def wan_post(block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, attn_out, context, context_mask, attention):
    sa, ca = block.self_attn, block.cross_attn
    hs, cq = _compiled("wan_post_a", _wan_post_a)(
        residual_x,
        gate_msa,
        attn_out,
        sa.o.weight,
        sa.o.bias,
        block.norm3.weight,
        block.norm3.bias,
        ca.q.weight,
        ca.q.bias,
        ca.norm_q.weight,
        float(block.norm3.eps),
        float(ca.norm_q.eps),
    )
    ck = ca.norm_k(ca.k(context))
    cv = ca.v(context)
    cx = attention(cq, ck, cv, ca.num_heads, context_mask)
    f1, f2 = block.ffn[0], block.ffn[2]
    return _compiled("post_b", _post_b)(
        hs,
        cx,
        ca.o.weight,
        ca.o.bias,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        f1.weight,
        f1.bias,
        f2.weight,
        f2.bias,
        float(block.norm2.eps),
        None,
        None,
    )


# ---------------------------------------------------------------- action (SelfAttnActionDiTBlock)
def _act_pre(x, tmod, modulation, q_w, q_b, k_w, k_b, v_w, v_b, nq_w, nk_w, eps_ln: float, eps_q: float, eps_k: float):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (modulation.to(dtype=tmod.dtype) + tmod).chunk(
        6, dim=1
    )
    h = F.layer_norm(x, (x.shape[-1],), None, None, eps_ln) * (1 + scale_msa) + shift_msa
    q = _rms(F.linear(h, q_w, q_b), nq_w, eps_q)
    k = _rms(F.linear(h, k_w, k_b), nk_w, eps_k)
    v = F.linear(h, v_w, v_b)
    return q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp


def act_block_supported(block, t_mod) -> bool:
    return (
        _ENABLED
        and not torch.are_deterministic_algorithms_enabled()
        and t_mod.dim() == 3
        and block.self_attn_norm.weight is None
        and block.ffn_norm.weight is None
        and block.context_attn_norm.weight is not None
        and isinstance(block.ffn[1], torch.nn.GELU)
        and block.ffn[1].approximate == "tanh"
    )


def act_pre(block, x, t_mod):
    sa = block.self_attn
    return _compiled("act_pre", _act_pre)(
        x,
        t_mod,
        block.modulation,
        sa.q.weight,
        sa.q.bias,
        sa.k.weight,
        sa.k.bias,
        sa.v.weight,
        sa.v.bias,
        sa.norm_q.weight,
        sa.norm_k.weight,
        float(block.self_attn_norm.eps),
        float(sa.norm_q.eps),
        float(sa.norm_k.eps),
    )


def act_post(block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, attn_out, context, text_mask, attention):
    sa, ca = block.self_attn, block.cross_attn
    n = block.context_attn_norm
    hs, cq = _compiled("act_post_a", _wan_post_a)(
        residual_x,
        gate_msa,
        attn_out,
        sa.o.weight,
        sa.o.bias,
        n.weight,
        n.bias,
        ca.q.weight,
        ca.q.bias,
        ca.norm_q.weight,
        float(n.eps),
        float(ca.norm_q.eps),
    )
    ck = ca.norm_k(ca.k(context))
    cv = ca.v(context)
    cx = attention(cq, ck, cv, ca.num_heads, text_mask)
    f1, f2 = block.ffn[0], block.ffn[2]
    return _compiled("act_post_b", _post_b)(
        hs,
        cx,
        ca.o.weight,
        ca.o.bias,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        f1.weight,
        f1.bias,
        f2.weight,
        f2.bias,
        float(block.ffn_norm.eps),
        None,
        None,
    )


def masked_mha(q, k, v, num_heads: int, mask):
    """(B, Sq, H*D) x (B, Sk, H*D) masked SDPA -> (B, Sq, H*D), as the eager cross-attentions do."""
    from einops import rearrange

    from openwam.model.kernels.attention import sdpa

    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    if mask is not None and mask.dim() == 3:
        mask = mask.unsqueeze(1)
    x = sdpa(q, k, v, attn_mask=mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)
