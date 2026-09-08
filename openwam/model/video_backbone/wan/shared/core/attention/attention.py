import os

import torch
from einops import rearrange

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False


def initialize_attention_priority():
    if os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION") is not None:
        return os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION").lower()
    elif FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    elif FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    elif SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    elif XFORMERS_AVAILABLE:
        return "xformers"
    else:
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


def rearrange_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    required_in_pattern="b n s d",
    dims=None,
):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    attn_mask=None,
    scale=None,
):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def xformers_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def resolve_implementation(q: torch.Tensor) -> str:
    """Narrow ATTENTION_IMPLEMENTATION to one this input can actually run.

    FA2/FA3/sage are CUDA half-precision kernels and xformers is CUDA-only; all of
    them raise rather than degrade, so anything they cannot take falls back to SDPA.
    xformers keeps its fp32 support, hence the separate check.
    """
    if ATTENTION_IMPLEMENTATION in ("flash_attention_3", "flash_attention_2", "sage_attention"):
        if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
            return "torch"
    elif ATTENTION_IMPLEMENTATION == "xformers" and not q.is_cuda:
        return "torch"
    return ATTENTION_IMPLEMENTATION


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    attn_mask=None,
    scale=None,
    compatibility_mode=False,
):
    if compatibility_mode or (attn_mask is not None):
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale)
    else:
        impl = resolve_implementation(q)
        if impl == "flash_attention_3":
            return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif impl == "flash_attention_2":
            return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif impl == "sage_attention":
            return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif impl == "xformers":
            return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        else:
            return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
