"""MoT pre/post attention split for the Cosmos3-Edge gen stream.

Splits one decoder layer's gen half at the attention op so the dual-system MoT
driver can mix video and action Q/K/V in a single SDPA call:

- :func:`pre_self_attn` runs norm → GQA projections → per-head norms → rotary,
  prepends the cached und-stream K/V (the layer's ``k_norm_und_for_gen`` + rope
  already applied by ``run_und_tower``), and **expands the 8 KV heads to the 16
  Q heads** via ``repeat_interleave`` — mathematically identical to SDPA's
  ``enable_gqa`` grouping (query head ``h`` reads kv head ``h // groups``), and
  it keeps the driver's flat ``(B, S, H·D)`` single-``num_heads`` contract
  untouched. Keys are therefore ``prefix_kv_len`` tokens longer than queries;
  the driver widens the joint mask accordingly (``BlockLoopState.prefix_kv_*``).
- :func:`post_self_attn` applies the output projection, residual add, and the
  gen MLP half. It mutates nothing but the returned hidden states, satisfying
  the driver's ``_step_checkpointed`` invariant.

Parity contract: ``post(pre(x) → SDPA) == dit_forward._gen_block_forward(x)``
at fp32 tolerance (see tests/test_cosmos3_joint_self_attn.py).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.cosmos3.dit_forward import _apply_rope

__all__ = ["pre_self_attn", "post_self_attn"]


def pre_self_attn(
    layer,
    gen_seq: Tensor,
    k_und: Tensor,
    v_und: Tensor,
    cos_gen: Tensor,
    sin_gen: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, dict]:
    """Gen-half first stage. Returns ``(q, k, v, post_state)`` with
    ``q (B, S, H·D)`` and ``k/v (B, L_und + S, H·D)`` (KV heads pre-expanded)."""
    b, s, _ = gen_seq.shape
    attn = layer.self_attn
    n_heads = attn.num_attention_heads
    kv_heads = attn.num_key_value_heads
    head_dim = attn.head_dim
    groups = attn.num_key_value_groups

    normed = layer.input_layernorm_moe_gen(gen_seq)
    q = attn.norm_added_q(attn.add_q_proj(normed).view(b, s, n_heads, head_dim))
    k = attn.norm_added_k(attn.add_k_proj(normed).view(b, s, kv_heads, head_dim))
    v = attn.add_v_proj(normed).view(b, s, kv_heads, head_dim)
    q = _apply_rope(q, cos_gen, sin_gen)
    k = _apply_rope(k, cos_gen, sin_gen)

    all_k = torch.cat([k_und.to(k.dtype), k], dim=1)
    all_v = torch.cat([v_und.to(v.dtype), v], dim=1)
    if groups > 1:
        all_k = all_k.repeat_interleave(groups, dim=2)
        all_v = all_v.repeat_interleave(groups, dim=2)

    post_state = {"layer": layer, "residual": gen_seq}
    return (
        q.reshape(b, s, n_heads * head_dim),
        all_k.reshape(b, all_k.shape[1], n_heads * head_dim),
        all_v.reshape(b, all_v.shape[1], n_heads * head_dim),
        post_state,
    )


def post_self_attn(attn_out: Tensor, post_state: dict) -> Tensor:
    """Gen-half second stage: output projection → residual → gen MLP."""
    layer = post_state["layer"]
    residual = post_state["residual"]
    x = residual + layer.self_attn.to_add_out(attn_out)
    return x + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(x))


def state_pre_attn(net, block_id: int, state: BlockLoopState) -> Tuple[Tensor, Tensor, Tensor, dict]:
    """Adapter from BlockLoopState to :func:`pre_self_attn`."""
    layer = net.layers[block_id]
    k_und, v_und = state.extras["und_kv"][block_id]
    return pre_self_attn(
        layer,
        state.hidden_states,
        k_und,
        v_und,
        state.extras["cos_gen"],
        state.extras["sin_gen"],
    )


def state_post_attn(state: BlockLoopState, attn_out: Tensor, post_state: dict) -> BlockLoopState:
    """Adapter writing :func:`post_self_attn`'s result back onto the state."""
    state.hidden_states = post_self_attn(attn_out, post_state)
    return state
