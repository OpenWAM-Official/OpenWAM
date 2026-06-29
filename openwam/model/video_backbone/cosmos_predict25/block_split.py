"""CosmosPredict25 Block decomposition for joint self-attention (§17).

The upstream ``Block.forward``
(``third_party/cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py:1257``)
is monolithic. To participate in :class:`MoTJointDriver`'s mixed attention we
split it at the self-attention boundary:

- **pre half**: extras add → modulation → norm → flatten 5D→3D → Q/K/V
  projection → q_norm / k_norm / v_norm → RoPE (Q and K only, self-attention
  semantics). Returns ``(q, k, v, post_state)`` matching Wan's
  :meth:`pre_attn_at_layer` contract.
- **post half**: receive the mixed-attention result
  (pre-output_projection) → ``output_proj`` + dropout → reshape 3D→5D →
  residual + ``gate_self_attn`` → cross-attn sublayer → MLP sublayer. Returns
  the updated 5D hidden state.

Both halves call upstream submodules directly
(``block.layer_norm_*``, ``block.adaln_modulation_*``,
``block.self_attn.{compute_qkv, output_proj, output_dropout}``,
``block.cross_attn``, ``block.mlp``). This keeps the split numerically
equivalent to the monolithic forward (locked in by the GPU parity test in
``tests/test_cosmos_predict25_joint_self_attn.py``) but pins us to the upstream
attribute names. The parity test is the canary for any submodule pin bump.

The ``use_wan_fp32_strategy`` autocast branch on the upstream block
(line 1270) is intentionally omitted: the 2B Stage-c config sets
``use_wan_fp32_strategy=False``, so the autocast degenerates to a no-op on
our path. If a future variant flips this on, mirror the autocast here.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from einops import rearrange
from torch import Tensor


def _compute_modulation(block, emb_B_T_D: Tensor, adaln_lora_B_T_3D: Optional[Tensor]) -> dict:
    """Compute the nine ``(B, T, 1, 1, D)`` AdaLN modulation tensors for one block.

    Mirrors upstream lines 1271-1301: three sublayers (self-attn / cross-attn /
    MLP), each producing ``(shift, scale, gate)`` chunks. With
    ``use_adaln_lora=True`` (the only path CosmosPredict25-2B uses today), the LoRA
    output is summed with ``adaln_lora_B_T_3D`` BEFORE chunking.
    """
    if getattr(block, "use_adaln_lora", False):
        if adaln_lora_B_T_3D is None:
            raise ValueError("adaln_lora_B_T_3D is required when block.use_adaln_lora=True")
        msa = (block.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        mca = (block.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        mmlp = (block.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
    else:
        msa = block.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
        mca = block.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
        mmlp = block.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

    def _expand(t: Tensor) -> Tensor:
        return rearrange(t, "b t d -> b t 1 1 d")

    return {
        "shift_self_attn": _expand(msa[0]),
        "scale_self_attn": _expand(msa[1]),
        "gate_self_attn": _expand(msa[2]),
        "shift_cross_attn": _expand(mca[0]),
        "scale_cross_attn": _expand(mca[1]),
        "gate_cross_attn": _expand(mca[2]),
        "shift_mlp": _expand(mmlp[0]),
        "scale_mlp": _expand(mmlp[1]),
        "gate_mlp": _expand(mmlp[2]),
    }


def _adaln_modulate(x_5D: Tensor, norm_layer: Any, scale_5D: Tensor, shift_5D: Tensor) -> Tensor:
    """Upstream ``Block._fn``: ``norm_layer(x) * (1 + scale) + shift``.

    ``type_as`` matches the upstream cast (line 1291-1301) so the modulation
    tensors take the hidden-state dtype, not the SiLU/Linear output dtype.
    """
    return norm_layer(x_5D) * (1 + scale_5D.type_as(x_5D)) + shift_5D.type_as(x_5D)


def pre_self_attn(
    block: Any,
    x_B_T_H_W_D: Tensor,
    emb_B_T_D: Tensor,
    adaln_lora_B_T_3D: Optional[Tensor],
    rope_emb_L_1_1_D: Optional[Tensor],
    extra_per_block_pos_emb: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, dict]:
    """Pre half of one Cosmos Block — produces Q/K/V for joint attention.

    Returns ``(q, k, v, post_state)`` with ``q/k/v`` shaped ``(B, T·H·W, H·D)``
    (heads inlined into the trailing dim, matching the MoT driver contract).
    ``post_state`` carries every tensor :func:`post_self_attn` needs.
    """
    if extra_per_block_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

    mod = _compute_modulation(block, emb_B_T_D, adaln_lora_B_T_3D)
    normalized = _adaln_modulate(
        x_B_T_H_W_D,
        block.layer_norm_self_attn,
        mod["scale_self_attn"],
        mod["shift_self_attn"],
    )
    _B, T, H, W, _D = normalized.shape
    flat = rearrange(normalized, "b t h w d -> b (t h w) d")

    # ``compute_qkv`` applies q_proj / k_proj / v_proj + q_norm / k_norm /
    # v_norm + RoPE (on Q, K — self-attn) and returns ``(B, S, n_heads,
    # head_dim)`` tensors. Re-collapse heads so the driver's ``b s (n d)``
    # rearrange in ``_mixed_attention`` is well-formed.
    q_4d, k_4d, v_4d = block.self_attn.compute_qkv(flat, None, rope_emb=rope_emb_L_1_1_D)
    q = rearrange(q_4d, "b s h d -> b s (h d)")
    k = rearrange(k_4d, "b s h d -> b s (h d)")
    v = rearrange(v_4d, "b s h d -> b s (h d)")

    post_state = {
        "block": block,
        "T": T,
        "H": H,
        "W": W,
        "residual_x_5d": x_B_T_H_W_D,
        "gate_self_attn": mod["gate_self_attn"],
        "shift_cross_attn": mod["shift_cross_attn"],
        "scale_cross_attn": mod["scale_cross_attn"],
        "gate_cross_attn": mod["gate_cross_attn"],
        "shift_mlp": mod["shift_mlp"],
        "scale_mlp": mod["scale_mlp"],
        "gate_mlp": mod["gate_mlp"],
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
    }
    return q, k, v, post_state


def post_self_attn(attn_out_B_S_HD: Tensor, context: Tensor, post_state: dict) -> Tensor:
    """Post half of one Cosmos Block — finishes self-attn, then cross-attn + MLP.

    Args:
        attn_out_B_S_HD: ``(B, S, H·D)`` mixed-attention result, *before*
            ``self_attn.output_proj``. Matches what
            ``block.self_attn.attn_op`` would return inside the monolithic
            forward (see ``compute_attention`` body) — i.e., the head axis is
            already collapsed back into ``D``.
        context: ``(B, L, context_dim)`` cross-attention K/V source (same as
            ``state.context``).
        post_state: dict produced by :func:`pre_self_attn`.

    Returns:
        ``(B, T, H, W, D)`` updated hidden state — drop-in for ``state.x``.
    """
    block = post_state["block"]
    T = post_state["T"]
    H = post_state["H"]
    W = post_state["W"]
    residual_x_5d = post_state["residual_x_5d"]

    # Self-attn output projection + dropout (mirrors `Attention.compute_attention` tail).
    proj_3d = block.self_attn.output_dropout(block.self_attn.output_proj(attn_out_B_S_HD))
    sa_result_5d = rearrange(proj_3d, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    gate_sa = post_state["gate_self_attn"].type_as(residual_x_5d)
    x_5d = residual_x_5d + gate_sa * sa_result_5d

    # Cross-attn sublayer (upstream lines 1341-1372).
    normalized = _adaln_modulate(
        x_5d,
        block.layer_norm_cross_attn,
        post_state["scale_cross_attn"],
        post_state["shift_cross_attn"],
    )
    cross_out_3d = block.cross_attn(
        rearrange(normalized, "b t h w d -> b (t h w) d"),
        context,
        rope_emb=post_state["rope_emb_L_1_1_D"],
    )
    cross_out_5d = rearrange(cross_out_3d, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    gate_ca = post_state["gate_cross_attn"].type_as(x_5d)
    # Upstream uses ``result * gate + x`` ordering here (line 1372), unlike the
    # self-attn / MLP sublayers which use ``x + gate * result``. We mirror the
    # exact term order so bf16 reductions land on the same bit pattern.
    x_5d = cross_out_5d * gate_ca + x_5d

    # MLP sublayer (upstream lines 1374-1381).
    normalized = _adaln_modulate(
        x_5d,
        block.layer_norm_mlp,
        post_state["scale_mlp"],
        post_state["shift_mlp"],
    )
    mlp_out_5d = block.mlp(normalized)
    gate_mlp = post_state["gate_mlp"].type_as(x_5d)
    x_5d = x_5d + gate_mlp * mlp_out_5d
    return x_5d


__all__ = ["pre_self_attn", "post_self_attn"]
