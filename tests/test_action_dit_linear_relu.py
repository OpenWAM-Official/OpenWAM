"""ActionDiT ``attn_kernel`` switch tests.

Covers the Phase 3 ActionDiT changes from
plans/sana_mot_integration_plan.md §3.2-A:

- Default ``attn_kernel="softmax"`` preserves the existing post_state contract
  (no ``q_unrot`` / ``k_unrot`` / ``uses_linear_attn`` keys → Wan/Cosmos25
  callers and SDPA-based :class:`MoTJointDriver` see no change).
- ``attn_kernel="linear_relu"`` mirrors SANA's ``LiteLAReLURope`` preprocessing:
  RMSNorm → ReLU → RoPE, and stashes the unrotated ReLU'd Q/K in
  ``post_state`` in MoT ``(B, S, H*D)`` layout so
  :class:`SanaMoTJointDriver._mixed_attention` can build the SANA dual-track
  denominator.
- Invalid kernel strings raise at ``__init__`` time (fail fast — silent
  fallback to ``softmax`` would corrupt cross-modality inner products in MoT).
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.action_backbone.components import rope_apply_1d
from openwam.model.action_backbone.joint_action_dit import ActionDiT
from openwam.model.base import ActionState


def _make_dit(attn_kernel: str = "softmax", *, num_heads: int = 4, head_dim: int = 16, dim: int = 64) -> ActionDiT:
    return ActionDiT(
        action_dim=8,
        dim=dim,
        ffn_dim=128,
        num_heads=num_heads,
        num_layers=2,
        video_dim=dim,
        bridge_layers=(0, 1),
        variant="joint_self_attn",
        attn_head_dim=head_dim,
        attn_kernel=attn_kernel,
    )


def _make_astate(dit: ActionDiT, *, batch_size: int = 2, t_action: int = 6, seed: int = 0) -> ActionState:
    gen = torch.Generator().manual_seed(seed)
    actions = torch.randn(batch_size, t_action, dit.action_dim, generator=gen)
    timestep = torch.full((batch_size,), 0.3)
    context = torch.randn(batch_size, 3, dit.text_dim, generator=gen)
    return dit.prepare_state(actions, timestep, context=context)


def test_attn_kernel_property_default_softmax():
    """Default ``attn_kernel`` is ``"softmax"`` — no behavior change for old callers."""
    dit = _make_dit()
    assert dit.attn_kernel == "softmax"
    assert dit.attn_eps == 1e-15  # advertised default even on the softmax path


def test_attn_kernel_property_linear_relu():
    """``attn_kernel="linear_relu"`` is reflected on the property."""
    dit = _make_dit(attn_kernel="linear_relu")
    assert dit.attn_kernel == "linear_relu"


def test_attn_kernel_invalid_raises():
    """An unknown kernel name fails at __init__ rather than silently falling back."""
    with pytest.raises(ValueError, match="Unknown attn_kernel"):
        _make_dit(attn_kernel="flash")


def test_softmax_post_state_omits_unrot_qk():
    """Default softmax kernel must not introduce SANA-only keys into post_state.

    Old callers (SDPA :class:`MoTJointDriver`, deploy paths, Wan/Cosmos25 backbones)
    rely on the post_state dict being exactly the 6-key tuple it has always been.
    Adding ``q_unrot`` / ``k_unrot`` / ``uses_linear_attn`` unconditionally would
    leak SANA semantics into non-SANA paths.
    """
    dit = _make_dit(attn_kernel="softmax")
    astate = _make_astate(dit)
    q, k, v, post_state = dit.pre_attn_at_layer(0, astate)

    expected_keys = {"block", "residual_x", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"}
    assert set(post_state.keys()) == expected_keys
    assert "q_unrot" not in post_state
    assert "k_unrot" not in post_state
    assert "uses_linear_attn" not in post_state

    # Shape sanity on the regular outputs.
    B, T, _ = astate.payload.x_action.shape
    H, D = dit.num_heads, dit.head_dim
    assert q.shape == (B, T, H * D)
    assert k.shape == (B, T, H * D)
    assert v.shape == (B, T, H * D)


def test_linear_relu_post_state_exposes_unrot_qk():
    """``linear_relu`` populates post_state with ReLU'd, pre-RoPE Q/K in MoT layout."""
    dit = _make_dit(attn_kernel="linear_relu")
    astate = _make_astate(dit)
    q, k, v, post_state = dit.pre_attn_at_layer(0, astate)

    assert post_state.get("uses_linear_attn") is True
    assert "q_unrot" in post_state and "k_unrot" in post_state

    B, T, _ = astate.payload.x_action.shape
    H, D = dit.num_heads, dit.head_dim
    assert post_state["q_unrot"].shape == (B, T, H * D)
    assert post_state["k_unrot"].shape == (B, T, H * D)
    # Same layout as the rotated outputs the driver also concatenates.
    assert post_state["q_unrot"].shape == q.shape


def test_linear_relu_unrot_is_relu_not_rope():
    """``q_unrot`` / ``k_unrot`` are ReLU(norm_q(Q)) — *not* RoPE'd, not raw, not negated.

    The SANA dual-track form (plans/sana_mot_math_conflict.md §1) requires
    the denominator's track to be the kernel-applied but *un*-rotated Q/K.
    If we accidentally exposed the rotated version instead, the denominator
    would be wrong (rotated K can be negative after the complex rotation,
    breaking the positivity that the ``+ eps`` regularization assumes).

    We recompute the expected ``q_unrot`` from the published preprocessing
    (RMSNorm → reshape → ReLU) and assert exact equality.
    """
    dit = _make_dit(attn_kernel="linear_relu").eval()
    astate = _make_astate(dit, seed=42)

    with torch.no_grad():
        q, k, v, post_state = dit.pre_attn_at_layer(0, astate)

        # Re-derive the expected unrotated-relu Q from the documented sequence.
        payload = astate.payload
        block = dit.blocks[0]
        chunks = (block.modulation.to(dtype=payload.t_mod.dtype) + payload.t_mod).chunk(6, dim=1)
        shift_msa, scale_msa, *_ = chunks
        attn_input = block.self_attn_norm(payload.x_action) * (1 + scale_msa) + shift_msa
        sa = block.self_attn
        q_ref = sa.norm_q(sa.q(attn_input))
        k_ref = sa.norm_k(sa.k(attn_input))
        # Match the (B, n, s, d) → ReLU → (B, s, n*d) pivot order used in the
        # implementation so a reshape-order regression would be caught here.
        from einops import rearrange

        q_ref = rearrange(q_ref, "b s (n d) -> b n s d", n=dit.num_heads)
        k_ref = rearrange(k_ref, "b s (n d) -> b n s d", n=dit.num_heads)
        q_ref = torch.relu(q_ref)
        k_ref = torch.relu(k_ref)
        q_ref_mot = rearrange(q_ref, "b n s d -> b s (n d)", n=dit.num_heads)
        k_ref_mot = rearrange(k_ref, "b n s d -> b s (n d)", n=dit.num_heads)

    torch.testing.assert_close(post_state["q_unrot"], q_ref_mot)
    torch.testing.assert_close(post_state["k_unrot"], k_ref_mot)

    # Negative sanity: q_unrot should be non-negative (it's ReLU output).
    assert (post_state["q_unrot"] >= 0).all()
    assert (post_state["k_unrot"] >= 0).all()


def test_linear_relu_rotated_q_uses_relu_then_rope():
    """The driver-facing rotated Q must equal ``RoPE(ReLU(q))``, not ``RoPE(q)``.

    This pins the ordering RMSNorm → ReLU → RoPE (the SANA convention).
    Reversing them would commute incorrectly — RoPE rotation is linear in
    Q but ReLU is not, so ``ReLU(RoPE(x)) ≠ RoPE(ReLU(x))`` in general.
    """
    dit = _make_dit(attn_kernel="linear_relu").eval()
    astate = _make_astate(dit, seed=11)

    with torch.no_grad():
        q_out, _, _, post_state = dit.pre_attn_at_layer(0, astate)

        # Apply RoPE to the published q_unrot — should match q_out.
        from einops import rearrange

        q_unrot_heads = rearrange(post_state["q_unrot"], "b s (n d) -> b n s d", n=dit.num_heads)
        q_expected_heads = rope_apply_1d(q_unrot_heads, astate.payload.action_freqs)
        q_expected_mot = rearrange(q_expected_heads, "b n s d -> b s (n d)", n=dit.num_heads)

    torch.testing.assert_close(q_out, q_expected_mot)


def test_linear_relu_post_attn_still_works():
    """``post_attn_at_layer`` consumes the dict (including new keys) without crashing.

    Existing ``post_attn_at_layer`` extracts only the 5 AdaLN/residual keys
    out of the dict, so the new ``q_unrot`` / ``k_unrot`` / ``uses_linear_attn``
    fields must not interfere with the post path.
    """
    dit = _make_dit(attn_kernel="linear_relu").eval()
    astate = _make_astate(dit, seed=7)

    with torch.no_grad():
        q, k, v, post_state = dit.pre_attn_at_layer(0, astate)
        # Use a zero attn_out — we only care that the call path is intact.
        attn_out = torch.zeros_like(q)
        astate2 = dit.post_attn_at_layer(0, astate, attn_out, post_state)

    assert astate2.payload.x_action.shape == astate.payload.x_action.shape
    assert torch.isfinite(astate2.payload.x_action).all()
