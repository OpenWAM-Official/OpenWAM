"""SanaMoTJointDriver integration tests.

CPU-only, no SANA dependency — uses lightweight ``FakeLinearReluBackbone``
implementations that satisfy the ``VideoBackbone`` / ``ActionBackbone``
contracts that :class:`SanaMoTJointDriver` actually reads at runtime
(``num_layers``, ``num_heads``, ``head_dim``, ``attn_kernel``,
``pre_attn_at_layer``, ``post_attn_at_layer``, ``build_video_to_video_mask``
when ``run_joint_loop`` is invoked).

Covers Phase 3 acceptance from plans/sana_mot_integration_plan.md §3.4:

- Constructor rejects softmax-kernel backbones (fail-fast — silent fallback
  to SDPA produces mathematically meaningless cross-modality inner products).
- ``_mixed_attention`` requires the ``phi_q`` / ``phi_k`` dual-track inputs.
- ``_step_impl`` forwards Q/K/V from both backbones, runs SANA linear attn,
  and feeds slices back through ``post_attn_at_layer``.
- The chunk path is taken on a monotonic joint mask, and its output equals
  the expanded fallback (regression check that the dispatch is right).
- Backward propagates without NaN through the SANA linear-attn math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn

from openwam.model.architectures.dual_system.sana_linear_attn import (
    _expanded_linear_attn,
    _mask_to_chunk_index,
)
from openwam.model.architectures.dual_system.sana_mot_driver import SanaMoTJointDriver

# ---------------------------------------------------------------------------
# Fake backbones — minimal implementations sufficient for driver-level tests.
# ---------------------------------------------------------------------------


@dataclass
class _FakeVideoState:
    """Stand-in for ``BlockLoopState``. Only fields the driver / fakes touch."""

    x: torch.Tensor
    f: int = 1
    h: int = 1
    w: int = 1
    extras: dict = field(default_factory=dict)


@dataclass
class _FakeActionPayload:
    x_action: torch.Tensor


@dataclass
class _FakeActionState:
    payload: _FakeActionPayload


class _FakeLinearReluBackbone(nn.Module):
    """Backbone that implements the linear-relu pre/post contract using one
    real linear layer per (Q, K, V), so backward can verify gradient flow.

    The Q/K/V projections are shared across all layers — that's not realistic
    but keeps the test parameter count low and lets us assert ``.grad`` is
    non-zero after backward.
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, num_layers: int):
        super().__init__()
        self.dim = dim
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._num_layers = num_layers
        attn_hidden = num_heads * head_dim
        self.q_proj = nn.Linear(dim, attn_hidden, bias=False)
        self.k_proj = nn.Linear(dim, attn_hidden, bias=False)
        self.v_proj = nn.Linear(dim, attn_hidden, bias=False)
        self.o_proj = nn.Linear(attn_hidden, dim, bias=False)

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
    def attn_kernel(self) -> str:
        return "linear_relu"


class _FakeVideoBackbone(_FakeLinearReluBackbone):
    """Plays the VideoBackbone role for SanaMoTJointDriver tests."""

    video_attention_mask_mode = "first_frame_causal"

    def pre_attn_at_layer(
        self, layer_id: int, vstate: _FakeVideoState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = vstate.x
        q = torch.relu(self.q_proj(x))
        k = torch.relu(self.k_proj(x))
        v = self.v_proj(x)
        post = {
            "uses_linear_attn": True,
            "q_unrot": q,
            "k_unrot": k,
        }
        # Numerator track sees an additional "rotation": multiply by a fixed
        # scalar so a bug that confused tilde_* and phi_* would be visible.
        return q * 1.3, k * 1.3, v, post

    def post_attn_at_layer(
        self,
        layer_id: int,
        vstate: _FakeVideoState,
        attn_out: torch.Tensor,
        post_state: Dict[str, Any],
    ) -> _FakeVideoState:
        vstate.x = vstate.x + self.o_proj(attn_out)
        return vstate

    def build_video_to_video_mask(
        self, *, video_seq_len: int, video_tokens_per_frame: int, device: torch.device
    ) -> torch.Tensor:
        # Mirror MoTJointDriver._build_joint_mask's first_frame_causal layout
        # without depending on a real video backbone implementation.
        mask = torch.zeros(video_seq_len, video_seq_len, dtype=torch.bool, device=device)
        ff = min(video_tokens_per_frame, video_seq_len)
        mask[:ff, :ff] = True
        mask[ff:, :video_seq_len] = True
        return mask


class _FakeActionBackbone(_FakeLinearReluBackbone):
    """Plays the ActionBackbone role for SanaMoTJointDriver tests."""

    def pre_attn_at_layer(
        self, layer_id: int, astate: _FakeActionState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = astate.payload.x_action
        q = torch.relu(self.q_proj(x))
        k = torch.relu(self.k_proj(x))
        v = self.v_proj(x)
        post = {
            "uses_linear_attn": True,
            "q_unrot": q,
            "k_unrot": k,
        }
        return q * 0.7, k * 0.7, v, post

    def post_attn_at_layer(
        self,
        layer_id: int,
        astate: _FakeActionState,
        attn_out: torch.Tensor,
        post_state: Dict[str, Any],
    ) -> _FakeActionState:
        astate.payload.x_action = astate.payload.x_action + self.o_proj(attn_out)
        return astate


class _SoftmaxStubBackbone(nn.Module):
    """Minimal stub that advertises ``attn_kernel="softmax"`` for negative
    tests. Doesn't need to be runnable."""

    def __init__(self, num_heads: int = 4, head_dim: int = 16, num_layers: int = 2, kernel: str = "softmax"):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._num_layers = num_layers
        self._kernel = kernel

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
    def attn_kernel(self) -> str:
        return self._kernel

    video_attention_mask_mode = "bidirectional"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_driver(**override) -> SanaMoTJointDriver:
    """Construct a driver with linear_relu fake backbones."""
    kw = dict(num_heads=2, head_dim=8, num_layers=2)
    vb = _FakeVideoBackbone(dim=16, **kw)
    ab = _FakeActionBackbone(dim=24, **kw)  # heterogeneous residual width — supported by MoT
    return SanaMoTJointDriver(vb, ab, attention_mask_mode="joint", **override)


def _build_states(
    driver: SanaMoTJointDriver,
    *,
    s_video: int = 12,
    s_action: int = 4,
    batch_size: int = 1,
    seed: int = 0,
):
    gen = torch.Generator().manual_seed(seed)
    vx = torch.randn(batch_size, s_video, driver.vb.dim, generator=gen)
    ax = torch.randn(batch_size, s_action, driver.ab.dim, generator=gen)
    vstate = _FakeVideoState(x=vx, f=3, h=2, w=2)  # 12 = 3 * 2 * 2
    astate = _FakeActionState(payload=_FakeActionPayload(x_action=ax))
    return vstate, astate


def _joint_mask(s_video: int, s_action: int, video_tokens_per_frame: int) -> torch.Tensor:
    """Mirror :meth:`MoTJointDriver._build_joint_mask` for a first_frame_causal layout."""
    total = s_video + s_action
    mask = torch.zeros(total, total, dtype=torch.bool)
    ff = min(video_tokens_per_frame, s_video)
    mask[:ff, :ff] = True
    mask[ff:s_video, :s_video] = True
    mask[s_video:, s_video:] = True
    mask[s_video:, :s_video] = True
    return mask


# ---------------------------------------------------------------------------
# Constructor / negative tests
# ---------------------------------------------------------------------------


def test_driver_rejects_softmax_video_backbone():
    """Mixing a softmax video backbone with the SANA driver is a config error."""
    vb = _SoftmaxStubBackbone(kernel="softmax")
    ab = _FakeActionBackbone(dim=24, num_heads=4, head_dim=16, num_layers=2)
    with pytest.raises(ValueError, match="video_backbone.attn_kernel='linear_relu'"):
        SanaMoTJointDriver(vb, ab)


def test_driver_rejects_softmax_action_backbone():
    """Mixing a softmax action backbone with the SANA driver is a config error."""
    vb = _FakeVideoBackbone(dim=16, num_heads=4, head_dim=16, num_layers=2)
    ab = _SoftmaxStubBackbone(kernel="softmax", num_heads=4, head_dim=16, num_layers=2)
    with pytest.raises(ValueError, match="action_backbone.attn_kernel='linear_relu'"):
        SanaMoTJointDriver(vb, ab)


def test_driver_accepts_aligned_linear_relu_pair():
    """Both backbones at ``linear_relu`` — constructor succeeds."""
    driver = _build_driver()
    assert isinstance(driver, SanaMoTJointDriver)
    assert driver.num_heads == driver.vb.num_heads == driver.ab.num_heads
    assert driver.head_dim == driver.vb.head_dim == driver.ab.head_dim


def test_generic_mot_compile_skips_linear_relu_backbone():
    """The Wan/Cosmos SDPA compile helper must not wrap SANA's linear-attn driver."""

    from openwam.model.architectures.dual_system.joint_self_attn import _mot_loop_compile_skip_reason

    driver = _build_driver()

    assert "non-softmax" in _mot_loop_compile_skip_reason(driver.vb)


def test_mixed_attention_requires_phi_qk():
    """The 4-arg base signature is rejected — the SANA path is not a silent SDPA fallback."""
    driver = _build_driver()
    H, d = driver.num_heads, driver.head_dim
    B, S = 1, 6
    q = torch.randn(B, S, H * d)
    with pytest.raises(RuntimeError, match="requires phi_q and phi_k"):
        driver._mixed_attention(q, q, q, attn_mask=None)


# ---------------------------------------------------------------------------
# Forward / dispatch tests
# ---------------------------------------------------------------------------


def test_step_impl_forward_shape_no_mask():
    """``_step_impl`` returns updated states with shapes preserved (mask=None → expanded path)."""
    driver = _build_driver()
    vstate, astate = _build_states(driver, s_video=6, s_action=3, batch_size=2)

    orig_vx = vstate.x.clone()
    orig_ax = astate.payload.x_action.clone()

    new_vstate, new_astate = driver._step_impl(0, vstate, astate, attn_mask=None)

    assert new_vstate.x.shape == orig_vx.shape
    assert new_astate.payload.x_action.shape == orig_ax.shape
    # State must actually change — verifies post_attn_at_layer wired through.
    assert not torch.equal(new_vstate.x, orig_vx)
    assert not torch.equal(new_astate.payload.x_action, orig_ax)


def test_step_impl_uses_chunk_path_for_joint_mask(monkeypatch):
    """First-frame-causal joint mask → chunk path. Detect by patching the cumsum entry.

    Eval mode forces ``use_ckpt=False`` so we hit ``_chunked_linear_attn``
    directly rather than the ``_checkpointed`` variant — keeps the patch
    target unambiguous. The checkpointed variant has its own correctness
    test in ``test_sana_linear_attn_math.py``.
    """
    import openwam.model.architectures.dual_system.sana_mot_driver as drv

    driver = _build_driver()
    driver.vb.eval()
    driver.ab.eval()
    s_video, s_action, ff = 12, 4, 4
    vstate, astate = _build_states(driver, s_video=s_video, s_action=s_action)
    mask = _joint_mask(s_video, s_action, video_tokens_per_frame=ff)
    assert _mask_to_chunk_index(mask) is not None, "joint mask should factorize"

    chunk_calls = {"n": 0}
    ckpt_calls = {"n": 0}
    expanded_calls = {"n": 0}

    real_chunked = drv._chunked_linear_attn
    real_ckpt = drv._chunked_linear_attn_checkpointed
    real_expanded = drv._expanded_linear_attn

    def spy_chunked(*a, **kw):
        chunk_calls["n"] += 1
        return real_chunked(*a, **kw)

    def spy_ckpt(*a, **kw):
        ckpt_calls["n"] += 1
        return real_ckpt(*a, **kw)

    def spy_expanded(*a, **kw):
        expanded_calls["n"] += 1
        return real_expanded(*a, **kw)

    monkeypatch.setattr(drv, "_chunked_linear_attn", spy_chunked)
    monkeypatch.setattr(drv, "_chunked_linear_attn_checkpointed", spy_ckpt)
    monkeypatch.setattr(drv, "_expanded_linear_attn", spy_expanded)

    driver._step_impl(0, vstate, astate, attn_mask=mask)
    assert chunk_calls["n"] == 1, "joint mask should route to the cumsum chunk path"
    assert ckpt_calls["n"] == 0, "eval mode should NOT take the checkpointed cumsum variant"
    assert expanded_calls["n"] == 0, "expanded fallback should not be invoked when mask factorizes"


def test_step_impl_uses_chunk_checkpointed_path_when_training(monkeypatch):
    """In training mode with ``mot_checkpoint_mixed_attn=True`` (default), the
    chunked path is the *checkpointed* variant — that's the user-chosen
    Phase 3 memory profile (per-chunk gradient checkpointing)."""
    import openwam.model.architectures.dual_system.sana_mot_driver as drv

    driver = _build_driver()
    driver.vb.train()
    driver.ab.train()
    s_video, s_action, ff = 12, 4, 4
    vstate, astate = _build_states(driver, s_video=s_video, s_action=s_action)
    mask = _joint_mask(s_video, s_action, video_tokens_per_frame=ff)

    ckpt_calls = {"n": 0}
    real = drv._chunked_linear_attn_checkpointed

    def spy(*a, **kw):
        ckpt_calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(drv, "_chunked_linear_attn_checkpointed", spy)

    driver._step_impl(0, vstate, astate, attn_mask=mask)
    assert ckpt_calls["n"] == 1


def test_step_impl_falls_back_to_expanded_for_non_monotonic_mask(monkeypatch):
    """A non-monotonic mask (no factorization) routes to the expanded O(N²) fallback."""
    import openwam.model.architectures.dual_system.sana_mot_driver as drv

    driver = _build_driver()
    driver.vb.eval()
    driver.ab.eval()
    vstate, astate = _build_states(driver, s_video=4, s_action=4)
    n = 8
    # Random non-rectangular mask — guaranteed not block-causal.
    gen = torch.Generator().manual_seed(1)
    mask = (torch.rand(n, n, generator=gen) > 0.4).bool()
    # Ensure no empty rows (translator would also reject those).
    for i in range(n):
        mask[i, i] = True
    assert _mask_to_chunk_index(mask) is None

    chunk_calls = {"n": 0}
    expanded_calls = {"n": 0}
    real_chunked = drv._chunked_linear_attn
    real_expanded = drv._expanded_linear_attn

    def spy_chunked(*a, **kw):
        chunk_calls["n"] += 1
        return real_chunked(*a, **kw)

    def spy_expanded(*a, **kw):
        expanded_calls["n"] += 1
        return real_expanded(*a, **kw)

    monkeypatch.setattr(drv, "_chunked_linear_attn", spy_chunked)
    monkeypatch.setattr(drv, "_expanded_linear_attn", spy_expanded)

    driver._step_impl(0, vstate, astate, attn_mask=mask)
    assert chunk_calls["n"] == 0
    assert expanded_calls["n"] == 1


def test_mixed_attention_chunk_matches_expanded_on_joint_mask():
    """When the joint mask factorizes, chunk and expanded paths produce equal output.

    The driver dispatches to chunk; we manually run the expanded reference and
    require equality. This is the numerical correctness anchor: if a future
    refactor accidentally rotates one of the two tracks differently between
    paths, the test catches it.
    """
    driver = _build_driver()
    s_video, s_action, ff = 12, 4, 4
    vstate, astate = _build_states(driver, s_video=s_video, s_action=s_action, seed=2026)
    mask = _joint_mask(s_video, s_action, video_tokens_per_frame=ff)

    with torch.no_grad():
        vb, ab = driver.vb, driver.ab
        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(0, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(0, astate)
        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        phi_q = torch.cat([vpost["q_unrot"], apost["q_unrot"]], dim=1)
        phi_k = torch.cat([vpost["k_unrot"], apost["k_unrot"]], dim=1)

        out_chunk = driver._mixed_attention(
            q_cat, k_cat, v_cat, mask, phi_q=phi_q, phi_k=phi_k, use_ckpt=False
        )

        # Manually run expanded reference in the same layout the driver uses.
        from einops import rearrange

        H = driver.num_heads
        tilde_q = rearrange(q_cat, "b s (n d) -> b n s d", n=H)
        tilde_k = rearrange(k_cat, "b s (n d) -> b n s d", n=H)
        v_h = rearrange(v_cat, "b s (n d) -> b n s d", n=H)
        pq = rearrange(phi_q, "b s (n d) -> b n s d", n=H)
        pk = rearrange(phi_k, "b s (n d) -> b n s d", n=H)
        out_exp_h = _expanded_linear_attn(tilde_q, tilde_k, v_h, pq, pk, mask=mask, eps=driver.eps)
        out_exp = rearrange(out_exp_h, "b n s d -> b s (n d)", n=H)

    torch.testing.assert_close(out_chunk, out_exp, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def test_step_impl_backward_no_nan_and_flows_to_qkv_projections():
    """One layer of mixed attention backward — gradients are finite and reach the
    Q/K/V projections on both backbones."""
    driver = _build_driver()
    vstate, astate = _build_states(driver, s_video=8, s_action=3, batch_size=1, seed=7)
    mask = _joint_mask(8, 3, video_tokens_per_frame=4)

    driver.vb.train()
    driver.ab.train()

    # Make state tensors require grad so we can also verify input-side gradient.
    vstate.x = vstate.x.detach().requires_grad_(True)
    astate.payload.x_action = astate.payload.x_action.detach().requires_grad_(True)

    new_vstate, new_astate = driver._step_impl(0, vstate, astate, attn_mask=mask)
    loss = new_vstate.x.pow(2).sum() + new_astate.payload.x_action.pow(2).sum()
    loss.backward()

    for name, mod in (("video", driver.vb), ("action", driver.ab)):
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            p = getattr(mod, proj_name).weight
            assert p.grad is not None, f"{name}.{proj_name} got no grad"
            assert torch.isfinite(p.grad).all(), f"{name}.{proj_name} grad has NaN/Inf"
            assert p.grad.abs().sum() > 0, f"{name}.{proj_name} grad is exactly zero"
