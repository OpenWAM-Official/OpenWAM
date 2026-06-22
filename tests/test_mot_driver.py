"""Unit tests for :class:`DualSystemMoTDriver` — the joint-attention coordinator.

Covers:
- Construction-time validation (num_layers / num_heads / head_dim parity,
  mask-mode whitelist).
- Mixed-attention shape and dtype handling.
- ``run_joint_loop`` drives both backbones through the matching number of
  pre/post_attn_at_layer calls.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import ActionState
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    ISOLATED,
    MUTUAL,
    VIDEO_SEES_ACTION,
)
from openwam.model.video_backbone.base import BlockLoopState
from tests.test_openwam_trainer import _MockVideoBackbone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action_dit(*, dim: int = 32, num_heads: int = 4, num_layers: int = 2, action_dim: int = 7) -> ActionDiT:
    return ActionDiT(
        action_dim=action_dim,
        dim=dim,
        ffn_dim=dim * 2,
        num_heads=num_heads,
        num_layers=num_layers,
        video_dim=dim,
        bridge_layers=tuple(range(num_layers)),
        variant="joint_self_attn",
        text_dim=dim,
    )


def _make_action_context(ab: ActionDiT, B: int, T_ctx: int = 4, *, generator: torch.Generator | None = None):
    context = torch.randn(B, T_ctx, ab.text_dim, generator=generator)
    context_mask = torch.ones(B, T_ctx, dtype=torch.bool)
    return context, context_mask


def _make_states(
    vb: _MockVideoBackbone,
    ab: ActionDiT,
    *,
    B: int = 2,
    s_video: int = 9,
    s_action: int = 5,
) -> Tuple[BlockLoopState, ActionState]:
    """Build minimal vstate/astate ready for driver.step."""
    dim = vb.dim
    vstate = BlockLoopState(
        hidden_states=torch.randn(B, s_video, dim),
        time_mod=torch.randn(B, 6, dim),
        rope_freqs=torch.zeros(s_video, 1, 1),  # unused by mock
        context=torch.zeros(B, 1, dim),
        grid_frames=s_video,
        grid_height=1,
        grid_width=1,
    )
    actions = torch.randn(B, s_action, ab.action_dim)
    timestep = torch.randn(B)
    context, context_mask = _make_action_context(ab, B)
    astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
    return vstate, astate


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_driver_validates_num_layers():
    vb = _MockVideoBackbone(dim=32, num_layers=4, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=3)
    with pytest.raises(ValueError, match="num_layers"):
        DualSystemMoTDriver(vb, ab)


def test_driver_validates_num_heads():
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=8)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    with pytest.raises(ValueError, match="num_heads"):
        DualSystemMoTDriver(vb, ab)


def test_driver_validates_head_dim():
    # Different per-head sizes: vb has head_dim=8, ab has head_dim=16.
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=64, num_heads=4, num_layers=2)
    with pytest.raises(ValueError, match="head_dim"):
        DualSystemMoTDriver(vb, ab)


def test_driver_rejects_unknown_mask_mode():
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    with pytest.raises(ValueError, match="attention_mask_mode"):
        DualSystemMoTDriver(vb, ab, attention_mask_mode="causal")


# ---------------------------------------------------------------------------
# Step / run_joint_loop
# ---------------------------------------------------------------------------


def test_driver_step_calls_both_backbones():
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.eval()
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)

    vstate, astate = _make_states(vb, ab, B=2, s_video=9, s_action=5)

    # Counters via monkey-patch.
    pre_calls = {"v": 0, "a": 0}
    post_calls = {"v": 0, "a": 0}
    orig_v_pre = vb.pre_attn_at_layer
    orig_v_post = vb.post_attn_at_layer
    orig_a_pre = ab.pre_attn_at_layer
    orig_a_post = ab.post_attn_at_layer

    def _wrap(fn, key, bucket):
        def inner(*args, **kw):
            bucket[key] += 1
            return fn(*args, **kw)

        return inner

    vb.pre_attn_at_layer = _wrap(orig_v_pre, "v", pre_calls)
    vb.post_attn_at_layer = _wrap(orig_v_post, "v", post_calls)
    ab.pre_attn_at_layer = _wrap(orig_a_pre, "a", pre_calls)
    ab.post_attn_at_layer = _wrap(orig_a_post, "a", post_calls)

    try:
        with torch.no_grad():
            driver.run_joint_loop(vstate, astate)
    finally:
        vb.pre_attn_at_layer = orig_v_pre
        vb.post_attn_at_layer = orig_v_post
        ab.pre_attn_at_layer = orig_a_pre
        ab.post_attn_at_layer = orig_a_post

    assert pre_calls == {"v": vb.num_layers, "a": ab.num_layers}
    assert post_calls == {"v": vb.num_layers, "a": ab.num_layers}


def test_driver_dtype_mismatch_raises():
    """Driver must abort with a clear error if vb and ab produce different-dtype Q/K/V."""
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.eval()
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)

    vstate, astate = _make_states(vb, ab, B=1, s_video=4, s_action=3)
    # Force a mismatch by making the *video* mock stream emit bf16 Q/K/V while
    # the action stream stays fp32. The mock's pre_attn_at_layer just returns
    # state.hidden_states for q/k/v, so casting state.hidden_states to bf16 is enough.
    vstate.hidden_states = vstate.hidden_states.to(torch.bfloat16)

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        with torch.no_grad():
            driver.step(0, vstate, astate)


def test_driver_attention_mask_action_sees_video_layout():
    """action_sees_video mask: a→a + a→v True, v→a False, v→v from vb.video_attention_mask_mode."""
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    # _MockVideoBackbone defaults to bidirectional v↔v (full True).
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    driver = DualSystemMoTDriver(vb, ab, attention_mask_mode=ACTION_SEES_VIDEO)

    Sv, Sa = 6, 3
    mask = driver._build_attention_mask(s_video=Sv, s_action=Sa, video_tokens_per_frame=Sv, device=torch.device("cpu"))
    assert mask is not None
    assert mask.shape == (Sv + Sa, Sv + Sa)
    assert mask.dtype == torch.bool
    # v↔v: full (mock backbone reports bidirectional)
    assert mask[:Sv, :Sv].all()
    # a↔a: full
    assert mask[Sv:, Sv:].all()
    # a→v: full
    assert mask[Sv:, :Sv].all()
    # v→a: blocked
    assert not mask[:Sv, Sv:].any()


class _VBFirstFrameDouble(_MockVideoBackbone):
    """Video backbone double reporting first_frame_causal v↔v."""

    @property
    def video_attention_mask_mode(self):
        return "first_frame_causal"

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        ff = min(video_tokens_per_frame, video_seq_len)
        mask[:ff, ff:] = False
        return mask


@pytest.mark.parametrize(
    "mode, v_sees_a, a_sees_all_v",
    [
        (MUTUAL, True, True),
        (ACTION_SEES_VIDEO, False, True),
        (VIDEO_SEES_ACTION, True, False),
        (ISOLATED, False, False),
    ],
)
def test_driver_cross_modal_mode_layouts(mode, v_sees_a, a_sees_all_v):
    """Four modes share two invariants (v↔v from vb, a↔a full) and differ only
    in v→a (excluding first-frame rows) and a→v (all video vs first frame only)."""
    vb = _VBFirstFrameDouble(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    driver = DualSystemMoTDriver(vb, ab, attention_mask_mode=mode)

    Sv, Sa, ff = 6, 3, 2
    mask = driver._build_attention_mask(s_video=Sv, s_action=Sa, video_tokens_per_frame=ff, device=torch.device("cpu"))
    # a↔a always full.
    assert mask[Sv:, Sv:].all()

    # v→a: first-frame rows never see action; later rows match v_sees_a.
    assert not mask[:ff, Sv:].any(), "first-frame video rows must never see action"
    if v_sees_a:
        assert mask[ff:Sv, Sv:].all()
    else:
        assert not mask[ff:Sv, Sv:].any()

    # a→v: all video, or only the first frame.
    if a_sees_all_v:
        assert mask[Sv:, :Sv].all()
    else:
        assert mask[Sv:, :ff].all()
        assert not mask[Sv:, ff:Sv].any()


def test_driver_joint_mask_first_frame_causal():
    """When vb reports first_frame_causal, the action_sees_video mask's v↔v block
    matches FastWAM's layout: first frame only sees itself, others see all video."""
    vb = _VBFirstFrameDouble(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    driver = DualSystemMoTDriver(vb, ab, attention_mask_mode=ACTION_SEES_VIDEO)

    Sv, Sa, tokens_per_frame = 6, 3, 2
    mask = driver._build_attention_mask(
        s_video=Sv, s_action=Sa, video_tokens_per_frame=tokens_per_frame, device=torch.device("cpu")
    )
    # First frame's video query rows (rows 0,1) only see first-frame video keys (cols 0,1).
    assert mask[:tokens_per_frame, tokens_per_frame:Sv].sum() == 0
    assert mask[:tokens_per_frame, :tokens_per_frame].all()
    # Later frames see all video.
    assert mask[tokens_per_frame:Sv, :Sv].all()
    # action sees all of video and itself; video doesn't see action.
    assert mask[Sv:, :Sv].all()
    assert mask[Sv:, Sv:].all()
    assert not mask[:Sv, Sv:].any()


# ---------------------------------------------------------------------------
# Numerical isolation: with no v↔a coupling, each modality matches a
# stand-alone forward pass through its own pre/post_attn_at_layer + attention.
# ---------------------------------------------------------------------------


def test_driver_forwards_action_stream_unchanged_when_attention_is_identity():
    """Sanity: with mock vb's identity-style attention, the action stream still
    arrives at extract_prediction with finite values. (Real numerical isolation
    is gated on a true backbone where v↔a masking can be exercised.)"""
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.eval()
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)

    vstate, astate = _make_states(vb, ab, B=1, s_video=4, s_action=3)
    with torch.no_grad():
        vstate2, astate2 = driver.run_joint_loop(vstate, astate)
    assert vstate2.hidden_states.shape == (1, 4, 32)
    pred = ab.extract_prediction(astate2)
    assert pred.shape == (1, 3, ab.action_dim)
    assert torch.isfinite(pred).all()


def test_driver_joint_mask_blocks_video_to_action():
    """FastWAM-Joint property: video output is independent of action input.

    With ``attention_mask_mode='action_sees_video'`` the mask sets ``v→a = False`` so
    video queries cannot attend to action keys. Therefore changing the
    action input must not change the video output (within numerical noise).
    This is the prerequisite for video-KV prefill in inference.
    """
    torch.manual_seed(0)
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.eval()
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False, attention_mask_mode=ACTION_SEES_VIDEO)

    # Two runs that share video input but use different action inputs.
    B, s_video, s_action = 1, 4, 3
    video_x = torch.randn(B, s_video, vb.dim)

    def _run(action_seed: int) -> torch.Tensor:
        actions = torch.randn(B, s_action, ab.action_dim, generator=torch.Generator().manual_seed(action_seed))
        timestep = torch.zeros(B)  # deterministic across runs
        context, context_mask = _make_action_context(ab, B)
        astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        vstate = BlockLoopState(
            hidden_states=video_x.clone(),
            time_mod=torch.zeros(B, 6, vb.dim),
            rope_freqs=torch.zeros(s_video, 1, 1),
            context=torch.zeros(B, 1, vb.dim),
            grid_frames=s_video,
            grid_height=1,
            grid_width=1,
        )
        with torch.no_grad():
            vstate2, _ = driver.run_joint_loop(vstate, astate)
        return vstate2.hidden_states

    out_a = _run(action_seed=1)
    out_b = _run(action_seed=2)
    # Video output must be invariant to the action input → same x going in,
    # same x going out across both runs.
    assert torch.allclose(out_a, out_b, atol=1e-6), (
        f"joint mask leaked v→a coupling; max diff = {(out_a - out_b).abs().max().item()}"
    )


def test_driver_handles_heterogeneous_hidden_dim_end_to_end():
    """FastWAM-Joint: action hidden_dim=32, video hidden_dim=64. The driver
    must run a single mixed attention via the shared num_heads*attn_head_dim
    space and split outputs back to each modality's residual width."""
    torch.manual_seed(0)
    vb = _MockVideoBackbone(dim=64, num_layers=2, num_heads=4)
    # Action: hidden=32, attn_hidden = 4*16 = 64 (matches video's per-head layout).
    ab = ActionDiT(
        action_dim=7,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=2,
        video_dim=64,
        bridge_layers=(0, 1),
        variant="joint_self_attn",
        attn_head_dim=16,
        text_dim=32,
    )
    ab.eval()
    # vb head_dim is 64/4=16 — matches ab.attn_head_dim. Driver should accept.
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)

    B, s_video, s_action = 1, 4, 3
    actions = torch.randn(B, s_action, ab.action_dim)
    timestep = torch.zeros(B)
    context, context_mask = _make_action_context(ab, B)
    astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
    vstate = BlockLoopState(
        hidden_states=torch.randn(B, s_video, vb.dim),
        time_mod=torch.zeros(B, 6, vb.dim),
        rope_freqs=torch.zeros(s_video, 1, 1),
        context=torch.zeros(B, 1, vb.dim),
        grid_frames=s_video,
        grid_height=1,
        grid_width=1,
    )
    with torch.no_grad():
        vstate2, astate2 = driver.run_joint_loop(vstate, astate)
    # Each modality stays at its own hidden width post-split.
    assert vstate2.hidden_states.shape == (B, s_video, vb.dim)
    pred = ab.extract_prediction(astate2)
    assert pred.shape == (B, s_action, ab.action_dim)
    assert torch.isfinite(pred).all()


def test_driver_mutual_mask_does_couple_video_to_action():
    """Negative control: under ``mutual`` mode, the same setup as
    ``test_driver_joint_mask_blocks_video_to_action`` should produce a different
    video output for different actions (proving the joint test isn't a no-op).

    With ``grid_frames=4`` and a 1×1 grid, tokens_per_frame=1, so only the first
    video row is excluded from seeing action — rows 1..3 do see it.
    """
    torch.manual_seed(0)
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.eval()
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False, attention_mask_mode=MUTUAL)

    B, s_video, s_action = 1, 4, 3
    video_x = torch.randn(B, s_video, vb.dim)

    def _run(action_seed: int) -> torch.Tensor:
        actions = torch.randn(B, s_action, ab.action_dim, generator=torch.Generator().manual_seed(action_seed))
        timestep = torch.zeros(B)
        context, context_mask = _make_action_context(ab, B)
        astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        vstate = BlockLoopState(
            hidden_states=video_x.clone(),
            time_mod=torch.zeros(B, 6, vb.dim),
            rope_freqs=torch.zeros(s_video, 1, 1),
            context=torch.zeros(B, 1, vb.dim),
            grid_frames=s_video,
            grid_height=1,
            grid_width=1,
        )
        with torch.no_grad():
            vstate2, _ = driver.run_joint_loop(vstate, astate)
        return vstate2.hidden_states

    out_a = _run(action_seed=1)
    out_b = _run(action_seed=2)
    # In mutual mode the video does see the action — outputs differ.
    assert not torch.allclose(out_a, out_b, atol=1e-4), (
        "mutual mask: changing action did not change video output — the test mock or driver pipeline is broken."
    )


def test_action_uses_own_projected_text_context():
    """ActionDiT must use its own projected raw context from prepare_state.

    Two runs sharing every input except action raw context must produce
    different action hidden states. ``vstate.context`` is intentionally not
    used for action cross-attn anymore.
    """
    torch.manual_seed(0)
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.eval()
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)

    B, s_video, s_action, T_ctx = 1, 4, 3, 5
    actions = torch.randn(B, s_action, ab.action_dim)
    timestep = torch.zeros(B)
    video_x = torch.randn(B, s_video, vb.dim)

    def _run(context: torch.Tensor) -> torch.Tensor:
        context_mask = torch.ones(B, context.shape[1], dtype=torch.bool)
        astate = ab.prepare_state(actions.clone(), timestep, context=context, context_mask=context_mask)
        vstate = BlockLoopState(
            hidden_states=video_x.clone(),
            time_mod=torch.zeros(B, 6, vb.dim),
            rope_freqs=torch.zeros(s_video, 1, 1),
            context=torch.zeros(B, T_ctx, vb.dim),
            grid_frames=s_video,
            grid_height=1,
            grid_width=1,
        )
        with torch.no_grad():
            _, astate2 = driver.run_joint_loop(vstate, astate)
        return astate2.payload.x_action

    ctx_a = torch.randn(B, T_ctx, ab.text_dim, generator=torch.Generator().manual_seed(11))
    ctx_b = torch.randn(B, T_ctx, ab.text_dim, generator=torch.Generator().manual_seed(22))
    out_a = _run(ctx_a)
    out_b = _run(ctx_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5), (
        "Action hidden state is invariant to raw action context — the action-owned "
        "context embedding/cross-attn branch is not being driven."
    )


# ---------------------------------------------------------------------------
# Step-level gradient checkpointing parity
# ---------------------------------------------------------------------------


def _build_run_inputs(vb, ab, *, B, s_video, s_action, seed):
    """Reproduce identical (vstate, astate) inputs from a seed."""
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(B, s_action, ab.action_dim, generator=g)
    timestep = torch.zeros(B)
    context, context_mask = _make_action_context(ab, B, generator=g)
    astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
    vstate = BlockLoopState(
        hidden_states=torch.randn(B, s_video, vb.dim, generator=g, requires_grad=True),
        time_mod=torch.zeros(B, 6, vb.dim),
        rope_freqs=torch.zeros(s_video, 1, 1),
        context=torch.randn(B, 4, vb.dim, generator=g),
        grid_frames=s_video,
        grid_height=1,
        grid_width=1,
    )
    return vstate, astate


def test_driver_step_checkpoint_matches_non_checkpoint_forward_and_backward():
    """Step-level activation checkpointing must be a pure memory/compute
    trade — same forward output, same gradients on both backbones, same
    gradient on ``vstate.hidden_states``. Guards the joint_self_attn OOM fix.
    """
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.train()  # _step_checkpointed only fires when ab.training
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=True)

    B, s_video, s_action = 2, 4, 3

    # ---- Run 1: vanilla path
    ab.zero_grad(set_to_none=True)
    vstate1, astate1 = _build_run_inputs(vb, ab, B=B, s_video=s_video, s_action=s_action, seed=7)
    leaf_v1 = (
        vstate1.hidden_states
    )  # leaf tensor with requires_grad=True; ``vstate.hidden_states`` is rebound during the loop
    vstate1, astate1 = driver.run_joint_loop(vstate1, astate1, use_gradient_checkpointing=False)
    loss1 = vstate1.hidden_states.float().pow(2).sum() + astate1.payload.x_action.float().pow(2).sum()
    loss1.backward()
    grad_q1 = ab.blocks[0].self_attn.q.weight.grad.detach().clone()
    grad_ffn1 = ab.blocks[0].ffn[0].weight.grad.detach().clone()
    grad_x1 = leaf_v1.grad.detach().clone()

    # ---- Run 2: step-level checkpointing
    ab.zero_grad(set_to_none=True)
    vstate2, astate2 = _build_run_inputs(vb, ab, B=B, s_video=s_video, s_action=s_action, seed=7)
    leaf_v2 = vstate2.hidden_states
    vstate2, astate2 = driver.run_joint_loop(vstate2, astate2, use_gradient_checkpointing=True)
    loss2 = vstate2.hidden_states.float().pow(2).sum() + astate2.payload.x_action.float().pow(2).sum()
    loss2.backward()
    grad_q2 = ab.blocks[0].self_attn.q.weight.grad.detach().clone()
    grad_ffn2 = ab.blocks[0].ffn[0].weight.grad.detach().clone()
    grad_x2 = leaf_v2.grad.detach().clone()

    # Forward parity: identical inputs → identical outputs.
    assert torch.allclose(vstate1.hidden_states, vstate2.hidden_states, rtol=1e-5, atol=1e-6), (
        f"forward video mismatch; max diff = {(vstate1.hidden_states - vstate2.hidden_states).abs().max().item()}"
    )
    assert torch.allclose(astate1.payload.x_action, astate2.payload.x_action, rtol=1e-5, atol=1e-6), (
        f"forward action mismatch; "
        f"max diff = {(astate1.payload.x_action - astate2.payload.x_action).abs().max().item()}"
    )
    # Backward parity: gradients on ab params and vstate.hidden_states leaf must match.
    assert torch.allclose(grad_q1, grad_q2, rtol=1e-4, atol=1e-6), (
        f"action self_attn.q grad mismatch; max diff = {(grad_q1 - grad_q2).abs().max().item()}"
    )
    assert torch.allclose(grad_ffn1, grad_ffn2, rtol=1e-4, atol=1e-6), (
        f"action ffn[0] grad mismatch; max diff = {(grad_ffn1 - grad_ffn2).abs().max().item()}"
    )
    assert torch.allclose(grad_x1, grad_x2, rtol=1e-4, atol=1e-6), (
        f"vstate.hidden_states grad mismatch; max diff = {(grad_x1 - grad_x2).abs().max().item()}"
    )


def test_driver_step_checkpoint_no_op_in_eval_mode():
    """``_step_checkpointed`` keys off ``ab.training``; in eval mode the flag
    is silently ignored so inference paths never take the checkpoint trip.
    """
    vb = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    ab = _make_action_dit(dim=32, num_heads=4, num_layers=2)
    ab.eval()
    driver = DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)

    vstate1, astate1 = _build_run_inputs(vb, ab, B=1, s_video=4, s_action=3, seed=11)
    vstate2, astate2 = _build_run_inputs(vb, ab, B=1, s_video=4, s_action=3, seed=11)

    with torch.no_grad():
        vstate1, astate1 = driver.run_joint_loop(vstate1, astate1, use_gradient_checkpointing=False)
        vstate2, astate2 = driver.run_joint_loop(vstate2, astate2, use_gradient_checkpointing=True)

    assert torch.allclose(vstate1.hidden_states, vstate2.hidden_states, atol=1e-6)
    assert torch.allclose(astate1.payload.x_action, astate2.payload.x_action, atol=1e-6)
