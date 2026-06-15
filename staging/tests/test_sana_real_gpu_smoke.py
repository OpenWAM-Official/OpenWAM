'Public implementation.'

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu


ASSET_PATH = Path(os.environ.get("SANA_ASSET_PATH", "/path/to/assets/SANA-Video_2B_480p"))


def _sana_importable() -> bool:
    try:
        import diffusion.model.nets.sana_multi_scale_video  # noqa: F401
    except Exception:
        return False
    return True


def _skip_unless_runnable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not ASSET_PATH.exists():
        pytest.skip(f"SANA-Video 2B asset bundle missing at {ASSET_PATH}.")
    if not (ASSET_PATH / "checkpoints").exists():
        pytest.skip(f"SANA-Video 2B checkpoints missing at {ASSET_PATH}/checkpoints.")
    if not _sana_importable():
        pytest.skip("third_party/Sana not importable (submodule init or timm version.py missing).")


_F_LATENT, _H_LATENT, _W_LATENT = 4, 16, 16
# After patch_size=(1,2,2): f_p=4, h_p=8, w_p=8 → 256 tokens, 64 per frame.
_F_PATCH, _H_PATCH, _W_PATCH = 4, 8, 8
_V_TOKENS_PER_FRAME = _H_PATCH * _W_PATCH
_S_VIDEO = _F_PATCH * _V_TOKENS_PER_FRAME

# SANA caption embedding geometry (Gemma-2-2B hidden dim = 2304, default
# ``model_max_length=300``; we shorten to 8 tokens for the smoke).
_CAPTION_CHANNELS = 2304
_CAPTION_LEN = 8

# 2B SANA model invariants (factory ``SanaMSVideo_2000M_P2_D20`` + 480p
# preset ``linear_head_dim=112``). The SANA upstream "default factory kwargs"
# don't match the published 480p ckpt — the preset is wired in
# ``pipeline_builder._SANA_VIDEO_2B_480P_PRESET`` and triggered by the
# ``config.json:model_name`` field in the asset bundle.
_EXPECTED_DEPTH = 20
_EXPECTED_HIDDEN = 2240
_EXPECTED_HEAD_DIM = 112  # LiteLAReLURope head dim @ 480p
_EXPECTED_SELF_HEADS = _EXPECTED_HIDDEN // _EXPECTED_HEAD_DIM  # = 20


def _build_synthetic_video_inputs(*, batch: int, device, dtype):
    """Random ``(x, timestep, y, mask)`` matching SanaVideoBackbone.prepare."""
    x = torch.randn(batch, 16, _F_LATENT, _H_LATENT, _W_LATENT, device=device, dtype=dtype)
    timestep = torch.tensor([100] * batch, device=device)
    y = torch.randn(batch, 1, _CAPTION_LEN, _CAPTION_CHANNELS, device=device, dtype=dtype)
    mask = torch.ones(batch, 1, 1, _CAPTION_LEN, dtype=torch.int16, device=device)
    return x, timestep, y, mask


# Session-scoped: the 2B ckpt is ~8 GB on disk and loading it dominates the
# test runtime. Keep one copy alive across all tests in this module.
@pytest.fixture(scope="module")
def real_backbone():
    _skip_unless_runnable()
    from openwam.model.video_backbone.sana import SanaVideoBackbone

    bb = SanaVideoBackbone.from_pretrained(
        str(ASSET_PATH),
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    bb.eval()
    return bb


# ----------------------------------------------------------------------
# 1. Real-weights load + geometry
# ----------------------------------------------------------------------


def test_real_2b_geometry(real_backbone):
    """Loaded 2B backbone advertises the geometry SanaMoTJointDriver requires."""
    assert real_backbone.num_layers == _EXPECTED_DEPTH, (
        f"Expected depth {_EXPECTED_DEPTH}, got {real_backbone.num_layers}"
    )
    assert real_backbone.dim == _EXPECTED_HIDDEN
    assert real_backbone.num_heads == _EXPECTED_SELF_HEADS, (
        f"LiteLAReLURope self-attn heads: expected {_EXPECTED_SELF_HEADS}, got {real_backbone.num_heads}"
    )
    assert real_backbone.head_dim == _EXPECTED_HEAD_DIM
    assert real_backbone.attn_kernel == "linear_relu"
    assert real_backbone.video_attention_mask_mode == "first_frame_causal"


# ----------------------------------------------------------------------
# 2. Block-loop forward smoke on real 2B weights
# ----------------------------------------------------------------------


def test_real_2b_block_loop_forward(real_backbone):
    """``prepare → run_block × 20 → finalize`` on synthetic latents.

    Equivalent to running ``SanaMSVideo.forward`` once. We don't compare to
    the upstream forward (covered by Phase 0's mini-config equivalence
    ``test_split_eq_native``); the point here is that strict=False weight
    load doesn't break the per-block invariants on the real 8 GB ckpt.
    """
    x, timestep, y, mask = _build_synthetic_video_inputs(batch=1, device=torch.device("cuda:0"), dtype=torch.bfloat16)
    with torch.no_grad():
        state = real_backbone.prepare(x=x, timestep=timestep, y=y, mask=mask)
        assert state.x.shape == (1, _S_VIDEO, _EXPECTED_HIDDEN), (
            f"prepare produced unexpected x shape: {state.x.shape}, expected (1, {_S_VIDEO}, {_EXPECTED_HIDDEN})"
        )
        for layer_id in range(real_backbone.num_layers):
            state = real_backbone.run_block(layer_id, state)
        out = real_backbone.finalize(state)
    assert out.shape[0] == 1
    assert torch.isfinite(out).all(), "finalize produced non-finite output"


# ----------------------------------------------------------------------
# 3. Real ActionDiT + SanaMoTJointDriver — one joint layer
# ----------------------------------------------------------------------


def _build_action_backbone(*, video_backbone, t_action: int, action_dim: int = 20):
    """Construct an ActionDiT with linear-relu attention paired to the video DiT.

    Matches the auto-resolve that ``DualSystemSelfAttnArchitecture.__init__``
    performs (joint_self_attn.py:46-56): num_dit_layers, num_heads,
    attn_head_dim, and attn_kernel are pulled from ``video_backbone``.
    """
    from openwam.model.action_backbone.joint_action_dit import ActionDiT

    ab = ActionDiT(
        action_dim=action_dim,
        dim=1152,
        ffn_dim=4608,
        num_heads=video_backbone.num_heads,
        num_layers=video_backbone.num_layers,
        video_dim=video_backbone.dim,
        bridge_layers=tuple(range(video_backbone.num_layers)),
        variant="joint_self_attn",
        attn_head_dim=video_backbone.head_dim,
        text_dim=video_backbone.dim,
        attn_kernel="linear_relu",
    )
    ab.to(device=next(video_backbone._dit.parameters()).device, dtype=torch.bfloat16)
    return ab


def _build_joint_inputs(video_backbone, action_backbone, *, t_action: int = 16):
    """Run ``vb.prepare`` + ``ab.prepare_state`` and return both states + mask."""
    device = next(video_backbone._dit.parameters()).device
    dtype = torch.bfloat16

    x, timestep, y, mask = _build_synthetic_video_inputs(batch=1, device=device, dtype=dtype)
    noisy_actions = torch.randn(1, t_action, action_backbone.action_dim, device=device, dtype=dtype)
    action_timestep = torch.tensor([0.3], device=device, dtype=dtype)
    context = torch.randn(1, _CAPTION_LEN, action_backbone.text_dim, device=device, dtype=dtype)
    context_mask = torch.ones(1, _CAPTION_LEN, dtype=torch.bool, device=device)

    vstate = video_backbone.prepare(x=x, timestep=timestep, y=y, mask=mask)
    astate = action_backbone.prepare_state(noisy_actions, action_timestep, context=context, context_mask=context_mask)
    return vstate, astate


def test_real_2b_joint_mot_driver_step(real_backbone):
    """One layer of ``SanaMoTJointDriver._step_impl`` on real 2B weights.

    Verifies:
      - Driver dispatch + construction succeed (kernel alignment check passes).
      - Joint forward returns finite tensors of the right shape on both sides.
      - ActionDiT layer 0 weights receive gradient through the mixed-attention
        join. Video DiT stays parameter-frozen (we keep ``requires_grad=False``
        on the loaded ckpt; verifying the driver doesn't blow that up).
    """
    from openwam.model.architectures.dual_system.sana_mot_driver import (
        SanaMoTJointDriver,
    )

    ab = _build_action_backbone(video_backbone=real_backbone, t_action=16)
    ab.train()

    driver = SanaMoTJointDriver(
        real_backbone,
        ab,
        mot_checkpoint_mixed_attn=False,
        attention_mask_mode="joint",
        video_attention_mask_mode="first_frame_causal",
    )

    vstate, astate = _build_joint_inputs(real_backbone, ab, t_action=16)
    s_video = vstate.x.shape[1]
    s_action = astate.payload.x_action.shape[1]
    assert s_video == _S_VIDEO

    attn_mask = driver._build_joint_mask(
        s_video=s_video,
        s_action=s_action,
        video_tokens_per_frame=_V_TOKENS_PER_FRAME,
        device=vstate.x.device,
    )
    assert attn_mask.shape == (s_video + s_action, s_video + s_action)
    # v↛a half of FastWAM-Joint: video queries don't see action keys.
    assert not attn_mask[:s_video, s_video:].any()
    # a↔a fully connected, a→v fully connected.
    assert attn_mask[s_video:, s_video:].all()
    assert attn_mask[s_video:, :s_video].all()
    # first_frame_causal v↔v: frame 0 query rows do NOT see frames 1+ keys.
    assert not attn_mask[:_V_TOKENS_PER_FRAME, _V_TOKENS_PER_FRAME:s_video].any()

    vstate, astate = driver._step_impl(0, vstate, astate, attn_mask=attn_mask)

    assert vstate.x.shape == (1, s_video, _EXPECTED_HIDDEN)
    assert torch.isfinite(vstate.x).all(), "video state non-finite after MoT layer"
    assert astate.payload.x_action.shape == (1, s_action, ab.dim)
    assert torch.isfinite(astate.payload.x_action).all(), "action state non-finite after MoT layer"

    # Gradient flow: ActionDiT layer 0 attention projections should pick up grad.
    loss = astate.payload.x_action.float().abs().sum() + vstate.x.float().abs().sum()
    loss.backward()
    q_proj = ab.blocks[0].self_attn.q
    assert q_proj.weight.grad is not None, "ActionDiT.q grad missing"
    assert torch.isfinite(q_proj.weight.grad).all()


def test_real_2b_v_not_attn_to_a_invariance(real_backbone):
    """Numerical ablation #3 (plan §4.3): zeroing/perturbing action tokens must
    leave video output unchanged under ``first_frame_causal`` joint mask.

    This is the v↛a half of the FastWAM-Joint mask. If the chunked cumsum
    accidentally let action mass into the video denominator (or numerator),
    this test catches it on real-scale 2B weights.
    """
    from openwam.model.architectures.dual_system.sana_mot_driver import (
        SanaMoTJointDriver,
    )

    ab = _build_action_backbone(video_backbone=real_backbone, t_action=16)
    ab.eval()

    driver = SanaMoTJointDriver(
        real_backbone,
        ab,
        mot_checkpoint_mixed_attn=False,
        attention_mask_mode="joint",
        video_attention_mask_mode="first_frame_causal",
    )

    # Two runs share x/timestep/y/mask and ActionDiT context; only the noisy
    # action tokens differ.
    device = next(real_backbone._dit.parameters()).device
    x, timestep, y, mask = _build_synthetic_video_inputs(batch=1, device=device, dtype=torch.bfloat16)
    context = torch.randn(1, _CAPTION_LEN, ab.text_dim, device=device, dtype=torch.bfloat16)
    context_mask = torch.ones(1, _CAPTION_LEN, dtype=torch.bool, device=device)
    action_timestep = torch.tensor([0.3], device=device, dtype=torch.bfloat16)

    torch.manual_seed(0)
    actions_a = torch.randn(1, 16, ab.action_dim, device=device, dtype=torch.bfloat16)
    torch.manual_seed(1)
    actions_b = torch.randn(1, 16, ab.action_dim, device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def _run(actions):
        vstate = real_backbone.prepare(x=x, timestep=timestep, y=y, mask=mask)
        astate = ab.prepare_state(actions, action_timestep, context=context, context_mask=context_mask)
        attn_mask = driver._build_joint_mask(
            s_video=vstate.x.shape[1],
            s_action=astate.payload.x_action.shape[1],
            video_tokens_per_frame=_V_TOKENS_PER_FRAME,
            device=device,
        )
        # 3 layers is enough to surface any cross-modal contamination; running
        # all 20 needlessly doubles the test runtime.
        for layer_id in range(3):
            vstate, astate = driver._step_impl(layer_id, vstate, astate, attn_mask=attn_mask)
        return vstate.x.detach().clone()

    v_a = _run(actions_a)
    v_b = _run(actions_b)
    torch.testing.assert_close(
        v_a,
        v_b,
        rtol=0.0,
        atol=0.0,
        msg=(
            "Video state changed when action tokens changed under first_frame_causal "
            "joint mask — v↛a invariance broken."
        ),
    )


def test_real_2b_chunked_eq_expanded_on_joint_mask(real_backbone):
    """Phase 3 chunk path numerical equivalence at real scale.

    Phase 2 already covered the math on synthetic ``(B, H, N, d)`` tensors,
    but real 2B weights produce different magnitude / sparsity in the
    post-ReLU q/k than synthetic random init. Verify the cumsum and expanded
    paths agree on real activations.
    """
    from einops import rearrange

    from openwam.model.architectures.dual_system.sana_linear_attn import (
        _chunked_linear_attn,
        _expanded_linear_attn,
        _mask_to_chunk_index,
    )
    from openwam.model.architectures.dual_system.sana_mot_driver import (
        SanaMoTJointDriver,
    )

    ab = _build_action_backbone(video_backbone=real_backbone, t_action=16)
    ab.eval()

    driver = SanaMoTJointDriver(
        real_backbone,
        ab,
        mot_checkpoint_mixed_attn=False,
        attention_mask_mode="joint",
        video_attention_mask_mode="first_frame_causal",
    )

    vstate, astate = _build_joint_inputs(real_backbone, ab, t_action=16)
    s_video = vstate.x.shape[1]
    s_action = astate.payload.x_action.shape[1]
    device = vstate.x.device

    attn_mask = driver._build_joint_mask(
        s_video=s_video,
        s_action=s_action,
        video_tokens_per_frame=_V_TOKENS_PER_FRAME,
        device=device,
    )

    # Pull real q/k/v from layer 0 of both backbones.
    with torch.no_grad():
        q_v, k_v, v_v, vpost = real_backbone.pre_attn_at_layer(0, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(0, astate)

        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        phi_q = torch.cat([vpost["q_unrot"], apost["q_unrot"]], dim=1)
        phi_k = torch.cat([vpost["k_unrot"], apost["k_unrot"]], dim=1)

        H = driver.num_heads
        tq, tk, vv, pq, pk = (
            rearrange(t, "b s (h d) -> b h s d", h=H).float() for t in (q_cat, k_cat, v_cat, phi_q, phi_k)
        )

        chunk_index = _mask_to_chunk_index(attn_mask)
        assert chunk_index is not None, (
            "FastWAM-Joint first_frame_causal mask should map to a monotonic chunk "
            "index; got None — _mask_to_chunk_index drift?"
        )

        out_chunk = _chunked_linear_attn(tq, tk, vv, pq, pk, chunk_index, eps=driver.eps)
        out_expanded = _expanded_linear_attn(tq, tk, vv, pq, pk, mask=attn_mask, eps=driver.eps)

    torch.testing.assert_close(
        out_chunk,
        out_expanded,
        rtol=1e-4,
        atol=1e-4,
        msg=("Chunked cumsum path disagrees with expanded reference on real 2B q/k/v under FastWAM-Joint mask."),
    )
