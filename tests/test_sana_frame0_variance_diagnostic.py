'Public implementation.'

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu


ASSET_PATH = Path(os.environ.get("SANA_ASSET_PATH", "/path/to/assets/SANA-Video_2B_480p"))

# Deployed SANA-Video 480p geometry. T_lat=21 (81 frames after VAE 4×),
# spatial 60×104 (480×832 after VAE 8×). After patch_size=(1,2,2):
# f_p=21, h_p=30, w_p=52 → 32760 video tokens, 1560 per frame.
_T_LAT, _H_LAT, _W_LAT = 21, 60, 104
_F_PATCH, _H_PATCH, _W_PATCH = 21, 30, 52
_V_TOKENS_PER_FRAME = _H_PATCH * _W_PATCH  # 1560
_S_VIDEO = _F_PATCH * _V_TOKENS_PER_FRAME  # 32760

_CAPTION_LEN = 8
_CAPTION_CHANNELS = 2304
_T_ACTION = 16

# Plan §5.3 acceptance threshold.
_FRAME0_STD_RATIO_THRESHOLD = 10.0


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
        pytest.skip("third_party/Sana not importable.")


def _build_action_backbone(*, vb):
    """ActionDiT paired to SANA video backbone (linear-relu kernel, depth matched)."""
    from openwam.model.action_backbone.joint_action_dit import ActionDiT

    ab = ActionDiT(
        action_dim=20,
        dim=1152,
        ffn_dim=4608,
        num_heads=vb.num_heads,
        num_layers=vb.num_layers,
        video_dim=vb.dim,
        bridge_layers=tuple(range(vb.num_layers)),
        variant="joint_self_attn",
        attn_head_dim=vb.head_dim,
        text_dim=vb.dim,
        attn_kernel="linear_relu",
    )
    ab.to(device=next(vb._dit.parameters()).device, dtype=torch.bfloat16).eval()
    return ab


@pytest.fixture(scope="module")
def real_vb_ab():
    """Module-scoped 2B backbone + paired ActionDiT.

    Loading the 8 GB SANA ckpt + materializing the matched ActionDiT
    dominates the test runtime. Diagnostic tests in this module only do
    ``torch.no_grad()`` forwards, so sharing the instances across tests
    is safe; each test builds its own ``SanaMoTJointDriver`` (cheap —
    holds references, no weight copies).
    """
    _skip_unless_runnable()
    from openwam.model.video_backbone.sana import SanaVideoBackbone

    vb = SanaVideoBackbone.from_pretrained(
        str(ASSET_PATH), device="cuda:0", dtype=torch.bfloat16
    )
    vb.eval()
    ab = _build_action_backbone(vb=vb)
    return vb, ab


def test_real_2b_frame0_variance_drift_81frame(real_vb_ab):
    """Quantify SANA pretrained K-scale drift on frame 0 vs frames 1+ at 81×480p.

    Loads real 2B weights, builds a paired ActionDiT, runs layer 0 of the
    joint self-attention through ``_chunked_linear_attn`` under the
    deployed first_frame_causal joint mask. Splits the output into
    (frame_0_video, frames_1+_video, action) regions and compares per-region
    output std.

    Phase 5 acceptance (plan §5.3): frame-0 std < 10 × frames-1+ std.
    """
    from einops import rearrange

    from openwam.model.architectures.dual_system.sana_linear_attn import (
        _chunked_linear_attn,
        _mask_to_chunk_index,
    )
    from openwam.model.architectures.dual_system.sana_mot_driver import (
        SanaMoTJointDriver,
    )

    vb, ab = real_vb_ab

    driver = SanaMoTJointDriver(
        vb,
        ab,
        mot_checkpoint_mixed_attn=False,
        attention_mask_mode="joint",
        video_attention_mask_mode="first_frame_causal",
    )

    # --- Synthetic inputs at deployed geometry ---
    # We don't need real frames — the diagnostic measures the SANA pretrained
    # weight statistics on standard-Gaussian noisy latents (which is what
    # diffusion training sees at mid-timestep anyway).
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    x = torch.randn(1, 16, _T_LAT, _H_LAT, _W_LAT, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device)  # mid-diffusion (σ ≈ 0.5)
    y = torch.randn(1, 1, _CAPTION_LEN, _CAPTION_CHANNELS, device=device, dtype=dtype)
    mask = torch.ones(1, 1, 1, _CAPTION_LEN, dtype=torch.int16, device=device)

    noisy_actions = torch.randn(1, _T_ACTION, ab.action_dim, device=device, dtype=dtype)
    action_timestep = torch.tensor([0.3], device=device, dtype=dtype)
    context = torch.randn(1, _CAPTION_LEN, ab.text_dim, device=device, dtype=dtype)
    context_mask = torch.ones(1, _CAPTION_LEN, dtype=torch.bool, device=device)

    with torch.no_grad():
        vstate = vb.prepare(x=x, timestep=timestep, y=y, mask=mask)
        astate = ab.prepare_state(
            noisy_actions, action_timestep, context=context, context_mask=context_mask
        )

        assert vstate.x.shape[1] == _S_VIDEO, (
            f"Expected {_S_VIDEO} video tokens at 81×480p, got {vstate.x.shape[1]} — "
            "geometry mismatch with SANA-Video 480p preset."
        )
        s_action = astate.payload.x_action.shape[1]

        # --- Build joint mask + chunk index ---
        attn_mask = driver._build_joint_mask(
            s_video=_S_VIDEO,
            s_action=s_action,
            video_tokens_per_frame=_V_TOKENS_PER_FRAME,
            device=device,
        )
        chunk_index = _mask_to_chunk_index(attn_mask)
        assert chunk_index is not None, (
            "FastWAM-Joint first_frame_causal mask should map to a 3-chunk index "
            "at 81×480p — got None. Mask topology drift?"
        )
        # Expected layout: [0, 1560, 32760, 32760+s_action]
        assert chunk_index[0] == 0
        assert chunk_index[1] == _V_TOKENS_PER_FRAME
        assert chunk_index[2] == _S_VIDEO
        assert chunk_index[3] == _S_VIDEO + s_action

        # --- Probe layer 0 q/k/v (rawest pretrained statistics) ---
        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(0, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(0, astate)

        q_cat = torch.cat([q_v, q_a], dim=1)
        k_cat = torch.cat([k_v, k_a], dim=1)
        v_cat = torch.cat([v_v, v_a], dim=1)
        phi_q = torch.cat([vpost["q_unrot"], apost["q_unrot"]], dim=1)
        phi_k = torch.cat([vpost["k_unrot"], apost["k_unrot"]], dim=1)

        H = driver.num_heads
        tq, tk, vv, pq, pk = (
            rearrange(t, "b s (h d) -> b h s d", h=H).float()
            for t in (q_cat, k_cat, v_cat, phi_q, phi_k)
        )

        out = _chunked_linear_attn(tq, tk, vv, pq, pk, chunk_index, eps=driver.eps)
        # out: (1, H=20, S=32760+s_action, d=112) in fp32

        # --- Per-chunk output stats ---
        out_frame_0 = out[:, :, : _V_TOKENS_PER_FRAME, :]
        out_frames_1plus = out[:, :, _V_TOKENS_PER_FRAME:_S_VIDEO, :]
        out_action = out[:, :, _S_VIDEO:, :]

        std_frame_0 = out_frame_0.std().item()
        std_frames_1plus = out_frames_1plus.std().item()
        std_action = out_action.std().item()
        ratio = std_frame_0 / max(std_frames_1plus, 1e-30)

        # --- Per-chunk denominator stats (diagnostic, no assertion) ---
        # Reconstruct what _chunked_linear_attn computes internally so we can
        # see how close frame 0's denominator gets to driver.eps.
        # Denominator at query i = phi_q_i @ z_c + eps, where z_c is the
        # cumulative phi_k sum up to and including chunk(i).
        z_c0 = pk[:, :, : _V_TOKENS_PER_FRAME, :].sum(dim=-2, keepdim=True)  # (1, H, 1, d)
        z_c1 = z_c0 + pk[:, :, _V_TOKENS_PER_FRAME:_S_VIDEO, :].sum(dim=-2, keepdim=True)
        z_c2 = z_c1 + pk[:, :, _S_VIDEO:, :].sum(dim=-2, keepdim=True)

        denom_frame_0 = (pq[:, :, : _V_TOKENS_PER_FRAME, :] @ z_c0.transpose(-1, -2)).squeeze(-1)
        denom_frames_1plus = (
            pq[:, :, _V_TOKENS_PER_FRAME:_S_VIDEO, :] @ z_c1.transpose(-1, -2)
        ).squeeze(-1)
        denom_action = (pq[:, :, _S_VIDEO:, :] @ z_c2.transpose(-1, -2)).squeeze(-1)

        # Geometric mean (log-space) — denominators span many orders of magnitude.
        def _logmean(t):
            t = t.clamp_min(1e-30)
            return t.log().mean().exp().item()

        gmean_denom_frame_0 = _logmean(denom_frame_0)
        gmean_denom_frames_1plus = _logmean(denom_frames_1plus)
        gmean_denom_action = _logmean(denom_action)

        eps_saturation_frac = (denom_frame_0 < driver.eps * 10).float().mean().item()

    # --- Report ---
    print("\n[Phase 5 §5.1 diagnostic — layer 0, 81×480p, M = first_frame_causal joint]")
    print(f"  geometry:          S_video={_S_VIDEO}, tokens/frame={_V_TOKENS_PER_FRAME}, "
          f"S_action={s_action}, H={H}, d={tq.shape[-1]}")
    print(f"  eps:               {driver.eps:.1e}")
    print(f"  output std         frame 0 = {std_frame_0:.4e}")
    print(f"                     frames 1+ = {std_frames_1plus:.4e}")
    print(f"                     action  = {std_action:.4e}")
    print(f"  ratio frame0/1+    = {ratio:.3f}× (threshold: < {_FRAME0_STD_RATIO_THRESHOLD}×)")
    print(f"  denominator gmean  frame 0  = {gmean_denom_frame_0:.4e}")
    print(f"                     frames 1+ = {gmean_denom_frames_1plus:.4e}")
    print(f"                     action  = {gmean_denom_action:.4e}")
    print(f"  frame-0 eps-saturation frac (denom < 10·eps): {eps_saturation_frac:.4f}")

    assert torch.isfinite(out).all(), "Layer-0 mixed-attention output is non-finite."
    assert ratio < _FRAME0_STD_RATIO_THRESHOLD, (
        f"Phase 5 §5.3 acceptance #1 FAILED: frame-0 std / frames-1+ std = "
        f"{ratio:.3f}× exceeds threshold {_FRAME0_STD_RATIO_THRESHOLD}×. "
        "SANA pretrained K-scale drifts under first_frame_causal at deployed geometry — "
        "Phase 5 §5.2 mitigation needed (small-LR finetune / per-chunk LayerNorm)."
    )


def test_real_2b_per_layer_variance_drift_81frame(real_vb_ab):
    """All-20-layers extension of the §5.1 diagnostic.

    The layer-0 test above probes only the rawest pretrained statistics —
    layer-0 q/k/v projections applied to standard-Gaussian latents. Each of
    SANA's 20 blocks has its own q/k/v projection, and the input distribution
    to layer N depends on the full output stack of layers 0..N-1. So the
    layer-0 robustness result doesn't automatically generalize.

    This test runs one full forward through ``SanaMoTJointDriver.run_joint_loop``
    on synthetic inputs at the deployed 81×480p geometry, monkey-patching
    ``driver._mixed_attention`` to capture per-chunk output std at every
    layer. It then asserts the worst-layer ``frame_0_std / frames_1+_std``
    ratio is still under the §5.3 acceptance threshold (10×).
    """
    from openwam.model.architectures.dual_system.sana_mot_driver import (
        SanaMoTJointDriver,
    )

    vb, ab = real_vb_ab

    driver = SanaMoTJointDriver(
        vb,
        ab,
        mot_checkpoint_mixed_attn=False,
        attention_mask_mode="joint",
        video_attention_mask_mode="first_frame_causal",
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(42)

    x = torch.randn(1, 16, _T_LAT, _H_LAT, _W_LAT, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device)  # mid-diffusion
    y = torch.randn(1, 1, _CAPTION_LEN, _CAPTION_CHANNELS, device=device, dtype=dtype)
    mask = torch.ones(1, 1, 1, _CAPTION_LEN, dtype=torch.int16, device=device)

    noisy_actions = torch.randn(1, _T_ACTION, ab.action_dim, device=device, dtype=dtype)
    action_timestep = torch.tensor([0.3], device=device, dtype=dtype)
    context = torch.randn(1, _CAPTION_LEN, ab.text_dim, device=device, dtype=dtype)
    context_mask = torch.ones(1, _CAPTION_LEN, dtype=torch.bool, device=device)

    # Capture per-layer per-chunk output std via monkey-patch on the bound
    # ``_mixed_attention``. ``_step_impl`` calls ``self._mixed_attention(...)``;
    # since we set the instance attribute, Python finds our function before
    # the class method and calls it as a plain function (no ``self`` binding).
    captured: list[dict] = []
    original_mixed = driver._mixed_attention

    def captured_mixed(q_cat, k_cat, v_cat, attn_mask, *, phi_q=None, phi_k=None, use_ckpt=False):
        out = original_mixed(
            q_cat, k_cat, v_cat, attn_mask, phi_q=phi_q, phi_k=phi_k, use_ckpt=use_ckpt
        )
        # out: (1, S_video + S_action, H*D) in bf16.
        out_f = out.float()
        captured.append(
            {
                "frame_0_std": out_f[:, :_V_TOKENS_PER_FRAME, :].std().item(),
                "frames_1plus_std": out_f[:, _V_TOKENS_PER_FRAME:_S_VIDEO, :].std().item(),
                "action_std": out_f[:, _S_VIDEO:, :].std().item(),
            }
        )
        return out

    driver._mixed_attention = captured_mixed
    try:
        with torch.no_grad():
            vstate = vb.prepare(x=x, timestep=timestep, y=y, mask=mask)
            astate = ab.prepare_state(
                noisy_actions, action_timestep, context=context, context_mask=context_mask
            )
            assert vstate.x.shape[1] == _S_VIDEO, (
                f"Expected {_S_VIDEO} video tokens at 81×480p, got {vstate.x.shape[1]}."
            )
            driver.run_joint_loop(vstate, astate)
    finally:
        driver._mixed_attention = original_mixed

    n_layers = vb.num_layers
    assert len(captured) == n_layers, (
        f"Expected {n_layers} captured mixed-attention calls (one per layer), "
        f"got {len(captured)}. _mixed_attention hook may have been bypassed."
    )

    ratios = [s["frame_0_std"] / max(s["frames_1plus_std"], 1e-30) for s in captured]
    max_ratio = max(ratios)
    worst_layer = int(max(range(n_layers), key=lambda i: ratios[i]))

    print(
        "\n[Phase 5 §5.1 per-layer diagnostic — full 20-block stack, "
        "81×480p, M = first_frame_causal joint]"
    )
    print(f"  {'layer':>5}  {'frame_0_std':>14}  {'frames_1+_std':>14}  {'action_std':>12}  {'ratio':>8}")
    for i, s in enumerate(captured):
        print(
            f"  {i:>5}  {s['frame_0_std']:>14.4e}  {s['frames_1plus_std']:>14.4e}  "
            f"{s['action_std']:>12.4e}  {ratios[i]:>7.3f}×"
        )
    print(f"  worst layer: {worst_layer} (ratio {max_ratio:.3f}×, threshold {_FRAME0_STD_RATIO_THRESHOLD}×)")

    assert max_ratio < _FRAME0_STD_RATIO_THRESHOLD, (
        f"Phase 5 §5.3 acceptance #1 FAILED at layer {worst_layer}: "
        f"frame-0 std / frames-1+ std = {max_ratio:.3f}× exceeds threshold "
        f"{_FRAME0_STD_RATIO_THRESHOLD}×. SANA pretrained K-scale drifts under "
        "first_frame_causal somewhere in the 20-block stack — §5.2 mitigation needed."
    )
