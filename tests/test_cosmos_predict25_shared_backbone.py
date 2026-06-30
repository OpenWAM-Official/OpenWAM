"""SharedBackbone (vanilla + moe) support for the CosmosPredict25 video backbone.

Cosmos keeps its DiT state as a 5D grid ``(B,T,H,W,D)`` with per-frame modulation,
so non-grid action/state tokens can't be appended directly. SharedBackbone runs
the ``[video|action|state]`` sequence through a flat 3D block forward
(``cosmos_predict25.shared_block``). These CPU tests use the ``_RichCosmosBlock``
fakes (shared with the joint_self_attn suite) to validate:

* the 3D block forward reproduces the monolithic 5D block on video-only input;
* inject/extract round-trips (5D grid restored, action tail recovered);
* the cross-modal mask isolates action from video queries;
* vanilla + MoE forward end-to-end on a Cosmos rich-fake backbone.
"""

from __future__ import annotations

import torch
from einops import rearrange

from tests.test_cosmos_predict25_joint_self_attn import _build_rich_wrapper


def _prep(backbone, *, force=True, seed=0, B=1):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(B, 16, 2, 8, 8, generator=g)  # grid T=2, H=4, W=4 → 32 video tokens
    context = torch.randn(B, 4, 12, generator=g)
    timestep = torch.randint(0, 1000, (B,), generator=g)
    return backbone.prepare(input_latents=latents, context=context, timestep=timestep, force_per_token_t_mod=force)


# ----------------------------------------------------------------------
# 3D block forward parity: video-only must match the monolithic 5D block.
# ----------------------------------------------------------------------


def test_run_block_3d_matches_5d_video_only():
    from openwam.model.video_backbone.cosmos_predict25 import shared_block

    backbone = _build_rich_wrapper(num_blocks=1)
    backbone.eval()

    s5 = _prep(backbone, seed=0)
    s3 = _prep(backbone, seed=0)

    s5 = backbone.run_block(0, s5)  # normal 5D path
    out_5d_flat = rearrange(s5.hidden_states, "b t h w d -> b (t h w) d")

    T, H, W = s3.grid_frames, s3.grid_height, s3.grid_width
    tpf = H * W
    x3d = rearrange(s3.hidden_states, "b t h w d -> b (t h w) d")
    emb = shared_block.expand_video_emb_to_tokens(s3.extras["t_embedding_B_T_D"], T, tpf)
    lora = shared_block.expand_video_emb_to_tokens(s3.extras["adaln_lora_B_T_3D"], T, tpf)
    block = backbone.dit.blocks[0]
    out_3d = shared_block.run_block_3d(block, x3d, emb, lora, s3.extras["rope_emb_L_1_1_D"], s3.context, None)

    assert torch.allclose(out_5d_flat, out_3d, atol=1e-5, rtol=1e-5), (
        f"3D shared block must reproduce the 5D block on video-only input; "
        f"max abs diff = {(out_5d_flat - out_3d).abs().max().item():.2e}"
    )


def test_expand_video_emb_matches_repeat_order():
    from openwam.model.video_backbone.cosmos_predict25 import shared_block

    # per-frame emb expands so each frame's value repeats over its H·W tokens.
    emb = torch.arange(2 * 3, dtype=torch.float32).view(1, 2, 3)  # (B=1, T=2, D=3)
    out = shared_block.expand_video_emb_to_tokens(emb, grid_frames=2, tokens_per_frame=4)
    assert out.shape == (1, 8, 3)
    assert torch.equal(out[0, :4], emb[0, 0].expand(4, 3))
    assert torch.equal(out[0, 4:], emb[0, 1].expand(4, 3))


# ----------------------------------------------------------------------
# inject / extract round-trip.
# ----------------------------------------------------------------------


def test_inject_extract_round_trip():
    backbone = _build_rich_wrapper(num_blocks=1)
    state = _prep(backbone, seed=1)
    B, dim = 1, 16
    n_action, n_state = 3, 2
    action = torch.randn(B, n_action, dim)
    state_tok = torch.randn(B, n_state, dim)
    T, H, W = state.grid_frames, state.grid_height, state.grid_width
    s_video = T * H * W

    state = backbone.inject_shared_tokens(
        state, action, n_action, state_tokens=state_tok, n_state=n_state, timestep=torch.tensor([0.5])
    )
    # 3D shared sequence
    assert state.hidden_states.shape == (B, s_video + n_action + n_state, dim)
    assert state.extras["shared_mode"] is True
    assert state.extras["shared_emb_B_S_D"].shape[1] == s_video + n_action + n_state
    assert state.extras["shared_rope"].shape[0] == s_video + n_action + n_state

    state, action_out = backbone.extract_shared_tokens(state, n_action, n_state=n_state)
    assert state.hidden_states.shape == (B, T, H, W, dim)  # 5D grid restored
    assert torch.allclose(action_out, action)  # no blocks ran → tail unchanged
    assert not state.extras.get("shared_mode")


# ----------------------------------------------------------------------
# Cross-modal mask isolates action from video queries.
# ----------------------------------------------------------------------


def test_shared_mask_blocks_action_from_video_queries():
    from openwam.model.architectures.shared_backbone.state import attach_shared_attention_mask
    from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO

    backbone = _build_rich_wrapper(num_blocks=1)
    backbone.video_attention_mask_mode = "bidirectional"
    backbone.eval()
    B, dim, n_action = 1, 16, 3

    def _run(action):
        state = _prep(backbone, seed=0)
        T, H, W = state.grid_frames, state.grid_height, state.grid_width
        s_video = T * H * W
        state = backbone.inject_shared_tokens(state, action, n_action, timestep=torch.tensor([0.5]))
        attach_shared_attention_mask(backbone, state, n_action, attention_mask_mode=ACTION_SEES_VIDEO)
        state = backbone.run_block(0, state)
        return state.hidden_states[:, :s_video]

    out_a = _run(torch.randn(B, n_action, dim))
    out_b = _run(torch.randn(B, n_action, dim) + 10.0)
    # ACTION_SEES_VIDEO: video queries never attend to action → video output invariant.
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=0)


# ----------------------------------------------------------------------
# vanilla + MoE end-to-end forward through the architecture.
# ----------------------------------------------------------------------


def _make_shared_arch(variant, num_blocks=2):
    from openwam.model import build_architecture

    backbone = _build_rich_wrapper(num_blocks=num_blocks)
    cfg = {
        "framework": "shared_backbone",
        "variant": variant,
        "action_dim": 7,
        "video_dim": 16,
        "attention_mask_mode": "action_sees_video",
        "video_attention_mask_mode": "first_frame_causal",
    }
    if variant == "moe":
        cfg["expert_ffn_dim"] = 32
        cfg["bridge_layers"] = tuple(range(num_blocks))
    arch = build_architecture(f"shared_backbone_{variant}", cfg)
    arch.video_backbone = backbone
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch.eval()
    return arch


def _forward_inputs(B=1):
    return dict(
        latents=torch.randn(B, 16, 2, 8, 8),
        context=torch.randn(B, 4, 12),
        timestep=torch.tensor([0.5]),
    )


def test_shared_backbone_vanilla_forward_cosmos():
    arch = _make_shared_arch("vanilla")
    actions = torch.randn(1, 3, 7)
    video_out, action_pred = arch(actions, torch.tensor([0.5]), **_forward_inputs())
    assert action_pred.shape == (1, 3, 7) and torch.isfinite(action_pred).all()
    assert video_out.shape == (1, 16, 2, 8, 8) and torch.isfinite(video_out).all()


def test_shared_backbone_moe_forward_cosmos():
    arch = _make_shared_arch("moe")
    actions = torch.randn(1, 3, 7)
    video_out, action_pred = arch(actions, torch.tensor([0.5]), **_forward_inputs())
    assert action_pred.shape == (1, 3, 7) and torch.isfinite(action_pred).all()
    assert torch.isfinite(video_out).all()


def test_shared_backbone_video_only_forward_cosmos():
    # noisy_actions=None → pure video path (no shared tokens, normal 5D loop).
    arch = _make_shared_arch("vanilla")
    video_out, action_pred = arch(None, None, **_forward_inputs())
    assert action_pred is None
    assert video_out.shape == (1, 16, 2, 8, 8) and torch.isfinite(video_out).all()
