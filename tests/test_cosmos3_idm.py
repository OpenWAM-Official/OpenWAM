"""IDM teacher-forcing support for cosmos3_edge — CPU tests.

Covers the backbone's branch merge/split contract and the prefix-KV mask
widening the IDM driver needs, since Cosmos3's per-layer keys carry the cached
und (text) stream that has no matching query rows.
"""

import types

import pytest
import torch

pytest.importorskip("diffusers")

from openwam.model.architectures.utils.mask_modes import widen_mask_for_prefix_kv  # noqa: E402
from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit_forward, idm_merge, text_pack  # noqa: E402
from openwam.model.video_backbone.cosmos3._vendor.transformer_cosmos3 import (  # noqa: E402
    Cosmos3OmniTransformer,
)

MINI = dict(
    attention_bias=False,
    head_dim=6,
    hidden_size=12,
    intermediate_size=24,
    latent_channel=2,
    latent_patch_size=1,
    num_attention_heads=2,
    num_hidden_layers=2,
    num_key_value_heads=1,
    patch_latent_dim=2,
    qk_norm_for_text=False,
    use_und_k_norm_for_gen=True,
    hidden_act="relu2",
    rms_norm_eps=1e-5,
    rope_axes_dim=[1, 1, 1],
    rope_theta=1e8,
    vocab_size=32,
)


def _net():
    torch.manual_seed(0)
    return Cosmos3OmniTransformer(**MINI).eval()


def _state(net, ids, lat, ncp=0):
    text_pos = text_pack.text_mrope_positions(ids.numel(), float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        ids.numel(), grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(lat.shape[0], ids.numel(), dtype=torch.bool)
    ctx, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0).expand(lat.shape[0], -1), und_mask, cos_u, sin_u)
    return dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((lat.shape[0],), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=ncp,
    )


def test_merge_concats_tokens_and_rotary():
    net = _net()
    ids = torch.tensor([1, 2, 3, 4])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))

    s_n_before, s_c_before = noisy.hidden_states.shape[1], cond.hidden_states.shape[1]
    merged, s_noisy, s_cond = noisy_merged = idm_merge.merge_branches(noisy, cond)
    assert (s_noisy, s_cond) == (s_n_before, s_c_before)
    assert merged.hidden_states.shape[1] == s_noisy + s_cond
    assert merged.extras["cos_gen"].shape[1] == s_noisy + s_cond
    assert merged.extras["sin_gen"].shape[1] == s_noisy + s_cond
    assert merged.grid_frames == noisy.grid_frames + cond.grid_frames
    # und cache is shared (same prompt) and the prefix declaration survives.
    assert merged.extras["und_kv"] is noisy.extras["und_kv"]
    assert merged.prefix_kv_len == noisy.prefix_kv_len
    del noisy_merged


def test_split_is_inverse_of_merge():
    net = _net()
    ids = torch.tensor([5, 6, 7])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    hn, hc = noisy.hidden_states.clone(), cond.hidden_states.clone()

    merged, _, _ = idm_merge.merge_branches(noisy, cond)
    noisy2, cond2 = idm_merge.split_branches(merged, noisy, cond)
    assert torch.equal(noisy2.hidden_states, hn)
    assert torch.equal(cond2.hidden_states, hc)


def test_merge_rejects_mismatched_spatial_layout():
    net = _net()
    ids = torch.tensor([1, 2])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 4, 2))
    with pytest.raises(ValueError, match="spatial token layout"):
        idm_merge.merge_branches(noisy, cond)


def test_merged_state_runs_through_blocks():
    net = _net()
    ids = torch.tensor([1, 2, 3, 4])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    with torch.no_grad():
        merged, s_noisy, s_cond = idm_merge.merge_branches(noisy, cond)
        for i in range(len(net.layers)):
            merged = dit_forward.run_block(net, i, merged)
        noisy2, cond2 = idm_merge.split_branches(merged, noisy, cond)
    assert noisy2.hidden_states.shape[1] == s_noisy
    assert cond2.hidden_states.shape[1] == s_cond
    assert torch.isfinite(noisy2.hidden_states).all() and torch.isfinite(cond2.hidden_states).all()


def test_prefix_widening_shapes_and_semantics():
    # Square mask + a declared prefix -> rectangular mask with visible prefix cols.
    st = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=None)
    m = torch.zeros((5, 5), dtype=torch.bool)
    w = widen_mask_for_prefix_kv(m, st)
    assert w.shape == (5, 8)
    assert w[:, :3].all() and not w[:, 3:].any()

    # Per-sample padding gate broadcasts to (B, 1, S, prefix + S).
    pm = torch.tensor([[True, True, False]])
    st2 = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=pm)
    w2 = widen_mask_for_prefix_kv(torch.ones((5, 5), dtype=torch.bool), st2)
    assert w2.shape == (1, 1, 5, 8)
    assert w2[0, 0, :, 2].sum() == 0  # padded und column masked for every query row

    # No prefix declared -> untouched (byte-identical for Wan / predict2.5).
    st3 = types.SimpleNamespace(prefix_kv_len=0, prefix_kv_mask=None)
    m3 = torch.ones((4, 4), dtype=torch.bool)
    assert widen_mask_for_prefix_kv(m3, st3) is m3


def test_backbone_exposes_idm_methods():
    vb = Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)
    assert callable(vb.merge_idm_video_branches)
    assert callable(vb.split_idm_video_branches)
