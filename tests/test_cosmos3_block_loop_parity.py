"""CPU parity: the batched cosmos3 engine vs the vendored native forward.

Runs the tiny random Edge-flavoured config through both paths with identical
inputs and asserts the noisy-frame velocity fields match at fp32 tolerance —
the canary for any drift in the und-tower / gen-block / patchify / rope /
timestep-scatter reimplementation. (The native forward zero-fills clean frames
and decodes only noisy tokens; our finalize returns the full grid, so clean
frames are compared for the upstream-zeros invariant only.)
"""

import pytest
import torch

pytest.importorskip("diffusers")

from openwam.model.video_backbone.cosmos3 import dit_forward, text_pack  # noqa: E402
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


def _build(cfg):
    torch.manual_seed(0)
    return Cosmos3OmniTransformer(**cfg).eval()


def _native_velocity(net, ids, lat, text_pos, vis_pos, grid, ncp, t_val):
    grid_t, grid_h, grid_w = grid
    length = ids.numel()
    stride = grid_h * grid_w
    s = grid_t * stride
    noisy = torch.arange(ncp, grid_t)
    mse_idx = torch.cat([torch.arange(length + f * stride, length + (f + 1) * stride) for f in noisy.tolist()])
    with torch.no_grad():
        out = net(
            input_ids=ids,
            text_indexes=torch.arange(length),
            position_ids=torch.cat([text_pos, vis_pos], dim=1),
            und_len=length,
            sequence_length=length + s,
            vision_tokens=[lat],
            vision_token_shapes=[grid],
            vision_sequence_indexes=torch.arange(length, length + s),
            vision_mse_loss_indexes=mse_idx,
            vision_timesteps=torch.full((noisy.numel() * stride,), t_val),
            vision_noisy_frame_indexes=[noisy],
        )
    return out.sample[0]


def _mine_velocity(net, ids, lat, text_pos, vis_pos, ncp, t_val):
    with torch.no_grad():
        cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
        und_mask = torch.ones(1, ids.numel(), dtype=torch.bool)
        context, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0), und_mask, cos_und, sin_und)
        state = dit_forward.prepare_block_loop(
            net,
            latents=lat,
            timestep=torch.tensor([t_val]),
            context=context,
            context_mask=und_mask,
            und_kv=und_kv,
            vision_positions=vis_pos.unsqueeze(1),
            num_clean_prefix_frames=ncp,
        )
        for i in range(len(net.layers)):
            state = dit_forward.run_block(net, i, state)
        return dit_forward.finalize_block_loop(net, state)


def _positions(net, und_len, latent_grid):
    p = int(net.config.latent_patch_size)
    grid = text_pack.patch_grid(*latent_grid, p)
    text_pos = text_pack.text_mrope_positions(und_len, float_positions=True)
    _, vis_pos = text_pack.build_joint_positions(
        und_len,
        grid,
        modality_margin=int(net.config.unified_3d_mrope_temporal_modality_margin),
        fps=24.0,
        base_fps=float(net.config.base_fps),
        temporal_compression_factor=4,
    )
    return text_pos, vis_pos, grid


def test_engine_matches_native_forward_p1():
    net = _build(MINI)
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    text_pos, vis_pos, grid = _positions(net, ids.numel(), (3, 2, 2))

    native = _native_velocity(net, ids, lat, text_pos, vis_pos, grid, ncp=1, t_val=500.0)
    mine = _mine_velocity(net, ids, lat, text_pos, vis_pos, ncp=1, t_val=500.0)

    assert native[:, :, :1].abs().sum() == 0
    diff = (mine[:, :, 1:] - native[:, :, 1:]).abs().max().item()
    assert torch.allclose(mine[:, :, 1:], native[:, :, 1:], atol=1e-5), f"max diff {diff}"


def test_engine_matches_native_forward_p2_with_padding():
    cfg = dict(MINI, latent_patch_size=2, patch_latent_dim=2 * 2 * MINI["latent_channel"])
    net = _build(cfg)
    ids = torch.tensor([5, 6, 7])
    lat = torch.randn(1, cfg["latent_channel"], 2, 3, 5)  # odd H/W → zero-pad path
    text_pos, vis_pos, grid = _positions(net, ids.numel(), (2, 3, 5))

    native = _native_velocity(net, ids, lat, text_pos, vis_pos, grid, ncp=1, t_val=995.0)
    mine = _mine_velocity(net, ids, lat, text_pos, vis_pos, ncp=1, t_val=995.0)

    diff = (mine[:, :, 1:] - native[:, :, 1:]).abs().max().item()
    assert torch.allclose(mine[:, :, 1:], native[:, :, 1:], atol=1e-5), f"max diff {diff}"


def test_batched_engine_consistent_with_single():
    net = _build(MINI)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    lat1 = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    lat = torch.cat([lat1, lat1], dim=0)
    text_pos, vis_pos, _ = _positions(net, 4, (3, 2, 2))

    with torch.no_grad():
        cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
        und_mask = torch.ones(2, 4, dtype=torch.bool)
        context, und_kv = dit_forward.run_und_tower(net, ids, und_mask, cos_und, sin_und)
        state = dit_forward.prepare_block_loop(
            net,
            latents=lat,
            timestep=torch.tensor([500.0, 500.0]),
            context=context,
            context_mask=und_mask,
            und_kv=und_kv,
            vision_positions=vis_pos.unsqueeze(1),
            num_clean_prefix_frames=1,
        )
        for i in range(len(net.layers)):
            state = dit_forward.run_block(net, i, state)
        batched = dit_forward.finalize_block_loop(net, state)

    single = _mine_velocity(net, ids[0], lat1, text_pos, vis_pos, ncp=1, t_val=500.0)
    assert torch.allclose(batched[0], batched[1], atol=1e-6)
    assert torch.allclose(batched[:1], single, atol=1e-6)


def test_freeze_unused_native_heads():
    from openwam.model.video_backbone.cosmos3.pipeline_builder import _freeze_unused_native_heads

    net = _build(dict(MINI, action_gen=True, action_dim=4, num_embodiment_domains=3))
    frozen = _freeze_unused_native_heads(net)
    expected = sum(p.numel() for p in net.action_proj_in.parameters())
    expected += sum(p.numel() for p in net.action_proj_out.parameters())
    expected += net.action_modality_embed.numel()
    assert frozen == expected
    assert not any(p.requires_grad for p in net.action_proj_in.parameters())
    assert not any(p.requires_grad for p in net.action_proj_out.parameters())
    assert not net.action_modality_embed.requires_grad
    # The gen pathway stays trainable.
    assert net.layers[0].self_attn.add_q_proj.weight.requires_grad
    assert net.proj_out.weight.requires_grad


def test_gradients_reach_gen_pathway():
    net = _build(MINI)
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    text_pos, vis_pos, _ = _positions(net, ids.numel(), (3, 2, 2))

    cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(1, 4, dtype=torch.bool)
    context, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0), und_mask, cos_und, sin_und)
    state = dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.tensor([500.0]),
        context=context,
        context_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=1,
    )
    for i in range(len(net.layers)):
        state = dit_forward.run_block(net, i, state)
    out = dit_forward.finalize_block_loop(net, state)
    out[:, :, 1:].square().mean().backward()

    gen = net.layers[0].self_attn.add_q_proj.weight.grad
    und = net.layers[0].self_attn.to_q.weight.grad
    assert gen is not None and gen.abs().sum() > 0
    assert und is None  # und tower ran under no_grad
