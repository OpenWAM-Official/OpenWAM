"""joint_self_attn (MoT) support for cosmos3_edge — CPU parity tests.

1. Split parity: ``post(pre → driver-style SDPA) == run_block`` per layer on the
   tiny random Edge-flavoured config. The split path uses the driver's exact
   flat single-``num_heads`` layout (KV heads pre-expanded 8→16-style), so this
   is the canary for GQA-expand / rope / residual mistakes.
2. Driver integration: with ``attention_mask_mode="isolated"`` (video cannot
   see action) and bidirectional v↔v, the video stream that comes out of
   ``DualSystemMoTDriver.run_joint_loop`` must equal the pure ``run_block``
   loop — exercising the rectangular prefix-KV mask end to end.
"""

import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("diffusers")

from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver  # noqa: E402
from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import block_split, dit_forward, text_pack  # noqa: E402
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


def _build_state(net, ids, lat, ncp=0, *, padded_und: bool = True):
    """``padded_und=False`` reproduces the unpadded production shape, where the
    backbone passes ``und_mask=None`` all the way into SDPA."""
    text_pos = text_pack.text_mrope_positions(ids.numel(), float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        ids.numel(), grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(lat.shape[0], ids.numel(), dtype=torch.bool) if padded_und else None
    ids_b = ids.unsqueeze(0).expand(lat.shape[0], -1)
    context, und_kv = dit_forward.run_und_tower(net, ids_b, und_mask, cos_und, sin_und)
    state = dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((lat.shape[0],), 500.0),
        context=context,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=ncp,
    )
    return state


@pytest.mark.parametrize("padded_und", [True, False], ids=["und_mask_tensor", "und_mask_none"])
def test_split_parity_matches_run_block(padded_und):
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)

    with torch.no_grad():
        ref = _build_state(net, ids, lat, padded_und=padded_und)
        split = _build_state(net, ids, lat, padded_und=padded_und)
        assert split.prefix_kv_len == 4
        assert (split.prefix_kv_mask is not None) is padded_und

        for i in range(len(net.layers)):
            ref = dit_forward.run_block(net, i, ref)

            q, k, v, post = block_split.state_pre_attn(net, i, split)
            assert k.shape[1] == q.shape[1] + split.prefix_kv_len
            # Driver-style mixed attention: single num_heads, rectangular mask.
            n = MINI["num_attention_heads"]
            qh = q.view(q.shape[0], q.shape[1], n, -1).transpose(1, 2)
            kh = k.view(k.shape[0], k.shape[1], n, -1).transpose(1, 2)
            vh = v.view(v.shape[0], v.shape[1], n, -1).transpose(1, 2)
            mask = torch.ones((q.shape[1], k.shape[1]), dtype=torch.bool)
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
            out = out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)
            split = block_split.state_post_attn(split, out, post)

            diff = (split.hidden_states - ref.hidden_states).abs().max().item()
            assert diff < 1e-5, f"layer {i}: split vs run_block diff {diff}"


class _TinyActionBackbone(nn.Module):
    """Minimal MoT-compatible action stream for driver integration tests."""

    def __init__(self, dim, num_layers, num_heads, head_dim):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        width = num_heads * head_dim
        self.q = nn.ModuleList(nn.Linear(dim, width) for _ in range(num_layers))
        self.k = nn.ModuleList(nn.Linear(dim, width) for _ in range(num_layers))
        self.v = nn.ModuleList(nn.Linear(dim, width) for _ in range(num_layers))
        self.o = nn.ModuleList(nn.Linear(width, dim) for _ in range(num_layers))

    def pre_attn_at_layer(self, layer_id, astate):
        x = astate.payload.x_action
        return self.q[layer_id](x), self.k[layer_id](x), self.v[layer_id](x), {"layer": layer_id}

    def post_attn_at_layer(self, layer_id, astate, attn_out, post_state):
        astate.payload.x_action = astate.payload.x_action + self.o[post_state["layer"]](attn_out)
        return astate


@pytest.mark.parametrize("padded_und", [True, False], ids=["und_mask_tensor", "und_mask_none"])
def test_driver_isolated_mode_matches_run_block_loop(padded_und):
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)

    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        vae=None,
        tokenizer=None,
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
    ).eval()
    ab = _TinyActionBackbone(
        MINI["hidden_size"], MINI["num_hidden_layers"], MINI["num_attention_heads"], MINI["head_dim"]
    ).eval()
    driver = DualSystemMoTDriver(vb, ab, attention_mask_mode="isolated")

    with torch.no_grad():
        ref = _build_state(net, ids, lat, padded_und=padded_und)
        for i in range(len(net.layers)):
            ref = dit_forward.run_block(net, i, ref)

        joint = _build_state(net, ids, lat, padded_und=padded_und)
        astate = types.SimpleNamespace(payload=types.SimpleNamespace(x_action=torch.randn(1, 5, MINI["hidden_size"])))
        joint_v, joint_a = driver.run_joint_loop(joint, astate)

    diff = (joint_v.hidden_states - ref.hidden_states).abs().max().item()
    assert diff < 1e-5, f"isolated-mode video stream deviates from run_block: {diff}"
    assert torch.isfinite(joint_a.payload.x_action).all()


def _padded_state(net, poison: bool, pad: int = 2):
    """B=2 state whose second prompt is right-padded; optionally poison the
    padded und K/V so a masking mistake shows up as a numeric change."""
    torch.manual_seed(7)
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    und_mask = torch.tensor([[True] * 5, [True] * (5 - pad) + [False] * pad])
    lat = torch.randn(2, MINI["latent_channel"], 2, 2, 2)
    text_pos = text_pack.text_mrope_positions(5, float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        5, grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    ctx, und_kv = dit_forward.run_und_tower(net, ids, und_mask, cos_u, sin_u)
    if poison:
        und_kv = [(k.clone(), v.clone()) for k, v in und_kv]
        for k, v in und_kv:  # sample 1's padded slots only
            k[1, 5 - pad : 5] = 1e4
            v[1, 5 - pad : 5] = 1e4
    return dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((2,), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=0,
    )


@pytest.mark.parametrize("mode", ["action_sees_video", "mutual", "isolated"])
def test_joint_loop_keeps_padded_und_keys_out(mode):
    """Padded und keys must not reach the video or action stream through MoT.

    ``run_joint_loop`` widens its cross-modal mask by ``prefix_kv_len`` and gates
    the prefix columns with ``prefix_kv_mask``. Poisoning the padded und K/V is
    the end-to-end check that the gate survives the widening for every mask mode
    — the mask helper's own unit tests cannot see a wiring mistake here.
    """
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        vae=None,
        tokenizer=None,
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
    ).eval()
    ab = _TinyActionBackbone(
        MINI["hidden_size"], MINI["num_hidden_layers"], MINI["num_attention_heads"], MINI["head_dim"]
    ).eval()
    driver = DualSystemMoTDriver(vb, ab, attention_mask_mode=mode, mot_checkpoint_mixed_attn=False)

    outs = []
    for poison in (False, True):
        state = _padded_state(net, poison)
        astate = types.SimpleNamespace(
            payload=types.SimpleNamespace(
                x_action=torch.arange(2 * 3 * MINI["hidden_size"]).float().view(2, 3, -1) * 1e-2
            )
        )
        with torch.no_grad():
            v, a = driver.run_joint_loop(state, astate)
        outs.append((v.hidden_states, a.payload.x_action))

    d_video = (outs[0][0] - outs[1][0]).abs().max().item()
    d_action = (outs[0][1] - outs[1][1]).abs().max().item()
    assert d_video < 1e-6, f"{mode}: padded und keys leaked into the video stream ({d_video:.3e})"
    assert d_action < 1e-6, f"{mode}: padded und keys leaked into the action stream ({d_action:.3e})"


def test_padded_und_poison_is_detectable():
    """Positive control for the test above: without the gate the poison must move
    the output, otherwise the leak assertions would pass vacuously."""
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    outs = []
    for poison in (False, True):
        state = _padded_state(net, poison)
        state.prefix_kv_mask = None  # drop the gate
        state.extras = {**state.extras, "und_mask": None}
        with torch.no_grad():
            for i in range(len(net.layers)):
                state = dit_forward.run_block(net, i, state)
            outs.append(dit_forward.finalize_block_loop(net, state))
    assert (outs[0] - outs[1]).abs().max().item() > 1e-3, "poison is inert; the leak tests prove nothing"
