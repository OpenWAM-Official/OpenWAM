"""single_system (vanilla / moe) support for cosmos3_edge — CPU tests.

Cosmos3's gen stream is already a flat ``(B, S, D)`` sequence and the model has
no AdaLN, so injection is a concat plus identity rotary rows and an additive
timestep embedding. These tests pin that contract, the round-trip through the
block loop, and the cross-modal mask reaching the attention.
"""

import pytest
import torch

pytest.importorskip("diffusers")

from openwam.model.architectures.single_system.state import attach_shared_attention_mask  # noqa: E402
from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit_forward, shared_block, text_pack  # noqa: E402
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
DIM = MINI["hidden_size"]


def _build():
    torch.manual_seed(0)
    net = Cosmos3OmniTransformer(**MINI).eval()
    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        vae=None,
        tokenizer=None,
        dim=DIM,
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=DIM,
    ).eval()
    return net, vb


def _state(net, ids, lat, *, padded_und: bool = True):
    """``padded_und=False`` is the production B=1 shape: no prompt padding, so
    the backbone hands down ``und_mask=None`` and ``_gen_block_forward`` takes
    its ``und_mask is None`` branch — the branch every deploy call uses and the
    one an all-True tensor fixture never reaches."""
    text_pos = text_pack.text_mrope_positions(ids.numel(), float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        ids.numel(), grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(lat.shape[0], ids.numel(), dtype=torch.bool) if padded_und else None
    ctx, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0).expand(lat.shape[0], -1), und_mask, cos_u, sin_u)
    return dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((lat.shape[0],), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=0,
    )


def test_inject_extract_roundtrip():
    net, vb = _build()
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)
    st = _state(net, ids, lat)
    s_video = st.hidden_states.shape[1]

    n_action, n_state = 5, 1
    action = torch.randn(1, n_action, DIM)
    state_tok = torch.randn(1, n_state, DIM)
    with torch.no_grad():
        st = vb.inject_shared_tokens(
            st, action, n_action, state_tokens=state_tok, n_state=n_state, timestep=torch.tensor([500.0])
        )
        assert st.hidden_states.shape[1] == s_video + n_action + n_state
        assert st.extras["cos_gen"].shape[1] == s_video + n_action + n_state
        assert st.extras["shared_mode"] is True
        # Identity rotation on the injected rows.
        assert torch.equal(st.extras["cos_gen"][:, s_video:], torch.ones_like(st.extras["cos_gen"][:, s_video:]))
        assert torch.equal(st.extras["sin_gen"][:, s_video:], torch.zeros_like(st.extras["sin_gen"][:, s_video:]))

        st2, out_action = vb.extract_shared_tokens(st, n_action, n_state=n_state)
    assert out_action.shape == (1, n_action, DIM)
    assert st2.hidden_states.shape[1] == s_video
    assert st2.extras["cos_gen"].shape[1] == s_video
    assert "shared_mode" not in st2.extras


def test_injected_tokens_carry_timestep_embedding():
    net, vb = _build()
    ids = torch.tensor([1, 2])
    st = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    s_video = st.hidden_states.shape[1]
    action = torch.zeros(1, 3, DIM)  # zero tokens isolate the additive embedding
    with torch.no_grad():
        st = vb.inject_shared_tokens(st, action, 3, timestep=torch.tensor([500.0]))
        injected = st.hidden_states[:, s_video:]
        expected = shared_block.shared_token_timestep_embedding(
            net, torch.tensor([500.0]), 3, 1, st.hidden_states.dtype
        )
    assert torch.allclose(injected, expected, atol=1e-6)


@pytest.mark.parametrize("padded_und", [True, False], ids=["und_mask_tensor", "und_mask_none"])
def test_shared_mask_reaches_attention_and_isolates_action(padded_und):
    """With an isolated mask the video rows must be unaffected by action tokens.

    Parametrized over the und-mask shape because ``_gen_block_forward`` combines
    the two into one branch: with ``und_mask=None`` (every B=1 deploy call) the
    only thing keeping the cross-modal mask alive is that the ``gen_mask``
    arm is checked too. Weakening the guard to ``if und_mask is None:`` passes
    the tensor case and silently drops the mask on the production branch.
    """
    net, vb = _build()
    ids = torch.tensor([1, 2, 3, 4])
    lat = torch.randn(1, MINI["latent_channel"], 3, 2, 2)

    with torch.no_grad():
        ref = _state(net, ids, lat, padded_und=padded_und)
        for i in range(len(net.layers)):
            ref = dit_forward.run_block(net, i, ref)

        st = _state(net, ids, lat, padded_und=padded_und)
        s_video = st.hidden_states.shape[1]
        n_action = 4
        st = vb.inject_shared_tokens(st, torch.randn(1, n_action, DIM), n_action, timestep=torch.tensor([500.0]))
        attach_shared_attention_mask(vb, st, n_action, attention_mask_mode="isolated")
        assert st.extras["shared_attention_mask"].shape == (s_video + n_action, s_video + n_action)
        for i in range(len(net.layers)):
            st = dit_forward.run_block(net, i, st)
        st, action_out = vb.extract_shared_tokens(st, n_action)

    diff = (st.hidden_states - ref.hidden_states).abs().max().item()
    assert diff < 1e-5, f"isolated shared mask leaked action into video: {diff}"
    assert torch.isfinite(action_out).all()


def test_inject_validates_shapes_and_timestep():
    net, vb = _build()
    ids = torch.tensor([1, 2])
    st = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    with pytest.raises(ValueError, match="timestep"):
        vb.inject_shared_tokens(st, torch.randn(1, 2, DIM), 2)
    with pytest.raises(ValueError, match="does not match"):
        vb.inject_shared_tokens(st, torch.randn(1, 2, DIM + 1), 2, timestep=torch.tensor([1.0]))


def test_extract_rejects_inconsistent_lengths():
    net, vb = _build()
    ids = torch.tensor([1, 2])
    st = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    with pytest.raises(ValueError, match="inject_shared_tokens"):
        vb.extract_shared_tokens(st, 0)


def test_assert_ready_is_noop():
    net, vb = _build()
    ids = torch.tensor([1, 2])
    st = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    assert vb.assert_ready_for_shared_tokens(st) is None
