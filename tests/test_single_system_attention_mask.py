"""SingleSystem attention mask layout and Wan adapter behavior tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from openwam.model.architectures.single_system.state import attach_shared_attention_mask
from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    ISOLATED,
    MUTUAL,
    VIDEO_SEES_ACTION,
    set_video_attention_mask_mode,
)
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.wan import action_tokens
from openwam.model.video_backbone.wan.models.dit import DiTBlock, modulate, rope_apply
from openwam.model.video_backbone.wan_backbone import Wan21


def _make_wan_backbone(*, dim: int = 24, num_heads: int = 4) -> Wan21:
    block = DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=48)
    block.eval()
    dit = SimpleNamespace(
        blocks=nn.ModuleList([block]),
        dim=dim,
        freq_dim=dim,
        time_embedding=nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)),
        time_projection=nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6)),
        video_attention_mask_mode="bidirectional",
    )
    return Wan21(SimpleNamespace(dit=dit))


def _identity_freqs(seq_len: int, head_dim: int) -> torch.Tensor:
    return torch.polar(
        torch.ones(seq_len, 1, head_dim // 2),
        torch.zeros(seq_len, 1, head_dim // 2),
    )


def _make_state(vb: Wan21, video: torch.Tensor, action: torch.Tensor, *, mask=None) -> BlockLoopState:
    x = torch.cat([video, action], dim=1)
    extras = {
        "dit": vb._dit,
        "vace": None,
        "time_embed": torch.zeros(x.shape[0], x.shape[1], x.shape[2]),
    }
    if mask is not None:
        extras["shared_attention_mask"] = mask
    return BlockLoopState(
        hidden_states=x,
        time_mod=torch.zeros(x.shape[0], x.shape[1], 6, x.shape[2]),
        rope_freqs=_identity_freqs(x.shape[1], vb.head_dim),
        context=torch.zeros(x.shape[0], 4, x.shape[2]),
        grid_frames=video.shape[1],
        grid_height=1,
        grid_width=1,
        extras=extras,
    )


def _old_masked_block_reference(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference copy of the old adapter-local masked block implementation."""
    has_seq = t_mod.dim() == 4
    chunk_dim = 2 if has_seq else 1
    chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        chunks = tuple(c.squeeze(2) for c in chunks)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    sa = block.self_attn
    q = sa.norm_q(sa.q(input_x))
    k = sa.norm_k(sa.k(input_x))
    v = sa.v(input_x)
    q = rope_apply(q, freqs, sa.num_heads)
    k = rope_apply(k, freqs, sa.num_heads)

    q = rearrange(q, "b s (n d) -> b n s d", n=sa.num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=sa.num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=sa.num_heads)
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    attn_out = rearrange(attn_out, "b n s d -> b s (n d)", n=sa.num_heads)

    x = block.gate(x, gate_msa, sa.o(attn_out))
    cross_mask = context_mask.unsqueeze(1).expand(-1, x.shape[1], -1).unsqueeze(1)
    x = x + block.cross_attn(block.norm3(x), context, ctx_mask=cross_mask)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    return block.gate(x, gate_mlp, block.ffn(input_x))


def _build_mask_via_attach(vb, *, n_video, n_action, n_state=0, mode=ACTION_SEES_VIDEO):
    """Drive the real shared path: attach builds the mask onto state.extras."""
    total = n_video + n_action + n_state
    state = BlockLoopState(
        hidden_states=torch.zeros(1, total, vb.dim),
        time_mod=torch.zeros(1, total, 6, vb.dim),
        rope_freqs=_identity_freqs(total, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        grid_frames=n_video,
        grid_height=1,
        grid_width=1,
        extras={},
    )
    attach_shared_attention_mask(vb, state, n_action, n_state=n_state, attention_mask_mode=mode)
    return state.extras["shared_attention_mask"]


def test_single_system_attach_mask_requires_extras():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        hidden_states=torch.zeros(1, 7, vb.dim),
        time_mod=torch.zeros(1, 7, 6, vb.dim),
        rope_freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        grid_frames=5,
        grid_height=1,
        grid_width=1,
        extras=None,
    )

    with pytest.raises(RuntimeError, match="shared_attention_mask"):
        attach_shared_attention_mask(vb, state, n_action=2, attention_mask_mode=ACTION_SEES_VIDEO)


def test_single_system_attach_mask_rejects_unknown_mode():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        hidden_states=torch.zeros(1, 7, vb.dim),
        time_mod=torch.zeros(1, 7, 6, vb.dim),
        rope_freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        grid_frames=5,
        grid_height=1,
        grid_width=1,
        extras={},
    )

    with pytest.raises(ValueError, match="attention_mask_mode"):
        attach_shared_attention_mask(vb, state, n_action=2, attention_mask_mode="bidirectional")


def test_single_system_set_video_attention_mask_mode_warns_when_not_settable(caplog):
    class ReadOnlyBackbone:
        @property
        def video_attention_mask_mode(self):
            return "bidirectional"

    with caplog.at_level("WARNING"):
        set_video_attention_mask_mode(ReadOnlyBackbone(), "first_frame_causal")

    assert "does not expose a settable property" in caplog.text


def test_single_system_action_rope_defaults_to_1d():
    vb = _make_wan_backbone(dim=32, num_heads=4)
    base = _identity_freqs(seq_len=2, head_dim=vb.head_dim)

    freqs = action_tokens.extend_freqs_with_action_tokens(base, n_action_tokens=3)

    assert freqs.shape == (5, 1, vb.head_dim // 2)
    assert torch.allclose(freqs[:2], base)
    # Action position 0 is identity, later action positions carry 1D RoPE phase.
    assert torch.allclose(freqs[2], base[0])
    assert not torch.allclose(freqs[3:], torch.ones_like(freqs[3:]))


def test_wan_action_tmod_broadcasts_scalar_to_batch():
    vb = _make_wan_backbone(dim=24, num_heads=4)

    t_mod = action_tokens.build_action_t_mod(torch.tensor([0.5]), n_action_tokens=3, dit=vb._dit, batch_size=2)

    assert t_mod.shape == (2, 3, 6, vb.dim)


def test_wan_action_tmod_accepts_per_sample_and_per_token():
    vb = _make_wan_backbone(dim=24, num_heads=4)

    per_sample = action_tokens.build_action_t_mod(
        torch.tensor([0.5, 0.8]), n_action_tokens=3, dit=vb._dit, batch_size=2
    )
    per_token = action_tokens.build_action_t_mod(torch.rand(2, 3), n_action_tokens=3, dit=vb._dit, batch_size=2)

    assert per_sample.shape == (2, 3, 6, vb.dim)
    assert per_token.shape == (2, 3, 6, vb.dim)


def test_wan_action_tmod_rejects_mismatched_shapes():
    vb = _make_wan_backbone(dim=24, num_heads=4)

    with pytest.raises(ValueError, match="action_timestep"):
        action_tokens.build_action_t_mod(torch.rand(3), n_action_tokens=3, dit=vb._dit, batch_size=2)
    with pytest.raises(ValueError, match="action_timestep"):
        action_tokens.build_action_t_mod(torch.rand(2, 2), n_action_tokens=3, dit=vb._dit, batch_size=2)


def test_single_system_attention_mask_action_sees_video_layout():
    vb = _make_wan_backbone()
    mask = _build_mask_via_attach(vb, n_video=5, n_action=3, mode=ACTION_SEES_VIDEO)
    Sv, Sa = 5, 3
    assert mask.shape == (Sv + Sa, Sv + Sa)
    assert mask.dtype == torch.bool
    assert mask[:Sv, :Sv].all()
    assert not mask[:Sv, Sv:].any()
    assert mask[Sv:, :Sv].all()
    assert mask[Sv:, Sv:].all()


def test_single_system_attention_mask_layout_with_state():
    vb = _make_wan_backbone()
    mask = _build_mask_via_attach(vb, n_video=5, n_action=3, n_state=2, mode=ACTION_SEES_VIDEO)
    Sv, Sa, Ss = 5, 3, 2
    v = slice(0, Sv)
    a = slice(Sv, Sv + Sa)
    s = slice(Sv + Sa, Sv + Sa + Ss)

    assert mask.shape == (Sv + Sa + Ss, Sv + Sa + Ss)
    assert mask[v, v].all()
    assert not mask[v, a].any()
    assert mask[v, s].all()
    assert mask[a, :].all()
    assert not mask[s, v].any()
    assert not mask[s, a].any()
    assert mask[s, s].all()


@pytest.mark.parametrize(
    "mode, v_sees_a, a_sees_all_v",
    [
        (MUTUAL, True, True),
        (ACTION_SEES_VIDEO, False, True),
        (VIDEO_SEES_ACTION, True, False),
        (ISOLATED, False, False),
    ],
)
def test_single_system_cross_modal_modes_with_state_tail(mode, v_sees_a, a_sees_all_v):
    """All four modes drive the shared mask; the state token stays a read-only
    tail (everyone sees it, it sees only itself). v↔v here is bidirectional
    (mock backbone), so tokens_per_frame=1 makes only the first video row a
    first-frame row."""
    vb = _make_wan_backbone()
    Sv, Sa, Ss, ff = 5, 3, 2, 1
    mask = _build_mask_via_attach(vb, n_video=Sv, n_action=Sa, n_state=Ss, mode=mode)
    u_start = Sv + Sa
    # a↔a full.
    assert mask[Sv:u_start, Sv:u_start].all()
    # state read-only tail.
    assert mask[:u_start, u_start:].all()
    assert mask[u_start:, u_start:].all()
    assert not mask[u_start:, :u_start].any()
    # v→a: first-frame row excluded; later rows follow the mode.
    assert not mask[:ff, Sv:u_start].any()
    assert mask[ff:Sv, Sv:u_start].all() if v_sees_a else not mask[ff:Sv, Sv:u_start].any()
    # a→v: all video, or first frame only.
    if a_sees_all_v:
        assert mask[Sv:u_start, :Sv].all()
    else:
        assert mask[Sv:u_start, :ff].all()
        assert not mask[Sv:u_start, ff:Sv].any()


def test_single_system_joint_mask_blocks_action_from_video_queries():
    torch.manual_seed(0)
    vb = _make_wan_backbone()
    vb.video_attention_mask_mode = "bidirectional"

    B, Sv, Sa, D = 1, 4, 3, vb.dim
    video = torch.randn(B, Sv, D)
    action_a = torch.randn(B, Sa, D)
    action_b = torch.randn(B, Sa, D) + 10.0

    mask = _build_mask_via_attach(vb, n_video=Sv, n_action=Sa, mode=ACTION_SEES_VIDEO)

    with torch.no_grad():
        out_joint_a = vb.run_block(0, _make_state(vb, video, action_a, mask=mask)).hidden_states[:, :Sv]
        out_joint_b = vb.run_block(0, _make_state(vb, video, action_b, mask=mask)).hidden_states[:, :Sv]
        out_bidir_a = vb.run_block(0, _make_state(vb, video, action_a, mask=None)).hidden_states[:, :Sv]
        out_bidir_b = vb.run_block(0, _make_state(vb, video, action_b, mask=None)).hidden_states[:, :Sv]

    assert torch.allclose(out_joint_a, out_joint_b, atol=0, rtol=0)
    assert not torch.allclose(out_bidir_a, out_bidir_b)


def test_dit_block_masked_forward_matches_old_adapter_reference():
    torch.manual_seed(0)
    dim, num_heads, seq_len, context_len = 24, 4, 7, 5
    block = DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=48).eval()
    x = torch.randn(2, seq_len, dim)
    context = torch.randn(2, context_len, dim)
    context_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    t_mod = torch.randn(2, seq_len, 6, dim) * 0.01
    freqs = _identity_freqs(seq_len, dim // num_heads)
    self_attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    self_attn_mask[:3, 4:] = False

    with torch.no_grad():
        old = _old_masked_block_reference(block, x, context, context_mask, t_mod, freqs, self_attn_mask)
        new = block(
            x,
            context,
            t_mod,
            freqs,
            context_mask=context_mask.unsqueeze(1).expand(-1, seq_len, -1),
            self_attn_mask=self_attn_mask,
        )

    assert torch.allclose(new, old, atol=1e-6, rtol=1e-5)


def test_dit_block_masked_forward_shape():
    torch.manual_seed(0)
    dim, num_heads, seq_len = 24, 4, 7
    block = DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=48).eval()
    x = torch.randn(2, seq_len, dim)
    context = torch.randn(2, 5, dim)
    t_mod = torch.randn(2, seq_len, 6, dim) * 0.01
    freqs = _identity_freqs(seq_len, dim // num_heads)
    self_attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    with torch.no_grad():
        out = block(x, context, t_mod, freqs, self_attn_mask=self_attn_mask)

    assert out.shape == x.shape


def test_wan_shared_token_injection_extends_tmod_freqs_and_extracts_action_only():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Sa, Ss, D = 2, 4, 3, 1, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
        extras={"dit": vb._dit, "vace": None, "time_embed": torch.zeros(B, Sv, D)},
    )
    action_tokens = torch.randn(B, Sa, D)
    state_tokens = torch.randn(B, Ss, D)
    timestep = torch.tensor([0.25, 0.75])

    state = vb.inject_shared_tokens(
        state,
        action_tokens,
        Sa,
        state_tokens=state_tokens,
        n_state=Ss,
        timestep=timestep,
    )

    assert state.hidden_states.shape == (B, Sv + Sa + Ss, D)
    assert state.time_mod.shape == (B, Sv + Sa + Ss, 6, D)
    assert state.rope_freqs.shape[0] == Sv + Sa + Ss

    state, action_tail = vb.extract_shared_tokens(state, Sa, n_state=Ss)
    assert action_tail.shape == (B, Sa, D)
    assert torch.allclose(action_tail, action_tokens)
    assert state.hidden_states.shape == (B, Sv, D)
    assert state.time_mod.shape == (B, Sv, 6, D)
    assert state.rope_freqs.shape[0] == Sv


def test_wan_shared_token_injection_supports_state_only_video_conditioning():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Ss, D = 2, 4, 1, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
        extras={"dit": vb._dit, "vace": None},
    )
    state_tokens = torch.randn(B, Ss, D)

    state = vb.inject_shared_tokens(
        state,
        None,
        0,
        state_tokens=state_tokens,
        n_state=Ss,
        timestep=torch.tensor([0.25, 0.75]),
    )

    assert state.hidden_states.shape == (B, Sv + Ss, D)
    assert state.time_mod.shape == (B, Sv + Ss, 6, D)
    assert state.rope_freqs.shape[0] == Sv + Ss
    state, action_tail = vb.extract_shared_tokens(state, 0, n_state=Ss)
    assert action_tail.shape == (B, 0, D)
    assert state.hidden_states.shape == (B, Sv, D)
    assert state.time_mod.shape == (B, Sv, 6, D)
    assert state.rope_freqs.shape[0] == Sv


def test_wan_shared_token_injection_rejects_batch_mismatch():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Sa, Ss, D = 2, 4, 3, 1, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
    )

    with pytest.raises(ValueError, match="video batch=2, action batch=3"):
        vb.inject_shared_tokens(
            state,
            torch.randn(3, Sa, D),
            Sa,
            state_tokens=torch.randn(B, Ss, D),
            n_state=Ss,
            timestep=torch.tensor([0.25, 0.75]),
        )

    with pytest.raises(ValueError, match="video batch=2, state batch=3"):
        vb.inject_shared_tokens(
            state,
            torch.randn(B, Sa, D),
            Sa,
            state_tokens=torch.randn(3, Ss, D),
            n_state=Ss,
            timestep=torch.tensor([0.25, 0.75]),
        )


def test_wan_shared_token_injection_requires_timestep_for_per_token_tmod():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Sa, D = 2, 4, 3, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
    )

    with pytest.raises(ValueError, match="requires `timestep`"):
        vb.inject_shared_tokens(state, torch.randn(B, Sa, D), Sa)


def test_wan_state_tmod_uses_sample_level_timestep_from_per_token_input():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    per_token = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

    state_tmod = action_tokens.build_sample_t_mod(per_token, n_tokens=2, dit=vb._dit, batch_size=2)
    first_tmod = action_tokens.build_sample_t_mod(per_token[:, 0], n_tokens=2, dit=vb._dit, batch_size=2)

    assert state_tmod.shape == (2, 2, 6, vb.dim)
    assert torch.allclose(state_tmod, first_tmod)
