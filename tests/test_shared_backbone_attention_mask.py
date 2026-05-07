"""SharedBackbone attention mask layout and Wan adapter behavior tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from openwam.model.architectures.shared_backbone.mask import (
    attach_shared_attention_mask,
    build_shared_backbone_attention_mask,
    set_video_attention_mask_mode,
)
from openwam.model.video_backbone.adapter import BlockLoopState
from openwam.model.video_backbone.wan.dit import DiTBlock
from openwam.model.video_backbone.wan_adapter import WanVideoBackbone


def _make_wan_backbone(*, dim: int = 24, num_heads: int = 4) -> WanVideoBackbone:
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
    return WanVideoBackbone(SimpleNamespace(dit=dit))


def _identity_freqs(seq_len: int, head_dim: int) -> torch.Tensor:
    return torch.polar(
        torch.ones(seq_len, 1, head_dim // 2),
        torch.zeros(seq_len, 1, head_dim // 2),
    )


def _make_state(vb: WanVideoBackbone, video: torch.Tensor, action: torch.Tensor, *, mask=None) -> BlockLoopState:
    x = torch.cat([video, action], dim=1)
    extras = {"dit": vb._dit, "vace": None, "use_usp": False}
    if mask is not None:
        extras["shared_attention_mask"] = mask
    return BlockLoopState(
        x=x,
        t_mod=torch.zeros(x.shape[0], x.shape[1], 6, x.shape[2]),
        freqs=_identity_freqs(x.shape[1], vb.head_dim),
        context=torch.zeros(x.shape[0], 4, x.shape[2]),
        f=video.shape[1],
        h=1,
        w=1,
        t=torch.zeros(x.shape[0], x.shape[1], x.shape[2]),
        extras=extras,
    )


def test_shared_backbone_attention_mask_bidirectional_returns_none():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        x=torch.zeros(1, 7, vb.dim),
        t_mod=torch.zeros(1, 7, 6, vb.dim),
        freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        f=5,
        h=1,
        w=1,
    )

    mask = build_shared_backbone_attention_mask(vb, state, n_action=2, attention_mask_mode="bidirectional")
    assert mask is None


def test_shared_backbone_attach_mask_bidirectional_allows_missing_extras():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        x=torch.zeros(1, 7, vb.dim),
        t_mod=torch.zeros(1, 7, 6, vb.dim),
        freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        f=5,
        h=1,
        w=1,
        extras=None,
    )

    attach_shared_attention_mask(vb, state, n_action=2, attention_mask_mode="bidirectional")


def test_shared_backbone_attach_mask_joint_requires_extras():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        x=torch.zeros(1, 7, vb.dim),
        t_mod=torch.zeros(1, 7, 6, vb.dim),
        freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        f=5,
        h=1,
        w=1,
        extras=None,
    )

    with pytest.raises(RuntimeError, match="shared_attention_mask"):
        attach_shared_attention_mask(vb, state, n_action=2, attention_mask_mode="joint")


def test_shared_backbone_attach_mask_joint_rejects_usp():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        x=torch.zeros(1, 7, vb.dim),
        t_mod=torch.zeros(1, 7, 6, vb.dim),
        freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        f=5,
        h=1,
        w=1,
        extras={"use_usp": True},
    )

    with pytest.raises(NotImplementedError, match="unified sequence parallel"):
        attach_shared_attention_mask(vb, state, n_action=2, attention_mask_mode="joint")


def test_shared_backbone_set_video_attention_mask_mode_warns_when_not_settable(caplog):
    class ReadOnlyBackbone:
        @property
        def video_attention_mask_mode(self):
            return "bidirectional"

    with caplog.at_level("WARNING"):
        set_video_attention_mask_mode(ReadOnlyBackbone(), "first_frame_causal")

    assert "does not expose a settable property" in caplog.text


def test_shared_backbone_action_rope_defaults_to_1d():
    vb = _make_wan_backbone(dim=32, num_heads=4)
    base = _identity_freqs(seq_len=2, head_dim=vb.head_dim)

    freqs = vb._extend_freqs_with_action_tokens(base, n_action_tokens=3)

    assert freqs.shape == (5, 1, vb.head_dim // 2)
    assert torch.allclose(freqs[:2], base)
    # Action position 0 is identity, later action positions carry 1D RoPE phase.
    assert torch.allclose(freqs[2], base[0])
    assert not torch.allclose(freqs[3:], torch.ones_like(freqs[3:]))


def test_wan_action_tmod_broadcasts_scalar_to_batch():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    bias = torch.zeros(1, 1, 6, vb.dim)

    t_mod = vb._build_action_t_mod(torch.tensor([0.5]), bias, n_action_tokens=3, batch_size=2)

    assert t_mod.shape == (2, 3, 6, vb.dim)


def test_wan_action_tmod_accepts_per_sample_and_per_token():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    bias = torch.zeros(1, 1, 6, vb.dim)

    per_sample = vb._build_action_t_mod(torch.tensor([0.5, 0.8]), bias, n_action_tokens=3, batch_size=2)
    per_token = vb._build_action_t_mod(torch.rand(2, 3), bias, n_action_tokens=3, batch_size=2)

    assert per_sample.shape == (2, 3, 6, vb.dim)
    assert per_token.shape == (2, 3, 6, vb.dim)


def test_wan_action_tmod_rejects_mismatched_shapes():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    bias = torch.zeros(1, 1, 6, vb.dim)

    with pytest.raises(ValueError, match="action_timestep"):
        vb._build_action_t_mod(torch.rand(3), bias, n_action_tokens=3, batch_size=2)
    with pytest.raises(ValueError, match="action_timestep"):
        vb._build_action_t_mod(torch.rand(2, 2), bias, n_action_tokens=3, batch_size=2)


def test_shared_backbone_attention_mask_joint_layout():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        x=torch.zeros(1, 8, vb.dim),
        t_mod=torch.zeros(1, 8, 6, vb.dim),
        freqs=_identity_freqs(8, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        f=5,
        h=1,
        w=1,
    )

    mask = build_shared_backbone_attention_mask(vb, state, n_action=3, attention_mask_mode="joint")
    Sv, Sa = 5, 3
    assert mask.shape == (Sv + Sa, Sv + Sa)
    assert mask.dtype == torch.bool
    assert mask[:Sv, :Sv].all()
    assert not mask[:Sv, Sv:].any()
    assert mask[Sv:, :Sv].all()
    assert mask[Sv:, Sv:].all()


def test_shared_backbone_joint_mask_blocks_action_from_video_queries():
    torch.manual_seed(0)
    vb = _make_wan_backbone()
    vb.video_attention_mask_mode = "bidirectional"

    B, Sv, Sa, D = 1, 4, 3, vb.dim
    video = torch.randn(B, Sv, D)
    action_a = torch.randn(B, Sa, D)
    action_b = torch.randn(B, Sa, D) + 10.0

    mask_state = BlockLoopState(
        x=torch.zeros(B, Sv + Sa, D),
        t_mod=torch.zeros(B, Sv + Sa, 6, D),
        freqs=_identity_freqs(Sv + Sa, vb.head_dim),
        context=torch.zeros(B, 4, D),
        f=Sv,
        h=1,
        w=1,
    )
    mask = build_shared_backbone_attention_mask(vb, mask_state, n_action=Sa, attention_mask_mode="joint")

    with torch.no_grad():
        out_joint_a = vb.run_block(0, _make_state(vb, video, action_a, mask=mask)).x[:, :Sv]
        out_joint_b = vb.run_block(0, _make_state(vb, video, action_b, mask=mask)).x[:, :Sv]
        out_bidir_a = vb.run_block(0, _make_state(vb, video, action_a, mask=None)).x[:, :Sv]
        out_bidir_b = vb.run_block(0, _make_state(vb, video, action_b, mask=None)).x[:, :Sv]

    assert torch.allclose(out_joint_a, out_joint_b, atol=0, rtol=0)
    assert not torch.allclose(out_bidir_a, out_bidir_b)
