"""Tests for FastWAM-style proprio context conditioning."""

import pytest
import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture


class _ContextProprioArch(DualSystemSelfAttnArchitecture):
    """Tiny architecture shell that only exercises BaseWAMArchitecture helpers."""

    def __init__(self, *, state_dim: int = 14, text_dim: int = 32):
        super().__init__(None)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._init_proprio_context(
            {
                "use_proprioception": True,
                "state_dim": state_dim,
                "text_dim": text_dim,
            },
            text_dim=text_dim,
        )


def test_proprio_context_token_extends_context_and_mask():
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    inputs = {
        "context": torch.randn(2, 4, 16),
        "seq_lens": torch.tensor([2, 4]),
    }
    proprio = torch.randn(2, 7)

    out = arch._append_proprio_context_token(inputs, proprio)

    assert out["context"].shape == (2, 5, 16)
    assert out["context_mask"].shape == (2, 5)
    assert out["context_mask"][:, -1].all()
    assert out["context_mask"][0].tolist() == [True, True, False, False, True]
    assert out["seq_lens"].tolist() == [2, 4]


def test_proprio_context_requires_dataset_field():
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    with pytest.raises(ValueError, match="requires `proprio`"):
        arch._append_proprio_context_token({"context": torch.randn(1, 4, 16)}, None)


def test_proprio_context_validates_dim():
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    with pytest.raises(ValueError, match="last dim must be 7"):
        arch._append_proprio_context_token({"context": torch.randn(1, 4, 16)}, torch.randn(1, 6))


def test_action_dit_rejects_per_token_timestep():
    from openwam.model.action_backbone.separate_action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=1,
        video_dim=32,
        bridge_layers=(0,),
        variant="joint_self_attn",
    )
    with pytest.raises(ValueError, match="action timestep must be 1D"):
        context = torch.randn(2, 4, dit.text_dim)
        context_mask = torch.ones(2, 4, dtype=torch.bool)
        dit.prepare_state(torch.randn(2, 5, 7), torch.randn(2, 5), context=context, context_mask=context_mask)


def test_action_dit_rejects_proprio_constructor_flags():
    from openwam.model.action_backbone.separate_action_dit import ActionDiT

    with pytest.raises(TypeError, match="use_proprioception"):
        ActionDiT(
            action_dim=7,
            dim=32,
            ffn_dim=64,
            num_heads=4,
            num_layers=1,
            video_dim=32,
            bridge_layers=(0,),
            variant="joint_self_attn",
            use_proprioception=True,
            state_dim=7,
        )
