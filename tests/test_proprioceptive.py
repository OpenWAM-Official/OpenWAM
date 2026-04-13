"""Tests for proprioceptive state conditioning."""

import pytest
import torch

from openwam.model.action_model.proprioceptive import ProprioceptiveEncoder


def test_import():
    """ProprioceptiveEncoder should be importable from models package."""
    from openwam.model import ProprioceptiveEncoder

    assert callable(ProprioceptiveEncoder)


def test_add_mode_shape():
    """Add mode should preserve sequence length."""
    enc = ProprioceptiveEncoder(state_dim=14, hidden_dim=64, mode="add")
    action_embeds = torch.randn(2, 49, 64)
    state = torch.randn(2, 14)
    out = enc(action_embeds, state)

    assert out.shape == (2, 49, 64)


def test_concat_mode_shape():
    """Concat mode should add extra tokens."""
    enc = ProprioceptiveEncoder(state_dim=14, hidden_dim=64, mode="concat", num_state_tokens=4)
    action_embeds = torch.randn(2, 49, 64)
    state = torch.randn(2, 14)
    out = enc(action_embeds, state)

    assert out.shape == (2, 53, 64)  # 49 + 4


def test_extra_tokens_property():
    """extra_tokens should match mode."""
    enc_add = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="add")
    enc_cat = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="concat", num_state_tokens=8)
    assert enc_add.extra_tokens == 0
    assert enc_cat.extra_tokens == 8


def test_zero_init():
    """Output should be near-zero at initialization (preserves pretrained behavior)."""
    enc = ProprioceptiveEncoder(state_dim=14, hidden_dim=64, mode="add")
    action_embeds = torch.randn(1, 10, 64)
    state = torch.randn(1, 14)
    out = enc(action_embeds, state)

    # With zero-init, the state contribution should be ~0
    diff = (out - action_embeds).abs().max().item()
    assert diff < 1e-5, f"Expected near-zero init, got max diff {diff}"


def test_zero_init_concat():
    """Concat mode state tokens should also start near-zero."""
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="concat", num_state_tokens=4)
    state = torch.randn(1, 7)
    action_embeds = torch.randn(1, 10, 32)
    out = enc(action_embeds, state)

    # State tokens (first 4) should be near-zero
    state_tokens = out[:, :4, :]
    assert state_tokens.abs().max().item() < 1e-5


def test_gradient_flow():
    """Gradients should flow through the encoder."""
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="add")
    action_embeds = torch.randn(1, 10, 32, requires_grad=True)
    state = torch.randn(1, 7, requires_grad=True)

    out = enc(action_embeds, state)
    loss = out.sum()
    loss.backward()

    assert state.grad is not None
    assert action_embeds.grad is not None


def test_gradient_flow_concat():
    """Concat mode should also propagate gradients."""
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="concat", num_state_tokens=2)
    state = torch.randn(1, 7, requires_grad=True)
    action_embeds = torch.randn(1, 10, 32, requires_grad=True)

    out = enc(action_embeds, state)
    loss = out.sum()
    loss.backward()

    assert state.grad is not None


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="invalid")


def test_batch_independence():
    """Different batch elements should produce different outputs."""
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="add")
    # Break zero-init for test
    with torch.no_grad():
        enc.encoder[-1].weight.fill_(0.1)

    action_embeds = torch.randn(2, 10, 32)
    state = torch.randn(2, 7)
    state[0] = 0.0  # Different states
    state[1] = 1.0

    out = enc(action_embeds, state)
    # Outputs should differ because states differ
    assert not torch.allclose(out[0], out[1])


def test_various_state_dims():
    """Should work with common proprioceptive dims."""
    for state_dim in [7, 13, 14, 20]:
        enc = ProprioceptiveEncoder(state_dim=state_dim, hidden_dim=64, mode="add")
        out = enc(torch.randn(1, 49, 64), torch.randn(1, state_dim))
        assert out.shape == (1, 49, 64)


def test_dropout():
    """Dropout should be applied in training mode."""
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="add", dropout=0.5)
    # Break zero-init for test
    with torch.no_grad():
        enc.encoder[-1].weight.fill_(1.0)

    enc.train()
    action_embeds = torch.randn(1, 10, 32)
    state = torch.ones(1, 7)

    # Run multiple times — with dropout, outputs should vary
    outs = [enc(action_embeds, state) for _ in range(10)]
    # At least some should differ
    all_same = all(torch.allclose(outs[0], o) for o in outs[1:])
    assert not all_same, "Dropout should cause variation in training mode"
