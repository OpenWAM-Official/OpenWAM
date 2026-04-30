"""Tests for proprioceptive state conditioning."""

import pytest
import torch

from openwam.model.action_backbone.proprioceptive import ProprioceptiveEncoder


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


def test_sequence_concat_mode_shape():
    """sequence_concat mode should add extra tokens."""
    enc = ProprioceptiveEncoder(state_dim=14, hidden_dim=64, mode="sequence_concat", num_state_tokens=4)
    action_embeds = torch.randn(2, 49, 64)
    state = torch.randn(2, 14)
    out = enc(action_embeds, state)

    assert out.shape == (2, 53, 64)  # 49 + 4


def test_concat_alias_backcompat():
    """mode='concat' must remain a working alias of 'sequence_concat'."""
    enc_alias = ProprioceptiveEncoder(state_dim=14, hidden_dim=64, mode="concat", num_state_tokens=4)
    enc_canonical = ProprioceptiveEncoder(state_dim=14, hidden_dim=64, mode="sequence_concat", num_state_tokens=4)
    assert enc_alias.extra_tokens == enc_canonical.extra_tokens == 4
    action_embeds = torch.randn(2, 49, 64)
    state = torch.randn(2, 14)
    # Shape parity is enough — identical numerical init isn't guaranteed
    # because nn.Linear params are independent RNG draws.
    assert enc_alias(action_embeds, state).shape == enc_canonical(action_embeds, state).shape


def test_extra_tokens_property():
    """extra_tokens should match mode."""
    enc_add = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="add")
    enc_cat = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="sequence_concat", num_state_tokens=8)
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
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="sequence_concat", num_state_tokens=4)
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
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="sequence_concat", num_state_tokens=2)
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


# --- channel_concat mode (channel-wise concat per figure) ---


def test_channel_concat_mode_preserves_seq_len():
    """channel_concat should keep the sequence length unchanged."""
    enc = ProprioceptiveEncoder(state_dim=14, hidden_dim=64, mode="channel_concat")
    action_embeds = torch.randn(2, 49, 64)
    state = torch.randn(2, 14)
    out = enc(action_embeds, state)

    assert out.shape == (2, 49, 64)
    assert enc.extra_tokens == 0


def test_channel_concat_zero_init_matches_baseline():
    """At init the channel_concat encoder should be a no-op on action_embeds.

    This matters because it guarantees that enabling proprioception on a
    pretrained checkpoint starts training from identical behavior to the
    baseline (no regression at step 0).
    """
    enc = ProprioceptiveEncoder(state_dim=14, hidden_dim=32, mode="channel_concat")
    action_embeds = torch.randn(3, 16, 32)
    # Use a non-zero state to make the test meaningful — the encoder's
    # zero-init MUST make the state contribution vanish regardless of value.
    state = torch.randn(3, 14) * 10.0

    out = enc(action_embeds, state)

    diff = (out - action_embeds).abs().max().item()
    assert diff < 1e-5, f"Expected channel_concat identity at init, got max diff {diff}"


def test_channel_concat_gradient_flow():
    """Gradients should flow through both the action path and state path."""
    enc = ProprioceptiveEncoder(state_dim=7, hidden_dim=32, mode="channel_concat")
    # Break zero-init on both the encoder MLP AND the state half of
    # channel_merge so the state path carries a non-zero gradient.
    with torch.no_grad():
        enc.encoder[-1].weight.fill_(0.1)
        enc.channel_merge.weight[:, enc.hidden_dim :].fill_(0.05)

    action_embeds = torch.randn(2, 10, 32, requires_grad=True)
    state = torch.randn(2, 7, requires_grad=True)

    out = enc(action_embeds, state)
    out.sum().backward()

    assert action_embeds.grad is not None
    assert state.grad is not None
    assert state.grad.abs().max().item() > 0


def test_action_dit_end_to_end_channel_concat():
    """ActionDiT with channel_concat proprio should produce correctly-shaped predictions."""
    from openwam.model.action_backbone.action_dit import ActionDiT

    bridge_layers = (0, 1)
    dit = ActionDiT(
        action_dim=14,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=len(bridge_layers),
        video_dim=48,
        bridge_layers=bridge_layers,
        variant="joint_cross_attn",
        detach_bridge=False,
        use_proprioception=True,
        proprio_fusion="channel_concat",
    )

    B, T_a, T_v = 2, 8, 5
    action_tokens = torch.randn(B, T_a, 14)
    timestep = torch.randint(0, 1000, (B,)).float()
    video_features = [torch.randn(B, T_v, 48) for _ in bridge_layers]
    proprio = torch.randn(B, 14)

    out = dit(action_tokens, video_features, timestep, proprio_state=proprio)
    assert out.shape == (B, T_a, 14)


def test_action_dit_end_to_end_sequence_concat():
    """ActionDiT with sequence_concat should slice state tokens off the output."""
    from openwam.model.action_backbone.action_dit import ActionDiT

    bridge_layers = (0, 1)
    num_state_tokens = 3
    dit = ActionDiT(
        action_dim=14,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=len(bridge_layers),
        video_dim=48,
        bridge_layers=bridge_layers,
        variant="joint_cross_attn",
        detach_bridge=False,
        use_proprioception=True,
        proprio_fusion="sequence_concat",
        num_state_tokens=num_state_tokens,
    )

    B, T_a, T_v = 2, 8, 5
    action_tokens = torch.randn(B, T_a, 14)
    timestep = torch.randint(0, 1000, (B,)).float()
    video_features = [torch.randn(B, T_v, 48) for _ in bridge_layers]
    proprio = torch.randn(B, 14)

    out = dit(action_tokens, video_features, timestep, proprio_state=proprio)
    # Output should be aligned with action seq length, not (T_a + num_state_tokens).
    assert out.shape == (B, T_a, 14)
    assert dit.num_proprio_tokens == num_state_tokens


def test_action_dit_joint_self_attn_sequence_concat():
    """joint_self_attn + sequence_concat must not corrupt the video prefix.

    Regression test for a name-collision bug: an earlier version reused
    ``ActionDiTState.skip_prefix_tokens`` — a field read by wan_video.py to
    strip *video* reference-frame prefix — to carry the *action*-side
    proprio prefix count. That leaked the action count into video slicing
    when this combo was exercised.

    Here we just confirm the dual-stream forward runs end-to-end with the
    combined setting and that ``skip_prefix_tokens`` (video semantics)
    stays at 0 while ``action_prefix_tokens`` picks up the proprio count.
    """
    from openwam.model.action_backbone.action_dit import ActionDiT

    bridge_layers = (0, 1)
    num_state_tokens = 3
    dit = ActionDiT(
        action_dim=14,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=len(bridge_layers),
        video_dim=32,
        bridge_layers=bridge_layers,
        variant="joint_self_attn",
        use_proprioception=True,
        proprio_fusion="sequence_concat",
        num_state_tokens=num_state_tokens,
    )

    B, T_a, T_v = 2, 8, 5
    action_tokens = torch.randn(B, T_a, 14)
    timestep = torch.randint(0, 1000, (B,)).float()
    video_features = [torch.randn(B, T_v, 32) for _ in bridge_layers]
    proprio = torch.randn(B, 14)

    # Full forward — exercises the joint_self_attn code path.
    out = dit(action_tokens, video_features, timestep, proprio_state=proprio)
    assert out.shape == (B, T_a, 14)

    # prepare_action_state is what feeds wan_video.py; verify the field
    # split is correct.
    state = dit.prepare_action_state(action_tokens, timestep, proprio_state=proprio)
    assert state.skip_prefix_tokens == 0, (
        "skip_prefix_tokens is reserved for video reference-frame slicing; the proprio count must not leak into it."
    )
    assert state.action_prefix_tokens == num_state_tokens
    final = dit.finalize_action_output(state)
    assert final.shape == (B, T_a, 14)


def test_action_dit_asserts_when_state_missing():
    """Building with use_proprioception=True but forwarding None should fail loudly."""
    from openwam.model.action_backbone.action_dit import ActionDiT

    bridge_layers = (0,)
    dit = ActionDiT(
        action_dim=14,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=1,
        video_dim=32,
        bridge_layers=bridge_layers,
        variant="joint_cross_attn",
        detach_bridge=False,
        use_proprioception=True,
        proprio_fusion="channel_concat",
    )
    action_tokens = torch.randn(1, 4, 14)
    video_features = [torch.randn(1, 3, 32)]
    timestep = torch.tensor([0.0])
    with pytest.raises(AssertionError, match="proprio_state=None"):
        dit(action_tokens, video_features, timestep, proprio_state=None)


def test_action_dit_proprio_disabled_by_default():
    """When use_proprioception=False the encoder should be None."""
    from openwam.model.action_backbone.action_dit import ActionDiT

    bridge_layers = (0,)
    dit = ActionDiT(
        action_dim=14,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=1,
        video_dim=32,
        bridge_layers=bridge_layers,
        variant="joint_cross_attn",
        detach_bridge=False,
    )
    assert dit.proprio_encoder is None
    assert dit.num_proprio_tokens == 0
