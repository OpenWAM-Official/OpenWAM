"""End-to-end smoke test: train → save → load → infer.

Exercises the full ActionDiT lifecycle through the BaseWAMArchitecture
interface using a tiny model (dim=64, 1 layer) and random data.
No GPU, no real pipeline, completes in seconds.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from openwam.model.dual_system import DualSystemArchitecture


def _make_tiny_architecture():
    """Create a minimal DualSystemArchitecture for testing."""
    cfg = {
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "video_dim": 64,
        "bridge_layers": (0,),
        "bridge_type": "cross_attn",
    }
    return DualSystemArchitecture(cfg=cfg)


def test_e2e_train_save_load_infer():
    """Smoke test: one training step, save, load, inference pass."""
    B, T, action_dim, video_dim = 1, 4, 7, 64

    # --- 1. Create tiny architecture ---
    arch = _make_tiny_architecture()
    arch.train()

    # --- 2. Fake data ---
    action_data = torch.randn(B, T, action_dim)
    action_noise = torch.randn_like(action_data)
    sigma = 0.5
    noisy_actions = (1 - sigma) * action_data + sigma * action_noise
    target = action_noise - action_data  # flow matching velocity

    timestep = torch.tensor([500.0])
    bridge_feature = torch.randn(B, T * 4, video_dim)  # fake video hidden

    # --- 3. Forward pass through architecture interface ---
    state = arch.prepare_action_tokens(noisy_actions, timestep)
    _, state = arch.on_dit_block(0, bridge_feature, state)
    pred = arch.extract_action_prediction(state)

    assert pred.shape == (B, T, action_dim), f"Expected {(B, T, action_dim)}, got {pred.shape}"

    # --- 4. Backward + optimizer step ---
    loss = F.mse_loss(pred, target)
    optimizer = torch.optim.Adam(arch.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"

    # --- 5. Save weights ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "action_dit.safetensors"
        from safetensors.torch import load_file, save_file

        save_file(arch.action_dit.state_dict(), str(ckpt_path))
        assert ckpt_path.exists()

        # --- 6. Load into fresh model ---
        arch2 = _make_tiny_architecture()
        loaded = load_file(str(ckpt_path))
        arch2.action_dit.load_state_dict(loaded, strict=True)

        # Verify weights match
        for (k1, v1), (k2, v2) in zip(
            arch.action_dit.state_dict().items(),
            arch2.action_dit.state_dict().items(),
        ):
            assert k1 == k2, f"Key mismatch: {k1} vs {k2}"
            assert torch.equal(v1, v2), f"Weight mismatch for {k1}"

    # --- 7. Inference pass ---
    arch2.eval()
    with torch.no_grad():
        infer_actions = torch.randn(B, T, action_dim)
        infer_timestep = torch.tensor([300.0])
        infer_bridge = torch.randn(B, T * 4, video_dim)

        state = arch2.prepare_action_tokens(infer_actions, infer_timestep)
        _, state = arch2.on_dit_block(0, infer_bridge, state)
        infer_pred = arch2.extract_action_prediction(state)

        assert infer_pred.shape == (B, T, action_dim)

    # --- 8. Verify denormalization properties ---
    assert arch2.action_mean.shape == (action_dim,)
    assert arch2.action_std.shape == (action_dim,)
    assert arch2.action_dim == action_dim
    assert arch2.bridge_layers == (0,)
    assert arch2.is_interleaved is False


def test_e2e_interleaved_forward_pass():
    """Smoke test: joint_self_attn architecture forward pass."""
    B, T, action_dim, video_dim = 1, 4, 7, 64

    cfg = {
        "action_dim": action_dim,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "video_dim": video_dim,
        "bridge_layers": (0,),
        "bridge_type": "joint_self_attn",
    }
    arch = DualSystemArchitecture(cfg=cfg)
    arch.eval()

    assert arch.is_interleaved is True

    noisy_actions = torch.randn(B, T, action_dim)
    timestep = torch.tensor([500.0])
    video_hidden = torch.randn(B, T * 4, video_dim)

    with torch.no_grad():
        state = arch.prepare_action_tokens(noisy_actions, timestep)
        assert "dit_state" in state.extra

        video_out, state = arch.on_dit_block(0, video_hidden, state)
        assert video_out.shape == video_hidden.shape

        pred = arch.extract_action_prediction(state)
        assert pred.shape == (B, T, action_dim)
