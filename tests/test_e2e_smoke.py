"""End-to-end smoke test: train → save → load → infer.

Exercises the full ActionDiT lifecycle through the BaseWAMArchitecture
interface using a tiny model (dim=64, 1 layer) and random data.
No GPU, no real pipeline, completes in seconds.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture
from openwam.model.base import ExecutionPlan


def _make_tiny_architecture():
    """Create a minimal DualSystemCrossAttnArchitecture for testing."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "video_dim": 64,
        "bridge_layers": (0,),
    }
    return DualSystemCrossAttnArchitecture(cfg=cfg)


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
    state = arch.action_backbone.prepare_state(noisy_actions, timestep)
    state.bridge_features.append(bridge_feature)
    pred = arch.action_backbone.extract_prediction(state)

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

        save_file(arch.action_backbone.state_dict(), str(ckpt_path))
        assert ckpt_path.exists()

        # --- 6. Load into fresh model ---
        arch2 = _make_tiny_architecture()
        loaded = load_file(str(ckpt_path))
        arch2.action_backbone.load_state_dict(loaded, strict=True)

        # Verify weights match
        for (k1, v1), (k2, v2) in zip(
            arch.action_backbone.state_dict().items(),
            arch2.action_backbone.state_dict().items(),
        ):
            assert k1 == k2, f"Key mismatch: {k1} vs {k2}"
            assert torch.equal(v1, v2), f"Weight mismatch for {k1}"

    # --- 7. Inference pass ---
    arch2.eval()
    with torch.no_grad():
        infer_actions = torch.randn(B, T, action_dim)
        infer_timestep = torch.tensor([300.0])
        infer_bridge = torch.randn(B, T * 4, video_dim)

        state = arch2.action_backbone.prepare_state(infer_actions, infer_timestep)
        state.bridge_features.append(infer_bridge)
        infer_pred = arch2.action_backbone.extract_prediction(state)

        assert infer_pred.shape == (B, T, action_dim)

    # --- 8. Verify denormalization properties ---
    assert arch2.action_mean.shape == (action_dim,)
    assert arch2.action_std.shape == (action_dim,)
    assert arch2.action_dim == action_dim
    assert arch2.bridge_layers == (0,)
    assert arch2.execution_plan == ExecutionPlan.BRIDGE_COLLECTION


def test_e2e_interleaved_forward_pass():
    """Smoke test: joint_self_attn architecture forward pass."""
    from openwam.model.architectures.dual_system import DualSystemSelfAttnArchitecture

    B, T, action_dim, video_dim = 1, 4, 7, 64

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": action_dim,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "video_dim": video_dim,
        "bridge_layers": (0,),
    }
    arch = DualSystemSelfAttnArchitecture(cfg=cfg)
    arch.eval()

    assert arch.execution_plan == ExecutionPlan.INTERLEAVED_SPLIT_SELF_ATTENTION

    noisy_actions = torch.randn(B, T, action_dim)
    timestep = torch.tensor([500.0])
    video_hidden = torch.randn(B, T * 4, video_dim)

    with torch.no_grad():
        state = arch.action_backbone.prepare_state(noisy_actions, timestep)
        assert state.runtime_state is not None
        assert state.runtime_state.variant == "joint_self_attn"

        class _MockVState:
            def __init__(self, x, t_mod, f, h, w):
                self.x = x
                self.reference_prefix_len = 0
                self.t_mod = t_mod
                self.f = f
                self.h = h
                self.w = w

        class _MockVB:
            def run_block(self, _bid, vs):
                return vs

        # video_hidden has T*4 tokens — pretend it's a 1D temporal axis.
        vstate = _MockVState(video_hidden, t_mod=torch.zeros(B, 6, video_dim), f=video_hidden.shape[1], h=1, w=1)
        vstate, state = arch.action_backbone.run_block(0, _MockVB(), vstate, state)
        assert vstate.x.shape == video_hidden.shape

        pred = arch.action_backbone.extract_prediction(state)
        assert pred.shape == (B, T, action_dim)
