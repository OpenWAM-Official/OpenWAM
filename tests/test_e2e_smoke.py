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
    bridges = {bid: torch.randn(B, T * 4, video_dim) for bid in arch.bridge_layers}

    # --- 3. Forward pass through ActionDiT.forward (cross-attn variant) ---
    pred = arch.action_backbone(noisy_actions, bridges, timestep)
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
        infer_bridges = {bid: torch.randn(B, T * 4, video_dim) for bid in arch2.bridge_layers}
        infer_pred = arch2.action_backbone(infer_actions, infer_bridges, infer_timestep)
        assert infer_pred.shape == (B, T, action_dim)

    # --- 8. Verify action geometry properties ---
    assert arch2.action_dim == action_dim
    assert arch2.bridge_layers == (0,)


def test_e2e_interleaved_forward_pass():
    """Smoke test: joint_self_attn architecture pre/post_attn_at_layer round-trip."""
    from openwam.model.architectures.dual_system import DualSystemSelfAttnArchitecture

    B, T, action_dim, video_dim = 1, 4, 7, 64

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": action_dim,
        "dim": video_dim,
        "ffn_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "video_dim": video_dim,
        "bridge_layers": (0,),
    }
    arch = DualSystemSelfAttnArchitecture(cfg=cfg)
    arch.eval()

    noisy_actions = torch.randn(B, T, action_dim)
    timestep = torch.tensor([500.0])

    with torch.no_grad():
        ab = arch.action_backbone
        context = torch.randn(B, 4, ab.text_dim)
        context_mask = torch.ones(B, 4, dtype=torch.bool)
        astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)
        assert astate.payload is not None

        # MoT-style action half-step: pull q/k/v, simulate a mixed-attention
        # output, push it back through post_attn_at_layer.
        for layer_id in range(ab.num_layers):
            q, k, v, post = ab.pre_attn_at_layer(layer_id, astate)
            attn_out = torch.randn_like(q)
            astate = ab.post_attn_at_layer(layer_id, astate, attn_out, post)

        pred = ab.extract_prediction(astate)
        assert pred.shape == (B, T, action_dim)
