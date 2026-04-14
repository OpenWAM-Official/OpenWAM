"""GPU-based end-to-end DualSystem architecture test.

Loads video backbone + ActionDiT from YAML configs, runs joint forward pass
(video DiT + bridge feature extraction + ActionDiT), and prints all shapes.

Tests all three bridge_type modes:
  - cross_attn: unidirectional video->action
  - cross_attn_detach: same but with gradient detachment
  - joint_self_attn: bidirectional MMDiT-style (modifies video hidden)

Usage:
    python scripts/model/dual_system_gpu_test.py
    python scripts/model/dual_system_gpu_test.py --model_path /path/to/model
    python scripts/model/dual_system_gpu_test.py --bridge_type joint_self_attn

GPU memory requirements (approximate):
    VACE-1.3B + ActionDiT:  ~17 GB
    TI2V-5B + ActionDiT:    ~25 GB
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "scripts/model")
sys.path.insert(0, "scripts/model/video_backbone")
from _config_utils import apply_freeze, load_architecture_config, merge_arch_params, print_freeze_status
from video_backbone_gpu_load import load_backbone


def build_arch(cfg, video_dim, bridge_type_override=None, device="cuda"):
    """Build DualSystem architecture, optionally overriding bridge_type."""
    params = merge_arch_params(cfg, video_dim=video_dim)
    if bridge_type_override:
        params["bridge_type"] = bridge_type_override
    bridge_layers = tuple(params.get("bridge_layers", []))

    from openwam.model import build_architecture

    arch = build_architecture(params.pop("type"), params)
    arch.action_dit.to(dtype=torch.bfloat16, device=device)
    return arch, bridge_layers


def test_cross_attn_mode(arch, bridge_layers, noisy_actions, bridge_features, a_timestep, bridge_type):
    """Test cross_attn / cross_attn_detach: collect features then run ActionDiT."""
    print(f"\n--- Testing bridge_type='{bridge_type}' ---")

    # Verify block type
    block_cls = type(arch.action_dit.blocks[0]).__name__
    print(f"  Block class: {block_cls}")
    assert block_cls == "ActionDiTBlock", f"Expected ActionDiTBlock, got {block_cls}"
    assert not hasattr(arch.action_dit, "video_back_projs") or not isinstance(
        arch.action_dit.video_back_projs, torch.nn.ModuleList
    ), "cross_attn should NOT have video_back_projs"
    print(f"  video_back_projs: absent (correct for {bridge_type})")

    # Forward pass
    with torch.no_grad():
        action_pred = arch.action_dit(noisy_actions, bridge_features, a_timestep)
    print(f"  action_pred: {action_pred.shape}")

    # Verify is NOT interleaved
    assert not arch.is_interleaved, f"{bridge_type} should NOT be interleaved"
    print(f"  is_interleaved: {arch.is_interleaved} (correct)")

    return action_pred


def test_joint_self_attn_mode(arch, bridge_layers, noisy_actions, bridge_features, a_timestep):
    """Test joint_self_attn: interleaved execution with video back-projection."""
    print("\n--- Testing bridge_type='joint_self_attn' ---")

    # Verify block type
    block_cls = type(arch.action_dit.blocks[0]).__name__
    print(f"  Block class: {block_cls}")
    assert block_cls == "JointActionDiTBlock", f"Expected JointActionDiTBlock, got {block_cls}"

    # Verify back-projection exists and is zero-initialized
    assert hasattr(arch.action_dit, "video_back_projs"), "joint_self_attn MUST have video_back_projs"
    bp_weight = arch.action_dit.video_back_projs[0].weight
    is_zero = torch.all(bp_weight == 0).item()
    print(f"  video_back_projs: present, zero-init={is_zero}")

    # Verify is interleaved
    assert arch.is_interleaved, "joint_self_attn SHOULD be interleaved"
    print(f"  is_interleaved: {arch.is_interleaved} (correct)")

    # Test via architecture hooks (prepare → on_dit_block → extract)
    state = arch.prepare_action_tokens(noisy_actions, a_timestep)
    video_hidden = bridge_features[0].clone()

    # Simulate video DiT loop with interleaved action blocks
    bridge_idx = 0
    max_block = max(bridge_layers) + 1
    for block_id in range(max_block):
        video_hidden_new, state = arch.on_dit_block(block_id, video_hidden, state)
        if block_id in set(bridge_layers):
            # Check video_hidden was modified by back-projection
            changed = not torch.allclose(video_hidden_new, video_hidden, atol=1e-6)
            print(f"  block {block_id}: video_hidden modified={changed}")
            video_hidden = video_hidden_new
            bridge_idx += 1

    action_pred = arch.extract_action_prediction(state)
    print(f"  action_pred: {action_pred.shape}")
    return action_pred


def main():
    parser = argparse.ArgumentParser(description="End-to-end DualSystem GPU test (all bridge types)")
    parser.add_argument("--arch", type=str, default="configs/model/dual_system.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--bridge_type", type=str, default=None, help="Test specific bridge_type. Default: test all three."
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_architecture_config(args.arch)
    video_cfg = cfg["video_backbone"]

    model_path = args.model_path or video_cfg.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        parser.error("model_path not found. Set in video_backbone YAML or pass --model_path.")

    bridge_types = [args.bridge_type] if args.bridge_type else ["cross_attn", "cross_attn_detach", "joint_self_attn"]

    print("=" * 60)
    print("DualSystem End-to-End GPU Test (all bridge types)")
    print(f"  Architecture:  {args.arch}")
    print(f"  Model path:    {model_path}")
    print(f"  Device:        {args.device}")
    print(f"  bridge_types:  {bridge_types}")
    print("=" * 60)

    # --- Load video backbone (once, shared across bridge types) ---
    print("\n[1/5] Loading video backbone...")
    pipe = load_backbone(model_path, device=args.device)
    video_dim = pipe.dit.dim
    print(f"  video_dim: {video_dim}, layers: {len(pipe.dit.blocks)}")

    # --- Apply freeze strategy ---
    print("\n[2/5] Applying freeze strategy...")
    freeze_list = cfg.get("freeze", ["text_encoder", "vae", "image_encoder"])
    apply_freeze(pipe, freeze_list)

    # Build one arch instance for status display
    _display_arch, _ = build_arch(cfg, video_dim, device=args.device)
    print_freeze_status(pipe, _display_arch.action_dit, "dual_system", freeze_list)
    del _display_arch

    # --- Create fake inputs (shared) ---
    print("\n[3/5] Creating fake inputs...")
    height = video_cfg["resolution"]["height"]
    width = video_cfg["resolution"]["width"]
    B, action_dim = 1, int(cfg["architecture"]["action_dim"])

    latent_h, latent_w = height // 8, width // 8
    latent_t = (9 - 1) // 4 + 1
    in_dim = pipe.dit.in_dim if hasattr(pipe.dit, "in_dim") else 16

    latents = torch.randn(B, in_dim, latent_t, latent_h, latent_w, dtype=torch.bfloat16, device=args.device)
    context = torch.randn(B, 64, 4096, dtype=torch.bfloat16, device=args.device)
    v_timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=args.device)
    a_timestep = torch.tensor([300.0], dtype=torch.bfloat16, device=args.device)
    noisy_actions = torch.randn(B, 33, action_dim, dtype=torch.bfloat16, device=args.device)

    # --- Collect bridge features (once) ---
    print("\n[4/5] Running video DiT + bridge feature collection...")
    orig_bridge_layers = tuple(cfg["architecture"].get("bridge_layers", [3, 7, 11, 15, 19, 23, 26, 29]))
    bridge_features = []
    with torch.no_grad():
        noise_pred = pipe.model_fn(
            dit=pipe.dit,
            latents=latents,
            timestep=v_timestep,
            context=context,
            bridge_feature_store=bridge_features,
            bridge_feature_layers=set(orig_bridge_layers),
            bridge_feature_detach=True,
        )
    print(f"  video noise_pred: {noise_pred.shape}")
    assert noise_pred.shape == latents.shape, f"Video output shape {noise_pred.shape} != input shape {latents.shape}"
    print("  shape check:      noise_pred == latents shape (PASSED)")
    print(f"  bridge features:  {len(bridge_features)} layers")

    # --- Test each bridge type ---
    print("\n[5/5] Testing bridge types...")
    for bt in bridge_types:
        arch, bridge_layers = build_arch(cfg, video_dim, bridge_type_override=bt, device=args.device)
        param_count = sum(p.numel() for p in arch.action_dit.parameters()) / 1e6

        if bt in ("cross_attn", "cross_attn_detach"):
            pred = test_cross_attn_mode(arch, bridge_layers, noisy_actions, bridge_features, a_timestep, bt)
        else:
            pred = test_joint_self_attn_mode(arch, bridge_layers, noisy_actions, bridge_features, a_timestep)

        print(f"  PASSED: bridge_type='{bt}', {param_count:.1f}M params, output={pred.shape}")

    print("\n" + "=" * 60)
    print(f"All {len(bridge_types)} bridge type(s) passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
