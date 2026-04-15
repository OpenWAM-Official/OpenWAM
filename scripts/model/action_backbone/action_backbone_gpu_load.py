"""GPU-based ActionDiT model test.

Loads ActionDiT parameters from YAML config (architecture + action_backbone
inline in dual_system.yaml), creates fake video features, runs forward pass,
and prints output shapes.

Usage:
    python scripts/model/action_backbone/action_backbone_gpu_load.py
    python scripts/model/action_backbone/action_backbone_gpu_load.py --arch configs/model/dual_system.yaml
    python scripts/model/action_backbone/action_backbone_gpu_load.py --device cpu

GPU memory requirements: ~1 GB (ActionDiT only, no video backbone)
"""

import argparse
import sys

import torch

sys.path.insert(0, "scripts/model")
from _config_utils import load_architecture_config, merge_arch_params


def main():
    parser = argparse.ArgumentParser(description="GPU test for ActionDiT forward pass")
    parser.add_argument("--arch", type=str, default="configs/model/dual_system.yaml", help="Architecture YAML")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_architecture_config(args.arch)
    # Use a fake video_dim (3072 = typical for TI2V-5B / VACE-1.3B)
    params = merge_arch_params(cfg, video_dim=3072)

    bridge_layers = tuple(params.get("bridge_layers", []))
    num_layers = params.get("num_layers", len(bridge_layers))

    print("=" * 60)
    print("ActionDiT GPU Test (from YAML)")
    print(f"  Config:        {args.arch}")
    print(f"  Device:        {args.device}")
    print(f"  action_dim:    {params.get('action_dim')}")
    print(f"  dim:           {params.get('dim')}")
    print(f"  ffn_dim:       {params.get('ffn_dim')}")
    print(f"  num_heads:     {params.get('num_heads')}")
    print(f"  num_layers:    {num_layers} (from {len(bridge_layers)} bridge layers)")
    print(f"  bridge_type:   {params.get('bridge_type')}")
    print(f"  video_dim:     {params.get('video_dim')} (fake)")
    print("=" * 60)

    from openwam.model.action_model.action_dit import ActionDiT

    print("\n[1/3] Instantiating ActionDiT...")
    dit = ActionDiT(
        action_dim=int(params["action_dim"]),
        dim=int(params["dim"]),
        ffn_dim=int(params["ffn_dim"]),
        num_heads=int(params["num_heads"]),
        num_layers=num_layers,
        video_dim=int(params["video_dim"]),
        bridge_layers=bridge_layers,
        bridge_type=params.get("bridge_type", "cross_attn_detach"),
    ).to(dtype=torch.bfloat16, device=args.device)

    param_count = sum(p.numel() for p in dit.parameters()) / 1e6
    print(f"  Parameters: {param_count:.1f}M")

    print("\n[2/3] Creating fake inputs...")
    B, T_action, T_video = 1, 33, 3600
    action_dim = int(params["action_dim"])
    video_dim = int(params["video_dim"])
    actions = torch.randn(B, T_action, action_dim, dtype=torch.bfloat16, device=args.device)
    video_features = [
        torch.randn(B, T_video, video_dim, dtype=torch.bfloat16, device=args.device) for _ in range(num_layers)
    ]
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=args.device)

    print(f"  actions:        {actions.shape}  (B, T_action, action_dim)")
    print(f"  video_features: {num_layers} x {video_features[0].shape}  (B, T_video, video_dim)")
    print(f"  timestep:       {timestep.shape}")

    print("\n[3/3] Running forward pass...")
    with torch.no_grad():
        out = dit(actions, video_features, timestep)

    print(f"  output: {out.shape}  (B, T_action, action_dim)")

    print("\n" + "=" * 60)
    print(f"ActionDiT GPU test passed. ({param_count:.1f}M params)")
    print("=" * 60)


if __name__ == "__main__":
    main()
