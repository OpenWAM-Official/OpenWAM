"""GPU-based end-to-end SharedBackbone architecture test.

Loads video backbone + SharedBackbone from YAML configs, runs joint forward pass
through model_fn (action tokens concatenated to video sequence, shared DiT),
and prints all shapes.

Usage:
    python scripts/model/shared_backbone_gpu_test.py
    python scripts/model/shared_backbone_gpu_test.py --model_path /path/to/model
    python scripts/model/shared_backbone_gpu_test.py --device cpu

GPU memory requirements (approximate):
    VACE-1.3B + SharedBackbone:  ~16 GB  (minimal extra params)
    TI2V-5B + SharedBackbone:    ~24 GB
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "scripts/model")
sys.path.insert(0, "scripts/model/video_backbone")
from _config_utils import apply_freeze, load_architecture_config, merge_arch_params, print_freeze_status
from video_backbone_gpu_load import load_backbone


def main():
    parser = argparse.ArgumentParser(description="End-to-end SharedBackbone GPU test")
    parser.add_argument("--arch", type=str, default="configs/model/shared_backbone.yaml")
    parser.add_argument("--model_path", type=str, default=None, help="Override video backbone model path")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_architecture_config(args.arch)
    video_cfg = cfg["video_backbone"]

    model_path = args.model_path or video_cfg.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        parser.error(
            f"model_path not found. Set in video_backbone YAML or pass --model_path.\n"
            f"  YAML model_path: {video_cfg.get('model_path')}"
        )

    action_dim = int(cfg["architecture"]["action_dim"])
    num_action_tokens = int(cfg["architecture"].get("num_action_tokens", 33))

    print("=" * 60)
    print("SharedBackbone End-to-End GPU Test (from YAML)")
    print(f"  Architecture:       {args.arch}")
    print(f"  Model path:         {model_path}")
    print(f"  Device:             {args.device}")
    print(f"  action_dim:         {action_dim}")
    print(f"  num_action_tokens:  {num_action_tokens}")
    print("=" * 60)

    # --- Load video backbone ---
    print("\n[1/5] Loading video backbone...")
    pipe = load_backbone(model_path, device=args.device)
    video_dim = pipe.dit.dim
    num_video_layers = len(pipe.dit.blocks)
    print(f"  video_dim: {video_dim}, layers: {num_video_layers}")

    # --- Apply freeze strategy ---
    print("\n[2/6] Applying freeze strategy...")
    freeze_list = cfg.get("freeze", ["text_encoder", "vae", "image_encoder"])
    apply_freeze(pipe, freeze_list)

    # --- Build SharedBackbone architecture ---
    print("\n[3/6] Building SharedBackbone architecture...")
    params = merge_arch_params(cfg, video_dim=video_dim)
    from openwam.model import build_architecture

    arch = build_architecture(params.pop("type"), params)
    arch.to(dtype=torch.bfloat16, device=args.device)

    param_count = sum(p.numel() for p in arch.parameters()) / 1e6
    print(f"  SharedBackbone: {param_count:.1f}M params")

    print_freeze_status(pipe, arch, "shared_backbone", freeze_list)

    # --- Create fake inputs ---
    print("\n[4/6] Creating fake inputs...")
    height = video_cfg["resolution"]["height"]
    width = video_cfg["resolution"]["width"]
    B, num_frames = 1, 9

    latent_h, latent_w = height // 8, width // 8
    latent_t = (num_frames - 1) // 4 + 1
    in_dim = pipe.dit.in_dim if hasattr(pipe.dit, "in_dim") else 16

    latents = torch.randn(B, in_dim, latent_t, latent_h, latent_w, dtype=torch.bfloat16, device=args.device)
    context = torch.randn(B, 64, 4096, dtype=torch.bfloat16, device=args.device)
    v_timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=args.device)
    a_timestep = torch.tensor([300.0], dtype=torch.bfloat16, device=args.device)
    noisy_actions = torch.randn(B, num_action_tokens, action_dim, dtype=torch.bfloat16, device=args.device)

    print(f"  latents:       {latents.shape}")
    print(f"  noisy_actions: {noisy_actions.shape}")

    # --- Prepare SharedBackbone state ---
    print("\n[5/6] Running video DiT + SharedBackbone forward...")
    action_state = arch.prepare_action_tokens(noisy_actions, a_timestep)
    sb_state = action_state.extra.get("shared_backbone_state")

    print(f"  action_latents (projected): {action_state.action_latents.shape}  (B, T_action, video_dim)")

    # Run through model_fn with shared_backbone_state
    with torch.no_grad():
        noise_pred = pipe.model_fn(
            dit=pipe.dit,
            latents=latents,
            timestep=v_timestep,
            context=context,
            shared_backbone_state=sb_state,
        )

    print(f"  video noise_pred: {noise_pred.shape}")
    assert noise_pred.shape == latents.shape, f"Video output shape {noise_pred.shape} != input shape {latents.shape}"
    print("  video shape check: noise_pred == latents shape (PASSED)")

    # --- Extract action prediction ---
    print("\n[6/6] Extracting action prediction...")
    with torch.no_grad():
        action_pred = arch.extract_action_prediction(action_state)

    print(f"  action_pred: {action_pred.shape}")
    expected_action_shape = (B, num_action_tokens, action_dim)
    assert action_pred.shape == expected_action_shape, (
        f"Action output shape {action_pred.shape} != expected {expected_action_shape}"
    )
    print("  action shape check: PASSED")

    print("\n" + "=" * 60)
    print("SharedBackbone end-to-end test passed.")
    print(f"  Video:  {latents.shape} -> {noise_pred.shape}")
    print(f"  Action: {noisy_actions.shape} -> {action_pred.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
