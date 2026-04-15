"""GPU-based video backbone model loading and forward pass test.

Loads a real video backbone model from a local path (specified via YAML config
or CLI argument), creates fake inputs, runs a forward pass, and prints
output shapes. Requires GPU and model weights.

Usage:
    # Use dual_system config (default, reads video_backbone.model_path)
    python scripts/model/video_backbone/video_backbone_gpu_load.py

    # Override model path
    python scripts/model/video_backbone/video_backbone_gpu_load.py --model_path /path/to/model

    # Use a different architecture config
    python scripts/model/video_backbone/video_backbone_gpu_load.py --config configs/model/moe_expert.yaml

GPU memory requirements (approximate, only for loading, not for training):
    VACE-1.3B:  ~16 GB
    TI2V-5B:    ~24 GB
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import torch
import yaml


def discover_model_files(model_dir: str) -> list:
    """Auto-discover and group model files from a directory.

    Groups sharded safetensors by prefix, each .pth as standalone.
    Returns a list suitable for ModelConfig construction.
    """
    safetensors = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    pth_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    if not safetensors and not pth_files:
        raise FileNotFoundError(f"No *.safetensors or *.pth files found in {model_dir}")

    shard_groups = defaultdict(list)
    standalone = []
    for f in safetensors:
        basename = os.path.basename(f)
        m = re.match(r"^(.+)-\d{5}-of-\d{5}\.safetensors$", basename)
        if m:
            shard_groups[m.group(1)].append(f)
        else:
            standalone.append(f)

    entries = []
    for prefix in sorted(shard_groups):
        entries.append(sorted(shard_groups[prefix]))
    for f in standalone:
        entries.append(f)
    for f in pth_files:
        entries.append(f)

    return entries


def load_backbone(model_path: str, device: str = "cuda"):
    """Load WanVideoPipeline from a local model directory."""
    from openwam.deployment.model_config import ModelConfig
    from openwam.model.video_backbone import WanVideoPipeline

    model_entries = discover_model_files(model_path)
    model_configs = [ModelConfig(entry) for entry in model_entries]

    # Auto-detect tokenizer
    tokenizer_dir = os.path.join(model_path, "google", "umt5-xxl")
    if os.path.isdir(tokenizer_dir):
        tokenizer_config = ModelConfig(tokenizer_dir)
    else:
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    return pipe


def main():
    parser = argparse.ArgumentParser(description="GPU test for video backbone model loading and forward pass")
    parser.add_argument("--model_path", type=str, default=None, help="Local path to model directory (overrides YAML)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/dual_system.yaml",
        help="Path to architecture YAML config (reads video_backbone.model_path)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to load model on")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    video_cfg = raw_cfg.get("video_backbone", {})

    # Resolve model_path: CLI override > YAML > error
    model_path = args.model_path or video_cfg.get("model_path")
    if not model_path or not os.path.isdir(model_path):
        parser.error(
            f"model_path not found. "
            f"Either set video_backbone.model_path in {args.config} or pass --model_path.\n"
            f"  YAML model_path: {video_cfg.get('model_path')}\n"
            f"  CLI  model_path: {args.model_path}"
        )

    print("=" * 60)
    print("Video Backbone GPU Test")
    print(f"  Config:     {args.config}")
    print(f"  Model path: {model_path}")
    print(f"  Device:     {args.device}")
    print(f"  Backbone:   {video_cfg.get('name', 'unknown')}")
    print("=" * 60)

    # --- Load model ---
    print("\n[1/4] Loading video backbone...")
    pipe = load_backbone(model_path, device=args.device)

    video_dim = pipe.dit.dim
    num_layers = len(pipe.dit.blocks)
    print(f"  video_dim (from model): {video_dim}")
    print(f"  num_layers:             {num_layers}")
    print(f"  dit type:               {type(pipe.dit).__name__}")
    if pipe.vae is not None:
        print(f"  vae type:               {type(pipe.vae).__name__}")
    if pipe.vace is not None:
        print(f"  vace type:              {type(pipe.vace).__name__}")

    # --- Create fake inputs ---
    print("\n[2/4] Creating fake inputs...")
    num_frames = 9
    batch_size = 1

    # VAE latent dimensions: spatial / 8, temporal / 4, channels = in_dim
    latent_h = 480 // 8  # default
    latent_w = 640 // 8
    latent_t = (num_frames - 1) // 4 + 1  # Wan VAE temporal compression
    in_dim = pipe.dit.in_dim if hasattr(pipe.dit, "in_dim") else 16

    latents = torch.randn(batch_size, in_dim, latent_t, latent_h, latent_w, dtype=torch.bfloat16, device=args.device)
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=args.device)
    context = torch.randn(batch_size, 64, 4096, dtype=torch.bfloat16, device=args.device)

    print(f"  latents:   {latents.shape}  (B, C, T, H, W)")
    print(f"  timestep:  {timestep.shape}")
    print(f"  context:   {context.shape}  (B, L, text_dim)")

    # --- Forward pass via model_fn (includes bridge feature extraction) ---
    print("\n[3/4] Running forward pass via model_fn...")
    arch_cfg = raw_cfg.get("architecture", {})
    bridge_layers = arch_cfg.get("bridge_layers", [3, 7, 11, 15, 19, 23, 26, 29])
    valid_bridge_layers = [layer for layer in bridge_layers if layer < num_layers]

    bridge_features = []
    with torch.no_grad():
        noise_pred = pipe.model_fn(
            dit=pipe.dit,
            latents=latents,
            timestep=timestep,
            context=context,
            bridge_feature_store=bridge_features,
            bridge_feature_layers=set(valid_bridge_layers),
            bridge_feature_detach=True,
        )

    print(f"  noise_pred: {noise_pred.shape}  (B, C, T, H, W)")

    # --- Bridge features ---
    print("\n[4/4] Bridge feature shapes...")
    print(f"  Requested bridge layers: {valid_bridge_layers}")
    print(f"  Collected features:      {len(bridge_features)}")
    for i, feat in enumerate(bridge_features):
        print(f"    layer {valid_bridge_layers[i]:2d}: {feat.shape}  (B, seq_len, {video_dim})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  video_dim:    {video_dim}")
    print(f"  num_layers:   {num_layers}")
    print(f"  latent shape: {latents.shape}")
    print(f"  output shape: {noise_pred.shape}")
    print("=" * 60)
    print("GPU test passed.")


if __name__ == "__main__":
    main()
