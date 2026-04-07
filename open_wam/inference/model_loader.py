"""Model-loading helpers for package-native inference entrypoints."""

from __future__ import annotations

import json
from typing import Any

import torch

from open_wam.models.action_dit import ActionDiT
from open_wam.inference.model_config import ModelConfig
from open_wam.inference.video_pipeline import WanVideoPipeline


def load_wam_models(cfg: Any, device: str = "cuda"):
    """Load the Wan pipeline and ActionDiT from Hydra config.

    This centralizes the model-loading path so evaluation, inference, and
    serving do not need to depend on each other's script-level helpers.
    """
    eval_cfg = cfg.eval
    model_cfg = cfg.model
    backbone_cfg = cfg.model.backbone

    model_paths = getattr(eval_cfg, "model_paths", None)
    tokenizer_path = getattr(eval_cfg, "tokenizer_path", None)

    if model_paths is None:
        model_paths = [
            "models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth",
        ]
    if isinstance(model_paths, str):
        model_paths = json.loads(model_paths)
    if tokenizer_path is None:
        tokenizer_path = "models/Wan-AI/Wan2.1-VACE-1.3B/google/umt5-xxl"

    model_configs = [ModelConfig(path) for path in model_paths]
    tokenizer_config = ModelConfig(tokenizer_path)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    bridge_layers = tuple(int(x) for x in model_cfg.bridge_layers)
    action_dit = ActionDiT(
        action_dim=int(model_cfg.action_dim),
        dim=int(model_cfg.dim),
        ffn_dim=int(model_cfg.ffn_dim),
        num_heads=int(model_cfg.num_heads),
        num_layers=int(model_cfg.num_layers),
        video_dim=int(backbone_cfg.video_dim),
        bridge_layers=bridge_layers,
        bridge_type=model_cfg.bridge_type,
    ).to(dtype=torch.bfloat16, device=device)

    ckpt_path = eval_cfg.ckpt_path
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    action_keys = {k: v for k, v in state_dict.items() if k.startswith("action_dit.")}
    if action_keys:
        cleaned = {k.removeprefix("action_dit."): v for k, v in action_keys.items()}
        action_dit.load_state_dict(cleaned, strict=False)

    candidate_keys = {k: v for k, v in state_dict.items() if not k.startswith("action_dit.")}
    if candidate_keys and hasattr(pipe, "vace"):
        vace_expected = set(pipe.vace.state_dict().keys())
        vace_keys = {k: v for k, v in candidate_keys.items() if k in vace_expected}
        if vace_keys:
            pipe.vace.load_state_dict(vace_keys, strict=False)

    action_dit.eval()
    return pipe, action_dit
