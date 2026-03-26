"""RoboTwin policy adapter — bridges legacy VAMPolicy with the new open_wam interfaces.

Provides ``get_model``, ``eval_one_step``, ``reset_model`` for compatibility
with RoboTwin's ``eval_policy.py`` evaluation harness.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

from eval_robotwin import (  # noqa: E402
    VAMPolicy as LegacyVAMPolicy,
    get_model as _legacy_get_model,
)


def get_model(
    task_name: str,
    ckpt_path: str,
    model_paths: str = None,
    tokenizer_path: str = None,
    action_dim: int = 14,
    action_dit_dim: int = 768,
    action_dit_ffn_dim: int = 3072,
    action_dit_num_heads: int = 12,
    action_dit_num_layers: int = 8,
    action_dit_bridge_layers: str = "3,7,11,15,19,23,26,29",
    video_dim: int = 1536,
    bridge_type: str = "joint_self_attn",
    num_frames: int = 49,
    height: int = 480,
    width: int = 832,
    target_camera: str = "head_camera",
    multiview: bool = False,
    device: str = "cuda",
    num_denoise_steps: int = 20,
    **kwargs,
) -> LegacyVAMPolicy:
    """Load a VAM policy for RoboTwin evaluation.

    Delegates entirely to the legacy ``get_model`` implementation.
    All parameters are forwarded unchanged.

    Returns:
        Legacy ``VAMPolicy`` instance ready for closed-loop evaluation.
    """
    return _legacy_get_model(
        task_name=task_name,
        ckpt_path=ckpt_path,
        model_paths=model_paths,
        tokenizer_path=tokenizer_path,
        action_dim=action_dim,
        action_dit_dim=action_dit_dim,
        action_dit_ffn_dim=action_dit_ffn_dim,
        action_dit_num_heads=action_dit_num_heads,
        action_dit_num_layers=action_dit_num_layers,
        action_dit_bridge_layers=action_dit_bridge_layers,
        video_dim=video_dim,
        bridge_type=bridge_type,
        num_frames=num_frames,
        height=height,
        width=width,
        target_camera=target_camera,
        multiview=multiview,
        device=device,
        num_denoise_steps=num_denoise_steps,
        **kwargs,
    )


def eval_one_step(model: LegacyVAMPolicy, obs_dict: dict) -> np.ndarray:
    """Predict one action step given the current observation."""
    return model.predict_action(obs_dict)


def reset_model(model: LegacyVAMPolicy):
    """Reset the policy state between episodes."""
    model.reset()
