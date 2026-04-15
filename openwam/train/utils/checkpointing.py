"""Checkpoint save / load / management utilities."""

import glob as _glob
import logging
import os
import re

import torch

logger = logging.getLogger(__name__)


def save_trainable_checkpoint(
    path: str,
    action_dit: torch.nn.Module,
    pipe,
    lambda_action: float,
):
    """Export trainable state dict to safetensors or .pt.

    Saves all parameters with ``requires_grad=True`` from both the
    action model and the video pipeline, plus action normalization
    buffers (``action_mean``, ``action_std``).

    Args:
        path: Output file path (``.safetensors`` or ``.pt``).
        action_dit: Action model (ActionDiT / MoEExpertDiT / etc.).
        pipe: WanVideoPipeline instance.
        lambda_action: Action loss weight — if 0, action params are skipped.
    """
    state_dict = {}

    # ActionDiT parameters and buffers
    if lambda_action > 0:
        for name, param in action_dit.named_parameters():
            if param.requires_grad:
                state_dict[f"action_dit.{name}"] = param.data
        for buf_name in ("action_mean", "action_std"):
            buf = getattr(action_dit, buf_name, None)
            if buf is not None:
                state_dict[f"action_dit.{buf_name}"] = buf

    # Video pipeline trainable parameters
    for name, param in pipe.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.data

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(state_dict, path)
    else:
        torch.save(state_dict, path)

    logger.info("Saved checkpoint to %s (%d keys)", path, len(state_dict))


def load_trainable_checkpoint(
    path: str,
    action_dit: torch.nn.Module,
    pipe,
):
    """Load a checkpoint into action_dit and pipeline.

    Keys prefixed with ``action_dit.`` are loaded into the action model;
    all other keys are loaded into the pipeline.

    Args:
        path: Checkpoint file path (``.safetensors`` or ``.pt``).
        action_dit: Action model to load weights into.
        pipe: WanVideoPipeline to load weights into.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")

    action_keys = {k: v for k, v in state_dict.items() if k.startswith("action_dit.")}
    if action_keys:
        cleaned = {k.removeprefix("action_dit."): v for k, v in action_keys.items()}
        action_dit.load_state_dict(cleaned, strict=False)

    pipe_keys = {k: v for k, v in state_dict.items() if not k.startswith("action_dit.")}
    if pipe_keys:
        pipe.load_state_dict(pipe_keys, strict=False)

    logger.info("Loaded checkpoint from %s", path)


def manage_checkpoints(output_dir: str, keep_last_k: int):
    """Delete old checkpoints, keeping only the most recent *keep_last_k*.

    Looks for files matching ``checkpoint_step_*`` in *output_dir*
    and removes the oldest ones.

    Args:
        output_dir: Directory containing checkpoint files.
        keep_last_k: Number of most recent checkpoints to keep.
    """
    pattern = os.path.join(output_dir, "checkpoint_step_*")
    files = _glob.glob(pattern)

    # Sort numerically by step number
    def _step_num(path):
        m = re.search(r"checkpoint_step_(\d+)", path)
        return int(m.group(1)) if m else 0

    files.sort(key=_step_num)
    while len(files) > keep_last_k:
        old = files.pop(0)
        if os.path.isfile(old):
            os.remove(old)
            logger.info("Removed old checkpoint: %s", old)
