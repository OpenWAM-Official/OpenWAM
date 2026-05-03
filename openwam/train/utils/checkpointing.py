"""Checkpoint save / load / management utilities."""

import glob as _glob
import logging
import os
import re

import torch

logger = logging.getLogger(__name__)


def save_config(output_dir: str, cfg):
    """Save Hydra DictConfig as config.yaml in the checkpoint directory.

    Only written once (skipped if the file already exists).

    Args:
        output_dir: Checkpoint directory.
        cfg: Hydra DictConfig to serialize.
    """
    config_path = os.path.join(output_dir, "config.yaml")
    if os.path.exists(config_path):
        return
    os.makedirs(output_dir, exist_ok=True)
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, config_path)
    logger.info("Saved config to %s", config_path)


def save_action_stats(output_dir: str, dataset) -> None:
    """Copy the dataset's resolved action-stats .npy into the checkpoint dir.

    Written once (skipped if ``action_stats.npy`` already exists in
    *output_dir*).  Silently no-ops when the dataset has no stats path
    (e.g. normalization disabled or unsupported dataset type).

    The copied file preserves the nested ``{"joint": ..., "eef": ...}``
    schema so deployment can pick whichever sub-dict matches the saved
    config's ``action_mode``.
    """
    import shutil

    dst = os.path.join(output_dir, "action_stats.npy")
    if os.path.exists(dst):
        logger.info(
            "[normalizer] action_stats.npy already present in checkpoint dir: %s (skip copy)",
            dst,
        )
        return
    src = getattr(dataset, "action_stats_path", None)
    if not src:
        logger.info(
            "[normalizer] Dataset has no action_stats_path (normalization likely disabled); "
            "nothing copied into checkpoint dir."
        )
        return
    if not os.path.exists(src):
        logger.warning(
            "[normalizer] Dataset reports action_stats_path=%s but file does not exist; "
            "nothing copied into checkpoint dir.",
            src,
        )
        return
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src, dst)
    logger.info("[normalizer] Copied action stats into checkpoint dir:\n  src: %s\n  dst: %s", src, dst)


_MIXED_PRECISION_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "no": torch.float32,
}


def _parse_dtype(mixed_precision: str) -> torch.dtype:
    dtype = _MIXED_PRECISION_TO_DTYPE.get(str(mixed_precision).strip().lower())
    if dtype is None:
        raise ValueError(
            f"Unknown mixed_precision={mixed_precision!r}. Expected one of: {list(_MIXED_PRECISION_TO_DTYPE)}"
        )
    return dtype


_MISSING_MIXED_PRECISION = object()


def save_trainable_checkpoint(
    path: str,
    action_backbone: torch.nn.Module,
    pipe,
    lambda_action: float,
    mixed_precision=_MISSING_MIXED_PRECISION,
):
    """Export full model state dict to safetensors or .pt.

    Args:
        path: Output file path (``.safetensors`` or ``.pt``).
        action_backbone: Action backbone (``ActionDiT`` / ``SharedMoEActionBackbone`` / ``SharedVanillaActionBackbone``).
        pipe: WanVideoPipeline instance.
        lambda_action: Action loss weight (unused, kept for API compat).
        mixed_precision: ``"bf16"`` / ``"fp16"`` / ``"no"`` — target dtype for
            floating-point tensors. If omitted, defaults to ``"bf16"`` with
            a WARNING so the dtype decision is explicit.
    """
    if mixed_precision is _MISSING_MIXED_PRECISION:
        logger.warning("save_trainable_checkpoint: mixed_precision not provided, defaulting to 'bf16'")
        mixed_precision = "bf16"
    target_dtype = _parse_dtype(mixed_precision)
    state_dict = {}

    def _maybe_cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=target_dtype) if t.is_floating_point() else t

    def _non_persistent_names(root) -> set[str]:
        """Fully-qualified names of non-persistent buffers to skip on save.

        ``named_buffers`` yields non-persistent buffers (e.g. RoPE freqs)
        too — and complex dtypes like ``complex64`` are not supported by
        safetensors. Falls back to an empty set for plain objects that
        don't expose ``named_modules`` (test mocks).
        """
        if not hasattr(root, "named_modules"):
            return set()
        names: set[str] = set()
        for mod_prefix, submodule in root.named_modules():
            nonp = getattr(submodule, "_non_persistent_buffers_set", set())
            for bname in nonp:
                names.add(f"{mod_prefix}.{bname}" if mod_prefix else bname)
        return names

    # ActionBackbone: all parameters + persistent buffers
    for name, param in action_backbone.named_parameters():
        state_dict[f"action_backbone.{name}"] = _maybe_cast(param.data)
    skip_action = _non_persistent_names(action_backbone)
    for name, buf in action_backbone.named_buffers():
        if name in skip_action:
            continue
        state_dict[f"action_backbone.{name}"] = _maybe_cast(buf)

    # Video pipeline: all parameters + persistent buffers
    for name, param in pipe.named_parameters():
        state_dict[name] = _maybe_cast(param.data)
    skip_pipe = _non_persistent_names(pipe)
    for name, buf in pipe.named_buffers():
        if name in skip_pipe:
            continue
        state_dict[name] = _maybe_cast(buf)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(state_dict, path)
    else:
        torch.save(state_dict, path)

    logger.info(
        "Saved checkpoint to %s (%d keys, dtype=%s)",
        path,
        len(state_dict),
        target_dtype,
    )


def load_trainable_checkpoint(
    path: str,
    action_backbone: torch.nn.Module,
    pipe,
):
    """Load a checkpoint into action_backbone and pipeline.

    Keys prefixed with ``action_backbone.`` are loaded into the action model;
    all other keys are loaded into the pipeline.

    Args:
        path: Checkpoint file path (``.safetensors`` or ``.pt``).
        action_backbone: Action backbone to load weights into.
        pipe: WanVideoPipeline to load weights into.
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(path)
    else:
        state_dict = torch.load(path, map_location="cpu")

    action_keys = {k: v for k, v in state_dict.items() if k.startswith("action_backbone.")}
    if action_keys:
        cleaned = {k.removeprefix("action_backbone."): v for k, v in action_keys.items()}
        action_backbone.load_state_dict(cleaned, strict=False)

    pipe_keys = {k: v for k, v in state_dict.items() if not k.startswith("action_backbone.")}
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
