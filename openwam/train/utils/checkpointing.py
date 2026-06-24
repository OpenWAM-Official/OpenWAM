"""Checkpoint save / load / management utilities."""

import glob as _glob
import logging
import os
import re

logger = logging.getLogger(__name__)


def save_config(output_dir: str, cfg):
    """Save Hydra DictConfig as config.yaml in the checkpoint directory.

    Only written once (skipped if the file already exists).
    """
    config_path = os.path.join(output_dir, "config.yaml")
    if os.path.exists(config_path):
        return
    os.makedirs(output_dir, exist_ok=True)
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, config_path)
    logger.info("Saved config to %s", config_path)


def save_normalization_stats(output_dir: str, dataset) -> None:
    """Copy the dataset's resolved action-stats .npy into the checkpoint dir.

    Written once (skipped if ``normalization_stats.npy`` already exists). Silently
    no-ops when the dataset has no stats path. The copied file preserves the nested
    ``{"joint": ..., "eef": ...}`` schema so deployment can pick the sub-dict
    matching the saved config's ``action_mode``.
    """
    import shutil

    dst = os.path.join(output_dir, "normalization_stats.npy")
    if os.path.exists(dst):
        logger.info(
            "[normalizer] normalization_stats.npy already present in checkpoint dir: %s (skip copy)",
            dst,
        )
        return
    src = getattr(dataset, "normalization_stats_path", None)
    if not src:
        logger.info(
            "[normalizer] Dataset has no normalization_stats_path (normalization likely disabled); "
            "nothing copied into checkpoint dir."
        )
        return
    if not os.path.exists(src):
        logger.warning(
            "[normalizer] Dataset reports normalization_stats_path=%s but file does not exist; "
            "nothing copied into checkpoint dir.",
            src,
        )
        return
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src, dst)
    logger.info("[normalizer] Copied action stats into checkpoint dir:\n  src: %s\n  dst: %s", src, dst)


def _step_num(path: str, prefix: str = "checkpoint_step_") -> int:
    m = re.search(rf"{prefix}(\d+)", path)
    return int(m.group(1)) if m else 0


def manage_checkpoints(output_dir: str, keep_last_k: int):
    """Keep only the most recent *keep_last_k* checkpoints.

    Prunes ``checkpoint_step_*`` (weights, files) and ``accel_state_step_*``
    (resume state, dirs) in lockstep so a kept weights file always retains its
    sibling state dir.
    """
    import shutil

    files = _glob.glob(os.path.join(output_dir, "checkpoint_step_*"))
    files.sort(key=lambda p: _step_num(p, "checkpoint_step_"))
    while len(files) > keep_last_k:
        old = files.pop(0)
        if os.path.isfile(old):
            os.remove(old)
            logger.info("Removed old checkpoint: %s", old)

    state_dirs = [p for p in _glob.glob(os.path.join(output_dir, "accel_state_step_*")) if os.path.isdir(p)]
    state_dirs.sort(key=lambda p: _step_num(p, "accel_state_step_"))
    while len(state_dirs) > keep_last_k:
        old = state_dirs.pop(0)
        try:
            shutil.rmtree(old)
            logger.info("Removed old accelerate state dir: %s", old)
        except OSError as e:
            logger.warning("Failed to remove old accelerate state dir %s: %s", old, e)


def find_latest_weights(run_dir: str) -> str:
    """Return the highest-step ``checkpoint_step_N.safetensors`` in *run_dir*.

    Used by the finetune path. Malformed names are skipped; step-0-only triggers
    a warning (likely a crash before the first real save).
    """
    files = _glob.glob(os.path.join(run_dir, "checkpoint_step_*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No checkpoint_step_*.safetensors found in {run_dir}")
    step_re = re.compile(r"checkpoint_step_(\d+)\.safetensors$")
    numbered: list[tuple[int, str]] = []
    for f in files:
        m = step_re.search(os.path.basename(f))
        if m is not None:
            numbered.append((int(m.group(1)), f))
        else:
            logger.warning("Skipping malformed checkpoint name: %s", f)
    if not numbered:
        raise FileNotFoundError(f"No file in {run_dir} matches checkpoint_step_<int>.safetensors")
    numbered.sort(key=lambda p: p[0])
    latest_step, latest_path = numbered[-1]
    if latest_step == 0:
        logger.warning("Latest checkpoint in %s is step 0 (%s); verify before finetune.", run_dir, latest_path)
    return latest_path


def find_latest_accel_state(run_dir: str) -> str | None:
    """Return the highest-step *usable* ``accel_state_step_N/`` in *run_dir*, or None.

    Usable = has both ``trainer_state.json`` and at least one ``random_states_*.pkl``
    (the latter proves the collective ``save_state`` actually ran, not just a mkdir
    from a crash). Half-written dirs are skipped.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None
    best: tuple[int, str] | None = None
    for name in os.listdir(run_dir):
        if not name.startswith("accel_state_step_"):
            continue
        state_dir = os.path.join(run_dir, name)
        if not os.path.isfile(os.path.join(state_dir, "trainer_state.json")):
            continue
        if not _glob.glob(os.path.join(state_dir, "random_states_*.pkl")):
            logger.warning("[resume] skipping incomplete state dir (no random_states_*.pkl): %s", state_dir)
            continue
        step = _step_num(name, "accel_state_step_")
        if best is None or step > best[0]:
            best = (step, state_dir)
    return best[1] if best else None


def finalize_keep_weights_only(output_dir: str):
    """Training-complete cleanup: drop all resume state, keep only the final weights.

    Removes every ``accel_state_step_*`` dir and every ``checkpoint_step_*.safetensors``
    except the highest step. Rank-0 only — caller must guard.
    """
    import shutil

    for d in _glob.glob(os.path.join(output_dir, "accel_state_step_*")):
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
                logger.info("Removed accelerate state dir: %s", d)
            except OSError as e:
                logger.warning("Failed to remove accelerate state dir %s: %s", d, e)

    files = _glob.glob(os.path.join(output_dir, "checkpoint_step_*.safetensors"))
    files.sort(key=lambda p: _step_num(p, "checkpoint_step_"))
    for old in files[:-1]:
        if os.path.isfile(old):
            os.remove(old)
            logger.info("Removed non-final checkpoint: %s", old)
