"""Checkpoint save / load / management utilities."""

import glob as _glob
import logging
import os
import re

logger = logging.getLogger(__name__)

_NORMALIZATION_TRANSFORM_KEYS = ("mean", "std", "min", "max", "q01", "q99")
_NORMALIZATION_SEMANTIC_KEYS = (
    "gripper_convention",
    "representation",
    "osc_position_scale",
    "osc_rotation_scale",
)


# --- Deploy assets (write-once) ---


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


def _normalization_stats_values_equal(left, right) -> bool:
    """Deep equality for nested normalization_stats.npy payloads."""
    import numpy as np

    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_normalization_stats_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_normalization_stats_values_equal(a, b) for a, b in zip(left, right))
    left_arr = np.asarray(left) if isinstance(left, np.ndarray) or isinstance(right, np.ndarray) else None
    if left_arr is not None:
        right_arr = np.asarray(right)
        if left_arr.shape != right_arr.shape:
            return False
        if np.issubdtype(left_arr.dtype, np.floating) or np.issubdtype(right_arr.dtype, np.floating):
            # Strict resume must preserve the exact transform. Tolerant
            # comparison can accept small regenerated-stat differences even
            # though restored weights and optimizer state were trained in the
            # original coordinates.
            return bool(np.array_equal(left_arr, right_arr, equal_nan=True))
        return bool(np.array_equal(left_arr, right_arr))
    return left == right


def _normalization_transforms_equal(left, right) -> bool:
    """Compare effective transforms while allowing full-vs-legacy payloads.

    Older checkpoints contain only the six vectors deployment consumes, while
    unified dataset artifacts may additionally carry quantiles, row counts and
    provenance.  Those extra fields do not change normalization coordinates.
    Semantic markers remain strict whenever both sides provide them; a legacy
    reduced checkpoint legitimately has none.
    """
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_modes = {
        key: value
        for key, value in left.items()
        if isinstance(value, dict) and any(field in value for field in _NORMALIZATION_TRANSFORM_KEYS)
    }
    right_modes = {
        key: value
        for key, value in right.items()
        if isinstance(value, dict) and any(field in value for field in _NORMALIZATION_TRANSFORM_KEYS)
    }
    if not left_modes or left_modes.keys() != right_modes.keys():
        return False
    for mode in left_modes:
        left_stats = left_modes[mode]
        right_stats = right_modes[mode]
        for field in _NORMALIZATION_TRANSFORM_KEYS:
            if (field in left_stats) != (field in right_stats):
                return False
            if field in left_stats and not _normalization_stats_values_equal(left_stats[field], right_stats[field]):
                return False
        for field in _NORMALIZATION_SEMANTIC_KEYS:
            if field in left_stats and field in right_stats:
                if not _normalization_stats_values_equal(left_stats[field], right_stats[field]):
                    return False
    return True


def verify_resume_normalization_stats(output_dir: str, dataset) -> None:
    """Fail fast when a resumed run's deploy stats diverge from the dataset's.

    Strict resume reuses the checkpoint-directory ``normalization_stats.npy``
    rather than refreshing it from the dataset. If that artifact is missing,
    unreadable, or its effective six-vector transform differs from the dataset,
    training would normalize with one transform while deployment kept another —
    and restored weights/optimizer state remain in the old coordinates. A legacy
    reduced artifact is accepted when those vectors match the unified full file;
    shared semantic markers are still compared strictly. Direct the user to
    ``finetune_ckpt_path`` instead of silently continuing or replacing.
    """
    import pickle

    import numpy as np

    src = getattr(dataset, "normalization_stats_path", None)
    if not src:
        return

    dst = os.path.join(output_dir, "normalization_stats.npy")
    if not os.path.isfile(dst):
        raise FileNotFoundError(
            f"resume blocked: {dst} is missing while the dataset reports "
            f"normalization_stats_path={src}. Use finetune_ckpt_path to warm-start "
            "with freshly copied stats, or restore the run's normalization_stats.npy."
        )
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"resume blocked: dataset normalization_stats_path={src} does not exist "
            f"while comparing against checkpoint artifact {dst}."
        )

    try:
        ckpt_payload = np.load(dst, allow_pickle=True).item()
        data_payload = np.load(src, allow_pickle=True).item()
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(
            f"resume blocked: failed to read normalization stats for comparison "
            f"(checkpoint={dst}, dataset={src}): {exc}. "
            "Use finetune_ckpt_path to warm-start instead of resume_ckpt_path."
        ) from exc

    if not _normalization_stats_values_equal(ckpt_payload, data_payload) and not _normalization_transforms_equal(
        ckpt_payload, data_payload
    ):
        raise ValueError(
            "resume blocked: checkpoint normalization_stats.npy does not match the "
            f"dataset stats at {src}. Restored weights/optimizer were trained under "
            "different normalization coordinates than the current dataset transform. "
            "Use finetune_ckpt_path to warm-start instead of resume_ckpt_path.\n"
            f"  checkpoint: {dst}\n"
            f"  dataset:    {src}"
        )


# --- Checkpoint I/O (read/write training state) ---


def save_weights(accelerator, architecture, output_path: str, global_step: int, *, final: bool) -> None:
    """Write the weights safetensors (the deploy artifact).

    Only rank-0 writes. ZeRO-1/2 (this repo's only stages, see
    ``training.zero_stage``) replicates the bf16 params on every rank, so
    ``get_state_dict`` is rank-local: non-writers return immediately —
    otherwise every rank can clone the full state dict to host memory at once
    and exhaust the node during checkpointing. Pruning is the caller's job, run
    after the full state is also written so the two lines stay in lockstep.
    """
    from tqdm import tqdm

    if not accelerator.is_main_process:
        return
    ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
    msg = f"[checkpoint] Saving {'final ' if final else ''}step {global_step} -> {ckpt_path}"
    logger.info(msg)
    tqdm.write(msg)
    state_dict = accelerator.get_state_dict(architecture)
    accelerator.unwrap_model(architecture).save_checkpoint(ckpt_path, state_dict=state_dict)
    msg = f"[checkpoint] Saved{' final' if final else ''}: {ckpt_path}"
    logger.info(msg)
    tqdm.write(msg)


def save_full_state(accelerator, output_path: str, global_step: int, opt_step: int, epoch: int) -> None:
    """Write full Accelerate state to ``accel_state_step_N/`` for resume.

    ALL ranks enter (DeepSpeed shards optimizer state per-rank). rank-0 writes the
    ``trainer_state.json`` marker last (atomic) so a half-written dir is never picked
    by ``find_latest_accel_state``.
    """
    import json

    state_dir = os.path.join(output_path, f"accel_state_step_{global_step}")
    if accelerator.is_main_process:
        os.makedirs(state_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        meta = {"global_step": int(global_step), "opt_step": int(opt_step), "epoch": int(epoch)}
        meta_path = os.path.join(state_dir, "trainer_state.json")
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(meta, f)
        os.replace(tmp_path, meta_path)


def load_full_state(accelerator, state_dir: str) -> dict:
    """Restore optimizer/scheduler/RNG/model from ``state_dir`` (call AFTER prepare).

    Returns the ``trainer_state.json`` contents (global_step / opt_step / epoch).
    """
    import json

    accelerator.load_state(state_dir)
    meta_path = os.path.join(state_dir, "trainer_state.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {"global_step": 0, "opt_step": 0, "epoch": 0}


# --- Discovery (locate latest step) ---


def step_num(path: str, prefix: str = "checkpoint_step_") -> int:
    m = re.search(rf"{prefix}(\d+)", path)
    return int(m.group(1)) if m else 0


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

    Usable = ``trainer_state.json`` present. ``save_full_state`` writes that marker
    atomically AFTER ``accelerator.save_state`` returns, so its presence proves the
    (possibly large / sharded) state finished writing. DeepSpeed's ``save_state``
    writes a ``pytorch_model/`` dir and no ``random_states_*.pkl``, so the marker —
    not RNG files — is the completion signal. Half-written dirs lack it and are skipped.
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
        step = step_num(name, "accel_state_step_")
        if best is None or step > best[0]:
            best = (step, state_dir)
    return best[1] if best else None


# --- Resume position (pure math) ---


def compute_resume_position(global_step: int, batches_per_epoch: int, grad_accum: int) -> tuple[int, int, int]:
    """Map a resumed ``global_step`` to ``(start_epoch, skip_first_batches, aligned_global_step)``.

    ``skip`` is floored to a grad_accum boundary so the first optimizer step after
    resume sees a full accumulation cycle; ``aligned_global_step`` pulls ``global_step``
    back to that same boundary so the floored-off batches are not re-trained and the
    per-step seed (keyed on global_step) stays matched. No-op at grad_accum=1.
    """
    batches_per_epoch = max(batches_per_epoch, 1)
    start_epoch = global_step // batches_per_epoch
    skip = global_step % batches_per_epoch
    if grad_accum > 1 and skip % grad_accum != 0:
        skip = (skip // grad_accum) * grad_accum
    aligned_global_step = start_epoch * batches_per_epoch + skip
    return start_epoch, skip, aligned_global_step


# --- Retention / finalize (prune) ---


def manage_checkpoints(output_dir: str, keep_last_k: int):
    """Keep only the most recent *keep_last_k* checkpoints.

    Prunes ``checkpoint_step_*`` (weights, files) and ``accel_state_step_*``
    (resume state, dirs) in lockstep so a kept weights file always retains its
    sibling state dir.
    """
    import shutil

    files = _glob.glob(os.path.join(output_dir, "checkpoint_step_*"))
    files.sort(key=lambda p: step_num(p, "checkpoint_step_"))
    while len(files) > keep_last_k:
        old = files.pop(0)
        if os.path.isfile(old):
            os.remove(old)
            logger.info("Removed old checkpoint: %s", old)

    state_dirs = [p for p in _glob.glob(os.path.join(output_dir, "accel_state_step_*")) if os.path.isdir(p)]
    state_dirs.sort(key=lambda p: step_num(p, "accel_state_step_"))
    while len(state_dirs) > keep_last_k:
        old = state_dirs.pop(0)
        try:
            shutil.rmtree(old)
            logger.info("Removed old accelerate state dir: %s", old)
        except OSError as e:
            logger.warning("Failed to remove old accelerate state dir %s: %s", old, e)


def finalize_keep_weights_only(output_dir: str, keep_last_k: int = 1):
    """Training-complete cleanup: drop all resume state, keep recent weights.

    Removes every ``accel_state_step_*`` dir and every ``checkpoint_step_*.safetensors``
    except the most recent *keep_last_k*. Rank-0 only — caller must guard.
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
    files.sort(key=lambda p: step_num(p, "checkpoint_step_"))
    num_to_remove = max(0, len(files) - keep_last_k)
    for old in files[:num_to_remove]:
        if os.path.isfile(old):
            os.remove(old)
            logger.info("Removed old checkpoint after training completion: %s", old)
