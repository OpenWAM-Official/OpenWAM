"""VLABench primitive-task reader.

VLABench's standard primitive finetune dataset (LeRobot v3, bucket
``vlabench_primitive_ft_lerobot_video``): Franka Panda single-arm, 5000
episodes / 575k frames / 128 task variants at 10 fps, 480x480 AV1 mp4 per
camera. This is the official 10-task x 500-episode release described in the
VLABench README.

Cameras
-------
``image`` (front) -> head slot, ``wrist_image`` -> left wrist slot, and
``second_image`` (a second external view) -> right wrist slot, so all three
views reach the multiview canvas.

State / action
--------------
Both share the 7-D layout ``[x, y, z, roll, pitch, yaw, gripper]`` (Euler XYZ,
radians) — an absolute EE pose in the ROBOT BASE frame, not the world frame:
VLABench's converter subtracts the base position
(``episode_config["robot"]["position"]``, default ``[0, -0.4, 0.78]``) from the
recorded world pose. The eval client re-adds it before handing the target to
VLABench's IK. Layout is identical to the OXE euler7 stream, so the OXE
conversion helper applies unchanged, and Euler wraparound at +-pi disappears
in the rot6d representation.

Gripper polarity (upstream trap)
--------------------------------
``state`` and ``action`` use OPPOSITE gripper conventions in this dataset —
measured correlation between the two columns is -0.93:

* ``state[6]``  is ``robot.get_ee_open_state()``, which for the Franka returns
  True when the fingers are CLOSED. That is an acknowledged upstream bug
  (``VLABench/robots/single_arm/franka.py:57`` carries the comment
  ``# BUG: should be False``; the WidowX implementation has the opposite,
  correct, polarity). So **state 1 = closed**.
* ``actions[6]`` is the commanded finger width binarized by the converter with
  ``> 0.03 -> 1`` against a 0.04 m open width. So **action 1 = open**.

Both are passed through verbatim. The point is consistency with the training
distribution, not correctness of the upstream convention — the eval client
reads the same buggy accessor at rollout time, so the inversion cancels out.
"Fixing" either side here would silently take the policy off-distribution.

Action space
------------
Raw EEF10 ``[xyz3, rot6d6, gripper1]``, normalized first and then scattered
into the unified 80-D left-arm slots by ``unify_action`` /
``unify_action_map: ["0-9"]`` (see ``configs/dataloader/vlabench.yaml``). Those
ten slots are pretrained semantic dims, which is what makes warm-starting from
``OpenWAM/Pretrained_OpenWAM`` meaningful.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization
from openwam.dataloader.utils.oxe_schema import euler7_action_to_arm10

logger = logging.getLogger(__name__)

_ACTION_MODE = "eef"
EEF10_DIM = 10


class VLABenchDataset(LeRobotV3Reader):
    """Single-bucket VLABench reader for the LeRobot v3 primitive finetune set."""

    DATASET_NAME = "VLABench"
    HEAD_CAMERA = "image"
    LEFT_WRIST_CAMERA = "wrist_image"
    RIGHT_WRIST_CAMERA = "second_image"
    NEEDED_COLS = ("state", "actions", "task_index")
    ACTION_DIM = EEF10_DIM
    PROMPT_SOURCE = "task_index"
    PROMPT_FILE_REQUIRED = True
    # Sim data is clean (no outliers), so plain min-max matches the RoboTwin /
    # LIBERO deploy convention; the stats payload carries all six stat fields so a
    # config can still opt into quantile / z-score.
    DEFAULT_NORMALIZE_MODE = "min-max"
    STATS_FILENAME = "vlabench_normalization_stats.npy"
    STATS_DIM = EEF10_DIM
    STATS_STRICT_MINMAX = True
    # Serve the unified action but stay deployable: the deploy server gathers
    # the model's 80-D output back to raw EEF10, then unnormalizes with the RAW
    # stats written below.
    DEPLOY_ACTION_MODE = _ACTION_MODE

    def _add_data_offsets(self, eps: pd.DataFrame) -> None:
        """Rebuild the episode -> (data shard, row offset) map from real row counts.

        **The upstream episode metadata is corrupt.** In
        ``VLABench/vlabench_primitive_ft_lerobot_video`` (verified against the
        Hub, so this is not a damaged download) the writer emitted

            dataset_from_index = length * episode_index
            dataset_to_index   = length * (episode_index + 1)

        as if every episode had the same length as the current one, and
        ``data/chunk_index`` / ``data/file_index`` are wrong in step with it —
        e.g. episodes 0, 1 and 2 all claim file 0, but file 0 holds only
        93 + 74 = 167 rows, exactly episodes 0 and 1. Trusting those columns
        reads the wrong parquet rows, which pairs each video clip with another
        episode's actions and prompt: silent training-data corruption, or an
        ``IndexError`` on an empty slice when the offset runs past the file.

        ``length`` itself is sound — it sums to ``info.json``'s total_frames and
        matches each episode's video frame count one-for-one — and the episodes
        are packed into the data shards contiguously in ``episode_index`` order
        (verified exactly across all 3,953 shards / 5,000 episodes). So the true
        layout is recoverable: accumulate ``length`` in ``episode_index`` order
        for each episode's global row start, then locate it with the real
        per-file row counts read from the parquet footers.

        The video metadata needs no repair — one mp4 per episode, correct
        indices, ``to_timestamp - from_timestamp == length / fps`` throughout.
        """
        paths = list((self._dataset_dir / "data").glob("chunk-*/file-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No data parquet files under {self._dataset_dir}/data")

        def _read_meta(path):
            chunk_m = re.search(r"chunk-(\d+)$", path.parent.name)
            file_m = re.search(r"file-(\d+)$", path.stem)
            if chunk_m is None or file_m is None:
                return None
            return (int(chunk_m.group(1)), int(file_m.group(1)), pq.ParquetFile(path).metadata.num_rows)

        with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as pool:
            results = list(pool.map(_read_meta, paths))
        data_files = [r for r in results if r is not None]
        if not data_files:
            raise FileNotFoundError(f"No usable data parquet files under {self._dataset_dir}/data")
        # Order by the PARSED (chunk, file) ints, not by path string: lexicographic
        # order only coincides with numeric order while file_index stays 3 digits
        # ("file-1000" sorts before "file-999"). This release keeps it there via
        # chunks_size=1000, but nothing here reads or enforces that, and the
        # cumulative row offsets below are only valid in true packing order.
        data_files.sort(key=lambda r: (r[0], r[1]))

        file_rows = np.array([n for _, _, n in data_files], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(file_rows)]).astype(np.int64)

        # Global row start per episode = cumulative length in episode_index order.
        order = np.argsort(eps["episode_index"].to_numpy(), kind="stable")
        lengths = eps["length"].to_numpy().astype(np.int64)
        cumulative = np.concatenate([[0], np.cumsum(lengths[order])[:-1]])
        global_starts = np.empty(len(eps), dtype=np.int64)
        global_starts[order] = cumulative

        total_rows = int(starts[-1])
        total_len = int(lengths.sum())
        if total_len != total_rows:
            raise ValueError(
                f"{self.DATASET_NAME}: sum(length)={total_len} but the data parquets hold "
                f"{total_rows} rows. The episode metadata cannot be repaired by repacking; "
                "the dataset copy is inconsistent beyond the known upstream index bug."
            )

        file_pos = np.searchsorted(starts, global_starts, side="right") - 1
        row_offset = global_starts - starts[file_pos]
        # An episode that straddled a shard boundary would be silently truncated
        # by the single-file slice in _getitem_impl, so refuse rather than train
        # on short windows.
        overflow = row_offset + lengths > file_rows[file_pos]
        if overflow.any():
            bad = int(np.flatnonzero(overflow)[0])
            raise ValueError(
                f"{self.DATASET_NAME}: episode {int(eps['episode_index'].to_numpy()[bad])} spans a data "
                f"shard boundary (offset {int(row_offset[bad])} + length {int(lengths[bad])} > "
                f"{int(file_rows[file_pos[bad]])} rows). The single-shard window slice cannot serve it."
            )

        chunks = np.array([data_files[i][0] for i in file_pos], dtype=np.int64)
        files = np.array([data_files[i][1] for i in file_pos], dtype=np.int64)
        n_moved = int(
            ((eps["data/chunk_index"].to_numpy() != chunks) | (eps["data/file_index"].to_numpy() != files)).sum()
        )
        if n_moved:
            logger.warning(
                "%s: repaired %d/%d episode->shard assignments from real parquet row counts "
                "(upstream metadata writes dataset_from_index = length * episode_index).",
                self.DATASET_NAME,
                n_moved,
                len(eps),
            )
        eps["data/chunk_index"] = chunks
        eps["data/file_index"] = files
        eps["_data_row_offset"] = row_offset

    def _build_stats_rank0(self, path) -> None:
        """Auto-build the pooled stats file: rank 0 scans, other ranks wait."""
        import os
        import time

        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        if dist_ready:
            rank = dist.get_rank()
        else:
            # torchrun sets RANK before init_process_group; honor it so
            # pre-init constructions still elect a single builder.
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            from openwam.dataloader.utils.stats_computation.vlabench_stats_computation import (
                build_and_save_vlabench_stats,
            )

            build_and_save_vlabench_stats(self._dataset_dir, output=path)
        else:
            deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
            poll_interval_s = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
            while not path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for rank 0 to build VLABench stats: {path}")
                time.sleep(poll_interval_s)

    def _load_stats(self, info: dict):
        """Load ``meta/vlabench_normalization_stats.npy`` (auto-built on first
        use — rank 0 scans, other ranks wait) and emit the deploy denormalizer
        artifact.

        The base implementation only materializes the in-reader stats; without
        also writing ``meta/normalization_stats.npy`` the trained checkpoint
        would have no denormalizer and deploy would return normalized actions.
        """
        if self._normalize_mode and self._normalize_mode not in ("none", "null"):
            stats_path = self._dataset_dir / "meta" / self.STATS_FILENAME
            if not stats_path.is_file():
                self._build_stats_rank0(stats_path)
        stats = super()._load_stats(info)
        if stats is not None:
            self._write_deploy_normalizer_stats(stats, STAT_KEYS)
        return stats

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        action = np.stack(win["actions"].values).astype(np.float32)  # (T_actual, 7)
        return apply_normalization(euler7_action_to_arm10(action), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        state = np.stack(win["state"].values[:1]).astype(np.float32)  # (1, 7)
        return apply_normalization(euler7_action_to_arm10(state), self._normalization_stats, self._normalize_mode)


__all__ = ["VLABenchDataset"]
