"""RoboMIND dataloader for LeRobot v3 format datasets.

Reads converted RoboMIND buckets
through the shared :class:`~openwam.dataloader.bases.lerobot_v3_reader.LeRobotV3Reader`
machinery, with **real action / state supervision** enabled — functionally it
behaves like RoboCOIN / RoboTwin (real EEF targets + proprio), not like the
video-only EgoDex reader.

Heterogeneous embodiments in one root
-------------------------------------
RoboMIND mixes several embodiments, each with a different raw EEF layout and a
different camera set. They are distinguished purely by the camera keys present
in ``info.json`` (more robust than parsing ``robot_type`` strings):

  ===============  ===================================  ====  ===========================
  embodiment       cameras (head / left / right)        dim   raw EEF layout
  ===============  ===================================  ====  ===========================
  franka_1/3rgb    camera_top / —      / —               7    [xyz3, rpy3, grip1]
  ur_1rgb          camera_top / —      / —               7    [xyz3, rpy3, grip1]
  franka_sim       camera_front_external /
                   camera_handeye / —                    8    [xyz3, quat4(xyzw), grip1]
  agilex_3rgb      camera_front / camera_left_wrist /
                   camera_right_wrist                    14   [L(xyz3,rpy3,grip1), R(...)]
  ===============  ===================================  ====  ===========================

Notes:
  * franka_3rgb additionally has fixed third-person ``camera_left`` /
    ``camera_right`` (NOT wrist cameras); they are intentionally dropped — the
    bottom wrist slots stay black for that embodiment.
  * franka_sim additionally has ``camera_left_external`` / ``camera_right_external``;
    only ``camera_front_external`` + ``camera_handeye`` are used.

All embodiments emit the canonical 20-D EEF schema and a 384×320 multiview
L-canvas, so RoboMIND buckets collate cleanly with OXE / RoboCOIN / EgoDex in
``configs/dataloader/mixture.yaml``.

20-D conversion (pure geometry, see :func:`robomind_raw_to_20d`)
  single-arm (franka / ur / sim) → fill left half [0:10], zero-pad right half,
  masked by ``LEFT_ARM_DIM_MASK``; agilex → all 20 dims real (mask = None).

Per-embodiment stats + in-reader normalization
  Each ``robot_type`` has independent 20-D stats at
  ``{root}/meta/stats_{robot_type}.json`` (top-level key ``eef``, pooled over
  action+state). The reader normalizes inside ``_action_20d`` / ``_proprio_20d``
  via ``_normalize_array``; samples leave pre-normalized and
  ``normalization_stats`` returns None. Mirrors RoboCOIN exactly.

Action / state temporal alignment
  The conversion already row-aligned the parquet so that row ``t`` holds
  ``state[t]`` and ``action[t] := state[t+1]``. So ``win["action"]`` is used
  directly with no +1 offset (same contract as RoboCOIN). The window emits
  ``T_action = num_frames - 1`` action targets.

enable_action_supervision:
  False → action_mask / proprio_mask all-False (video-only auxiliary source).
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import (
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    assemble_single_arm_left,
    assert_unit_quaternion,
    eef14_to_eef20,
    euler_xyz_to_rot6d,
    quat_xyzw_to_rot6d,
)
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

logger = logging.getLogger(__name__)

_IMG_PREFIX = "observation.images."

# RoboMIND raw-EEF layouts, keyed by the embodiment's camera set.
ROBOMIND_EEF_KINDS = ("single_euler", "single_quat", "dual_euler")


# ---------------------------------------------------------------------------
# Layout resolution + raw → 20-D conversion (module-level, shared with the
# stats script — mirrors how robocoin_stats_computation imports
# _eef14_to_eef20 from robocoin.py).
# ---------------------------------------------------------------------------


def resolve_robomind_layout(features: dict) -> Tuple[str, Optional[str], Optional[str], str]:
    """Resolve (head, left_wrist, right_wrist, eef_kind) from info.json features.

    Decision is by camera-key presence (robust to robot_type string drift):

      * agilex      → both ``camera_left_wrist`` + ``camera_right_wrist`` present
      * franka_sim  → ``camera_handeye`` present (single wrist)
      * franka/ur   → ``camera_top`` present (no wrist)

    Raises ``ValueError`` when no head camera can be resolved (e.g. the
    franka_2rgb 2-cam subset, which the converter no longer emits).
    """
    keys = set(features.keys())
    left_wrist = _IMG_PREFIX + "camera_left_wrist"
    right_wrist = _IMG_PREFIX + "camera_right_wrist"
    handeye = _IMG_PREFIX + "camera_handeye"
    top = _IMG_PREFIX + "camera_top"

    if {left_wrist, right_wrist} <= keys:  # agilex dual-arm
        return _IMG_PREFIX + "camera_front", left_wrist, right_wrist, "dual_euler"
    if handeye in keys:  # franka_sim single-arm with wrist cam
        return _IMG_PREFIX + "camera_front_external", handeye, None, "single_quat"
    if top in keys:  # franka_1/3rgb, ur single-arm, no wrist
        return top, None, None, "single_euler"
    raise ValueError(
        "RoboMIND: no head camera resolved from features "
        f"{sorted(k for k in keys if k.startswith(_IMG_PREFIX))} — suspected unsupported 2-cam subset."
    )


def robomind_raw_to_20d(arr: np.ndarray, kind: str) -> np.ndarray:
    """Convert a RoboMIND raw EEF array to the canonical 20-D EEF schema.

    **Pure geometry, no normalization** — shared verbatim by the reader
    (``_action_20d`` / ``_proprio_20d``) and ``robomind_stats_computation``
    so the two produce bit-identical 20-D points. The reader applies
    ``_normalize_array`` on top; the stats script accumulates the raw output.

    Args:
        arr: ``(T, D)`` raw rows from the parquet ``action`` /
            ``observation.state`` column. ``D`` depends on ``kind``:
              * ``single_euler`` (franka_1/3rgb, ur): 7  = [xyz3, rpy3, grip1]
              * ``single_quat``  (franka_sim):        8  = [xyz3, quat4(xyzw), grip1]
              * ``dual_euler``   (agilex):            14 = [L(xyz3,rpy3,grip1), R(...)]
        kind: one of :data:`ROBOMIND_EEF_KINDS`.

    Returns:
        ``(T, 20)`` float array. Single-arm kinds fill the left half ``[0:10]``
        and zero-pad ``[10:20]`` (masked by ``LEFT_ARM_DIM_MASK`` downstream);
        ``dual_euler`` fills all 20 dims.
    """
    if kind == "single_euler":
        arm10 = np.concatenate([arr[:, 0:3], euler_xyz_to_rot6d(arr[:, 3:6]), arr[:, 6:7]], axis=-1)
        return assemble_single_arm_left(arm10)
    if kind == "single_quat":
        arm10 = np.concatenate([arr[:, 0:3], quat_xyzw_to_rot6d(arr[:, 3:7]), arr[:, 7:8]], axis=-1)
        return assemble_single_arm_left(arm10)
    if kind == "dual_euler":
        # Reslice agilex's interleaved [L_pos,L_rpy,L_grip,R_pos,R_rpy,R_grip]
        # into the (eef12, grip2) layout eef14_to_eef20 expects, so RoboMIND's
        # bimanual path is bit-identical to RoboCOIN's (covered by its tests).
        eef12 = np.concatenate([arr[:, 0:3], arr[:, 3:6], arr[:, 7:10], arr[:, 10:13]], axis=-1)
        grip2 = np.concatenate([arr[:, 6:7], arr[:, 13:14]], axis=-1)
        return eef14_to_eef20(eef12, grip2)
    raise ValueError(f"unknown RoboMIND eef kind: {kind!r} (expected one of {ROBOMIND_EEF_KINDS})")


# ---------------------------------------------------------------------------
# Single-bucket reader
# ---------------------------------------------------------------------------


class RoboMINDDataset(LeRobotV3Reader):
    """Single-bucket reader for one RoboMIND embodiment×benchmark bucket.

    The embodiment-specific bits (camera mapping, raw EEF layout, per-dim mask)
    are resolved per-instance in :meth:`_resolve_cameras`; everything else —
    window / offset / video / pickle / prompt / normalization — is inherited
    from :class:`LeRobotV3Reader`.
    """

    DATASET_NAME = "RoboMIND"
    NEEDED_COLS = ("task_index", "observation.state", "action")
    ACTION_DIM = EEF_DIM
    PROMPT_SOURCE = "task_index"
    PROMPT_FILE_REQUIRED = True  # the converter always writes meta/tasks.parquet
    DEFAULT_NORMALIZE_MODE = None  # class default off; robomind.yaml opts into quantile
    # Tolerate any wrist (auxiliary-camera) decode failure → black slot, same as RoboCOIN.
    WRIST_DECODE_TOLERATED = (Exception,)

    # ----- hooks ------------------------------------------------------------

    def _resolve_cameras(self, info: dict):
        """Resolve cameras + record embodiment kind / robot_type / per-dim mask.

        ``ACTION_DIM_MASK`` is set as an *instance* attribute (overriding the
        ``None`` class default) so each bucket in root mode carries its own
        single-vs-dual mask. Called once at __init__ before any window build.
        """
        features = info.get("features", {})
        head, left_wrist, right_wrist, kind = resolve_robomind_layout(features)
        self._eef_kind = kind
        self._robot_type = info.get("robot_type", "unknown")
        # Single-arm: only the left 10 dims are real; agilex: all 20 valid.
        self.ACTION_DIM_MASK = None if kind == "dual_euler" else LEFT_ARM_DIM_MASK
        return head, left_wrist, right_wrist

    def _post_init(self, info: dict) -> None:
        """For franka_sim, sanity-check the state quaternion is unit-norm (xyzw).

        Reads up to 64 rows from the first parquet shard; catches an
        un-normalized / wrongly-routed quaternion before it yields garbage
        rot6d. Mirrors the OXE RT-1 check.
        """
        if self._eef_kind != "single_quat":
            return
        first_path = self._dataset_dir / self._data_path_template.format(chunk_index=0, file_index=0)
        if not first_path.exists():
            return
        table = pq.read_table(first_path, memory_map=True, columns=["observation.state"])
        rows = table.column("observation.state").to_pylist()[:64]
        if not rows:
            return
        sample = np.stack(rows).astype(np.float32)
        if sample.shape[1] >= 7:
            assert_unit_quaternion(sample[:, 3:7], tol=0.05, sample_n=len(sample))

    def _load_stats(self, info: dict):
        """Load per-robot-type 20-D stats (``{root}/meta/stats_<robot_type>.json`` 'eef').

        Stats are always 20-D (single-arm right half is zero-range; the
        normalizer floors every scale at ``NORM_EPS`` so those dims map to a
        constant and are masked out by ``LEFT_ARM_DIM_MASK``). Mirrors RoboCOIN.
        """
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir.parent / "meta" / f"stats_{self._robot_type}.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but stats file is missing: {stats_path}. "
                f"Run python -m openwam.dataloader.utils.stats_computation.robomind_stats_computation to generate it, or set normalize_mode=null."
            )
        with open(stats_path) as f:
            raw = json.load(f)
        eef = raw.get("eef", {})
        return materialize_eef_stats(
            eef,
            self._normalize_mode,
            dim=EEF_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}: eef.* — re-run python -m openwam.dataloader.utils.stats_computation.robomind_stats_computation",
        )

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply per-bucket normalization to a (..., 20) array (no-op when null)."""
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        raw = np.stack(win["action"].values).astype(np.float32)  # (T_actual, 7|8|14)
        return self._normalize_array(robomind_raw_to_20d(raw, self._eef_kind))  # (T_actual, 20)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        raw = np.stack(win["observation.state"].values[:1]).astype(np.float32)  # (1, 7|8|14)
        return self._normalize_array(robomind_raw_to_20d(raw, self._eef_kind))  # (1, 20)

    @property
    def robot_type(self) -> str:
        return self._robot_type

    @property
    def eef_kind(self) -> str:
        return self._eef_kind

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiBucketRoboMINDDataset


# ---------------------------------------------------------------------------
# Multi-bucket wrapper (root mode)
# ---------------------------------------------------------------------------


class MultiBucketRoboMINDDataset(MultiLeRobotV3Reader):
    """Aggregate of N RoboMIND embodiment×benchmark buckets.

    Used in root mode: ``RoboMINDDataset.from_config({dataset_dir: <root>})``
    where ``<root>`` holds the per-embodiment subdirs (each with
    ``meta/info.json``). All buckets share ``action_dim=20`` and the 384×320
    canvas; each resolves its own cameras / EEF kind / dim mask.
    """

    def __init__(self, buckets: List["RoboMINDDataset"]):
        super().__init__(buckets)
        robot_types = sorted(set(b.robot_type for b in self._buckets))
        logger.info(
            "MultiBucketRoboMINDDataset: %d buckets, %d windows, %d robot types: %s",
            len(self._buckets),
            len(self),
            len(robot_types),
            robot_types,
        )

    @property
    def buckets(self) -> List["RoboMINDDataset"]:
        return self._buckets


__all__ = [
    "RoboMINDDataset",
    "MultiBucketRoboMINDDataset",
    "resolve_robomind_layout",
    "robomind_raw_to_20d",
    "ROBOMIND_EEF_KINDS",
]
