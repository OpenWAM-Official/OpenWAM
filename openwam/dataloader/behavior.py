"""BEHAVIOR-1K dataloader (2025 Challenge demos, robot R1Pro).

Reads the official LeRobot dataset ``behavior-1k/2025-challenge-demos`` and maps
it into OpenWAM's unified 80-D EEF action space, reusing the
:class:`~openwam.dataloader.bases.lerobot_v3_reader.LeRobotV3Reader` machinery
(windowing, video decode, multiview L-shape, unify-scatter, masks, normalization)
almost verbatim — BEHAVIOR is, in unified terms, a *grippered bimanual EEF* robot
exactly like a standard RoboCOIN bucket, with two deltas:

  1. **Mobile base.** R1Pro is a wheeled mobile manipulator. Following the
     1st-place challenge solution (I. Larchenko), the base action is a 3-D
     base-frame velocity ``[vx, vy, vyaw]`` (NOT a pose delta). It occupies the
     unified reserved slots ``[68:71)``; the rest of the reserved tail stays
     masked. This is the one thing RoboCOIN's grippered path does not populate —
     wired here via the unify map ``["0-9", "34-43", "68-70"]`` (raw dims 20:23 →
     unified 68:70).
  2. **EEF source + format.** RoboCOIN reads pre-computed ``eef_sim_pose_*``
     (euler) columns; BEHAVIOR has none. The per-arm EEF pose is read from the
     256-D ``observation.state`` (xyzw quaternions, base frame) and the dataset
     is **LeRobot v2.1** (one parquet per episode + ``meta/episodes.jsonl``),
     not v3 — so the IO layer (episode index + path templates + prompts) is
     overridden while every semantic helper is reused.

Unified 80-D layout (== robocoin.yaml): L[0:34] xyz3+rot6d6+grip1+dex24,
R[34:68] same, reserved[68:80]. R1Pro has parallel grippers (no dexterous hand),
so the 24 dex dims/arm are zero-padded + loss-masked; only L[0:10], R[34:44] and
base[68:71] carry real data.

Raw 23-D pre-scatter vector (action & proprio):
  [L_pos(3), L_rot6d(6), L_grip(1), R_pos(3), R_rot6d(6), R_grip(1), base_vel(3)]

Action/state temporal alignment:
  The EEF *action target* at window step t is the NEXT-frame achieved pose
  ``eef(state[t+1])`` (shifted +1, last step clamped → T_action = num_frames-1
  targets, matching the other readers). The gripper command (``action[:,14/22]``,
  binary {-1,+1}, +1=open) and base velocity (``action[:,0:3]``) are taken at t
  (row-aligned commands). Proprio is the current-frame (t=0) pose + commands.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import quat_xyzw_to_rot6d
from openwam.dataloader.utils.lerobotv3 import apply_info_splits
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

logger = logging.getLogger(__name__)

# ── observation.state[256] layout (reverse-engineered + validated on real data:
#    quat norms == 1.0, arm sin^2+cos^2 == 1.0, L/R eef pos symmetric in y) ──
_L_EEF_POS = slice(186, 189)
_L_EEF_QUAT = slice(189, 193)  # xyzw
_R_EEF_POS = slice(225, 228)
_R_EEF_QUAT = slice(228, 232)  # xyzw
# ── action[23] layout (OmniGibson ACTION_QPOS_INDICES['R1Pro']) ──
_ACT_BASE = slice(0, 3)  # [vx, vy, vyaw] base-frame velocity
_ACT_LGRIP = 14
_ACT_RGRIP = 22

# Raw pre-scatter width: EEF 20 (pos3+rot6d6+grip1 ×2) + base velocity 3.
_RAW_DIM = 23
_EEF_DIM = 20
_BASE_DIM = 3

# R1Pro RGB camera feature keys (depth / seg_instance are intentionally ignored).
_HEAD_CAMERA = "observation.images.rgb.head"
_LEFT_WRIST_CAMERA = "observation.images.rgb.left_wrist"
_RIGHT_WRIST_CAMERA = "observation.images.rgb.right_wrist"

_NEEDED_COLS = ("action", "observation.state")


def _state_to_eef18(state: np.ndarray) -> np.ndarray:
    """``(T, 256)`` state → ``(T, 18)`` pose ``[L_pos3, L_rot6d6, R_pos3, R_rot6d6]``.

    EEF poses are in the robot base frame; orientation is an xyzw unit quaternion
    converted to the 6-D rotation representation (first two rotation-matrix cols).
    """
    l_pos = state[:, _L_EEF_POS]
    l_rot6d = quat_xyzw_to_rot6d(state[:, _L_EEF_QUAT])
    r_pos = state[:, _R_EEF_POS]
    r_rot6d = quat_xyzw_to_rot6d(state[:, _R_EEF_QUAT])
    return np.concatenate([l_pos, l_rot6d, r_pos, r_rot6d], axis=-1).astype(np.float32)


def _assemble_raw23(eef18: np.ndarray, l_grip: np.ndarray, r_grip: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Interleave grippers + base into the canonical raw 23-D layout
    ``[L_pos3, L_rot6d6, L_grip1, R_pos3, R_rot6d6, R_grip1, base3]``."""
    return np.concatenate([eef18[:, 0:9], l_grip, eef18[:, 9:18], r_grip, base], axis=-1).astype(np.float32)


class BehaviorDataset(LeRobotV3Reader):
    """Single-bucket BEHAVIOR-1K reader (LeRobot v2.1, R1Pro, unified 80-D EEF)."""

    DATASET_NAME = "BEHAVIOR"
    NEEDED_COLS = _NEEDED_COLS
    # Raw pre-scatter width (eef20 + base3). With unify_action=True the public
    # ACTION_DIM becomes UNIFY_DIM (80); _raw_action_dim stays 23 (read from this
    # instance attr by the base before it resets ACTION_DIM → see base __init__).
    ACTION_DIM = _RAW_DIM
    # All 23 raw dims are real → leave ACTION_DIM_MASK None; under unify the
    # scattered _unify_dim_mask marks exactly the mapped slots {0:10, 34:44,
    # 68:71} valid and everything else (dex, reserved tail) masked.
    ACTION_DIM_MASK = None
    # Prompt is per-episode in meta/episodes.jsonl (no tasks.parquet).
    PROMPT_SOURCE = "episode_annotated"
    DEFAULT_NORMALIZE_MODE = "quantile"
    # Tolerate any wrist-camera decode failure (→ black slot), mirroring RoboCOIN.
    WRIST_DECODE_TOLERATED = (Exception,)

    # ----- hooks ------------------------------------------------------------

    def _resolve_cameras(self, info: dict):
        """Fixed R1Pro RGB cameras (head + 2 wrists); skip depth / seg streams."""
        features = info.get("features", {})
        if _HEAD_CAMERA not in features:
            raise ValueError(f"BEHAVIOR({self._dataset_id}): head camera {_HEAD_CAMERA!r} missing from info.features")
        left = _LEFT_WRIST_CAMERA if _LEFT_WRIST_CAMERA in features else None
        right = _RIGHT_WRIST_CAMERA if _RIGHT_WRIST_CAMERA in features else None
        return _HEAD_CAMERA, left, right

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        """Build the eps DataFrame from LeRobot **v2.1** ``meta/episodes.jsonl``.

        v2.1 stores one parquet per episode at
        ``data/task-{chunk:04d}/episode_{episode_index:08d}.parquet`` with
        ``chunk = episode_index // chunks_size`` (== task id). So every episode
        is its own file → all data/video row offsets are 0. Episodes whose data
        parquet is not present on disk (partial download) are dropped.
        """
        chunks_size = int(info.get("chunks_size", 10000))
        cams = [c for c in (self._head_camera, self._left_wrist_camera, self._right_wrist_camera) if c]

        eps_path = self._dataset_dir / "meta" / "episodes.jsonl"
        records = []
        prompts = {}
        with open(eps_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ei = int(d["episode_index"])
                records.append((ei, int(d["length"])))
                tasks = d.get("tasks") or []
                prompts[ei] = tasks[0].strip() if tasks else ""
        self._ep_prompt = prompts  # consumed by _load_prompts

        # Keep only episodes whose data parquet AND head video are both on disk
        # (robust to partial downloads — head-video decode is fatal otherwise).
        def _present(paths):
            out = set()
            for p in paths:
                m = re.search(r"episode_(\d+)\.(?:parquet|mp4)$", p.name)
                if m:
                    out.add(int(m.group(1)))
            return out

        present = _present((self._dataset_dir / "data").glob("task-*/episode_*.parquet"))
        present &= _present((self._dataset_dir / "videos").glob(f"task-*/{self._head_camera}/episode_*.mp4"))
        records = [(ei, n) for ei, n in records if ei in present]
        if not records:
            raise FileNotFoundError(
                f"BEHAVIOR({self._dataset_id}): no episode parquet found under {self._dataset_dir}/data "
                f"(downloaded {len(present)} files). Check dataset_dir / download."
            )

        ep_idx = np.array([ei for ei, _ in records], dtype=np.int64)
        length = np.array([n for _, n in records], dtype=np.int64)
        chunk = ep_idx // chunks_size
        zeros = np.zeros(len(records), dtype=np.int64)
        cols = {
            "episode_index": ep_idx,
            "length": length,
            "data/chunk_index": chunk,
            "data/file_index": ep_idx,
            "_data_row_offset": zeros,
        }
        for cam in cams:
            cols[f"videos/{cam}/chunk_index"] = chunk
            cols[f"videos/{cam}/file_index"] = ep_idx
            cols[self._video_offset_col(cam)] = zeros
        df = pd.DataFrame(cols)

        # Honor info.json[splits] exactly like the v3 default (apply_info_splits):
        # train → all on-disk episodes, a declared split → its episode_index range,
        # any *undeclared* non-train split → empty. Without this a split="val" loader
        # would silently serve the whole training set (the dataset declares only a
        # train split), diverging from every sibling reader's "empty val" contract.
        info_splits = info.get("splits", {}) or {}
        df = apply_info_splits(df, self._split, info_splits, source_name=f"BEHAVIOR({self._dataset_id})")

        # The episode_annotated resolver does NOT guard emptiness (unlike the
        # task_index path, which raises). Fail fast here so a blank-`tasks` episode
        # can't feed an empty prompt into the model.
        blank = [int(ei) for ei in df["episode_index"].tolist() if not prompts.get(int(ei), "").strip()]
        if blank:
            raise ValueError(
                f"BEHAVIOR({self._dataset_id}): {len(blank)} served episode(s) have an empty 'tasks' prompt "
                f"in meta/episodes.jsonl (e.g. {blank[:5]}); per-episode prompts must be non-empty."
            )
        logger.info(
            "BEHAVIOR(%s): %d episodes (split=%s; %d on disk, %d in jsonl)",
            self._dataset_id,
            len(df),
            self._split,
            len(records),
            len(prompts),
        )
        return df

    def _load_prompts(self) -> None:
        """Per-episode prompts from meta/episodes.jsonl (cached in _build_episode_index)."""
        self._episode_idx_to_text = getattr(self, "_ep_prompt", {})

    def _post_init(self, info: dict) -> None:
        """Rewrite v2.1 path templates to the base's ``{chunk_index}/{file_index}``
        placeholders, and sanity-check the (reverse-engineered) quat offsets."""
        # info.json templates use {episode_chunk}/{episode_index}; the base formats
        # with chunk_index=/file_index=. Our eps df sets chunk_index=task chunk,
        # file_index=episode_index, so renaming the placeholders makes the inherited
        # _read_data_file_uncached / _decode_one_camera work unchanged.
        self._data_path_template = "data/task-{chunk_index:04d}/episode_{file_index:08d}.parquet"
        self._video_path_template = "videos/task-{chunk_index:04d}/{video_key}/episode_{file_index:08d}.mp4"
        if len(self._eps_df) == 0:
            return  # empty split (e.g. val on a train-only dataset) — nothing to check
        # Fail fast if a future re-upload changes the state packing.
        try:
            ep0 = int(self._eps_df["episode_index"].iloc[0])
            chunk0 = int(self._eps_df["data/chunk_index"].iloc[0])
            path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk0, file_index=ep0)
            import pyarrow.parquet as pq

            vals = pq.read_table(path, columns=["observation.state"]).to_pandas()["observation.state"].values[:64]
            if len(vals) == 0:
                return  # 0-row episode parquet — nothing to sanity-check
            st = np.stack(vals)
            for sl, name in ((_L_EEF_QUAT, "left"), (_R_EEF_QUAT, "right")):
                norms = np.linalg.norm(st[:, sl].astype(np.float64), axis=-1)
                if np.abs(norms - 1.0).max() > 0.05:
                    raise ValueError(
                        f"BEHAVIOR({self._dataset_id}): {name} eef quat at state[{sl.start}:{sl.stop}] is not "
                        f"unit-norm (max|‖q‖-1|={np.abs(norms - 1.0).max():.3f}); observation.state layout may "
                        "have changed — re-verify the EEF offsets."
                    )
        except FileNotFoundError:
            pass  # episode file not present yet (partial download) — skip the check

    def _load_stats(self, info: dict):
        """Load ``meta/stats_R1Pro.json`` → combined 23-D (eef20 + base_vel3) stats.

        Mirrors RoboCOIN's per-robot-type stats, with rot6d pinned to identity in
        the stats file (see behavior_stats_computation). The base velocity block is
        a BEHAVIOR-specific addition (no rot6d pin; real stats)."""
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir / "meta" / "stats_R1Pro.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but {stats_path} is missing. Run "
                "python -m openwam.dataloader.utils.stats_computation.behavior_stats_computation "
                f"--dataset_dir {self._dataset_dir}, or set normalize_mode=null."
            )
        with open(stats_path) as f:
            raw = json.load(f)
        eef = materialize_eef_stats(
            raw.get("eef", {}),
            self._normalize_mode,
            dim=_EEF_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}: eef.*",
        )
        base = materialize_eef_stats(
            raw.get("base_vel", {}),
            self._normalize_mode,
            dim=_BASE_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}: base_vel.*",
        )
        keys = ("mean", "std", "min", "max", "q01", "q99")
        for blk, name, dim in ((eef, "eef", _EEF_DIM), (base, "base_vel", _BASE_DIM)):
            for k in keys:
                if blk[k].shape[0] != dim:
                    raise ValueError(
                        f"BEHAVIOR({self._dataset_id}): '{name}' stats '{k}' width {blk[k].shape[0]} "
                        f"in {stats_path} != expected {dim}. Re-run behavior_stats_computation."
                    )
        return {k: np.concatenate([eef[k], base[k]]).astype(np.float32) for k in keys}

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply per-bucket normalization to a ``(..., 23)`` raw vector (no-op when
        normalize_mode is null / stats absent)."""
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    # ----- action / proprio -------------------------------------------------

    def _action_20d(self, win) -> np.ndarray:
        """Raw ``(actual_raw_len, 23)`` action: next-frame EEF pose + grip/base cmd at t."""
        state = np.stack(win["observation.state"].values).astype(np.float32)  # (L, 256)
        action = np.stack(win["action"].values).astype(np.float32)  # (L, 23)
        eef = _state_to_eef18(state)  # (L, 18) current-frame poses
        # action target = next-frame achieved pose (shift +1; clamp the last step,
        # which T_action = num_frames-1 drops for a full window anyway).
        eef_next = np.concatenate([eef[1:], eef[-1:]], axis=0) if len(eef) > 1 else eef
        raw = _assemble_raw23(
            eef_next,
            action[:, _ACT_LGRIP : _ACT_LGRIP + 1],
            action[:, _ACT_RGRIP : _ACT_RGRIP + 1],
            action[:, _ACT_BASE],
        )
        return self._normalize_array(raw)

    def _proprio_20d(self, win) -> np.ndarray:
        """Raw ``(1, 23)`` proprio: current-frame (t=0) EEF pose + grip/base cmd."""
        state = np.stack(win["observation.state"].values[:1]).astype(np.float32)  # (1, 256)
        action = np.stack(win["action"].values[:1]).astype(np.float32)  # (1, 23)
        eef = _state_to_eef18(state)  # (1, 18)
        raw = _assemble_raw23(
            eef, action[:, _ACT_LGRIP : _ACT_LGRIP + 1], action[:, _ACT_RGRIP : _ACT_RGRIP + 1], action[:, _ACT_BASE]
        )
        return self._normalize_array(raw)


__all__ = ["BehaviorDataset"]
