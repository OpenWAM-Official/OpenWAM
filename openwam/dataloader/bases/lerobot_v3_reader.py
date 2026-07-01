"""Unified single-bucket LeRobot v3 reader base.

One base class for every LeRobot v3 single-bucket reader in this package —
the 4 OXE readers (BC-Z / Bridge / RT-1 / DROID), RoboCOIN, and EgoDex are
all **direct, equal subclasses** of :class:`LeRobotV3Reader`. There is no
per-family intermediate base; the differences between readers are expressed
through a small set of hook methods + class attributes, and the genuinely
shared building blocks (single-arm EEF assembly, stats materialization,
prompt resolution, multi-bucket construction) live as stateless helpers in
``openwam.dataloader.utils``.

The canonical sample dict every subclass emits::

    {
      "video": [PIL.Image x num_video_frames],
      "vace_video": None,
      "first_frame_image": [video[0]],
      "action": (T-1, action_dim) float32,
      "action_mask": (T-1, action_dim) bool,
      "video_mask": (num_video_frames,) bool,
      "proprio": (1, action_dim) float32,
      "proprio_mask": (1, action_dim) bool,
      "prompt": str,
    }

Subclass surface
----------------
Class attributes:
  * ``DATASET_NAME``        — log / error label.
  * ``NEEDED_COLS``         — parquet columns the reader consumes.
  * ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` / ``RIGHT_WRIST_CAMERA`` — used by
    the default :meth:`_resolve_cameras`.
  * ``ACTION_DIM_MASK``     — per-dim validity mask (``LEFT_ARM_DIM_MASK`` for
    single-arm OXE; ``None`` = all dims valid, for bimanual RoboCOIN).
  * ``ACTION_DIM``          — per-step action/proprio width (20 = EEF).
  * ``PROMPT_FILE_REQUIRED``— default :meth:`_load_prompts` behavior on a
    missing ``tasks.parquet`` (True = raise, False = empty map).

Hooks (override as needed; defaults cover the common case):
  * ``_resolve_cameras(info)``     — (head, left_wrist, right_wrist).
  * ``_add_data_offsets(eps)``     — set ``eps['_data_row_offset']``.
  * ``_train_min_window_len()``    — min episode length for a train window.
  * ``_load_prompts()``            — populate the prompt lookup.
  * ``_resolve_prompt(row, win)``  — per-sample prompt text.
  * ``_load_stats(info)``          — return in-reader normalization stats / None.
  * ``_action_20d(win)`` / ``_proprio_20d(win)`` — normalized (..., 20) payload,
    or ``None`` to emit zeros (video-only / disabled supervision).
  * ``_post_init(info)``           — one-time per-dataset setup.
  * ``_multibucket_wrapper()``     — root-mode aggregate wrapper class or None.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases.dataset import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils.eef import (
    EEF_DIM,
    build_action_mask_2d,
    build_proprio_mask_2d,
)
from openwam.dataloader.utils.lerobotv3 import (
    apply_info_splits,
    build_multibucket,
    compute_file_local_offsets,
    load_episodes_parquet,
    load_tasks_annotated,
    parse_info_json,
    resolve_prompt_by_episode,
    subsample_episodes_by_hours,
)
from openwam.dataloader.utils.normalization import materialize_eef_stats
from openwam.dataloader.utils.unify_action import UNIFY_DIM, map_to_unify, parse_unify_spec
from openwam.dataloader.utils.video_io import decode_video_frames as _decode_video_frames

logger = logging.getLogger(__name__)

# Multiview L-shape slot sizes (must match assemble_multiview_layout defaults).
_HEAD_SLOT_H, _HEAD_SLOT_W = 256, 320
_WRIST_SLOT_H, _WRIST_SLOT_W = 128, 160
_GETITEM_MAX_RETRIES = 64

# Sentinel so an explicit ``normalize_mode=None`` (disable normalization) is
# distinguishable from "not passed" (fall back to the subclass default).
_NORMALIZE_MODE_UNSET = object()

# Sentinel for from_config key-presence checks: distinguishes a config key that
# is absent from one explicitly set to None.
_CONFIG_MISSING = object()


class LeRobotV3Reader(BaseDataset):
    """Shared single-bucket LeRobot v3 reading machinery (see module docstring)."""

    # --- subclass-overridable class attributes ---
    DATASET_NAME: ClassVar[str] = "LeRobotV3"
    NEEDED_COLS: ClassVar[Tuple[str, ...]] = ()
    HEAD_CAMERA: ClassVar[Optional[str]] = None
    LEFT_WRIST_CAMERA: ClassVar[Optional[str]] = None
    RIGHT_WRIST_CAMERA: ClassVar[Optional[str]] = None
    ACTION_DIM_MASK: ClassVar[Optional[np.ndarray]] = None
    ACTION_DIM: ClassVar[int] = EEF_DIM
    # Prompt convention: "task_index" (tasks.parquet, index=text — RoboCOIN /
    # EgoDex) or "episode_annotated" (tasks_annotated.parquet by episode_index — OXE).
    PROMPT_SOURCE: ClassVar[str] = "task_index"
    PROMPT_FILE_REQUIRED: ClassVar[bool] = True
    # In-reader normalization stats: when STATS_FILENAME is set, the default
    # ``_load_stats`` loads ``meta/<STATS_FILENAME>`` (a flat stats dict) with
    # the given dim / min-max strictness (OXE single-arm = 10-D, strict).
    # RoboCOIN overrides ``_load_stats`` for its dynamic per-robot-type path.
    STATS_FILENAME: ClassVar[Optional[str]] = None
    STATS_DIM: ClassVar[int] = EEF_DIM
    STATS_STRICT_MINMAX: ClassVar[bool] = False
    # Default normalize_mode when the caller doesn't pass one. OXE readers
    # override to "quantile" (their historical default); RoboCOIN/EgoDex keep
    # None (no in-reader normalization unless a config opts in).
    DEFAULT_NORMALIZE_MODE: ClassVar[Optional[str]] = None
    # Exception types a wrist (auxiliary-camera) decode failure is tolerated for
    # (missing / corrupt mp4 → black slot rather than a fatal sample). OXE used
    # the specific triple; RoboCOIN tolerated all. Subclasses override as needed.
    WRIST_DECODE_TOLERATED: ClassVar[tuple] = (FileNotFoundError, OSError, RuntimeError)

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_frames: int = 33,
        video_stride: int = 4,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        multiview: bool = False,
        normalize_mode: Any = _NORMALIZE_MODE_UNSET,
        enable_action_supervision: bool = True,
        dataset_id: Optional[str] = None,
        target_camera: Optional[str] = None,
        camera_layout: Optional[List[str]] = None,
        # Unified action space. unify_action=True scatters this reader's raw
        # ACTION_DIM-wide action/proprio into a UNIFY_DIM-wide vector and marks
        # only the mapped dims valid. unify_action_map is the yaml spec (see
        # openwam.dataloader.utils.unify_action); None + unify_action=True maps
        # the raw dims in order (0..raw_dim-1). The unified width is the global
        # UNIFY_DIM constant (not per-dataset configurable).
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        # Optional load-time video color jitter, applied consistently across a
        # clip's frames and ONLY on the train split. None / False / {} → disabled
        # (default; byte-identical to before). Truthy → enabled; a dict overrides
        # the per-channel strengths {brightness, contrast, saturation, hue}.
        color_jitter: Optional[Any] = None,
        # Optional data-budget knobs (None = use full bucket; the default
        # path is byte-identical to the pre-budget behavior).
        max_hours: Optional[float] = None,
        subsample_seed: int = 42,
        **_unused: Any,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = dataset_id or self._dataset_dir.name
        self._num_frames = int(num_frames)
        self._video_stride = max(1, int(video_stride))
        self._window_stride = max(1, int(window_stride))
        self._height = int(height)
        self._width = int(width)
        self._split = split
        self._multiview = bool(multiview)
        self._normalize_mode = (
            self.DEFAULT_NORMALIZE_MODE if normalize_mode is _NORMALIZE_MODE_UNSET else normalize_mode
        )
        self._enable_action_supervision = bool(enable_action_supervision)
        self._target_camera = target_camera
        self._camera_layout_param = list(camera_layout) if camera_layout else None
        self._max_hours = max_hours
        self._subsample_seed = int(subsample_seed)

        # ── load-time video augmentation ──────────────────────────────────
        # Color jitter is applied in _getitem_impl to the decoded clip (same
        # random factors across all frames, via VideoColorJitter). Built only
        # for the train split; val / disabled keeps video byte-identical.
        self._color_jitter = None
        if color_jitter and split == "train":
            cj_get = color_jitter.get if hasattr(color_jitter, "get") else (lambda k, d: d)
            self._color_jitter = VideoColorJitter(
                brightness=float(cj_get("brightness", 0.1)),
                contrast=float(cj_get("contrast", 0.1)),
                saturation=float(cj_get("saturation", 0.1)),
                hue=float(cj_get("hue", 0.0)),
            )

        # ── unified action space ──────────────────────────────────────────
        # _raw_action_dim is what this reader's _action_20d/_proprio_20d emit
        # (the class ACTION_DIM). When unify is on, the public ACTION_DIM (and
        # thus the finalized action/proprio width + downstream model action_dim)
        # becomes unify_dim, and _finalize_* scatters raw -> unified via
        # _unify_dst_index. Off (default) → byte-identical to before.
        # Instance attr (not type(self).ACTION_DIM) so a subclass can override the
        # raw action width per bucket BEFORE super().__init__ — RoboCOIN sets a
        # wider raw dim (pose + dexterous-hand fingers) for dex-hand buckets under
        # unify. Defaults to the class ACTION_DIM, so existing readers are unchanged.
        self._raw_action_dim = int(self.ACTION_DIM)
        self._unify = bool(unify_action)
        self._unify_dim = int(UNIFY_DIM)
        self._unify_dst_index: Optional[np.ndarray] = None
        if self._unify:
            if unify_action_map is None:
                # No spec → identity map: raw dims 0..raw_dim-1 in order.
                # Single-list form (top-level ints), NOT [[...]] which the
                # parser would read as one malformed src->dst pair.
                spec = list(range(self._raw_action_dim))
            else:
                spec = unify_action_map
            self._unify_dst_index = parse_unify_spec(spec, self._unify_dim)
            if self._unify_dst_index.shape[0] != self._raw_action_dim:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): unify_action_map maps "
                    f"{self._unify_dst_index.shape[0]} source dims but this reader emits "
                    f"{self._raw_action_dim}-D action. They must match."
                )
            # Instance attr shadows the class ACTION_DIM so finalized payloads,
            # the action_dim property, and the model action head are all unify_dim.
            self.ACTION_DIM = self._unify_dim

        # Rate-limited failure counters for _safe_get / wrist decode.
        self._fail_count = 0
        self._wrist_fail_count = 0
        self._fail_log_every = 100

        # ── info.json ─────────────────────────────────────────────────────
        info = parse_info_json(self._dataset_dir)
        self._fps = float(info["fps"])
        self._data_path_template = info["data_path"]
        self._video_path_template = info["video_path"]

        # ── cameras + multiview layout ────────────────────────────────────
        self._head_camera, self._left_wrist_camera, self._right_wrist_camera = self._resolve_cameras(info)
        if self._head_camera is None:
            raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): no head camera resolved")
        if self._multiview:
            self._camera_layout = self._camera_layout_param or [
                self._head_camera,
                self._left_wrist_camera or "__missing_left__",
                self._right_wrist_camera or "__missing_right__",
            ]
        else:
            self._camera_layout = [self._head_camera]

        # ── video sub-sampling ────────────────────────────────────────────
        # video_stride takes every Nth frame within the window. num_video_frames
        # is whatever that yields; for clean encoder temporal downsampling it
        # should match the encoder's contract (Wan VAE: (num_video_frames - 1) % 4
        # == 0) — NOT enforced here, a mismatch surfaces downstream at encode time.
        self._video_sample_indices = np.arange(0, self._num_frames, self._video_stride, dtype=np.int64)
        self._num_video_frames = int(self._video_sample_indices.size)

        # ── episodes + offsets + split ────────────────────────────────────
        eps = load_episodes_parquet(self._dataset_dir)
        self._add_episode_offsets(eps)
        info_splits = info.get("splits", {}) or {}
        self._eps_df = apply_info_splits(
            eps, split, info_splits, source_name=f"{self.DATASET_NAME}({self._dataset_id})"
        )

        # ── optional episode-level subsample to fit a per-bucket hour budget ──
        if self._max_hours is not None:
            n_before = len(self._eps_df)
            if n_before == 0:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): eps_df is already empty before "
                    "subsample (likely split/info.json mismatch)."
                )
            self._eps_df = subsample_episodes_by_hours(
                self._eps_df,
                target_hours=float(self._max_hours),
                fps=self._fps,
                seed=self._subsample_seed,
            )
            if len(self._eps_df) == 0:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): subsampling to max_hours="
                    f"{self._max_hours}h left 0 episodes. target_hours may be smaller than any episode."
                )
            logger.info(
                "%s(%s): subsampled %d/%d episodes (max_hours=%.3f, seed=%d)",
                self.DATASET_NAME,
                self._dataset_id,
                len(self._eps_df),
                n_before,
                self._max_hours,
                self._subsample_seed,
            )

        self._ep_data_row_offset = self._eps_df["_data_row_offset"].to_numpy().astype(np.int64)
        self._ep_video_frame_offsets: Dict[str, np.ndarray] = {}
        for cam in (self._head_camera, self._left_wrist_camera, self._right_wrist_camera):
            if cam is None:
                continue
            col = self._video_offset_col(cam)
            if col in self._eps_df.columns:
                self._ep_video_frame_offsets[cam] = self._eps_df[col].to_numpy().astype(np.int64)

        # ── window index ──────────────────────────────────────────────────
        length = self._eps_df["length"].to_numpy().astype(np.int64)
        min_window_len = self._num_frames if self._split == "val" else self._train_min_window_len()
        n_starts = np.where(
            length >= min_window_len,
            (length - min_window_len) // self._window_stride + 1,
            0,
        ).astype(np.int64)
        self._cum_n_starts = np.concatenate([[0], np.cumsum(n_starts)]).astype(np.int64)
        self._n_total = int(self._cum_n_starts[-1])

        # ── prompts + per-dataset normalization stats (hooks) ─────────────
        self._load_prompts()
        self._normalization_stats = self._load_stats(info)

        # ── LRU cache for parquet shards ──────────────────────────────────
        self._load_data_table = functools.lru_cache(maxsize=32)(self._read_data_file_uncached)

        # ── subclass hook ─────────────────────────────────────────────────
        self._post_init(info)

        # ── unified per-dim validity mask (static) ────────────────────────
        # Honors the raw ACTION_DIM_MASK: single-arm OXE / RoboMIND zero-pad
        # the right-arm half of the 20-D EEF, which must stay masked out even
        # after the scatter — a unified slot is valid iff its source raw dim
        # was valid. Built HERE (not in the unify block above) because a
        # reader's instance ACTION_DIM_MASK may be set in _resolve_cameras
        # (RoboMIND sets it per-embodiment), which runs after that block.
        self._unify_dim_mask: Optional[np.ndarray] = None
        if self._unify:
            self._unify_dim_mask = np.zeros(self._unify_dim, dtype=bool)
            raw_mask = self.ACTION_DIM_MASK
            if raw_mask is None:
                self._unify_dim_mask[self._unify_dst_index] = True
            else:
                self._unify_dim_mask[self._unify_dst_index] = np.asarray(raw_mask, dtype=bool)

        logger.info(
            "%s(%s, %s): %d eps, %d windows, fps=%.1f, multiview=%s, normalize=%s, supervision=%s",
            self.DATASET_NAME,
            self._dataset_id,
            split,
            len(self._eps_df),
            self._n_total,
            self._fps,
            self._multiview,
            self._normalize_mode,
            self._enable_action_supervision,
        )

    # ----- hooks (defaults) -------------------------------------------------

    def _resolve_cameras(self, info: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Resolve (head, left_wrist, right_wrist) camera keys.

        Default: a single ``target_camera`` if set (EgoDex-style), else the
        class-level ``HEAD_CAMERA`` / ``LEFT_WRIST_CAMERA`` / ``RIGHT_WRIST_CAMERA``
        (OXE-style). RoboCOIN overrides to resolve by info.features priority.
        """
        if self._target_camera is not None:
            return (self._target_camera, None, None)
        return (self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA)

    def _add_data_offsets(self, eps: pd.DataFrame) -> None:
        """Set ``eps['_data_row_offset']`` (generic LeRobot v3 groupby-cumsum)."""
        eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")

    def _train_min_window_len(self) -> int:
        """Min episode length to yield a train window. 1 = any single labeled step."""
        return 1

    def _load_prompts(self) -> None:
        """Populate the prompt lookup according to ``PROMPT_SOURCE``."""
        if self.PROMPT_SOURCE == "episode_annotated":
            self._episode_idx_to_text = load_tasks_annotated(
                self._dataset_dir, source_name=f"{self.DATASET_NAME}({self._dataset_id})"
            )
            return
        # "task_index": meta/tasks.parquet, index=text, "task_index" column=int.
        tasks_path = self._dataset_dir / "meta" / "tasks.parquet"
        if not self.PROMPT_FILE_REQUIRED and not tasks_path.exists():
            self._task_idx_to_text: Dict[int, str] = {}
            return
        tasks_df = pd.read_parquet(tasks_path)  # FileNotFoundError if required + missing
        task_idx = tasks_df["task_index"].to_numpy()
        task_str = tasks_df.index.to_numpy()
        self._task_idx_to_text = dict(zip(task_idx.tolist(), task_str.tolist()))

    def _resolve_prompt(self, row, win: pd.DataFrame) -> str:
        """Resolve this window's prompt text according to ``PROMPT_SOURCE``.

        Raises on a missing entry / empty string (a data bug should not slip
        into training as a fabricated prompt).
        """
        if self.PROMPT_SOURCE == "episode_annotated":
            return resolve_prompt_by_episode(self._episode_idx_to_text, int(row["episode_index"]), self.DATASET_NAME)
        task_idx = int(win["task_index"].iloc[0])
        if task_idx not in self._task_idx_to_text:
            raise KeyError(
                f"{self.DATASET_NAME} prompt lookup failed: task_index={task_idx} not present "
                "in this bucket's tasks.parquet."
            )
        text = self._task_idx_to_text[task_idx].strip()
        if not text:
            raise ValueError(
                f"{self.DATASET_NAME} task_index={task_idx} maps to an empty prompt string. "
                "Indicates a tasks.parquet entry with blank text — data must be fixed upstream."
            )
        return text

    def _load_stats(self, info: dict) -> Optional[dict]:
        """Load ``meta/<STATS_FILENAME>`` stats when declared, else None.

        Default covers the OXE per-dataset ``eef_stats.json`` (flat dict).
        RoboCOIN overrides for its dynamic per-robot-type stats path.
        """
        if self.STATS_FILENAME is None:
            return None
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir / "meta" / self.STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but {stats_path} is missing. "
                f"Run the matching compute_stats script for {self.DATASET_NAME}, or set normalize_mode=null."
            )
        import json

        with open(stats_path) as f:
            raw = json.load(f)
        return materialize_eef_stats(
            raw,
            self._normalize_mode,
            dim=self.STATS_DIM,
            strict_minmax=self.STATS_STRICT_MINMAX,
            source_hint=str(stats_path),
        )

    def _action_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        """Return normalized ``(actual_raw_len, ACTION_DIM)`` action, or None (disabled)."""
        return None

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        """Return normalized ``(1, ACTION_DIM)`` proprio, or None (disabled)."""
        return None

    def _post_init(self, info: dict) -> None:
        """Optional one-time per-dataset setup (e.g. RT-1 quat sanity check)."""

    @classmethod
    def _multibucket_wrapper(cls):
        """Root-mode aggregate wrapper class, or None when only single-bucket is supported."""
        return None

    # ----- pickle -----------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_load_data_table", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._load_data_table = functools.lru_cache(maxsize=32)(self._read_data_file_uncached)

    # ----- offsets / IO -----------------------------------------------------

    def _add_episode_offsets(self, eps: pd.DataFrame) -> None:
        self._add_data_offsets(eps)
        for cam in (self._head_camera, self._left_wrist_camera, self._right_wrist_camera):
            if cam is None:
                continue
            chunk_col = f"videos/{cam}/chunk_index"
            file_col = f"videos/{cam}/file_index"
            if chunk_col in eps.columns and file_col in eps.columns:
                eps[self._video_offset_col(cam)] = compute_file_local_offsets(eps, chunk_col, file_col)

    @staticmethod
    def _video_offset_col(camera: str) -> str:
        return f"_video_frame_offset/{camera}"

    def _read_data_file_uncached(self, chunk_idx: int, file_idx: int):
        path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
        return pq.read_table(path, memory_map=True, columns=list(self.NEEDED_COLS))

    # ----- Dataset ----------------------------------------------------------

    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> dict:
        return self._safe_get(idx)

    def _safe_get(self, idx: int) -> dict:
        for attempt in range(_GETITEM_MAX_RETRIES):
            try:
                return self._getitem_impl(idx)
            except Exception as e:
                if attempt == _GETITEM_MAX_RETRIES - 1:
                    raise
                self._fail_count += 1
                if self._fail_count == 1 or self._fail_count % self._fail_log_every == 0:
                    logger.warning(
                        "%s(%s): %d cumulative __getitem__ failures (latest: idx=%d, %s, attempt=%d)",
                        self.DATASET_NAME,
                        self._dataset_id,
                        self._fail_count,
                        idx,
                        type(e).__name__,
                        attempt,
                    )
                idx = (idx + 1) % max(1, len(self))
        raise RuntimeError("unreachable")

    def _getitem_impl(self, idx: int) -> dict:
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        row = self._eps_df.iloc[ep_local]
        ep_len = int(row["length"])
        actual_raw_len = min(self._num_frames, ep_len - offset)

        # 1) parquet window rows
        local_start = int(self._ep_data_row_offset[ep_local]) + offset
        table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
        win = table.slice(local_start, actual_raw_len).to_pandas()

        # 2) prompt (hook)
        prompt = self._resolve_prompt(row, win)

        # 3) action + 4) proprio (hooks → normalized 20-D payload or None)
        action, action_mask = self._finalize_action(self._action_20d(win), actual_raw_len)
        proprio, proprio_mask = self._finalize_proprio(self._proprio_20d(win))

        # 5) video — decode head (+ optional wrist) frames into the canvas
        real_local_indices = self._video_sample_indices[self._video_sample_indices < actual_raw_len]
        video = self._decode_window_video(row, ep_local, offset, real_local_indices, idx)
        if self._color_jitter is not None:
            # Same jitter factors across the whole clip (temporal consistency).
            video = self._color_jitter.apply({"video": video})["video"]
        video_mask = torch.from_numpy(self._video_sample_indices < actual_raw_len)

        return {
            "video": video,
            "vace_video": None,
            "first_frame_image": [video[0]] if video else [],
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "video_mask": video_mask,
            "proprio": torch.from_numpy(proprio).float(),
            "proprio_mask": torch.from_numpy(proprio_mask),
            "prompt": prompt,
        }

    # ----- action / proprio finalization -----------------------------------

    def _finalize_action(self, action_20d: Optional[np.ndarray], actual_raw_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad an action payload to ``(T_action, ACTION_DIM)`` + build its 2-D mask.

        ``action_20d is None`` → all-zero action + all-False mask (video-only
        / disabled supervision). When unify is on, the raw ``_raw_action_dim``
        payload is scattered into ``unify_dim`` slots (mask follows the mapped
        dims) — this happens AFTER normalization (``action_20d`` is already
        normalized by ``_action_20d``).
        """
        T_action = self._num_frames - 1
        width = self._raw_action_dim  # fill at raw width first; unify-scatter below
        action = np.zeros((T_action, width), dtype=np.float32)
        n_valid = 0
        if action_20d is not None:
            n_valid = min(actual_raw_len, T_action)
            if n_valid > 0:
                action[:n_valid] = action_20d[:n_valid]

        if self._unify:
            # (T, raw) -> (T, unify_dim). The dim mask is the precomputed
            # ACTION_DIM_MASK-honoring one (map_to_unify's own mask would mark
            # every mapped slot valid, leaking single-arm right-arm padding).
            action, _ = map_to_unify(action, self._unify_dst_index, self._unify_dim)
            dim_mask = self._unify_dim_mask
        else:
            dim_mask = self.ACTION_DIM_MASK

        action_mask = build_action_mask_2d(
            T_action=T_action,
            action_dim=self.ACTION_DIM,
            n_valid_time=n_valid if self._enable_action_supervision else 0,
            dim_mask=dim_mask,
        )
        return action, action_mask

    def _finalize_proprio(self, proprio_20d: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Build ``(1, ACTION_DIM)`` proprio + its 2-D mask.

        ``proprio_20d is None`` → all-zero proprio + all-False mask. When unify
        is on, the raw proprio is scattered into ``unify_dim`` slots (same map
        as the action), after normalization.
        """
        if proprio_20d is None:
            proprio = np.zeros((1, self.ACTION_DIM), dtype=np.float32)
            mask = build_proprio_mask_2d(action_dim=self.ACTION_DIM, enabled=False)
            return proprio, mask

        proprio = np.asarray(proprio_20d, dtype=np.float32)
        if self._unify:
            # Same scatter + ACTION_DIM_MASK-honoring mask as _finalize_action.
            proprio, _ = map_to_unify(proprio, self._unify_dst_index, self._unify_dim)
            dim_mask = self._unify_dim_mask
        else:
            dim_mask = self.ACTION_DIM_MASK

        mask = build_proprio_mask_2d(
            action_dim=self.ACTION_DIM,
            enabled=self._enable_action_supervision,
            dim_mask=dim_mask,
        )
        return proprio, mask

    # ----- video ------------------------------------------------------------

    def _decode_window_video(self, row, ep_local: int, offset: int, real_local_indices: np.ndarray, idx: int) -> List:
        """Decode the window's frames into a list of PIL images (single-view) or
        L-shape multiview canvases, with last-real-frame padding.

        Head decode failure is fatal (raises → ``_safe_get`` retries). Wrist
        (auxiliary) decode failure is tolerated (black slot)."""
        head_frames = self._decode_one_camera(self._head_camera, row, ep_local, offset, real_local_indices, kind="head")
        if not head_frames:
            raise RuntimeError(f"empty head-video decode for {self.DATASET_NAME}({self._dataset_id}) at idx={idx}")

        if self._multiview:
            left_frames = (
                self._decode_one_camera(
                    self._left_wrist_camera, row, ep_local, offset, real_local_indices, kind="wrist"
                )
                if self._left_wrist_camera
                else []
            )
            right_frames = (
                self._decode_one_camera(
                    self._right_wrist_camera, row, ep_local, offset, real_local_indices, kind="wrist"
                )
                if self._right_wrist_camera
                else []
            )
        else:
            left_frames = []
            right_frames = []

        # Pad to num_video_frames using the last real head frame.
        n_real = len(head_frames)
        if n_real < self._num_video_frames:
            pad = self._num_video_frames - n_real
            head_frames = head_frames + [head_frames[-1]] * pad
            if left_frames:
                left_frames = left_frames + [left_frames[-1]] * pad
            if right_frames:
                right_frames = right_frames + [right_frames[-1]] * pad

        if not self._multiview:
            return head_frames

        video = []
        for fi in range(self._num_video_frames):
            frames_dict: Dict[str, Any] = {self._head_camera: head_frames[fi]}
            if left_frames and self._left_wrist_camera:
                frames_dict[self._left_wrist_camera] = left_frames[fi]
            if right_frames and self._right_wrist_camera:
                frames_dict[self._right_wrist_camera] = right_frames[fi]
            video.append(
                assemble_multiview_layout(frames_dict, self._camera_layout, out_h=self._height, out_w=self._width)
            )
        return video

    def _decode_one_camera(self, camera, row, ep_local, offset, real_local_indices, *, kind: str) -> List:
        """Decode one camera's frames. ``kind='head'`` is fatal-on-error; ``kind='wrist'``
        tolerates a missing / corrupt clip by returning ``[]`` (black slot)."""
        if camera is None:
            return []
        chunk_col = f"videos/{camera}/chunk_index"
        file_col = f"videos/{camera}/file_index"
        if chunk_col not in row.index:
            return []
        if camera not in self._ep_video_frame_offsets:
            return []

        if kind == "head":
            path = self._dataset_dir / self._video_path_template.format(
                video_key=camera, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])
            )
            v_base = int(self._ep_video_frame_offsets[camera][ep_local]) + offset
            frame_indices = (real_local_indices + v_base).tolist()
            h, w = (_HEAD_SLOT_H, _HEAD_SLOT_W) if self._multiview else (self._height, self._width)
            return _decode_video_frames(str(path), frame_indices, h, w)

        # wrist: tolerated decode — path/offset computation lives INSIDE the try so a
        # corrupt chunk/file index or offset lookup degrades to a black slot rather
        # than failing the whole sample (matches the pre-refactor robocoin behavior).
        try:
            path = self._dataset_dir / self._video_path_template.format(
                video_key=camera, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])
            )
            v_base = int(self._ep_video_frame_offsets[camera][ep_local]) + offset
            frame_indices = (real_local_indices + v_base).tolist()
            return _decode_video_frames(str(path), frame_indices, _WRIST_SLOT_H, _WRIST_SLOT_W)
        except self.WRIST_DECODE_TOLERATED as e:
            self._wrist_fail_count += 1
            if self._wrist_fail_count == 1 or self._wrist_fail_count % self._fail_log_every == 0:
                logger.warning(
                    "%s(%s): %d cumulative wrist decode failures (latest: %s, camera=%s)",
                    self.DATASET_NAME,
                    self._dataset_id,
                    self._wrist_fail_count,
                    type(e).__name__,
                    camera,
                )
            return []

    # ----- BaseDataset ------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        # Samples leave the reader pre-normalized; downstream layers see None.
        return None

    # ----- from_config ------------------------------------------------------

    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        "num_frames",
        "video_stride",
        "window_stride",
        "height",
        "width",
        "multiview",
        "normalize_mode",
        "enable_action_supervision",
        "target_camera",
        "camera_layout",
        "unify_action",
        "unify_action_map",
        "color_jitter",
    )

    @classmethod
    def from_config(cls, config, split: str = "train") -> "BaseDataset":
        from openwam.dataloader.utils import get_cfg as _get

        dataset_dir = _get(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError(f"{cls.__name__}: missing dataset_dir")
        root = Path(dataset_dir)

        common: Dict[str, Any] = {"split": split}
        for key in cls.CONFIG_KEYS:
            v = _get(config, key, _CONFIG_MISSING)
            if v is _CONFIG_MISSING:
                continue
            # ``normalize_mode`` uses key-presence semantics: an explicit
            # ``null`` in the config must reach the ctor as None (disable
            # normalization) instead of being dropped — otherwise the ctor's
            # _NORMALIZE_MODE_UNSET sentinel falls back to DEFAULT_NORMALIZE_MODE
            # (e.g. OXE "quantile"). Other keys keep treating None as "unset".
            if v is None and key != "normalize_mode":
                continue
            common[key] = v

        # Single-bucket mode.
        if (root / "meta" / "info.json").is_file():
            kwargs = dict(common)
            dataset_id = _get(config, "dataset_id")
            if dataset_id is not None:
                kwargs["dataset_id"] = dataset_id
            # Honor total_hours in single-bucket mode too (subsample this bucket
            # to the budget) — otherwise a yaml-level total_hours would be
            # silently ignored when dataset_dir points at one bucket.
            total_hours = _get(config, "total_hours")
            if total_hours is not None:
                kwargs["max_hours"] = float(total_hours)
                kwargs["subsample_seed"] = int(_get(config, "seed", 42))
            return cls(dataset_dir=str(root), **kwargs)

        # Root mode (only when the reader declares a multi-bucket wrapper).
        wrapper = cls._multibucket_wrapper()
        if wrapper is None:
            raise FileNotFoundError(f"{cls.__name__}: {root}/meta/info.json missing")
        if not root.is_dir():
            raise FileNotFoundError(f"{cls.__name__}: {root} does not exist")
        sub_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "meta" / "info.json").is_file())
        if not sub_dirs:
            raise FileNotFoundError(f"{cls.__name__}: no sub-buckets with meta/info.json under {root}")
        logger.info("%s.from_config: root mode, %d buckets under %s", cls.__name__, len(sub_dirs), root)
        base_seed = int(_get(config, "seed", 42))
        total_hours = _get(config, "total_hours")
        return build_multibucket(
            cls,
            sub_dirs,
            common,
            base_seed=base_seed,
            total_hours=total_hours,
            wrapper_cls=wrapper,
            source_name=cls.__name__,
        )


__all__ = ["LeRobotV3Reader"]
