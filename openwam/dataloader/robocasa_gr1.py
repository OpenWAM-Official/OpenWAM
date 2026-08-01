"""RoboCasa GR1 LeRobot v3 dataloader.

The reader accepts the FK-enriched 33-D EEF representation only::

    [L xyz3, L rot6d6, L hand6, R xyz3, R rot6d6, R hand6, waist3]

``unify_action`` scatters those physical dimensions into the shared 80-D
EEF/dex-hand/reserved layout. Native 44-D joint vectors are deliberately
rejected: they must first pass through the simulator-backed FK enrichment.

Normalization statistics live at ONE fixed, config-free location — the
training root's ``meta/normalization_stats.npy`` — and are auto-built there on
first use (see :meth:`RoboCasaGR1Dataset.from_config`). There is deliberately
no ``normalization_stats_path`` config knob: a GR1 root holds ~25 task buckets,
each constructed as its own reader, so a per-bucket path would give every task
its own transform instead of the pooled one the deploy denormalizer assumes.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.normalization import STAT_KEYS, apply_normalization, load_stats_file

logger = logging.getLogger(__name__)

_ACTION_MODE = "eef"
EEF33_DIM = 33

# Fixed basename of the pooled EEF33 statistics, resolved against the training
# root's meta/ dir. Same basename as the deploy denormalizer artifact
# (LeRobotV3Reader._write_deploy_normalizer_stats), which is written per BUCKET
# — in root mode those are different files (root/meta vs root/<bucket>/meta).
# When dataset_dir points straight at a single bucket the two alias: the reader
# then rewrites the file with just the six stat vectors (identical values, minus
# the stats script's provenance fields), which is stable across reruns.
NORMALIZATION_STATS_FILENAME = "normalization_stats.npy"

# Sentinel distinguishing "key absent" from an explicit null in a config.
_CONFIG_UNSET = object()


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        pass
    return [value]


def _as_bool_mask(value: Any, dim: int, *, field: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.shape != (dim,):
        raise ValueError(f"{field} must be a boolean list of length {dim}, got shape {arr.shape}")
    return arr


def _pick_feature(features: dict, priorities: Sequence[str]) -> Optional[str]:
    for key in priorities:
        if key and key in features:
            return key
    return None


def _config_with(config: Any, **overrides: Any) -> Dict[str, Any]:
    """Shallow plain-dict copy of ``config`` with ``overrides`` applied.

    Values are forwarded untouched (a DictConfig's nested nodes stay nodes),
    exactly as ``LeRobotV3Reader.from_config`` already forwards them to the
    reader ctor. Returning a plain dict — rather than mutating the caller's
    config — keeps the injection out of Hydra's struct mode; ``get_cfg`` reads
    dicts and DictConfigs alike.
    """
    if hasattr(config, "keys"):
        plain = {key: config[key] for key in config.keys()}
    else:
        plain = dict(vars(config))
    plain.update(overrides)
    return plain


def _stats_builder_rank() -> int:
    """Rank that owns the stats scan (0 builds, the others wait)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    # torchrun sets RANK before init_process_group; honor it so pre-init
    # constructions still elect a single builder.
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def _wait_for_stats(path: Path) -> None:
    deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
    poll_interval = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for rank 0 to build RoboCasaGR1 normalization stats: {path}")
        time.sleep(poll_interval)


class RoboCasaGR1Dataset(LeRobotV3Reader):
    """Single-bucket RoboCasa GR1 reader for LeRobot v3 datasets."""

    DATASET_NAME = "RoboCasaGR1"
    PROMPT_FILE_REQUIRED = False
    # Bounded [-1, 1] targets by default; a config may still set z-score /
    # quantile, or null to disable in-reader normalization entirely.
    DEFAULT_NORMALIZE_MODE = "min-max"

    HEAD_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.ego_view",
        "observation.images.ego_view_rgb",
        "observation.images.ego_rgb",
        "observation.images.head_rgb",
        "observation.images.cam_head_rgb",
        "observation.images.cam_high_rgb",
        "video.ego_view_pad_res256_freq20",
    )
    LEFT_WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.left_wrist",
        "observation.images.left_wrist_rgb",
        "observation.images.cam_left_wrist_rgb",
    )
    RIGHT_WRIST_CAMERA_PRIORITY: ClassVar[Tuple[str, ...]] = (
        "observation.images.right_wrist",
        "observation.images.right_wrist_rgb",
        "observation.images.cam_right_wrist_rgb",
    )

    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "eef_action_column",
        "eef_state_column",
        "prompt_columns",
        "head_camera_priority",
        "left_wrist_camera_priority",
        "right_wrist_camera_priority",
        # NOT a user-facing knob: from_config always overwrites this with the
        # single path it resolved from dataset_dir, so a value left over in a
        # yaml / CLI override is ignored rather than splitting the buckets
        # across different transforms. It stays in CONFIG_KEYS because that is
        # the channel the base from_config uses to hand kwargs to every bucket.
        "normalization_stats_path",
        "unify_action",
        "unify_action_map",
        "state_mask",
        "action_mask",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = "eef",
        eef_action_column: str = "eef33_action",
        eef_state_column: str = "eef33_state",
        prompt_columns: Optional[Sequence[str]] = None,
        head_camera_priority: Optional[Sequence[str]] = None,
        left_wrist_camera_priority: Optional[Sequence[str]] = None,
        right_wrist_camera_priority: Optional[Sequence[str]] = None,
        normalization_stats_path: Optional[str] = None,
        unify_action: Optional[bool] = None,
        unify_action_map: Optional[Any] = None,
        action_mask: Optional[Sequence[bool]] = None,
        state_mask: Optional[Sequence[bool]] = None,
        **kwargs: Any,
    ):
        mode = str(action_mode).strip().lower()
        if mode != _ACTION_MODE:
            raise ValueError(
                f"RoboCasaGR1 currently supports only action_mode='eef', got {action_mode!r}. "
                "EEF is raw 33-D: [L xyz3+rot6d6+hand6, R xyz3+rot6d6+hand6, waist3]."
            )
        unify_on = bool(unify_action)
        if unify_on and unify_action_map is None:
            raise ValueError(
                "RoboCasaGR1 unify_action=true requires an explicit unify_action_map; "
                "set ['0-8', '10-15', '34-42', '44-49', '68-70'] for canonical EEF33 mapping"
            )
        self.action_mode = mode
        self.DEPLOY_ACTION_MODE = _ACTION_MODE
        # Set by from_config (the shared, root-level stats file). A directly
        # constructed reader may pass one; otherwise _load_stats falls back to
        # this bucket's own meta/ dir.
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._resolved_stats_path: Optional[str] = None  # set by _load_stats when normalization is on

        self._prompt_columns = [str(x) for x in _as_list(prompt_columns)]
        self._head_priority = tuple(str(x) for x in (head_camera_priority or self.HEAD_CAMERA_PRIORITY))
        self._left_wrist_priority = tuple(
            str(x) for x in (left_wrist_camera_priority or self.LEFT_WRIST_CAMERA_PRIORITY)
        )
        self._right_wrist_priority = tuple(
            str(x) for x in (right_wrist_camera_priority or self.RIGHT_WRIST_CAMERA_PRIORITY)
        )

        self._action_column = str(eef_action_column)
        self._state_column = str(eef_state_column)

        self.ACTION_DIM = EEF33_DIM
        action_dim_mask = _as_bool_mask(action_mask, self.ACTION_DIM, field="action_mask")
        state_dim_mask = _as_bool_mask(state_mask, self.ACTION_DIM, field="state_mask")
        if (
            action_dim_mask is not None
            and state_dim_mask is not None
            and not np.array_equal(action_dim_mask, state_dim_mask)
        ):
            raise ValueError("RoboCasaGR1 action_mask and state_mask must match; the shared reader uses one raw mask")
        self.ACTION_DIM_MASK = action_dim_mask if action_dim_mask is not None else state_dim_mask

        cols: List[str] = []
        for col in (
            self._action_column,
            self._state_column,
            *self._prompt_columns,
        ):
            if col:
                cols.append(str(col))
        cols.append("task_index")
        self.NEEDED_COLS = tuple(dict.fromkeys(cols))

        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=unify_on,
            unify_action_map=unify_action_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict):
        features = info.get("features", {}) or {}
        if self._target_camera is not None:
            return (self._target_camera, None, None)
        return (
            _pick_feature(features, self._head_priority),
            _pick_feature(features, self._left_wrist_priority),
            _pick_feature(features, self._right_wrist_priority),
        )

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        expected_size = (384, 320) if self._multiview else (256, 320)
        if (self._height, self._width) != expected_size:
            mode = "multiview" if self._multiview else "single-view"
            raise ValueError(
                f"RoboCasaGR1 {mode} requires height={expected_size[0]}, width={expected_size[1]}, "
                f"got height={self._height}, width={self._width}"
            )
        required = [
            self._action_column,
            self._state_column,
            "task_index",
        ]
        missing_required = sorted(col for col in required if col and col not in features)
        if missing_required:
            raise KeyError(
                f"RoboCasaGR1 action_mode={self.action_mode!r} requires columns absent from info.features: "
                f"{missing_required}. Generate a reliable EEF-enriched conversion first; "
                "the public NVIDIA GR1 joint44 columns must not be relabeled as EEF."
            )
        for column, expected_dim in (
            (self._action_column, EEF33_DIM),
            (self._state_column, EEF33_DIM),
        ):
            if not column:
                continue
            shape = tuple(features[column].get("shape", ()))
            if shape and shape != (expected_dim,):
                raise ValueError(
                    f"RoboCasaGR1 feature {column!r} must have shape [{expected_dim}] for "
                    f"action_mode={self.action_mode!r}, got {shape}"
                )

        configured = set(self._prompt_columns)
        self._prompt_columns = [col for col in self._prompt_columns if col in features]
        missing = configured.difference(self._prompt_columns)
        if missing:
            logger.info(
                "RoboCasaGR1(%s): ignoring prompt columns absent from info.features: %s",
                self._dataset_id,
                sorted(missing),
            )
        self.NEEDED_COLS = tuple(
            col for col in self.NEEDED_COLS if col not in configured or col in self._prompt_columns
        )

    def _resolve_prompt(self, row, win) -> str:
        for col in self._prompt_columns:
            if col in win:
                value = win[col].iloc[0]
                # The public GR1 data's annotation.human.coarse_action is an
                # integer class ID (e.g. 6), not language. Never stringify a
                # categorical ID into a bogus training prompt.
                if isinstance(value, str):
                    text = str(value).strip()
                    if text:
                        return text
        return super()._resolve_prompt(row, win)

    def _load_stats(self, info: dict):
        if not self._normalize_mode or self._normalize_mode in (None, "none", "null"):
            return None
        # from_config hands every bucket the ONE pooled file it resolved from
        # dataset_dir; a directly constructed reader falls back to its own
        # meta/ dir. A missing file is fatal HERE — the auto-build lives in
        # from_config, the only place that knows the training root.
        stats_path = (
            Path(self._source_stats_path)
            if self._source_stats_path
            else self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
        )
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"RoboCasaGR1({self._dataset_id}): normalize_mode={self._normalize_mode!r} but "
                f"{stats_path} is missing. Build it with\n"
                "  python -m openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation "
                f"--config configs/dataloader/robocasa_gr1.yaml --output {stats_path}\n"
                "(construction through from_config builds it automatically), or set normalize_mode=null."
            )
        self._resolved_stats_path = str(stats_path)
        global_stats = load_stats_file(
            stats_path,
            action_mode=self.action_mode,
            normalize_mode=self._normalize_mode,
            dim=self._raw_action_dim,
        )
        # Emit the deploy denormalizer artifact into THIS bucket's meta/. In
        # single-bucket mode that is the file just read: it comes back with the
        # same six stat vectors, so the reread is stable (see
        # NORMALIZATION_STATS_FILENAME).
        self._write_deploy_normalizer_stats(global_stats, STAT_KEYS)
        return global_stats

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win) -> np.ndarray:
        raw = self._raw_action(win)
        return self._normalize_array(raw)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = self._read_vector_window(win, vector_col=self._state_column, label="state", first_only=True)
        return self._normalize_array(raw)

    def _raw_action(self, win) -> np.ndarray:
        return self._read_vector_window(win, vector_col=self._action_column, label="action")

    def _raw_state(self, win) -> np.ndarray:
        return self._read_vector_window(win, vector_col=self._state_column, label="state")

    def _read_vector_window(
        self,
        win,
        *,
        vector_col: str,
        label: str,
        first_only: bool = False,
    ) -> np.ndarray:
        values = slice(0, 1) if first_only else slice(None)
        if vector_col not in win:
            raise KeyError(f"RoboCasaGR1 {label} column {vector_col!r} not found in parquet window")
        result = np.stack(win[vector_col].values[values]).astype(np.float32)
        if result.ndim != 2 or result.shape[-1] != EEF33_DIM:
            raise ValueError(f"RoboCasaGR1 {label} must be (T, {EEF33_DIM}), got {result.shape}")
        return result

    @classmethod
    def from_config(cls, config, split: str = "train"):
        """Resolve the ONE pooled stats file, then build the reader(s).

        The path is fixed at ``<dataset_dir>/meta/normalization_stats.npy`` and
        is NOT configurable: a GR1 root fans out into ~25 task buckets, each its
        own reader, so resolution has to happen here (where the root is known)
        rather than per bucket. Missing file → rank 0 scans the dataset and
        writes it; the other ranks poll. ``dist.barrier()`` is deliberately
        avoided — a minutes-long scan would trip NCCL's collective timeout (the
        ebench / libero precedent).
        """
        normalize_mode = get_cfg(config, "normalize_mode", _CONFIG_UNSET)
        if normalize_mode is _CONFIG_UNSET:
            normalize_mode = cls.DEFAULT_NORMALIZE_MODE
        if not normalize_mode or str(normalize_mode).strip().lower() in ("none", "null"):
            # Normalization off: drop any stale path so it cannot reach a bucket.
            return super().from_config(_config_with(config, normalization_stats_path=None), split)

        dataset_dir = get_cfg(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError(f"{cls.__name__}: missing dataset_dir")
        stats_path = Path(dataset_dir) / "meta" / NORMALIZATION_STATS_FILENAME
        if not stats_path.is_file():
            cls._build_shared_stats(config, stats_path)
        return super().from_config(_config_with(config, normalization_stats_path=str(stats_path)), split)

    @classmethod
    def _build_shared_stats(cls, config, path: Path) -> None:
        """Rank 0 pools EEF33 action+state rows into ``path``; other ranks wait."""
        # Lazy import: the stats module imports this reader at module level.
        from openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation import (
            build_and_save_robocasa_gr1_stats,
        )

        if _stats_builder_rank() != 0:
            _wait_for_stats(path)
            return

        logger.info(
            "RoboCasaGR1: no normalization stats at %s — computing them from the dataset "
            "(rank 0 scans; other ranks wait)",
            path,
        )
        # Always pooled from the TRAIN split, whatever split was requested: a
        # val reader must normalize with the transform training uses.
        # normalize_mode=None keeps this probe out of the resolution above.
        probe = super().from_config(_config_with(config, normalize_mode=None), "train")
        action_mode, raw_dim, action_rows, state_rows = build_and_save_robocasa_gr1_stats(probe, path)
        logger.info(
            "RoboCasaGR1: wrote %s mode=%s pool=action_state dim=%d action_rows=%d state_rows=%d",
            path,
            action_mode,
            raw_dim,
            action_rows,
            state_rows,
        )

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiRoboCasaGR1Dataset


class MultiRoboCasaGR1Dataset(MultiLeRobotV3Reader):
    """Aggregate multiple RoboCasa GR1 LeRobot v3 buckets."""

    def __init__(self, buckets: List[RoboCasaGR1Dataset]):
        super().__init__(buckets)
        dims = {int(b.action_dim) for b in self._buckets}
        modes = {b.action_mode for b in self._buckets}
        if len(dims) != 1:
            raise ValueError(f"MultiRoboCasaGR1Dataset requires homogeneous action_dim, got {sorted(dims)}")
        if len(modes) != 1:
            raise ValueError(f"MultiRoboCasaGR1Dataset requires homogeneous action_mode, got {sorted(modes)}")
        # Compare the RESOLVED paths (None when normalization is off): every
        # bucket must normalize with the SAME pooled file, otherwise the deploy
        # denormalizer — which carries a single bucket's stats — would
        # un-normalize the other tasks with the wrong transform.
        stats_paths = {b._resolved_stats_path for b in self._buckets}
        if len(stats_paths) != 1:
            raise ValueError(
                "MultiRoboCasaGR1Dataset requires one shared normalization stats file for all buckets "
                f"(from_config resolves <dataset_dir>/meta/{NORMALIZATION_STATS_FILENAME}); buckets resolved "
                f"{sorted(str(p) for p in stats_paths)}"
            )
        logger.info(
            "MultiRoboCasaGR1Dataset: %d buckets, %d windows, action_mode=%s, action_dim=%d",
            len(self._buckets),
            len(self),
            next(iter(modes)),
            next(iter(dims)),
        )

    @property
    def normalization_stats_path(self) -> Optional[str]:
        """Deploy artifact generated by the first homogeneous bucket."""
        return self._buckets[0].normalization_stats_path

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return RoboCasaGR1Dataset.from_config(config, split)


__all__ = [
    "EEF33_DIM",
    "NORMALIZATION_STATS_FILENAME",
    "RoboCasaGR1Dataset",
    "MultiRoboCasaGR1Dataset",
]
