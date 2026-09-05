"""RoboCasa GR1 LeRobot v3 dataloader.

The reader accepts the FK-enriched 33-D EEF representation only::

    [L xyz3, L rot6d6, L hand6, R xyz3, R rot6d6, R hand6, waist3]

``unify_action`` scatters those physical dimensions into the shared 80-D
EEF/dex-hand/reserved layout. Native 44-D joint vectors are deliberately
rejected: they must first pass through the simulator-backed FK enrichment.

Normalization statistics live at ONE fixed, config-free location — the
training root's ``meta/robocasa_gr1_normalization_stats.npy`` — and are auto-built there on
first use (see :meth:`RoboCasaGR1Dataset.from_config`). There is deliberately
no ``normalization_stats_path`` config knob: a GR1 root holds ~25 task buckets,
each constructed as its own reader, so a per-bucket path would give every task
its own transform instead of the pooled one the deploy denormalizer assumes.
Pose/waist dimensions share pooled action/state statistics. The 12 hand
dimensions are directional because action stores discrete Fourier-hand
commands while state stores continuous joint angles.
"""

from __future__ import annotations

import logging
import os
import pickle
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

# Fixed basename of the directional EEF33 statistics, resolved against the
# training root's meta/ dir. This same file is copied into checkpoints for
# deployment because it carries both `eef` action and `eef_state` proprio
# blocks; an action-only per-bucket rewrite would lose train/deploy parity.
NORMALIZATION_STATS_FILENAME = "robocasa_gr1_normalization_stats.npy"
STATS_SCHEMA_VERSION = 2

# Sentinel distinguishing "key absent" from an explicit null in a config.
_CONFIG_UNSET = object()

# Stats-file triage used by from_config auto-rebuild vs fail-fast paths.
_STATS_COMPATIBLE = "compatible"
_STATS_REBUILDABLE = "rebuildable"


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


def _stats_block_is_compatible(block: Any) -> bool:
    """True when ``block`` carries finite ``STAT_KEYS`` vectors of shape ``(33,)``."""
    if not isinstance(block, dict):
        return False
    for key in STAT_KEYS:
        if key not in block:
            return False
        try:
            arr = np.asarray(block[key])
            is_real_numeric = np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.complexfloating)
            finite = bool(np.isfinite(arr).all()) if is_real_numeric else False
        except (TypeError, ValueError):
            return False
        if arr.shape != (EEF33_DIM,) or not is_real_numeric or not finite:
            return False
    return True


def _classify_stats_file(path: Path) -> str:
    """Classify a GR1 stats artifact as compatible or rebuildable.

    Raises:
        ValueError: schema version newer than this code understands. Those files
            must not be overwritten with schema-v2 by auto-rebuild.
    """
    if not path.is_file():
        return _STATS_REBUILDABLE
    try:
        payload = np.load(path, allow_pickle=True).item()
    except (OSError, ValueError, EOFError, pickle.UnpicklingError):
        return _STATS_REBUILDABLE
    if not isinstance(payload, dict):
        return _STATS_REBUILDABLE

    schema = payload.get("robocasa_gr1_stats_schema")
    if schema is None:
        # Pre-schema pooled-hand artifacts are eligible for directional rebuild.
        return _STATS_REBUILDABLE
    if not isinstance(schema, (int, np.integer)) or isinstance(schema, bool):
        return _STATS_REBUILDABLE
    schema_i = int(schema)
    if schema_i > STATS_SCHEMA_VERSION:
        raise ValueError(
            f"RoboCasaGR1 normalization stats at {path} use unsupported schema "
            f"version {schema_i} (this code understands {STATS_SCHEMA_VERSION}). "
            "Upgrade OpenWAM rather than overwriting the artifact with an older schema."
        )
    if schema_i < STATS_SCHEMA_VERSION:
        return _STATS_REBUILDABLE
    if _stats_block_is_compatible(payload.get(_ACTION_MODE)) and _stats_block_is_compatible(
        payload.get(f"{_ACTION_MODE}_state")
    ):
        return _STATS_COMPATIBLE
    # Schema claims current version but vectors are missing/wrong-shaped/non-finite.
    return _STATS_REBUILDABLE


def _stats_file_is_compatible(path: Path) -> bool:
    """Return whether ``path`` carries valid directional GR1 hand statistics.

    Missing, unreadable, legacy, or malformed current-schema files return
    ``False`` so ``from_config`` can rebuild them. Unsupported newer schema
    versions raise ``ValueError`` instead of being treated as rebuildable.
    """
    return _classify_stats_file(path) == _STATS_COMPATIBLE


def _wait_for_stats(path: Path) -> None:
    deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
    poll_interval = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
    while not _stats_file_is_compatible(path):
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

    # Fixed camera layout (head, left wrist, right wrist); None slots are
    # rendered black in multiview mode. Override via ``camera_layout``.
    DEFAULT_CAMERA_LAYOUT: ClassVar[Tuple[Optional[str], ...]] = (
        "observation.images.ego_view",
        None,
        None,
    )

    CONFIG_KEYS: ClassVar[Tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
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
        self._state_normalization_stats: Optional[dict] = None

        self._action_column = "eef_action"
        self._state_column = "observation.eef_state"

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

        self.NEEDED_COLS = (self._action_column, self._state_column, "task_index")

        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=unify_on,
            unify_action_map=unify_action_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict):
        if self._target_camera is not None:
            return (self._target_camera, None, None)
        layout = list(self._camera_layout_param or self.DEFAULT_CAMERA_LAYOUT)
        layout += [None] * (3 - len(layout))
        return tuple(str(cam) if cam else None for cam in layout[:3])

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
        if not _stats_file_is_compatible(stats_path):
            raise ValueError(
                f"RoboCasaGR1 normalization stats at {stats_path} are legacy or malformed "
                f"(need schema={STATS_SCHEMA_VERSION} with finite {STAT_KEYS} vectors of shape "
                f"({EEF33_DIM},) under both '{_ACTION_MODE}' and '{_ACTION_MODE}_state'). "
                "GR1 hand action is a discrete command while hand state is a continuous joint angle; "
                "delete/rebuild this file with robocasa_gr1_stats_computation before training or deployment."
            )
        self._resolved_stats_path = str(stats_path)
        action_stats = load_stats_file(
            stats_path,
            action_mode=self.action_mode,
            normalize_mode=self._normalize_mode,
            dim=self._raw_action_dim,
        )
        self._state_normalization_stats = load_stats_file(
            stats_path,
            action_mode=f"{self.action_mode}_state",
            normalize_mode=self._normalize_mode,
            dim=self._raw_action_dim,
        )
        # The source file is already the deploy artifact: it carries both the
        # action block used by unnormalize() and the state block used by
        # normalize(). Point checkpoint saving at it instead of rewriting a
        # per-bucket action-only artifact.
        self.normalization_stats_path = str(stats_path)
        return action_stats

    def _normalize_array(self, arr: np.ndarray, stats: Optional[dict] = None) -> np.ndarray:
        return apply_normalization(
            arr,
            self._normalization_stats if stats is None else stats,
            self._normalize_mode,
        )

    def _action_20d(self, win) -> np.ndarray:
        raw = self._raw_action(win)
        return self._normalize_array(raw)

    def _proprio_20d(self, win) -> np.ndarray:
        raw = self._read_vector_window(win, vector_col=self._state_column, label="state", first_only=True)
        return self._normalize_array(raw, self._state_normalization_stats)

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

        The path is fixed at ``<dataset_dir>/meta/robocasa_gr1_normalization_stats.npy`` and
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
        if not _stats_file_is_compatible(stats_path):
            cls._build_shared_stats(config, stats_path)
        return super().from_config(_config_with(config, normalization_stats_path=str(stats_path)), split)

    @classmethod
    def _build_shared_stats(cls, config, path: Path) -> None:
        """Rank 0 builds directional hand stats into ``path``; other ranks wait."""
        # Lazy import: the stats module imports this reader at module level.
        from openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation import (
            build_and_save_robocasa_gr1_stats,
        )

        if _stats_builder_rank() != 0:
            _wait_for_stats(path)
            return

        logger.info(
            "RoboCasaGR1: missing/incompatible directional stats at %s — computing them from the dataset "
            "(rank 0 scans; other ranks wait)",
            path,
        )
        # Always pooled from the TRAIN split, whatever split was requested: a
        # val reader must normalize with the transform training uses.
        # normalize_mode=None keeps this probe out of the resolution above.
        probe = super().from_config(_config_with(config, normalize_mode=None), "train")
        action_mode, raw_dim, action_rows, state_rows = build_and_save_robocasa_gr1_stats(probe, path)
        logger.info(
            "RoboCasaGR1: wrote %s mode=%s pool=action_state_except_hand dim=%d action_rows=%d state_rows=%d",
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
        # bucket must normalize with the same directional action/state file.
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
    "STATS_SCHEMA_VERSION",
    "RoboCasaGR1Dataset",
    "MultiRoboCasaGR1Dataset",
]
