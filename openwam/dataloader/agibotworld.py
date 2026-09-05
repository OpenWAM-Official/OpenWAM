"""AgiBotWorld-Beta reader for LeRobot v3 buckets.

The source exposes bimanual terminal-arm poses in the robot-base frame as
``[L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]``.  Grippered buckets insert
``[L_grip, R_grip]`` to form the canonical 20-D EEF layout; dexterous-hand
buckets instead assemble a 30-D pose-and-finger vector before scattering it
into the shared 80-D action space.  The pose endpoint is the rigid arm-side
attachment (flange-side), not a universal task TCP.

Action grippers use ``0=open, 1=closed`` and are inverted at the reader
boundary.  State grippers are converted from closing-actuator position to the
same ``0=closed, 1=open`` aperture convention.  Mobile-base x/yaw values map
to unified slots 68/70; the unsupported lateral slot remains masked.

Actions are already next-state relabeled, so rows are consumed without an
additional temporal shift.  The clamped terminal action of an episode is not
supervised.  Optional normalization uses separate action and proprio stats,
with rot6d dimensions pinned to identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.robocoin import GRIP_EXCLUDED_DIM_MASK, _build_dex_unify_map
from openwam.dataloader.utils.lerobotv3 import (
    DataContractError,
    digest_lerobot_v3_data_population,
    resolve_lerobot_v3_data_population,
)
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_EEF20,
    apply_normalization,
    materialize_eef_stats,
    pin_rot6d_identity,
)

logger = logging.getLogger(__name__)

# Fixed camera columns shared by every supported bucket.
HEAD_CAMERA = "observation.images.head"
LEFT_WRIST_CAMERA = "observation.images.hand_left"
RIGHT_WRIST_CAMERA = "observation.images.hand_right"


_DEX_PER_HAND = 6

# Native widths before optional mobile-base values are appended.
_EEF_RAW_DIM = 20
_DEX_RAW_DIM = 18 + 2 * _DEX_PER_HAND


# Differential drive publishes [forward, lateral, yaw]; lateral is unsupported.
_MOVE_SRC_DIMS = (0, 2)
_MOVE_DIM = len(_MOVE_SRC_DIMS)
_MOVE_SLOTS = (68, 70)  # shared 80-D forward/yaw slots
_MOVE_EPS = 1e-6


_GRIPPER_STATE_OPEN_POSITION_M = 0.035
_GRIPPER_STATE_CLOSED_POSITION_M = 0.125
# Persist the physical direction and state calibration with generated stats.
_GRIPPER_CONTRACT = {
    "schema_version": 1,
    "raw_action_semantics": "0_open_1_closed_command",
    "action_transform": "1-x",
    "raw_state_semantics": "closing_actuator_position_m",
    "state_transform": "clip((closed_position_m-x)/(closed_position_m-open_position_m),0,1)",
    "open_endpoint_m": _GRIPPER_STATE_OPEN_POSITION_M,
    "closed_endpoint_m": _GRIPPER_STATE_CLOSED_POSITION_M,
    "output_semantics": "0_closed_1_open_aperture_fraction",
}
_STATS_SCHEMA_VERSION = 3


# ``info.features`` declares both signal families, so bucket flavor is explicit.
_DEX_BUCKET_IDS = frozenset(
    {
        "475",
        "536",
        "549",
        "554",
        "577",
        "578",
        "595",
        "608",
        "620",
        "622",
        "660",
        "679",
        "705",
        "710",
        "727",
        "730",
        "731",
        "749",
        "753",
    }
)


_GRIP_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
    "action.gripper",
    "observation.state.gripper",
)
_DEX_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
    "action.dex",
    "observation.state.dex",
)

_POSE_ONLY_COLS = (
    "task_index",
    "action.ee_base",
    "observation.state.ee_base",
)

_MOVE_COLS = ("action.robot_velocity", "observation.state.robot_velocity")


# Dex raw layout: L pose9 + L fingers6 + R pose9 + R fingers6.
_ROT6D_DIMS_DEX30 = (3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23)

_STAT_FIELDS = ("min", "max", "mean", "std", "q01", "q99")


_UNIFIED_STATS_FILENAME = "stats_g2a.json"


_BUCKET_STATS_FILENAME = "stats.json"

_SEGMENT_FLAG_COL = "segment_flag"
_SEGMENT_DELTA_COL = "segment_delta"


def _validate_trim_ratio(value) -> float | None:
    """Normalize ``segment_max_trim_ratio`` to a float in (0, 1] or None.

    Raises rather than clamping: a config typo like ``70`` (percent instead of a
    fraction) would silently disable the filter, and ``0`` would drop every
    episode including the untrimmed ones.
    """

    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"segment_max_trim_ratio must be a float in (0, 1] or null; got {value!r}") from e
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"segment_max_trim_ratio must be a fraction in (0, 1] or null; got {ratio!r}. "
            "Use 0.7 for '70% or more of the episode trimmed', not 70."
        )
    return ratio


def _apply_segment_annotations(
    eps_df: pd.DataFrame,
    *,
    use_segment_annotations: bool,
    segment_max_trim_ratio: float | None,
) -> tuple[pd.DataFrame, dict]:
    """Return the exact episode population admitted by the segment cleanup.

    This is deliberately shared by the training reader and the normalization-
    stats producer.  Keeping the policy in one function prevents a stats rerun
    from accidentally pooling the leading/trailing frames that the reader can
    never serve.
    """

    ratio = _validate_trim_ratio(segment_max_trim_ratio)
    summary = {
        "missing_columns": False,
        "dropped_static": 0,
        "dropped_empty": 0,
        "dropped_over_trim": 0,
    }
    if not use_segment_annotations:
        return eps_df.reset_index(drop=True), summary
    if _SEGMENT_FLAG_COL not in eps_df.columns or _SEGMENT_DELTA_COL not in eps_df.columns:
        summary["missing_columns"] = True
        return eps_df.reset_index(drop=True), summary

    out = eps_df.copy()
    flag = out[_SEGMENT_FLAG_COL].fillna(0).astype(np.int64).to_numpy()
    delta = np.maximum(0, out[_SEGMENT_DELTA_COL].fillna(0).astype(np.int64).to_numpy())
    length = out["length"].astype(np.int64).to_numpy()

    valid_start = np.zeros(len(out), dtype=np.int64)
    valid_end = length.copy()
    start_mask = flag == 1
    end_mask = flag == 2
    valid_start[start_mask] = np.minimum(delta[start_mask], length[start_mask])
    valid_end[end_mask] = np.maximum(0, length[end_mask] - delta[end_mask])

    keep = (flag != 3) & (valid_end > valid_start)
    summary["dropped_static"] = int(np.sum(flag == 3))
    summary["dropped_empty"] = int(np.sum((flag != 3) & (valid_end <= valid_start)))

    if ratio is not None:
        trimmed = np.maximum(0, length - np.maximum(0, valid_end - valid_start))
        with np.errstate(invalid="ignore", divide="ignore"):
            trim_ratio = np.where(length > 0, trimmed / np.maximum(length, 1), 0.0)
        over = (flag != 3) & keep & (trim_ratio >= ratio)
        summary["dropped_over_trim"] = int(np.sum(over))
        keep = keep & ~over

    out["_valid_start"] = valid_start
    out["_valid_end"] = valid_end
    return out.loc[keep].reset_index(drop=True), summary


def _effective_segment_population_digest(eps_df: pd.DataFrame) -> str:
    """Hash the kept ``(episode, start, end)`` ranges in reader order."""
    hasher = hashlib.sha256(b"openwam:agibotworld-segment-population:v1\0")
    ordered = eps_df.sort_values("episode_index", kind="stable")
    for _, row in ordered.iterrows():
        start = int(row.get("_valid_start", 0))
        end = int(row.get("_valid_end", row["length"]))
        for value in (int(row["episode_index"]), start, end):
            hasher.update(value.to_bytes(8, "little", signed=False))
    return hasher.hexdigest()


def _bucket_base_motion_flags(dataset_dir) -> tuple[bool, bool]:
    """Return independent ``(action_moves, state_moves)`` flags for a bucket.

    Each flag is read from its OWN ``robot_velocity`` block in the bucket's
    shipped ``meta/stats.json`` and is true iff an exact min/max exceeds
    ``_MOVE_EPS``.  Keeping the streams separate matters for command-only
    mobile buckets: action slots 68/70 remain supervised while all-zero state
    slots are absent from the proprio mask.

    A missing / unreadable stats file returns ``(False, False)`` and logs a
    warning, so no fabricated constant-zero movement is supervised.
    """

    p = Path(dataset_dir) / "meta" / _BUCKET_STATS_FILENAME
    if not p.exists():
        logger.warning(
            "AgiBotWorld %s: no %s — treating base as stationary (move slots unmapped).",
            dataset_dir,
            _BUCKET_STATS_FILENAME,
        )
        return False, False
    try:
        with open(p) as f:
            st = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(
            "AgiBotWorld %s: could not read %s (%s) — treating base as stationary (move slots unmapped).",
            dataset_dir,
            p.name,
            e,
        )
        return False, False

    flags = []
    for key in ("action.robot_velocity", "observation.state.robot_velocity"):
        blk = st.get(key)
        if not blk:
            flags.append(False)
            continue
        lo = np.abs(np.asarray(blk.get("min", [0.0]), dtype=np.float64)).max()
        hi = np.abs(np.asarray(blk.get("max", [0.0]), dtype=np.float64)).max()
        flags.append(bool(max(lo, hi) > _MOVE_EPS))
    return flags[0], flags[1]


def _action_gripper_to_open_convention(grip: np.ndarray) -> np.ndarray:
    """Convert AgiBotWorld action gripper ``0=open,1=closed`` to ``0=closed,1=open``."""
    return (1.0 - np.asarray(grip, dtype=np.float32)).astype(np.float32, copy=False)


def _state_gripper_to_open_convention(grip: np.ndarray) -> np.ndarray:
    """Convert closing-actuator position to a clipped ``0=closed,1=open`` fraction."""
    raw = np.asarray(grip, dtype=np.float32)
    span = _GRIPPER_STATE_CLOSED_POSITION_M - _GRIPPER_STATE_OPEN_POSITION_M
    return np.clip((_GRIPPER_STATE_CLOSED_POSITION_M - raw) / span, 0.0, 1.0).astype(np.float32, copy=False)


def _eef18_to_eef20(ee18: np.ndarray, grip2: np.ndarray) -> np.ndarray:
    """Insert grippers into an already-rot6d 18-D pose → canonical 20-D EEF.

    Input layout::

        ee18:  [L_pos(3), L_rot6d(6), R_pos(3), R_rot6d(6)]   (already rot6d)
        grip2: [L_grip, R_grip]

    Output layout::

        [L_pos(3), L_rot6d(6), L_grip(1), R_pos(3), R_rot6d(6), R_grip(1)]

    Pure concatenation — NO euler→rot6d conversion (that is what distinguishes
    AgiBotWorld from RoboCOIN's :func:`eef14_to_eef20`).
    """

    l_pose9 = ee18[:, 0:9]
    r_pose9 = ee18[:, 9:18]
    l_grip = grip2[:, 0:1]
    r_grip = grip2[:, 1:2]
    return np.concatenate([l_pose9, l_grip, r_pose9, r_grip], axis=-1).astype(np.float32)


class AgiBotWorldDataset(LeRobotV3Reader):
    """Single-dataset reader for one AgiBotWorld-Beta task bucket (LeRobot v3).

    Bimanual EEF with real action/state supervision. All window / offset / video /
    prompt machinery is inherited from :class:`LeRobotV3Reader`; only the
    AgiBotWorld-specific bits are overridden below.
    """

    DATASET_NAME = "AgiBotWorld"
    HEAD_CAMERA = HEAD_CAMERA
    LEFT_WRIST_CAMERA = LEFT_WRIST_CAMERA
    RIGHT_WRIST_CAMERA = RIGHT_WRIST_CAMERA

    PROMPT_FILE_REQUIRED = True

    DEFAULT_NORMALIZE_MODE = None

    WRIST_DECODE_TOLERATED = (Exception,)
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("use_segment_annotations", "segment_max_trim_ratio")

    def __init__(
        self,
        dataset_dir,
        *,
        unify_action: bool = False,
        unify_action_map=None,
        use_segment_annotations: bool = True,
        segment_max_trim_ratio: float | None = None,
        **kwargs,
    ):
        """Detect the bucket flavor + base motion and wire the finger / move scatter.

        Under ``unify_action=True`` the reader builds the per-bucket map in code
        (OVERRIDING any config ``unify_action_map`` — the map varies per bucket, so a
        single yaml value cannot serve all). The raw width / map depend on:

        * Flavor — grippered raw ``EEF(20)`` (scatter ``0-9`` / ``34-43``) vs
          dexterous-hand raw ``[pose(18), fingers(12)]`` = 30 (RoboCOIN's dex scatter:
          pose→0-8/34-42, fingers→10-15/44-49).
        * Base motion — a 2-D ``robot_velocity`` x/yaw tail (→ move slots ``68`` / ``70``;
          y≡0 dropped) is appended when action OR state moves.  Independent raw
          dimension masks keep a command-only action valid while masking an absent
          all-zero state (and vice versa).

        Under ``unify_action=False`` no movement is emitted: grippered → 20-D EEF,
        dex → 20-D pose with grip slots masked.

        Args:
            use_segment_annotations: honor ``segment_flag`` / ``segment_delta``
                (see :meth:`_filter_episodes`). False → full segments, the
                pre-annotation behavior.
            segment_max_trim_ratio: drop an episode outright when the segment trim
                would remove at least this fraction of it (``0.7`` → an episode
                keeping under 30% of its frames is discarded). None disables the
                rule. Only meaningful with ``use_segment_annotations=True``.
        """

        self._is_dex = Path(dataset_dir).name in _DEX_BUCKET_IDS

        action_moves, state_moves = _bucket_base_motion_flags(dataset_dir) if unify_action else (False, False)
        self._action_has_move = bool(action_moves)
        self._proprio_has_move = bool(state_moves)
        self._has_move = self._action_has_move or self._proprio_has_move

        self.ACTION_DIM_MASK = None
        self.PROPRIO_DIM_MASK = None
        self._use_segment_annotations = bool(use_segment_annotations)
        self._segment_max_trim_ratio = _validate_trim_ratio(segment_max_trim_ratio)
        if self._segment_max_trim_ratio is not None and not self._use_segment_annotations:
            logger.warning(
                "AgiBotWorld %s: segment_max_trim_ratio=%s is ignored because "
                "use_segment_annotations=False (there is no trim to measure).",
                dataset_dir,
                self._segment_max_trim_ratio,
            )
        if unify_action:
            move_dim = _MOVE_DIM if self._has_move else 0
            move_slots = list(_MOVE_SLOTS) if self._has_move else []
            if self._is_dex:
                self.ACTION_DIM = _DEX_RAW_DIM + move_dim
                unify_action_map = _build_dex_unify_map(_DEX_PER_HAND, _DEX_PER_HAND) + move_slots
            else:
                self.ACTION_DIM = _EEF_RAW_DIM + move_dim
                unify_action_map = ["0-9", "34-43"] + move_slots
            if self._has_move:
                action_mask = np.ones(self.ACTION_DIM, dtype=bool)
                proprio_mask = np.ones(self.ACTION_DIM, dtype=bool)
                action_mask[-_MOVE_DIM:] = self._action_has_move
                proprio_mask[-_MOVE_DIM:] = self._proprio_has_move
                self.ACTION_DIM_MASK = action_mask
                self.PROPRIO_DIM_MASK = proprio_mask
        super().__init__(dataset_dir, unify_action=unify_action, unify_action_map=unify_action_map, **kwargs)

    def _filter_episodes(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        """Apply segment-boundary annotations to the trainable episode span.

        ``segment_flag`` selects an unchanged segment, a leading trim, a
        trailing trim, or a fully static segment to drop. ``segment_delta`` is
        interpreted as an integer frame count.  The resulting
        ``_valid_start``/``_valid_end`` bounds are consumed by the shared
        window builder, keeping parquet and video slices aligned.

        When configured, ``segment_max_trim_ratio`` drops a segment whose
        boundary trim would remove at least that fraction.  Mid-segment frames
        are never spliced or reordered.
        """

        out, summary = _apply_segment_annotations(
            eps_df,
            use_segment_annotations=self._use_segment_annotations,
            segment_max_trim_ratio=self._segment_max_trim_ratio,
        )
        if summary["missing_columns"]:
            logger.warning(
                "AgiBotWorld(%s): segment annotation columns %s/%s missing; using full segments.",
                self._dataset_id,
                _SEGMENT_FLAG_COL,
                _SEGMENT_DELTA_COL,
            )
        dropped_static = summary["dropped_static"]
        dropped_empty = summary["dropped_empty"]
        dropped_over_trim = summary["dropped_over_trim"]
        if dropped_static or dropped_empty or dropped_over_trim:
            logger.info(
                "AgiBotWorld(%s): segment annotations dropped %d static, %d empty-after-trim and "
                "%d over-trimmed (>=%s of the episode) episodes; kept %d episodes.",
                self._dataset_id,
                dropped_static,
                dropped_empty,
                dropped_over_trim,
                "n/a" if self._segment_max_trim_ratio is None else f"{self._segment_max_trim_ratio:.0%}",
                len(out),
            )
        return out

    def _resolve_cameras(self, info: dict):
        """Return the fixed head/left/right cameras, record robot_type, and pick
        the parquet columns + supervision mask for this bucket flavor.

        ``self._unify`` is already set by the base ``__init__`` before this hook
        runs, and the base reads the action/proprio masks after it.  The move-tail
        masks were set independently in ``__init__``; this hook only supplies the
        dex + unify-OFF grip mask (mirrors RoboCOIN).
        """

        self._robot_type = info.get("robot_type", "unknown")
        move_cols = _MOVE_COLS if self._has_move else ()
        if self._is_dex:
            if self._unify:
                self.NEEDED_COLS = _DEX_COLS + move_cols
            else:
                self.NEEDED_COLS = _POSE_ONLY_COLS
                self.ACTION_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
                self.PROPRIO_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
        else:
            self.NEEDED_COLS = (_GRIP_COLS + move_cols) if self._unify else _GRIP_COLS
        return self.HEAD_CAMERA, self.LEFT_WRIST_CAMERA, self.RIGHT_WRIST_CAMERA

    def _load_stats(self, info: dict):
        """Assemble SEPARATE action / state normalization stats from the unified stats file.

        Returns None (raw pass-through) unless a config opts into normalization.
        Action and state are DISTINCT distributions — e.g. ``gripper`` is a binary
        command in ``action.*`` but a continuous physical opening in
        ``observation.state.*`` — so each stream gets its OWN stats, built from its
        own ``{prefix}.ee_base`` / ``.gripper`` | ``.dex`` / ``.robot_velocity`` blocks
        (assembled to this bucket's raw layout, rot6d dims identity-pinned since the
        stats file is not pinned). The two blocks are stashed on the instance
        (``_action_norm_stats`` / ``_proprio_norm_stats``) and consumed by
        ``_action_20d`` / ``_proprio_20d`` respectively.

        Stats source: the UNIFIED ``<dataset_root>/meta/stats_g2a.json`` produced by
        agibotworld_stats_computation — one flavor-aware stats set pooled across all
        buckets (AgiBotWorld is a single embodiment, so the same physical action should
        normalize identically everywhere; this mirrors RoboCOIN's per-robot-type stats).
        Each column was pooled only over the buckets where it carries signal (ee_base:
        all; gripper: grippered buckets; dex: dex buckets; robot_velocity: the
        independently moving buckets for that action/state stream).
        mean/std/min/max are exact; q01/q99 are an approximate pooled estimate (merged
        reservoirs). Every mode is valid; missing file → raise (run the stats script).
        Under root mode ``_dataset_dir`` is a bucket, so the file lives at
        ``_dataset_dir.parent / meta``.
        """

        self._action_norm_stats = None
        self._proprio_norm_stats = None
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir.parent / "meta" / _UNIFIED_STATS_FILENAME
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but the unified stats file {stats_path} is missing. "
                f"Run `python -m openwam.dataloader.utils.stats_computation.agibotworld_stats_computation "
                f"--dataset_dir {self._dataset_dir.parent}` to generate it, or set normalize_mode=null to disable."
            )
        with open(stats_path) as f:
            raw = json.load(f)

        if raw.get("gripper_contract") != _GRIPPER_CONTRACT:
            raise DataContractError(
                f"AgiBotWorld stats {stats_path} use gripper_contract="
                f"{raw.get('gripper_contract')!r}; expected {_GRIPPER_CONTRACT!r}. "
                "Regenerate stats with the current canonical direction and endpoint calibration."
            )

        population = raw.get("population")
        if not isinstance(population, dict) or population.get("schema_version") != _STATS_SCHEMA_VERSION:
            raise DataContractError(
                f"AgiBotWorld stats {stats_path} use stale population schema "
                f"{None if not isinstance(population, dict) else population.get('schema_version')!r}; "
                f"expected {_STATS_SCHEMA_VERSION}. Regenerate it to bind the canonical gripper "
                "contract, independent action/state velocity populations, and terminal-action exclusion."
            )
        recorded_buckets = population.get("buckets")
        if not isinstance(recorded_buckets, dict) or not all(
            isinstance(name, str) and name for name in recorded_buckets
        ):
            raise DataContractError(
                f"AgiBotWorld stats {stats_path} has a malformed pooled contributor map; regenerate stats."
            )

        current_buckets = {
            path.name
            for path in self._dataset_dir.parent.iterdir()
            if path.is_dir() and (path / "meta" / "info.json").is_file()
        }
        recorded_bucket_names = set(recorded_buckets)
        if recorded_bucket_names != current_buckets:
            missing = sorted(current_buckets - recorded_bucket_names)
            extra = sorted(recorded_bucket_names - current_buckets)
            raise DataContractError(
                f"AgiBotWorld stats {stats_path} pooled contributor set no longer matches "
                f"the current dataset root (missing={missing}, extra={extra}); regenerate stats."
            )
        if population.get("split") != "train":
            raise DataContractError(
                f"AgiBotWorld stats {stats_path} were generated from split={population.get('split')!r}; "
                "normalization statistics must be train-derived. Regenerate with --split train."
            )
        expected_annotations = bool(population.get("use_segment_annotations"))
        expected_ratio = _validate_trim_ratio(population.get("segment_max_trim_ratio"))
        if expected_annotations != self._use_segment_annotations or expected_ratio != self._segment_max_trim_ratio:
            raise DataContractError(
                f"AgiBotWorld stats {stats_path} were generated with "
                f"use_segment_annotations={expected_annotations}, segment_max_trim_ratio={expected_ratio}, "
                f"but bucket {self._dataset_id} loads with "
                f"use_segment_annotations={self._use_segment_annotations}, "
                f"segment_max_trim_ratio={self._segment_max_trim_ratio}. Regenerate stats with matching settings."
            )
        bucket_population = recorded_buckets.get(self._dataset_id)
        if not isinstance(bucket_population, dict):
            raise DataContractError(
                f"AgiBotWorld bucket {self._dataset_id} has no population record in {stats_path}; regenerate stats."
            )
        disk_population = resolve_lerobot_v3_data_population(self._dataset_dir, info=info)
        disk_digest = digest_lerobot_v3_data_population(disk_population)
        if bucket_population.get("data_population_digest") != disk_digest:
            raise DataContractError(
                f"AgiBotWorld bucket {self._dataset_id} manifest/data mapping changed after {stats_path} was generated; "
                "regenerate normalization stats."
            )

        if self._split == population.get("split") and self._max_hours is None:
            effective_digest = _effective_segment_population_digest(self._eps_df)
            if bucket_population.get("effective_population_digest") != effective_digest:
                raise DataContractError(
                    f"AgiBotWorld bucket {self._dataset_id} segment/exclusion population changed after "
                    f"{stats_path} was generated; regenerate normalization stats."
                )

        def _mat(key: str, dim: int):
            if key not in raw:
                raise KeyError(f"AgiBotWorld bucket {self._dataset_id}: '{key}' missing from {stats_path}.")
            return materialize_eef_stats(
                raw[key], self._normalize_mode, dim=dim, strict_minmax=False, source_hint=f"{stats_path}: {key}"
            )

        def _build(prefix: str):
            """Assemble the width-matched stats block for one stream ('action' or
            'observation.state')."""

            ee = _mat(f"{prefix}.ee_base", 18)
            combined = {}
            if self._is_dex and self._unify:
                fing = _mat(f"{prefix}.dex", 2 * _DEX_PER_HAND)
                for k in _STAT_FIELDS:
                    e, h = ee[k], fing[k]
                    combined[k] = np.concatenate(
                        [e[0:9], h[0:_DEX_PER_HAND], e[9:18], h[_DEX_PER_HAND : 2 * _DEX_PER_HAND]]
                    ).astype(np.float32)
                pin_rot6d_identity(combined, _ROT6D_DIMS_DEX30)
            else:
                g = _mat(f"{prefix}.gripper", 2)
                for k in _STAT_FIELDS:
                    e, gg = ee[k], g[k]
                    combined[k] = np.concatenate([e[0:9], gg[0:1], e[9:18], gg[1:2]]).astype(np.float32)
                pin_rot6d_identity(combined, ROT6D_DIMS_EEF20)
            if self._has_move:
                stream_moves = self._action_has_move if prefix == "action" else self._proprio_has_move
                if stream_moves:
                    vel = _mat(f"{prefix}.robot_velocity", 3)
                else:
                    vel = {
                        "mean": np.zeros(3, dtype=np.float32),
                        "std": np.ones(3, dtype=np.float32),
                        "min": -np.ones(3, dtype=np.float32),
                        "max": np.ones(3, dtype=np.float32),
                        "q01": -np.ones(3, dtype=np.float32),
                        "q99": np.ones(3, dtype=np.float32),
                    }
                src = list(_MOVE_SRC_DIMS)
                for k in _STAT_FIELDS:
                    combined[k] = np.concatenate([combined[k], vel[k][src]]).astype(np.float32)
            return combined

        self._action_norm_stats = _build("action")
        self._proprio_norm_stats = _build("observation.state")

        return self._action_norm_stats

    def _train_min_window_len(self) -> int:
        """Require two rows so every train window has a real next-state target."""
        return 2

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        """Mask the clamped final action in an episode-truncated tail window.

        The source is already next-state relabeled: its final episode row has no
        successor and therefore contains a fabricated/clamped target. A full
        ``num_frames`` window never emits that last row because ``T_action`` is
        one shorter; a truncated window does, so drop exactly its final row.
        """

        if actual_raw_len >= self._num_frames:
            return actual_raw_len
        return max(0, actual_raw_len - 1)

    def _normalize_array(self, arr: np.ndarray, stats) -> np.ndarray:
        """Apply per-stream normalization with the given stats. No-op when null."""
        return apply_normalization(arr, stats, self._normalize_mode)

    def _grip_or_zeros(self, win, col: str, n: int) -> np.ndarray:
        """Stack ``n`` rows of a gripper column, or zeros for dex-hand buckets.

        Dex-hand buckets under unify OFF read pose-only columns; the (n, 2) zero
        fill flows through :func:`_eef18_to_eef20` into the grip slots, which are
        masked out of supervision by ``GRIP_EXCLUDED_DIM_MASK``.
        """

        if not self._is_dex:
            grip = np.stack(win[col].values[:n]).astype(np.float32)
            if col == "action.gripper":
                grip = _action_gripper_to_open_convention(grip)
            else:
                grip = _state_gripper_to_open_convention(grip)
            return grip
        return np.zeros((n, 2), dtype=np.float32)

    def _dex_pose_fingers(self, ee18: np.ndarray, dex12: np.ndarray) -> np.ndarray:
        """Assemble a 30-D dex-hand pose+fingers vector (UN-normalized).

        ``[L_pos(3), L_rot6d(6), L_fingers(6), R_pos(3), R_rot6d(6), R_fingers(6)]``
        — pose from ``ee_base`` (already rot6d), fingers from ``dex`` (``[:6]`` left,
        ``[6:12]`` right). The base unify scatter places these into the 80-D pose +
        hand slots. Normalization is applied by the caller after the move tail is
        appended (so the whole raw vector normalizes in one pass).
        """

        l_pose9 = ee18[:, 0:9]
        r_pose9 = ee18[:, 9:18]
        l_fing = dex12[:, 0:_DEX_PER_HAND]
        r_fing = dex12[:, _DEX_PER_HAND : 2 * _DEX_PER_HAND]
        return np.concatenate([l_pose9, l_fing, r_pose9, r_fing], axis=-1).astype(np.float32)

    def _append_move(self, raw: np.ndarray, win, col: str, n: int) -> np.ndarray:
        """Append the base-movement tail for mobile buckets: robot_velocity's x/yaw
        (``_MOVE_SRC_DIMS``) only — the y column is ≡0 and is dropped."""

        if not self._has_move:
            return raw
        stream_moves = self._action_has_move if col.startswith("action.") else self._proprio_has_move
        if stream_moves:
            vel = np.stack(win[col].values[:n]).astype(np.float32)[:, _MOVE_SRC_DIMS]
        else:
            vel = np.zeros((n, _MOVE_DIM), dtype=np.float32)
        return np.concatenate([raw, vel], axis=-1)

    def _action_20d(self, win) -> np.ndarray:
        ee = np.stack(win["action.ee_base"].values).astype(np.float32)
        n = len(ee)
        if self._is_dex and self._unify:
            dex = np.stack(win["action.dex"].values).astype(np.float32)
            raw = self._dex_pose_fingers(ee, dex)
        else:
            grip = self._grip_or_zeros(win, "action.gripper", n)
            raw = _eef18_to_eef20(ee, grip)
        raw = self._append_move(raw, win, "action.robot_velocity", n)
        return self._normalize_array(raw, self._action_norm_stats)

    def _proprio_20d(self, win) -> np.ndarray:
        ee = np.stack(win["observation.state.ee_base"].values[:1]).astype(np.float32)
        if self._is_dex and self._unify:
            dex = np.stack(win["observation.state.dex"].values[:1]).astype(np.float32)
            raw = self._dex_pose_fingers(ee, dex)
        else:
            grip = self._grip_or_zeros(win, "observation.state.gripper", 1)
            raw = _eef18_to_eef20(ee, grip)
        raw = self._append_move(raw, win, "observation.state.robot_velocity", 1)
        return self._normalize_array(raw, self._proprio_norm_stats)

    @property
    def robot_type(self):
        return self._robot_type

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiAgiBotWorldDataset


class MultiAgiBotWorldDataset(MultiLeRobotV3Reader):
    """Aggregate of N AgiBotWorld-Beta per-task buckets."""

    def __init__(self, buckets: List[AgiBotWorldDataset]):
        super().__init__(buckets)
        n_dex = sum(1 for b in self._buckets if getattr(b, "_is_dex", False))
        logger.info(
            "MultiAgiBotWorldDataset: %d datasets (%d dex-hand, %d grippered), %d windows",
            len(self._buckets),
            n_dex,
            len(self._buckets) - n_dex,
            len(self),
        )

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return AgiBotWorldDataset.from_config(config, split)


__all__ = ["AgiBotWorldDataset", "MultiAgiBotWorldDataset"]
