"""Public implementation. Dataset-specific audit notes were removed."""

















































































































































from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    assert_unit_quaternion,
    quat_wxyz_to_rot6d,
)
from openwam.dataloader.utils.lerobotv3 import build_multibucket
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats

logger = logging.getLogger(__name__)

_ACTION_DIM = EEF_DIM




_CONFIG_SENTINEL = object()


ROBOT_TYPE_TO_EMBODIMENT: Dict[str, str] = {
    "Franka": "franka",
    "ARX Lift-2": "lift2",
    "Genie-1": "genie1",
    "AgileX Split Aloha": "split_aloha",
}




































GRIPPER_FULL_OPEN = {
    "franka": 0.08,
    "lift2": 0.088,
    "genie1": 5.74,
    "split_aloha": 0.1,
}

GRIPPER_ALT_FULL_OPEN = {
    "franka": 1.0,
}


_GRIPPER_SANE_MAX = 1.25


def resolve_gripper_scale(bucket_dir: Path, embodiment: str, grip_col: str) -> float:
    """Public implementation. Dataset-specific audit notes were removed."""
















    primary = GRIPPER_FULL_OPEN.get(embodiment, 1.0)
    alt = GRIPPER_ALT_FULL_OPEN.get(embodiment)
    stats_path = bucket_dir / "meta" / "stats.json"
    try:
        with open(stats_path) as f:
            blk = json.load(f)[grip_col]
        observed_max = float(np.ravel(blk["max"])[0])





        try:
            observed_mean = float(np.ravel(blk["mean"])[0])
        except (KeyError, TypeError, IndexError, ValueError):
            observed_mean = float("nan")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        logger.debug(
            "InternData-A1: no usable %s max in %s; assuming the %s stroke %.4f",
            grip_col,
            stats_path,
            embodiment,
            primary,
        )
        return primary

    scale = primary
    if alt is not None and observed_max > 0:

        if abs(np.log(observed_max / alt)) < abs(np.log(observed_max / primary)):
            scale = alt














            if observed_mean > primary:
                logger.info(
                    "InternData-A1 %s: %s max %.4f -> alt (second-variant) stroke %.4f for %r "
                    "(mean %.4f corroborates).",
                    bucket_dir,
                    grip_col,
                    observed_max,
                    alt,
                    embodiment,
                    observed_mean,
                )
            else:
                logger.warning(
                    "InternData-A1 %s: %s max %.4f selected the alt stroke %.4f for %r, but mean "
                    "%.4f does not clear the primary stroke %.4f — if that max is a single glitch "
                    "row this bucket's gripper is being squashed by %.1fx. Verify the bucket.",
                    bucket_dir,
                    grip_col,
                    observed_max,
                    alt,
                    embodiment,
                    observed_mean,
                    primary,
                    alt / primary,
                )






    if observed_max / scale > _GRIPPER_SANE_MAX:
        logger.warning(
            "InternData-A1 %s: %s max %.4f is %.2fx the assumed full-open stroke %.4f "
            "for %r — out-of-range outliers, or GRIPPER_FULL_OPEN needs updating.",
            bucket_dir,
            grip_col,
            observed_max,
            observed_max / scale,
            scale,
            embodiment,
        )
    return scale




_BIMANUAL_COLS: Tuple[str, ...] = (
    "states.left_ee_to_robot_pose",
    "states.left_gripper.position",
    "states.right_ee_to_robot_pose",
    "states.right_gripper.position",
    "actions.left_ee_to_robot_pose",
    "actions.left_gripper.position",
    "actions.right_ee_to_robot_pose",
    "actions.right_gripper.position",
    "task_index",
)

_SINGLE_ARM_COLS: Tuple[str, ...] = (
    "states.ee_to_robot_pose",
    "states.gripper.position",
    "actions.ee_to_robot_pose",
    "actions.gripper.position",
    "task_index",
)


_BIMANUAL_SIDES = {
    "state": (
        ("states.left_ee_to_robot_pose", "states.left_gripper.position"),
        ("states.right_ee_to_robot_pose", "states.right_gripper.position"),
    ),
    "action": (
        ("actions.left_ee_to_robot_pose", "actions.left_gripper.position"),
        ("actions.right_ee_to_robot_pose", "actions.right_gripper.position"),
    ),
}
_SINGLE_ARM_SIDES = {
    "state": (("states.ee_to_robot_pose", "states.gripper.position"), None),
    "action": (("actions.ee_to_robot_pose", "actions.gripper.position"), None),
}


_HEAD_CAMERA = "images.rgb.head"

_LEFT_WRIST_BIMANUAL = "images.rgb.hand_left"
_RIGHT_WRIST_BIMANUAL = "images.rgb.hand_right"
_WRIST_SINGLE_ARM = "images.rgb.hand"


def detect_arm_layout(features: Dict[str, Any]) -> str:
    """Public implementation. Dataset-specific audit notes were removed."""





    if "states.left_ee_to_robot_pose" in features and "states.right_ee_to_robot_pose" in features:
        return "bimanual"
    if "states.ee_to_robot_pose" in features:
        return "single_arm"
    raise ValueError(
        "InternData-A1: info.features exposes neither the bimanual "
        "(states.left_ee_to_robot_pose + states.right_ee_to_robot_pose) nor the "
        "single-arm (states.ee_to_robot_pose) EEF schema; got keys "
        f"{sorted(k for k in features if k.startswith('states.'))}"
    )


def embodiment_key(robot_type: str, arm_layout: str) -> str:
    """Public implementation. Dataset-specific audit notes were removed."""






    if robot_type in ROBOT_TYPE_TO_EMBODIMENT:
        return ROBOT_TYPE_TO_EMBODIMENT[robot_type]
    slug = "".join(c if c.isalnum() else "_" for c in str(robot_type).lower()).strip("_") or "unknown"
    logger.warning(
        "InternData-A1: unrecognized robot_type %r (%s layout) -> stats key %r. "
        "Add it to ROBOT_TYPE_TO_EMBODIMENT and re-run interndata_a1_stats_computation.",
        robot_type,
        arm_layout,
        slug,
    )
    return slug


def discover_a1_buckets(root: Path) -> List[Path]:
    """Public implementation. Dataset-specific audit notes were removed."""






























    out: List[Path] = []
    seen: set = set()
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
        except OSError:


            dirnames[:] = []
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            dirnames[:] = []
            continue
        seen.add(key)
        if os.path.isfile(os.path.join(dirpath, "meta", "info.json")):
            out.append(Path(dirpath))
            dirnames[:] = []
            continue





        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in ("data", "videos"))
    return sorted(out)










_CHUNK_DIR_RE = re.compile(r"chunk-(\d+)")
_SHARD_FILE_RE = re.compile(r"file-(\d+)\.parquet")


def parse_shard_path(path) -> Optional[Tuple[int, int]]:
    """Public implementation. Dataset-specific audit notes were removed."""















    p = Path(path)
    chunk = _CHUNK_DIR_RE.fullmatch(p.parent.name)
    shard = _SHARD_FILE_RE.fullmatch(p.name)
    if chunk is None or shard is None:
        return None
    ci, fi = int(chunk.group(1)), int(shard.group(1))
    if p.parent.name != f"chunk-{ci:03d}" or p.name != f"file-{fi:03d}.parquet":
        return None
    return ci, fi


def iter_data_shards(bucket) -> List[Tuple[int, int, Path]]:
    """Public implementation. Dataset-specific audit notes were removed."""







    out: List[Tuple[int, int, Path]] = []
    for pth in (Path(bucket) / "data").glob("chunk-*/file-*.parquet"):
        ids = parse_shard_path(pth)
        if ids is not None:
            out.append((ids[0], ids[1], pth))
    return sorted(out, key=lambda t: (t[0], t[1]))


def load_excluded_episodes(bucket) -> set:
    """Public implementation. Dataset-specific audit notes were removed."""












    path = Path(bucket) / "meta" / "excluded_episodes.json"
    if not path.is_file():
        return set()
    with open(path) as fh:
        raw = json.load(fh)["episode_indices"]
    out = set()
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(
                f"{path}: episode_indices must be JSON integers, got {x!r} "
                f"({type(x).__name__}). The reader matches these against an integer "
                "episode_index column verbatim, so a quoted or float value excludes "
                "nothing there while excluding here — the two sides would then "
                "normalise over different populations with no error anywhere."
            )
        out.add(int(x))
    return out


def validate_manifest_ranges(from_idx, to_idx, lengths, episode_idx, who: str) -> None:
    """Public implementation. Dataset-specific audit notes were removed."""















    from_idx = np.asarray(from_idx, dtype=np.int64)
    to_idx = np.asarray(to_idx, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    episode_idx = np.asarray(episode_idx, dtype=np.int64)

    if (from_idx < 0).any() or (to_idx < from_idx).any():
        i = int(np.flatnonzero((from_idx < 0) | (to_idx < from_idx))[0])
        raise ValueError(
            f"{who}: episode {int(episode_idx[i])} has a malformed manifest range "
            f"[{int(from_idx[i])}, {int(to_idx[i])})."
        )
    bad = np.flatnonzero(to_idx - from_idx != lengths)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"{who}: episode {int(episode_idx[i])} spans "
            f"[{int(from_idx[i])}, {int(to_idx[i])}) = {int(to_idx[i] - from_idx[i])} rows but "
            f"declares length={int(lengths[i])}. The reader sizes windows from one and "
            "bounds them with the other, so they must describe the same episode."
        )







    order = np.argsort(from_idx, kind="stable")
    fs, ts = from_idx[order], to_idx[order]
    if fs.size > 1:
        overlap = np.flatnonzero(fs[1:] < ts[:-1])
        if overlap.size:
            k = int(overlap[0])
            raise ValueError(
                f"{who}: episode {int(episode_idx[order][k + 1])} starts at row "
                f"{int(fs[k + 1])} but episode {int(episode_idx[order][k])} runs to "
                f"{int(ts[k])}. Overlapping manifest ranges make two episodes read the "
                "same rows — the totals and the per-episode capacity check both pass "
                "while the data is paired with the wrong episode."
            )


def _shard_episode_bounds(pf) -> Optional[Tuple[int, int]]:
    """Public implementation. Dataset-specific audit notes were removed."""



















    try:
        col = pf.schema_arrow.names.index("episode_index")
    except ValueError:
        return None
    lo = hi = None
    meta = pf.metadata
    for g in range(meta.num_row_groups):
        st = meta.row_group(g).column(col).statistics
        if st is None or not st.has_min_max:
            vals = pf.read(columns=["episode_index"]).column("episode_index")
            if len(vals) == 0:
                return None
            return int(min(vals.to_pylist())), int(max(vals.to_pylist()))
        lo = st.min if lo is None else min(lo, st.min)
        hi = st.max if hi is None else max(hi, st.max)
    if lo is None or hi is None:
        return None
    return int(lo), int(hi)


_TRIM_SPEC_CACHE: Dict[str, Dict[str, Dict[int, Tuple[int, Optional[int], Optional[int]]]]] = {}


def _load_trim_spec(path) -> Dict[str, Dict[int, Tuple[int, Optional[int], Optional[int]]]]:
    """Public implementation. Dataset-specific audit notes were removed."""













    import csv

    key = str(path)
    if key in _TRIM_SPEC_CACHE:
        return _TRIM_SPEC_CACHE[key]
    spec: Dict[str, Dict[int, Tuple[int, Optional[int], Optional[int]]]] = {}
    n = 0
    try:
        with open(key, newline="") as fh:
            for row in csv.DictReader(fh):

                def _int(name):
                    v = (row.get(name) or "").strip()
                    return int(v) if v else None

                head = _int("trim_head_to") or 0
                tail = _int("trim_tail_from")
                if not head and tail is None:
                    continue
                spec.setdefault(row["dataset"], {})[int(row["episode_index"])] = (
                    head,
                    tail,
                    _int("total_frames"),
                )
                n += 1
    except (OSError, KeyError, ValueError) as e:




        raise ValueError(
            f"InternDataA1: trim_csv {key} could not be read ({e}). Fix the path or set "
            "trim_csv=null to run untrimmed — it will not be skipped silently."
        ) from e
    else:
        logger.info(
            "InternDataA1: loaded %d trim entries across %d buckets from %s", n, len(spec), key
        )
    _TRIM_SPEC_CACHE[key] = spec
    return spec


_TRIM_DIGEST_CACHE: Dict[str, str] = {}


class AmbiguousBucketKey(LookupError):
    """Public implementation. Dataset-specific audit notes were removed."""


def resolve_bucket_key(keys, dataset_id: str, bucket_dir, *, what: str,
                       source: str) -> Optional[str]:
    """Public implementation. Dataset-specific audit notes were removed."""






















    if dataset_id in keys:
        return dataset_id

    if "/" in dataset_id:
        return None



    parts = Path(bucket_dir).resolve().parts
    for n in range(min(len(parts), 6), 0, -1):
        cand = "/".join(parts[-n:])
        if cand in keys:
            return cand



    leaf = Path(bucket_dir).name
    cands = sorted(k for k in keys if k == leaf or k.endswith("/" + leaf))
    if len(cands) > 1:
        raise AmbiguousBucketKey(
            f"{source}: {what} has {len(cands)} buckets whose path ends in {leaf!r} "
            f"({', '.join(cands[:4])}{' ...' if len(cands) > 4 else ''}), and none of them "
            "matches this bucket's own path. Bucket leaf names repeat across tasks, so this "
            "one cannot be identified from its directory name. Pass dataset_id (or point "
            "--dataset_dir at the corpus root) so buckets are keyed by their path relative "
            "to the root, which is what these files key on."
        )
    return None


def resolve_trim_bounds(entry, length: int, min_len: int) -> Optional[Tuple[int, int]]:
    """Public implementation. Dataset-specific audit notes were removed."""














    head, tail_from, total = entry
    if total is not None and int(total) != int(length):
        return None
    tail = int(length) if tail_from is None else min(int(tail_from), int(length))
    head = max(0, min(int(head), tail))
    if head == 0 and tail == int(length):
        return None
    if tail - head < min_len:
        return None
    return head, tail


class InternDataA1Dataset(LeRobotV3Reader):
    """Public implementation. Dataset-specific audit notes were removed."""







    DATASET_NAME = "InternDataA1"
    ACTION_DIM = _ACTION_DIM


    NEEDED_COLS = _BIMANUAL_COLS
    PROMPT_SOURCE = "task_index"
    PROMPT_FILE_REQUIRED = True
    STATS_DIM = _ACTION_DIM
    STATS_STRICT_MINMAX = False
    DEFAULT_NORMALIZE_MODE = "quantile"






    DEPLOY_ACTION_MODE = None


    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("trim_csv",)

    def __init__(
        self,
        dataset_dir,
        *,
        a1_stats_root: Optional[str] = None,
        trim_csv: Optional[str] = None,
        **kwargs,
    ):
        """Public implementation. Dataset-specific audit notes were removed."""









        self._a1_stats_root = Path(a1_stats_root) if a1_stats_root else Path(dataset_dir)
        self._trim_csv = trim_csv





        load_excluded_episodes(dataset_dir)
        super().__init__(dataset_dir, **kwargs)



    def _resolve_cameras(self, info: dict):
        """Public implementation. Dataset-specific audit notes were removed."""





        features = info.get("features", {}) or {}
        self._arm_layout = detect_arm_layout(features)
        self._robot_type = info.get("robot_type", "unknown")
        self._embodiment = embodiment_key(self._robot_type, self._arm_layout)

        if self._arm_layout == "bimanual":
            self.NEEDED_COLS = _BIMANUAL_COLS
            self._sides = _BIMANUAL_SIDES

            self.ACTION_DIM_MASK = None
            left_wrist, right_wrist = _LEFT_WRIST_BIMANUAL, _RIGHT_WRIST_BIMANUAL
        else:
            self.NEEDED_COLS = _SINGLE_ARM_COLS
            self._sides = _SINGLE_ARM_SIDES

            self.ACTION_DIM_MASK = LEFT_ARM_DIM_MASK

            left_wrist, right_wrist = _WRIST_SINGLE_ARM, None

        missing = [c for c in self.NEEDED_COLS if c != "task_index" and c not in features]
        if missing:
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {self._arm_layout} layout is missing "
                f"feature(s) {missing} in info.json."
            )



        self._grip_scale = tuple(
            resolve_gripper_scale(self._dataset_dir, self._embodiment, spec[1]) if spec is not None else 1.0
            for spec in self._sides["state"]
        )

        head = _HEAD_CAMERA if _HEAD_CAMERA in features else None
        if head is None:
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: no {_HEAD_CAMERA!r} camera in info.features "
                f"(has {sorted(k for k in features if k.startswith('images.'))})"
            )


        if left_wrist is not None and left_wrist not in features:
            left_wrist = None
        if right_wrist is not None and right_wrist not in features:
            right_wrist = None
        return head, left_wrist, right_wrist

    def _add_data_offsets(self, eps) -> None:
        """Public implementation. Dataset-specific audit notes were removed."""























        from concurrent.futures import ThreadPoolExecutor

        import pyarrow.parquet as pq

        shards = iter_data_shards(self._dataset_dir)
        if not shards:
            raise FileNotFoundError(f"No data parquet files under {self._dataset_dir}/data")

        def _read_meta(entry):
            chunk, file_idx, path = entry
            pf = pq.ParquetFile(path)
            return chunk, file_idx, pf.metadata.num_rows, _shard_episode_bounds(pf)

        with ThreadPoolExecutor(max_workers=min(len(shards), 4)) as pool:
            data_files = list(pool.map(_read_meta, shards))

        who = f"{self.DATASET_NAME}({self._dataset_id})"





        validate_manifest_ranges(
            eps["dataset_from_index"].to_numpy(),
            eps["dataset_to_index"].to_numpy(),
            eps["length"].to_numpy(),
            eps["episode_index"].to_numpy(),
            who,
        )

        starts = np.concatenate([[0], np.cumsum([n for _, _, n, _ in data_files])]).astype(np.int64)







        manifest_end = int(eps["dataset_to_index"].to_numpy().max())
        if manifest_end != int(starts[-1]):
            raise ValueError(
                f"{who}: the data shards hold {int(starts[-1])} rows but the manifest ends at "
                f"{manifest_end}. A shard is missing, truncated or out of order — resolving "
                "offsets against this would map episodes onto another episode's rows."
            )

        global_starts = eps["dataset_from_index"].to_numpy().astype(np.int64)
        file_pos = np.searchsorted(starts, global_starts, side="right") - 1
        if (file_pos < 0).any() or (file_pos >= len(data_files)).any():
            raise ValueError(f"{who}: dataset_from_index outside the data parquet row range")





        offsets = global_starts - starts[file_pos]
        lengths = eps["dataset_to_index"].to_numpy().astype(np.int64) - global_starts
        capacity = np.array([data_files[i][2] for i in file_pos], dtype=np.int64)
        overflow = np.flatnonzero(offsets + lengths > capacity)
        if overflow.size:
            i = int(overflow[0])
            raise ValueError(
                f"{who}: episode {int(eps['episode_index'].to_numpy()[i])} needs rows "
                f"[{int(offsets[i])}, {int(offsets[i] + lengths[i])}) of a shard holding only "
                f"{int(capacity[i])}. The shard set does not match the manifest — resolving "
                "offsets against it would map episodes onto another episode's rows."
            )









        ep_vals = eps["episode_index"].to_numpy()
        for i, pos in enumerate(file_pos):
            bounds = data_files[pos][3]
            if bounds is None:
                continue
            ep = int(ep_vals[i])
            if not bounds[0] <= ep <= bounds[1]:
                raise ValueError(
                    f"{who}: the manifest puts episode {ep} in shard "
                    f"chunk-{data_files[pos][0]:03d}/file-{data_files[pos][1]:03d}, but that "
                    f"shard only holds episodes {bounds[0]}..{bounds[1]}. The shard set does "
                    "not match the manifest (a copied, reordered or substituted shard passes "
                    "the row-count checks above while holding another episode's data)."
                )

        eps["data/chunk_index"] = np.array([data_files[i][0] for i in file_pos], dtype=np.int64)
        eps["data/file_index"] = np.array([data_files[i][1] for i in file_pos], dtype=np.int64)
        eps["_data_row_offset"] = offsets

    def _add_episode_offsets(self, eps) -> None:
        """Public implementation. Dataset-specific audit notes were removed."""















        super()._add_episode_offsets(eps)
        for cam in self._video_cameras():
            col = f"videos/{cam}/from_timestamp"
            if col not in eps.columns:




                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has no "
                    f"{col!r} in meta/episodes, so its frame offset cannot be resolved."
                )
            if not np.isfinite(self._fps) or self._fps <= 0:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): fps={self._fps!r} is not a "
                    "positive finite number, so timestamps cannot be converted to frames."
                )
            ts = eps[col].to_numpy().astype(np.float64)
            if not np.isfinite(ts).all() or (ts < 0).any():
                bad = int((~np.isfinite(ts)).sum() + (ts < 0).sum())
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has {bad} "
                    "non-finite or negative from_timestamp values; casting those would "
                    "produce INT64_MIN or negative frame indices and read black slots."
                )
            frames = ts * self._fps
            if np.abs(frames - np.rint(frames)).max() > 0.25:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has "
                    "from_timestamp values that are not on frame boundaries at "
                    f"fps={self._fps}; the offsets would be rounded onto neighbouring frames."
                )
            if frames.max() > 2**53:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} timestamp * fps "
                    "exceeds the exactly-representable integer range; the cast would wrap."
                )
            off = np.rint(frames).astype(np.int64)












            ck = f"videos/{cam}/chunk_index"
            fk = f"videos/{cam}/file_index"
            if fk in eps.columns:
                ep_len = eps["length"].to_numpy().astype(np.int64)
                chunk_of = (eps[ck].to_numpy().astype(np.int64) if ck in eps.columns
                            else np.zeros(len(off), dtype=np.int64))
                file_of = eps[fk].to_numpy().astype(np.int64)
                ep_vals = eps["episode_index"].to_numpy()
                shard_key = np.stack([chunk_of, file_of], axis=1)
                for shard in np.unique(shard_key, axis=0):
                    m = np.flatnonzero((chunk_of == shard[0]) & (file_of == shard[1]))
                    order = m[np.argsort(off[m], kind="stable")]
                    for a, b in zip(order, order[1:]):
                        if off[a] + ep_len[a] > off[b]:
                            raise ValueError(
                                f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r}, video "
                                f"shard chunk-{int(shard[0]):03d}/file-{int(shard[1]):03d}: "
                                f"episode {int(ep_vals[a])} covers frames "
                                f"[{int(off[a])}, {int(off[a] + ep_len[a])}) which overlaps "
                                f"episode {int(ep_vals[b])} starting at {int(off[b])}. One of "
                                "them would read the other's frames."
                            )





            tcol = f"videos/{cam}/to_timestamp"
            if tcol in eps.columns:
                span = (eps[tcol].to_numpy().astype(np.float64) - ts) * self._fps
                declared = eps["length"].to_numpy().astype(np.int64)



                bad = np.flatnonzero(~np.isfinite(span) | (np.abs(span - declared) > 0.5))
                if bad.size:
                    i = int(bad[0])
                    raise ValueError(
                        f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} episode "
                        f"{int(eps['episode_index'].to_numpy()[i])} spans {span[i]:.2f} video "
                        f"frames between from/to_timestamp but declares length="
                        f"{int(declared[i])}. The manifest describes two different episodes."
                    )
            eps[self._video_offset_col(cam)] = off

    def _trim_min_len(self) -> int:
        """Public implementation. Dataset-specific audit notes were removed."""





        return self._num_frames if self._split == "val" else self._train_min_window_len()

    def _match_bucket_key(self, keys, what: str) -> Optional[str]:
        """Public implementation. Dataset-specific audit notes were removed."""














        return resolve_bucket_key(keys, self._dataset_id, self._dataset_dir,
                                  what=what, source=f"InternDataA1({self._dataset_id})")

    def _trim_key(self) -> Optional[str]:
        spec = _load_trim_spec(self._trim_csv)
        if not spec:
            return None
        return self._match_bucket_key(spec, "trim_csv")

    def _filter_episodes(self, eps_df):
        """Public implementation. Dataset-specific audit notes were removed."""

































        eps_df = super()._filter_episodes(eps_df)
        if not self._trim_csv:
            return eps_df
        key = self._trim_key()
        if key is None:
            return eps_df
        spec = _load_trim_spec(self._trim_csv).get(key)
        if not spec:
            return eps_df

        cam_cols = [c for c in eps_df.columns if c.startswith("_video_frame_offset/")]


        lengths = eps_df["length"].to_numpy().copy()
        row_off = eps_df["_data_row_offset"].to_numpy().copy()
        cam_off = {c: eps_df[c].to_numpy().copy() for c in cam_cols}
        min_len = self._trim_min_len()

        n_trim = n_skip = 0
        frames_before = int(lengths.sum())
        for pos, ep in enumerate(eps_df["episode_index"].to_numpy()):
            entry = spec.get(int(ep))
            if entry is None:
                continue
            length = int(lengths[pos])
            bounds = resolve_trim_bounds(entry, length, min_len)
            if bounds is None:


                if not (entry[0] == 0 and entry[1] in (None, length)):
                    n_skip += 1
                continue
            head, tail = bounds
            lengths[pos] = tail - head
            if head:
                row_off[pos] += head
                for c in cam_cols:
                    cam_off[c][pos] += head
            n_trim += 1

        if n_skip:
            logger.warning(
                "InternDataA1(%s): %d trim entries skipped — recorded total_frames disagrees "
                "with the manifest length, or the trim would leave < %d frames. A total_frames "
                "mismatch is what a STALE trim list looks like; regenerate it against this "
                "corpus.",
                self._dataset_id,
                n_skip,
                min_len,
            )
        if not n_trim:
            return eps_df

        eps_df = eps_df.copy()
        eps_df["length"] = lengths
        eps_df["_data_row_offset"] = row_off
        for c in cam_cols:
            eps_df[c] = cam_off[c]
        after = int(lengths.sum())
        logger.info(
            "InternDataA1(%s): trimmed %d/%d episodes, %d -> %d frames (-%.1f%%, %.2f h removed)",
            self._dataset_id,
            n_trim,
            len(eps_df),
            frames_before,
            after,
            100.0 * (frames_before - after) / max(frames_before, 1),
            (frames_before - after) / self._fps / 3600.0,
        )
        return eps_df.reset_index(drop=True)

    def _train_min_window_len(self) -> int:
        """Public implementation. Dataset-specific audit notes were removed."""




        return 2

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        """Public implementation. Dataset-specific audit notes were removed."""






        if actual_raw_len >= self._num_frames:
            return actual_raw_len
        return max(0, actual_raw_len - 1)

    def _check_stats_population(self, raw: dict, stats_path) -> None:
        """Public implementation. Dataset-specific audit notes were removed."""
























        pop = raw.get("population")
        rerun = (
            "Re-run python -m openwam.dataloader.utils.stats_computation."
            f"interndata_a1_stats_computation --dataset_dir {self._dataset_dir} "
            f"--stats_root {self._a1_stats_root}"
        )
        if not isinstance(pop, dict):
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} has no 'population' "
                f"block, so there is no way to tell whether it describes the rows this reader "
                f"loads (split, trimming and keep-bound all change the distribution). {rerun}."
            )








        if pop.get("split") != "train":
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} was computed over the "
                f"{pop.get('split')!r} split. Normalization statistics must describe the "
                f"training distribution — a val-derived file would scale training by numbers "
                f"drawn from data the model never fits. {rerun} --split train."
            )

        want_trim = bool(self._trim_csv)
        if bool(pop.get("trim_active")) != want_trim:
            state = "with" if pop.get("trim_active") else "without"
            mine = "is" if want_trim else "is not"
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} was computed {state} a "
                f"trim list but this reader {mine} trimming. Trimming removes the motionless "
                f"head/tail, so the two describe different distributions. {rerun}"
                + (f" --trim_csv {self._trim_csv}" if want_trim else "") + "."
            )




        want_keep = self._train_min_window_len()
        if want_trim and int(pop.get("min_keep", -1)) != int(want_keep):
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} used --min_keep="
                f"{pop.get('min_keep')} but this reader keeps episodes down to {want_keep} "
                f"frames. The same trim CSV under a different bound keeps a different set of "
                f"episodes. {rerun} --trim_csv {self._trim_csv} --min_keep {want_keep}."
            )

        buckets = pop.get("buckets")
        if not isinstance(buckets, list):
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} has no bucket list in "
                f"its 'population' block. {rerun}."
            )









        known = list(buckets) + list(pop.get("empty_buckets") or [])



        if self._match_bucket_key(dict.fromkeys(known), "stats population") is None:
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {stats_path} covers {len(known)} "
                f"buckets, none of them this one — the scan refused it (unreadable metadata, "
                f"or shards that disagree with the manifest) while it still loads the shared "
                f"per-embodiment file, so its actions would be scaled by other buckets' "
                f"numbers. {rerun}."
            )

    def _load_stats(self, info: dict):
        """Public implementation. Dataset-specific audit notes were removed."""
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._a1_stats_root / "meta" / f"stats_{self._embodiment}.json"
        if not stats_path.exists():





            raise FileNotFoundError(
                f"normalize_mode={self._normalize_mode!r} but stats file is missing: {stats_path}. "
                "Run python -m openwam.dataloader.utils.stats_computation."
                "interndata_a1_stats_computation --dataset_dir <extracted-root> "
                f"--stats_root {self._a1_stats_root}, or set normalize_mode=null."
            )
        with open(stats_path) as f:
            raw = json.load(f)

        self._check_stats_population(raw, stats_path)

        eef_raw = raw.get("eef", {})





        for k in ("mean", "std", "min", "max", "q01", "q99"):
            if k not in eef_raw:
                continue
            width = len(eef_raw[k])
            if width != _ACTION_DIM:
                raise ValueError(
                    f"InternData-A1 bucket {self._dataset_id}: 'eef' stats {k!r} width "
                    f"{width} in {stats_path} != expected {_ACTION_DIM}. "
                    "Re-run interndata_a1_stats_computation."
                )
        return materialize_eef_stats(
            eef_raw,
            self._normalize_mode,
            dim=_ACTION_DIM,
            strict_minmax=False,
            source_hint=(
                f"{stats_path}: eef.* — re-run python -m openwam.dataloader.utils."
                "stats_computation.interndata_a1_stats_computation"
            ),
        )

    def _post_init(self, info: dict) -> None:
        """Public implementation. Dataset-specific audit notes were removed."""






        if len(self._eps_df) == 0:
            return
        row = self._eps_df.iloc[0]
        try:
            table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
            pose_col = self._sides["state"][0][0]
            sample = np.stack(table.slice(0, 64).to_pandas()[pose_col].values).astype(np.float32)
        except Exception as e:
            logger.debug("%s(%s): quaternion probe skipped (%s)", self.DATASET_NAME, self._dataset_id, e)
            return
        assert_unit_quaternion(sample[:, 3:7])



    def _arm10(self, win, spec, n: int, grip_scale: float) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""




        pose_col, grip_col = spec
        pose = np.stack(win[pose_col].values[:n]).astype(np.float32)
        grip = np.stack(win[grip_col].values[:n]).astype(np.float32).reshape(n, 1) / grip_scale
        rot6d = quat_wxyz_to_rot6d(pose[:, 3:7])
        return np.concatenate([pose[:, 0:3], rot6d, grip], axis=-1)

    def _eef20(self, win, kind: str, n: int) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""





        left_spec, right_spec = self._sides[kind]
        out = np.zeros((n, _ACTION_DIM), dtype=np.float32)
        out[:, :ARM10_DIM] = self._arm10(win, left_spec, n, self._grip_scale[0])
        if right_spec is not None:
            out[:, ARM10_DIM:] = self._arm10(win, right_spec, n, self._grip_scale[1])
        return out

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Public implementation. Dataset-specific audit notes were removed."""
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win) -> np.ndarray:
        n = len(win)
        return self._normalize_array(self._eef20(win, "action", n))

    def _proprio_20d(self, win) -> np.ndarray:
        return self._normalize_array(self._eef20(win, "state", 1))



    @property
    def robot_type(self) -> str:
        return self._robot_type

    @property
    def embodiment(self) -> str:
        return self._embodiment

    @property
    def arm_layout(self) -> str:
        return self._arm_layout

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiInternDataA1Dataset



    @classmethod
    def from_config(cls, config, split: str = "train") -> Any:
        """Public implementation. Dataset-specific audit notes were removed."""







        from openwam.dataloader.utils import get_cfg as _get

        dataset_dir = _get(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError(f"{cls.__name__}: missing dataset_dir")
        root = Path(dataset_dir)

        common: Dict[str, Any] = {"split": split}
        for key in cls.CONFIG_KEYS:
            v = _get(config, key, _CONFIG_SENTINEL)
            if v is _CONFIG_SENTINEL:
                continue



            if v is None and key != "normalize_mode":
                continue
            common[key] = v



        stats_root = _get(config, "stats_root") or str(root)
        common["a1_stats_root"] = stats_root


        if (root / "meta" / "info.json").is_file():
            kwargs = dict(common)
            dataset_id = _get(config, "dataset_id")
            if dataset_id is not None:
                kwargs["dataset_id"] = dataset_id
            total_hours = _get(config, "total_hours")
            if total_hours is not None:
                kwargs["max_hours"] = float(total_hours)
                kwargs["subsample_seed"] = int(_get(config, "seed", 42))
            return cls(dataset_dir=str(root), **kwargs)

        if not root.is_dir():
            raise FileNotFoundError(f"{cls.__name__}: {root} does not exist")
        sub_dirs = discover_a1_buckets(root)
        if not sub_dirs:
            raise FileNotFoundError(
                f"{cls.__name__}: no buckets with meta/info.json found anywhere under {root}. "
                "Did the tar.gz archives get extracted? (see scripts/extract_interndata_a1_v30.sh)"
            )
        logger.info("%s.from_config: root mode, %d buckets under %s", cls.__name__, len(sub_dirs), root)

        def _per_bucket(sub: Path) -> Dict[str, Any]:



            try:
                return {"dataset_id": str(sub.relative_to(root))}
            except ValueError:
                return {"dataset_id": sub.name}

        return build_multibucket(
            cls,
            sub_dirs,
            common,
            base_seed=int(_get(config, "seed", 42)),
            total_hours=_get(config, "total_hours"),
            wrapper_cls=cls._multibucket_wrapper(),
            source_name=cls.__name__,
            per_bucket_kwargs=_per_bucket,
        )


class MultiInternDataA1Dataset(MultiLeRobotV3Reader):
    """Public implementation. Dataset-specific audit notes were removed."""

    def __init__(self, buckets):
        super().__init__(buckets)
        counts: Dict[str, int] = {}
        windows: Dict[str, int] = {}
        for b in self._buckets:
            emb = getattr(b, "embodiment", "unknown")
            counts[emb] = counts.get(emb, 0) + 1
            windows[emb] = windows.get(emb, 0) + len(b)
        logger.info(
            "MultiInternDataA1Dataset: %d buckets, %d windows | %s",
            len(self._buckets),
            len(self),
            ", ".join(f"{k}: {counts[k]} buckets/{windows[k]} windows" for k in sorted(counts)),
        )
        self._embodiment_bucket_counts = counts
        self._embodiment_window_counts = windows

    @property
    def embodiment_bucket_counts(self) -> Dict[str, int]:
        return dict(self._embodiment_bucket_counts)

    @property
    def embodiment_window_counts(self) -> Dict[str, int]:
        return dict(self._embodiment_window_counts)


__all__ = [
    "InternDataA1Dataset",
    "MultiInternDataA1Dataset",
    "ROBOT_TYPE_TO_EMBODIMENT",
    "GRIPPER_FULL_OPEN",
    "GRIPPER_ALT_FULL_OPEN",
    "detect_arm_layout",
    "embodiment_key",
    "discover_a1_buckets",
    "resolve_gripper_scale",
]
