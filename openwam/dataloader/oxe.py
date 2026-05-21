"""OXE (Open X-Embodiment) dataset adapter (LeRobot v3 format).

All OXE single-arm datasets share a common action layout::

    raw 7-D = action.cartesian_position(6) + action.gripper_position(1)
            = xyz(3) + euler_xyz(3) + gripper(1)

The EEF transform produces the canonical 20-D dual-arm-aligned action::

    xyz(3) + rot6d(6) + gripper(1) + zeros(10) = 20-D

``action_dim_mask = [True] * 10 + [False] * 10`` is attached to each sample
so the loss can ignore the second-arm zero-padding.

The yaml schema lists OXE subsets explicitly (DROID, Bridge V2, …), each with
its own ``dataset_dir`` / camera layout / fps. A ``defaults`` block supplies
shared values that every subset inherits unless it overrides them
(see ``configs/dataloader/oxe.yaml``). Different subsets in the same yaml may
have completely different camera names — the per-subset ``target_camera`` /
``camera_layout`` keep them isolated. Subsets with ``enabled: false`` are
skipped, useful for placing data that's still being prepared (e.g. Bridge
before its ``meta/info.json`` and ``videos/`` are generated).

When more than one subset is enabled, :class:`OXEDataset` is a thin
:class:`~openwam.dataloader.lerobot_v3_base.MultiTaskLeRobot3Dataset` over
the per-subset :class:`~openwam.dataloader.lerobot_v3_base.LeRobot3Dataset`
instances; each subset uses its own ``meta/eef_stats.json`` so
``denormalize_action(..., task_name=sample["task_name"])`` recovers the
correct action range at inference.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from openwam.dataloader.lerobot_v3_base import LeRobot3Dataset, MultiTaskLeRobot3Dataset
from openwam.dataloader.transforms.rotation import RotationType, convert_rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OXE action layout (single-arm, padded to dual-arm width)
# ---------------------------------------------------------------------------

OXE_ACTION_FIELDS = ["action.cartesian_position", "action.gripper_position"]
OXE_DEFAULT_CAMERA = "observation.images.exterior_1_left"
OXE_CAMERAS_MULTIVIEW = [
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]
OXE_CAMERA_LAYOUT = list(OXE_CAMERAS_MULTIVIEW)
OXE_FPS = 15
OXE_SINGLE_ARM_DIM = 10  # xyz(3) + rot6d(6) + gripper(1)
OXE_ACTION_DIM_EEF = 20  # padded to dual-arm 20-D


def _make_oxe_eef_transform() -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """Return ``(transform_fn, action_dim_mask)`` for OXE single-arm EEF.

    Raw 7-D layout::

        [0:3]   xyz
        [3:6]   euler RPY (XYZ convention)
        [6:7]   gripper

    Output 20-D layout::

        [0:3]    xyz_L
        [3:9]    rot6d_L     (from euler_xyz via rotation matrix)
        [9]      gripper_L
        [10:20]  zeros       second-arm padding to align with dual-arm 20-D

    Returns:
        transform_fn:    ``(T, 7) → (T, 20)`` float32.
        action_dim_mask: ``(20,)`` bool — True for the 10 real dims, False
                         for the padded second-arm dims.
    """
    mask = np.zeros(OXE_ACTION_DIM_EEF, dtype=bool)
    mask[:OXE_SINGLE_ARM_DIM] = True

    def _transform(actions: np.ndarray) -> np.ndarray:
        xyz = actions[:, 0:3]
        euler_rpy = actions[:, 3:6]
        gripper = actions[:, 6:7]
        rot6d = convert_rotation(euler_rpy, RotationType.EULER_XYZ, RotationType.ROTATION_6D)
        single_arm = np.concatenate([xyz, rot6d, gripper], axis=-1)  # (T, 10)
        pad = np.zeros((actions.shape[0], OXE_SINGLE_ARM_DIM), dtype=np.float32)
        return np.concatenate([single_arm, pad], axis=-1).astype(np.float32)  # (T, 20)

    return _transform, mask


def _flatten_camera_layout(raw_layout) -> Optional[List[str]]:
    """Accept flat list or 2-D list-of-lists; flatten to a single string list."""
    if raw_layout is None:
        return None
    raw = list(raw_layout)
    if not raw:
        return []
    if isinstance(raw[0], str):
        return raw
    return [c for row in raw for c in list(row)]


def _cfg_get(cfg, key, default=None):
    """Read a key from a dict / DictConfig / object, returning default for None."""
    v = getattr(cfg, key, None)
    if v is None and hasattr(cfg, "get"):
        try:
            v = cfg.get(key, None)
        except TypeError:
            v = None
    return v if v is not None else default


def _to_plain_dict(cfg) -> dict:
    """Convert dict / OmegaConf node / SimpleNamespace into a plain ``dict``.

    Used so ``{**defaults, **subset}`` merging works regardless of whether
    Hydra hands us OmegaConf nodes, plain Python dicts, or SimpleNamespace
    (e.g. from tests).
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    if hasattr(cfg, "items"):
        return dict(cfg.items())
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    raise TypeError(f"cannot convert {type(cfg).__name__} to dict")


def _is_lerobot_v3_root(path: str) -> bool:
    """Return True iff ``path/meta/info.json`` exists."""
    return os.path.isfile(os.path.join(path, "meta", "info.json"))


# ---------------------------------------------------------------------------
# OXEDataset
# ---------------------------------------------------------------------------


class OXEDataset(MultiTaskLeRobot3Dataset):
    """Multi-subset OXE adapter.

    Built from a yaml ``subsets:`` list (see ``configs/dataloader/oxe.yaml``);
    each enabled entry becomes its own
    :class:`~openwam.dataloader.lerobot_v3_base.LeRobot3Dataset` and they're
    concatenated through this class. The OXE EEF transform (7-D → 20-D padded)
    and ``action_dim_mask = [True]*10 + [False]*10`` are shared across subsets
    when ``action_format='eef'``.

    With normalization active (default ``normalize_mode='min-max'`` +
    ``action_format='eef'``) every enabled subset is forced onto a single
    shared union stats npy (see ``from_config`` below), so at inference
    ``ds.denormalize_action(action)`` and
    ``ds.denormalize_action(action, task_name=sample["task_name"])``
    return the same answer — any subset's stats inverts every other
    subset's action correctly. ``task_name`` is still the recommended
    form for explicitness and forward-compat with the legacy per-subset
    mode (where each subset has its own ``action_stats_path``); in that
    mode ``denormalize_action(action)`` without ``task_name`` raises
    ``ValueError`` to prevent silently using the wrong subset's stats.

    See :class:`~openwam.dataloader.lerobot_v3_base.LeRobot3Dataset` for
    shared frame-sampling / multiview args.
    """

    @classmethod
    def from_config(cls, config, split: str = "train") -> "OXEDataset":
        """Build from a Hydra/OmegaConf config.

        Required yaml structure (flat — top-level fields are the shared
        defaults that every subset inherits unless it overrides them; this
        matches the AgiBot / Galaxea yaml style and avoids clashing with
        Hydra's reserved ``defaults:`` list keyword)::

            type: oxe
            num_frames: 33
            height: 192
            ... shared values ...
            subsets:
              - name: droid_1.0.1
                enabled: true
                dataset_dir: /path/to
                fps: 15
                target_camera: ...
                camera_layout: [...]
              - name: bridge
                enabled: false
                ...

        Each subset's effective config is ``{**top_level, **subset}`` (subset
        keys win). ``enabled: false`` subsets are skipped; an empty subsets
        list, all-disabled list, or duplicate ``name`` raise a clear error.
        """
        # Top-level fields (minus type/subsets) act as shared defaults.
        # mixture.yaml entries also pass `enabled` / `weight` down here — those
        # are mixture-level controls and just get ignored by LeRobot3Dataset.
        top_level = _to_plain_dict(config)
        top_level.pop("type", None)
        raw_subsets = top_level.pop("subsets", None)
        # `split` may live at the top level too (e.g. agibot.yaml has split: train);
        # the kwarg passed to from_config wins.
        top_level.pop("split", None)
        defaults = top_level

        if raw_subsets is None or len(raw_subsets) == 0:
            raise ValueError(
                "OXEDataset: yaml requires non-empty 'subsets' list "
                "(see configs/dataloader/oxe.yaml)"
            )

        subsets = [_to_plain_dict(s) for s in raw_subsets]
        enabled = [s for s in subsets if s.get("enabled", True)]
        if not enabled:
            raise RuntimeError(
                f"OXEDataset: all {len(subsets)} subset(s) have enabled=false; "
                "nothing to load. Set enabled=true on at least one subset."
            )

        names = [s.get("name") for s in enabled]
        if any(n is None or n == "" for n in names):
            raise ValueError(f"OXEDataset: every enabled subset needs a 'name' field; got {names}")
        if len(set(names)) != len(names):
            raise ValueError(
                f"OXEDataset: duplicate subset names {names}; "
                "MultiTaskLeRobot3Dataset routes per-subset stats by name, names must be unique."
            )

        # Shared EEF transform / mask for action_format=eef. action_format is treated
        # as a family-level decision (every OXE subset uses the same 7-D → 20-D layout),
        # so we read it from the first subset's merged config.
        merged0 = {**defaults, **enabled[0]}
        action_format = merged0.get("action_format")
        if action_format == "eef":
            action_transform, action_dim_mask = _make_oxe_eef_transform()
            action_out_dim = OXE_ACTION_DIM_EEF
        else:
            action_transform = None
            action_dim_mask = None
            action_out_dim = None

        # ---- Resolve shared union stats (RoboTwin pattern) ----
        # OXE differs from AgiBot/Galaxea in that each subset has its own
        # ``dataset_dir`` (DROID and Bridge are separate roots, possibly on
        # different filesystems). We promote ``action_stats_path`` from the
        # top-level yaml: if user specified it explicitly, use it; otherwise
        # default to ``<first_subset_dir>/../oxe_eef_union_stats.npy``.
        # All enabled subsets are then forced to point at this single union
        # npy so deploy gets one well-defined normalizer file.
        _nm = defaults.get("normalize_mode")
        _nm_active = _nm is not None and (
            not isinstance(_nm, str) or _nm.lower() not in ("none", "null", "")
        )
        _top_stats = defaults.get("action_stats_path")
        if _nm_active and action_transform is not None:
            # Read dataset_dir from the *merged* (defaults+subset) view so users
            # can legitimately set dataset_dir at the top level even though OXE
            # usually configures it per-subset.
            merged_dirs = [
                {**defaults, **s}.get("dataset_dir", "") for s in enabled
            ]
            if _top_stats is None:
                _first_dir = merged_dirs[0].rstrip("/") if merged_dirs else ""
                if _first_dir:
                    _top_stats = os.path.join(
                        os.path.dirname(_first_dir) or ".", "oxe_eef_union_stats.npy"
                    )
            if _top_stats is not None:
                if not os.path.exists(_top_stats):
                    subset_roots = [d for d in merged_dirs if d]
                    logger.info(
                        "[normalizer] No pre-computed OXE union stats at %s; "
                        "computing across %d enabled subset(s) (this may take a while)",
                        _top_stats,
                        len(subset_roots),
                    )
                    from openwam.dataloader.lerobot_v3_stats_computation import (
                        compute_multitask_oxe_stats,
                        save_union_stats_npy,
                    )

                    stats = compute_multitask_oxe_stats(subset_roots)
                    save_union_stats_npy(_top_stats, stats)
                    logger.info("[normalizer] Saved OXE union stats → %s", _top_stats)
                else:
                    logger.info("[normalizer] Using pre-computed OXE union stats: %s", _top_stats)
                # Force every enabled subset onto the union file (override any
                # per-subset action_stats_path; the union supersedes them).
                for s in enabled:
                    s["action_stats_path"] = _top_stats

        datasets: List[LeRobot3Dataset] = []
        for sub in enabled:
            merged = {**defaults, **sub}
            name = merged["name"]
            dataset_dir = merged.get("dataset_dir")
            if not dataset_dir:
                raise ValueError(f"OXE subset {name!r}: missing required 'dataset_dir'")
            if not _is_lerobot_v3_root(dataset_dir):
                raise FileNotFoundError(
                    f"OXE subset {name!r}: {dataset_dir} is not a LeRobot v3 root "
                    f"(missing meta/info.json). For data still being prepared "
                    f"(e.g. Bridge missing meta/videos), set 'enabled: false' on "
                    f"this subset in the yaml, or generate meta/info.json + videos/ first."
                )

            camera_layout = _flatten_camera_layout(merged.get("camera_layout"))
            if camera_layout is None:
                camera_layout = list(OXE_CAMERA_LAYOUT)
            target_camera = merged.get("target_camera") or OXE_DEFAULT_CAMERA

            try:
                datasets.append(
                    LeRobot3Dataset(
                        data_root=dataset_dir,
                        task_name=name,
                        action_fields=list(merged.get("action_fields", OXE_ACTION_FIELDS)),
                        target_camera=target_camera,
                        cameras=list(camera_layout),
                        camera_layout=camera_layout,
                        fps=int(merged.get("fps", OXE_FPS)),
                        num_frames=int(merged.get("num_frames", 33)),
                        height=int(merged.get("height", 480)),
                        width=int(merged.get("width", 640)),
                        split=split,
                        val_ratio=float(merged.get("val_ratio", 0.1)),
                        seed=int(merged.get("seed", 42)),
                        multiview=bool(merged.get("multiview", False)),
                        normalize_mode=merged.get("normalize_mode", "none"),
                        window_stride=int(merged.get("window_stride", 1)),
                        video_stride=int(merged.get("video_stride", 4)),
                        repeat=int(merged.get("repeat", 1)),
                        num_val_samples=int(
                            merged.get("max_val_samples", merged.get("num_val_samples", 0))
                        ),
                        action_stats_path=merged.get("action_stats_path"),
                        action_transform=action_transform,
                        action_out_dim=action_out_dim,
                        action_dim_mask=action_dim_mask,
                    )
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
                # Match AgiBot/Galaxea behavior: a bad subset surfaces as a per-subset
                # warning, the rest still load. If every subset fails we raise below.
                # Narrowed from `Exception` so call-site bugs (TypeError/AttributeError)
                # propagate instead of being silently downgraded to warnings.
                logger.warning("Skipping OXE subset %r: %s", name, e)

        if not datasets:
            raise RuntimeError(
                f"OXEDataset: all {len(enabled)} enabled subset(s) failed to load; "
                "see warnings above for per-subset errors."
            )

        if len(datasets) > 1:
            logger.info(
                "OXEDataset [%s]: loaded %d subset(s): %s",
                split,
                len(datasets),
                [ds.task_name for ds in datasets],
            )

        return cls(datasets)
