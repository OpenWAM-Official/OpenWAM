"""Public implementation. Dataset-specific audit notes were removed."""






















































import copy
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from openwam.dataloader.base_dataset import BaseActionDataset

logger = logging.getLogger(__name__)


class MixtureDataset(BaseActionDataset):
    """Public implementation. Dataset-specific audit notes were removed."""
















    def __init__(
        self,
        datasets: Sequence[BaseActionDataset],
        weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        action_dim_override: Optional[int] = None,
    ):
        if not datasets:
            raise ValueError("MixtureDataset requires at least one sub-dataset")

        self._datasets = list(datasets)
        self._seed = seed

        dims = [d.action_dim for d in self._datasets]
        if action_dim_override is not None:
            self._action_dim = action_dim_override
        elif len(set(dims)) == 1:
            self._action_dim = dims[0]
        else:
            self._action_dim = max(dims)
            logger.warning(
                "Sub-datasets have different action dims %s; padding to max=%d. "
                "Set action_dim_override explicitly to suppress this warning.",
                dims,
                self._action_dim,
            )

        if weights is None:
            weights = [float(len(d)) for d in self._datasets]
        if len(weights) != len(self._datasets):
            raise ValueError(f"weights length ({len(weights)}) != datasets length ({len(self._datasets)})")
        total_w = sum(weights)
        self._weights = [w / total_w for w in weights]

        self._build_index_map()
        self._action_stats_cache = self._compute_mixture_stats()

        logger.info(
            "MixtureDataset: %d sub-datasets, total %d virtual samples, weights=%s",
            len(self._datasets),
            len(self),
            [f"{w:.3f}" for w in self._weights],
        )

    def _build_index_map(self):
        """Public implementation. Dataset-specific audit notes were removed."""




        total_real = sum(len(d) for d in self._datasets)
        parts: List[np.ndarray] = []
        for di, (ds, w) in enumerate(zip(self._datasets, self._weights)):
            virtual_n = max(1, round(total_real * w))
            sample_idx = np.arange(virtual_n, dtype=np.int64) % len(ds)
            di_arr = np.full(virtual_n, di, dtype=np.int64)
            parts.append(np.stack([di_arr, sample_idx], axis=1))
        combined = np.concatenate(parts, axis=0)
        rng = np.random.RandomState(self._seed)
        perm = rng.permutation(len(combined))
        self._index_map = combined[perm]

    def _compute_mixture_stats(self) -> Optional[dict]:
        all_stats = []
        for i, (ds, w) in enumerate(zip(self._datasets, self._weights)):
            stats = ds.action_stats
            if stats is None:
                logger.warning(
                    "MixtureDataset: sub-dataset %d (%s) has action_stats=None — "
                    "mixture stats disabled; action_norm_mode will be a no-op. "
                    "If using EEF action_format, provide a pre-computed stats file via action_stats_path.",
                    i,
                    getattr(ds, "task_name", type(ds).__name__),
                )
                return None
            all_stats.append((stats, w))

        mean = np.zeros(self._action_dim, dtype=np.float64)
        for stats, w in all_stats:
            m = stats["mean"].astype(np.float64)
            if len(m) < self._action_dim:
                m = np.pad(m, (0, self._action_dim - len(m)))
            mean += w * m[: self._action_dim]

        var = np.zeros(self._action_dim, dtype=np.float64)
        for stats, w in all_stats:
            s = stats["std"].astype(np.float64)
            m = stats["mean"].astype(np.float64)
            if len(s) < self._action_dim:
                s = np.pad(s, (0, self._action_dim - len(s)), constant_values=1e-3)
                m = np.pad(m, (0, self._action_dim - len(m)))
            var += w * (s[: self._action_dim] ** 2 + (m[: self._action_dim] - mean) ** 2)

        std = np.maximum(np.sqrt(var), 1e-3)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def __getitem__(self, idx: int) -> dict:
        di, si = self._index_map[idx]
        di, si = int(di), int(si)
        sample = self._datasets[di][si]

        action_traj = sample.get("action")
        if (
            action_traj is not None
            and isinstance(action_traj, torch.Tensor)
            and action_traj.shape[-1] < self._action_dim
        ):
            pad_size = self._action_dim - action_traj.shape[-1]
            sample["action"] = torch.nn.functional.pad(action_traj, (0, pad_size))

        sample["_dataset_index"] = di
        return sample

    def __len__(self) -> int:
        return len(self._index_map)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        return self._action_stats_cache

    @property
    def datasets(self) -> List[BaseActionDataset]:
        return self._datasets

    @property
    def weights(self) -> List[float]:
        return self._weights

    def dataset_sample_counts(self) -> Dict[int, int]:

        di_col = self._index_map[:, 0]
        counts = np.bincount(di_col, minlength=len(self._datasets))
        return {int(i): int(c) for i, c in enumerate(counts)}

    @classmethod
    def from_config(cls, config, split: str = "train") -> "MixtureDataset":
        """Public implementation. Dataset-specific audit notes were removed."""







        from concurrent.futures import ThreadPoolExecutor

        from openwam.dataloader.registry import build_dataset

        def _get(cfg, key, default=None):
            v = getattr(cfg, key, None)
            if v is not None:
                return v
            if hasattr(cfg, "get"):
                return cfg.get(key, default)
            return default

        def _copy_with_default(cfg, key, value):
            if value is None or _get(cfg, key) is not None:
                return cfg
            if hasattr(cfg, "items"):
                copied = {k: v for k, v in cfg.items()}
                copied[key] = value
                return copied
            copied = copy.copy(cfg)
            setattr(copied, key, value)
            return copied

        weight_strategy = _get(config, "weight_strategy", "manual")
        split_manifest = _get(config, "split_manifest")
        expose_bridging_meta = _get(config, "expose_bridging_meta")
        normalization = _get(config, "normalization")
        statistics_path = _get(config, "statistics_path")
        image_resize_mode = _get(config, "image_resize_mode")
        image_short_side = _get(config, "image_short_side")

        dataset_cfgs = _get(config, "datasets", [])
        enabled_cfgs = [
            _copy_with_default(
                _copy_with_default(c, "split_manifest", split_manifest),
                "expose_bridging_meta",
                expose_bridging_meta,
            )
            for c in dataset_cfgs
            if _get(c, "enabled", True)
        ]
        enabled_cfgs = [
            _copy_with_default(
                _copy_with_default(
                    _copy_with_default(
                        _copy_with_default(c, "normalization", normalization),
                        "statistics_path",
                        statistics_path,
                    ),
                    "image_resize_mode",
                    image_resize_mode,
                ),
                "image_short_side",
                image_short_side,
            )
            for c in enabled_cfgs
        ]
        for skipped in (c for c in dataset_cfgs if not _get(c, "enabled", True)):
            logger.info(
                "MixtureDataset: skipping disabled sub-dataset (type=%s)",
                _get(skipped, "type", "?"),
            )





        n_workers = min(len(enabled_cfgs), 16)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            sub_datasets = list(pool.map(lambda c: build_dataset(c, split=split), enabled_cfgs))

        if not sub_datasets:
            raise RuntimeError("MixtureDataset: all sub-datasets are disabled or failed to load")

        if weight_strategy == "uniform":
            weights = [1.0] * len(sub_datasets)
            logger.info("MixtureDataset: weight_strategy=uniform, all weights set to 1.0")
        elif weight_strategy == "token":




            weights = []
            for ds, sub_cfg in zip(sub_datasets, enabled_cfgs):
                nf = float(_get(sub_cfg, "num_frames", 49))
                weights.append(1.0 / max(len(ds) * nf, 1.0))
            logger.info(
                "MixtureDataset: weight_strategy=token, raw weights=%s",
                [f"{w:.2e}" for w in weights],
            )
        else:
            weights = [float(_get(sub_cfg, "weight", 1.0)) for sub_cfg in enabled_cfgs]
            logger.info("MixtureDataset: weight_strategy=manual, weights=%s", weights)

        return cls(
            datasets=sub_datasets,
            weights=weights,
            seed=int(_get(config, "seed", 42)),
            action_dim_override=_get(config, "action_dim_override"),
        )
