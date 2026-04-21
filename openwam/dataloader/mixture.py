"""Public implementation. Dataset-specific audit notes were removed."""






















































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
        """Public implementation. Dataset-specific audit notes were removed."""

















        if not datasets:
            raise ValueError("MixtureDataset requires at least one sub-dataset")

        self._datasets = list(datasets)
        self._seed = seed

        dims = [d.action_dim for d in self._datasets]
        if action_dim_override is not None:
            max_dim = max(dims)
            if action_dim_override < max_dim:
                raise ValueError(
                    f"action_dim_override ({action_dim_override}) must be >= "
                    f"max sub-dataset action_dim ({max_dim}); downscaling is not "
                    f"supported because larger-dim samples cannot be truncated safely."
                )
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
            raise ValueError(
                f"weights length ({len(weights)}) != datasets length ({len(self._datasets)})"
            )
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got: {list(weights)}")
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError(f"weights must sum to > 0, got: {list(weights)}")
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
        self._index_map: List[tuple] = []
        for di, (ds, w) in enumerate(zip(self._datasets, self._weights)):
            virtual_n = max(1, round(total_real * w))
            ds_len = len(ds)
            if ds_len == 0:
                raise ValueError(
                    f"Sub-dataset at index {di} ({type(ds).__name__}) is empty; "
                    f"cannot build mixture index map."
                )
            for vi in range(virtual_n):
                self._index_map.append((di, vi % ds_len))
        rng = np.random.RandomState(self._seed)
        rng.shuffle(self._index_map)





    def _compute_mixture_stats(self) -> Optional[dict]:
        """Public implementation. Dataset-specific audit notes were removed."""










        all_stats = []
        for ds, w in zip(self._datasets, self._weights):
            stats = ds.action_stats
            if stats is None:
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
        """Public implementation. Dataset-specific audit notes were removed."""











        di, si = self._index_map[idx]
        sample = self._datasets[di][si]

        action = sample["action"]
        if isinstance(action, torch.Tensor) and action.shape[-1] < self._action_dim:
            pad_size = self._action_dim - action.shape[-1]
            action = torch.nn.functional.pad(action, (0, pad_size))
            sample["action"] = action
            if "action_trajectory" in sample:



                traj = sample["action_trajectory"]
                if isinstance(traj, torch.Tensor) and traj.shape[-1] < self._action_dim:
                    sample["action_trajectory"] = torch.nn.functional.pad(
                        traj, (0, self._action_dim - traj.shape[-1])
                    )





            if "action_mask" in sample:
                mask = sample["action_mask"]
                if isinstance(mask, torch.Tensor) and mask.shape[-1] < self._action_dim:
                    mask_pad = self._action_dim - mask.shape[-1]
                    pad_value = False if mask.dtype == torch.bool else 0
                    sample["action_mask"] = torch.nn.functional.pad(
                        mask, (0, mask_pad), value=pad_value
                    )

        sample["_dataset_index"] = di
        return sample

    def __len__(self) -> int:
        """Public implementation. Dataset-specific audit notes were removed."""





        return len(self._index_map)

    @property
    def action_dim(self) -> int:
        """Public implementation. Dataset-specific audit notes were removed."""





        return self._action_dim

    @property
    def action_stats(self) -> Optional[dict]:
        """Public implementation. Dataset-specific audit notes were removed."""





        return self._action_stats_cache





    @property
    def datasets(self) -> List[BaseActionDataset]:
        """Public implementation. Dataset-specific audit notes were removed."""





        return self._datasets

    @property
    def weights(self) -> List[float]:
        """Public implementation. Dataset-specific audit notes were removed."""





        return self._weights

    def dataset_sample_counts(self) -> Dict[int, int]:
        """Public implementation. Dataset-specific audit notes were removed."""





        counts: Dict[int, int] = {}
        for di, _ in self._index_map:
            counts[di] = counts.get(di, 0) + 1
        return counts





    @classmethod
    def from_config(cls, config, split: str = "train") -> "MixtureDataset":
        """Public implementation. Dataset-specific audit notes were removed."""



























        from openwam.dataloader.registry import build_dataset

        def _get(cfg, key, default=None):
            """Public implementation. Dataset-specific audit notes were removed."""













            if hasattr(cfg, key):
                v = getattr(cfg, key)
                return v if v is not None else default
            if hasattr(cfg, "get"):
                try:
                    v = cfg.get(key, default)
                except TypeError:

                    v = cfg.get(key)
                return v if v is not None else default
            return default

        weight_strategy = _get(config, "weight_strategy", "manual")

        sub_datasets: List[BaseActionDataset] = []
        enabled_cfgs = []

        for sub_cfg in _get(config, "datasets", []):
            if not _get(sub_cfg, "enabled", True):
                logger.info(
                    "MixtureDataset: skipping disabled sub-dataset (type=%s)",
                    _get(sub_cfg, "type", "?"),
                )
                continue
            ds = build_dataset(sub_cfg, split=split)
            sub_datasets.append(ds)
            enabled_cfgs.append(sub_cfg)

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
