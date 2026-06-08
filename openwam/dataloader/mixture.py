"""Public implementation. Dataset-specific audit notes were removed."""







































































import copy
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from openwam.dataloader.bases import BaseDataset

logger = logging.getLogger(__name__)


class MixtureDataset(BaseDataset):
    """Public implementation. Dataset-specific audit notes were removed."""

























    def __init__(
        self,
        datasets: Sequence[BaseDataset],
        weights: Optional[Sequence[float]] = None,
        seed: int = 42,
        action_dim_override: Optional[int] = None,
        names: Optional[Sequence[str]] = None,
        strict_action_dim: bool = False,
    ):
        if not datasets:
            raise ValueError("MixtureDataset requires at least one sub-dataset")

        self._datasets = list(datasets)
        self._base_seed = seed
        self._seed = seed
        self._strict_action_dim = strict_action_dim

        if names is None:
            self._names = [f"source_{i}" for i in range(len(self._datasets))]
        else:
            names = list(names)
            if len(names) != len(self._datasets):
                raise ValueError(f"names length ({len(names)}) != datasets length ({len(self._datasets)})")
            self._names = [str(n) for n in names]

        dims = [d.action_dim for d in self._datasets]
        if strict_action_dim:
            if len(set(dims)) != 1:
                raise ValueError(
                    "MixtureDataset: all enabled sub-datasets must share the same action_dim, "
                    f"got {dict(zip(self._names, dims))}. Align action_dim across sub-sources "
                    "before mixing (e.g. pad in the per-source reader). To restore the legacy "
                    "auto-pad behavior, re-add ``action_dim_override`` to the yaml."
                )
            self._action_dim = dims[0]
        else:




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
        if any(w < 0 for w in weights):
            raise ValueError(f"MixtureDataset: negative weights are not allowed: {weights}")
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("MixtureDataset: all sub-sources are empty or zero-weight; nothing to sample.")
        self._weights = [w / total_w for w in weights]

        self._build_index_map()

        logger.info(
            "MixtureDataset: %d sub-datasets, total %d virtual samples, weights={%s}",
            len(self._datasets),
            len(self),
            ", ".join(f"{n}: {w:.3f}" for n, w in zip(self._names, self._weights)),
        )

    def set_epoch(self, epoch: int) -> None:
        """Public implementation. Dataset-specific audit notes were removed."""































        self._seed = self._base_seed + int(epoch) * 7919
        self._build_index_map()

    def _build_index_map(self):
        """Public implementation. Dataset-specific audit notes were removed."""







        total_real = sum(len(d) for d in self._datasets)
        parts: List[np.ndarray] = []
        for di, (ds, w) in enumerate(zip(self._datasets, self._weights)):
            n_real = len(ds)
            if n_real == 0 or w <= 0.0:




                logger.warning(
                    "MixtureDataset: skipping source '%s' (len=%d, weight=%.4f) — not sampled.",
                    self._names[di],
                    n_real,
                    w,
                )
                continue
            virtual_n = max(1, round(total_real * w))
            sample_idx = (np.arange(virtual_n, dtype=np.int64) % n_real).astype(np.int32)
            di_arr = np.full(virtual_n, di, dtype=np.int32)
            parts.append(np.stack([di_arr, sample_idx], axis=1))
        if not parts:
            raise RuntimeError("MixtureDataset: all sub-sources are empty or zero-weight; nothing to sample.")
        combined = np.concatenate(parts, axis=0)
        rng = np.random.RandomState(self._seed)
        perm = rng.permutation(len(combined))
        self._index_map = combined[perm]

    def __getitem__(self, idx: int) -> dict:
        di, si = self._index_map[idx]
        di, si = int(di), int(si)
        sample = self._datasets[di][si]





        if not self._strict_action_dim:
            action_traj = sample.get("action")
            if (
                action_traj is not None
                and isinstance(action_traj, torch.Tensor)
                and action_traj.shape[-1] < self._action_dim
            ):
                pad_size = self._action_dim - action_traj.shape[-1]
                sample["action"] = torch.nn.functional.pad(action_traj, (0, pad_size))

        sample["_dataset_index"] = di
        sample["_dataset_name"] = self._names[di]
        return sample

    def __len__(self) -> int:
        return len(self._index_map)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def normalization_stats(self) -> Optional[dict]:





        return None

    @property
    def datasets(self) -> List[BaseDataset]:
        return self._datasets

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def names(self) -> List[str]:
        return list(self._names)

    def get_dataset(self, name: str) -> BaseDataset:
        """Public implementation. Dataset-specific audit notes were removed."""



        try:
            idx = self._names.index(name)
        except ValueError:
            raise KeyError(f"MixtureDataset: no sub-dataset named '{name}'. Known: {self._names}") from None
        return self._datasets[idx]

    def dataset_sample_counts(self) -> Dict[str, int]:
        """Public implementation. Dataset-specific audit notes were removed."""
        di_col = self._index_map[:, 0]
        counts = np.bincount(di_col, minlength=len(self._datasets))
        return {self._names[i]: int(c) for i, c in enumerate(counts)}

    @classmethod
    def from_config(cls, config, split: str = "train") -> "MixtureDataset":
        """Public implementation. Dataset-specific audit notes were removed."""
















































        from concurrent.futures import ThreadPoolExecutor

        from openwam.dataloader.registry import build_dataset
        from openwam.dataloader.utils import get_cfg as _get

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

        def _normalize_entries(datasets_cfg):
            """Public implementation. Dataset-specific audit notes were removed."""





            if datasets_cfg is None:
                return []


            if hasattr(datasets_cfg, "items") and hasattr(datasets_cfg, "keys") and not isinstance(datasets_cfg, list):
                return [(str(name), c) for name, c in datasets_cfg.items()]
            entries = []
            seen = {}
            for i, c in enumerate(datasets_cfg):
                base = str(_get(c, "type", f"source_{i}"))

                n = seen.get(base, 0)
                seen[base] = n + 1
                name = base if n == 0 else f"{base}_{n}"
                entries.append((name, c))
            return entries

        weight_strategy = _get(config, "weight_strategy", "proportional")
        split_manifest = _get(config, "split_manifest")



        all_entries = _normalize_entries(_get(config, "datasets"))
        enabled_entries = []
        for name, c in all_entries:
            if _get(c, "enabled", True):
                enabled_entries.append((name, _copy_with_default(c, "split_manifest", split_manifest)))
            else:
                logger.info(
                    "MixtureDataset: skipping disabled sub-dataset '%s' (type=%s)",
                    name,
                    _get(c, "type", "?"),
                )

        enabled_names = [n for n, _ in enabled_entries]
        enabled_cfgs = [c for _, c in enabled_entries]




        shape_fields = ("num_frames", "video_stride", "height", "width")
        for key in shape_fields:
            present = [(n, _get(c, key)) for n, c in zip(enabled_names, enabled_cfgs) if _get(c, key) is not None]
            uniq = set(v for _, v in present)
            if len(uniq) > 1:
                raise ValueError(f"MixtureDataset: enabled sub-sources must share the same '{key}', got {present}")





        if not enabled_cfgs:
            raise RuntimeError("MixtureDataset: all sub-datasets are disabled or failed to load")
        n_workers = min(len(enabled_cfgs), 16)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            sub_datasets = list(pool.map(lambda c: build_dataset(c, split=split), enabled_cfgs))

        valid_strategies = ("manual", "uniform", "inverse_size", "proportional")
        if weight_strategy not in valid_strategies:
            raise ValueError(
                f"MixtureDataset: unknown weight_strategy={weight_strategy!r}. Valid options: {valid_strategies}."
            )

        if weight_strategy == "uniform":

            weights = [1.0] * len(sub_datasets)
            logger.info("MixtureDataset: weight_strategy=uniform, all weights set to 1.0")
        elif weight_strategy == "inverse_size":




            weights = []
            for ds, sub_cfg in zip(sub_datasets, enabled_cfgs):
                nf = float(_get(sub_cfg, "num_frames", 49))
                weights.append(1.0 / max(len(ds) * nf, 1.0))
            logger.info(
                "MixtureDataset: weight_strategy=inverse_size, raw weights=%s",
                [f"{w:.2e}" for w in weights],
            )
        elif weight_strategy == "proportional":


            weights = [float(len(ds)) for ds in sub_datasets]
            logger.info("MixtureDataset: weight_strategy=proportional, raw weights=%s (= sub-source sizes)", weights)
        else:
            weights = [float(_get(sub_cfg, "weight", 1.0)) for sub_cfg in enabled_cfgs]
            logger.info("MixtureDataset: weight_strategy=manual, weights=%s", weights)







        try:
            has_override_field = "action_dim_override" in config
        except TypeError:
            has_override_field = hasattr(config, "action_dim_override")

        return cls(
            datasets=sub_datasets,
            weights=weights,
            seed=int(_get(config, "seed", 42)),
            action_dim_override=_get(config, "action_dim_override") if has_override_field else None,
            names=enabled_names,
            strict_action_dim=not has_override_field,
        )
