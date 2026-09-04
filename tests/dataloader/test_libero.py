from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.registry import DATASET_REGISTRY


def test_canonical_config_and_registry_use_native_action_reader() -> None:
    config = yaml.safe_load(Path("configs/dataloader/libero.yaml").read_text(encoding="utf-8"))
    assert config["type"] == "libero"
    assert config["action_mode"] == "eef"
    assert DATASET_REGISTRY["libero"] is LiberoDataset


def test_reader_hard_preserves_rot6d_under_custom_stats() -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    # Deliberately non-identity rotation stats: the reader must still leave
    # rot6d untouched rather than relying only on the generated stats artifact.
    reader._normalization_stats = {
        "min": np.zeros(10, dtype=np.float32),
        "max": np.ones(10, dtype=np.float32),
    }
    raw = np.array(
        [[0.25, 0.5, 0.75, 0.998, 0.02, -0.04, 0.01, 0.999, 0.02, -0.5]],
        dtype=np.float32,
    )
    normalized = reader._normalize_array(raw)
    np.testing.assert_array_equal(normalized[:, 3:9], raw[:, 3:9])
    expected_non_rot = np.clip(raw[:, [0, 1, 2, 9]] * 2.0 - 1.0, -1.0, 1.0)
    np.testing.assert_allclose(normalized[:, [0, 1, 2, 9]], expected_non_rot)


def test_reader_rejects_incomplete_compatibility_stats(tmp_path) -> None:
    reader = object.__new__(LiberoDataset)
    reader._normalize_mode = "min-max"
    reader._source_stats_path = str(tmp_path / "normalization_stats.npy")
    reader._dataset_dir = tmp_path
    reader._raw_action_dim = 10
    np.save(reader._source_stats_path, {"eef": {}}, allow_pickle=True)
    with pytest.raises(KeyError, match="eef"):
        reader._load_stats({})
