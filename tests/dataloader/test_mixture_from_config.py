"""Tests for MixtureDataset.from_config dispatch (dict vs list datasets).

These tests bypass the real RoboCOIN / EgoDex readers via a small
monkeypatched registry — the focus is from_config's discovery logic
and weight_strategy validation, not actual reader IO.
"""

from __future__ import annotations

import pytest

from openwam.dataloader import registry as registry_mod
from openwam.dataloader.mixture import MixtureDataset


@pytest.fixture
def fake_registry(monkeypatch, fake_dataset_factory):
    """Patch build_dataset to return FakeActionDataset for known type names."""

    def _fake_build_dataset(cfg, split="train"):
        # cfg might be dict-like; pull optional "n" (test param) for sizing.
        n = cfg.get("n", 10) if hasattr(cfg, "get") else getattr(cfg, "n", 10)
        return fake_dataset_factory(int(n), action_dim=20)

    monkeypatch.setattr(registry_mod, "build_dataset", _fake_build_dataset)
    return _fake_build_dataset


class TestFromConfigDictDatasets:
    def test_dict_style_picks_names_from_keys(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "robocoin": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "egodex": {
                    "type": "egodex",
                    "enabled": True,
                    "weight": 0.5,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg, split="train")
        assert m.names == ["robocoin", "egodex"]
        assert pytest.approx(m.weights[0]) == 1.0 / 1.5
        assert pytest.approx(m.weights[1]) == 0.5 / 1.5

    def test_disabled_skipped(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "alpha": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "beta": {
                    "type": "egodex",
                    "enabled": False,
                    "weight": 1.0,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg, split="train")
        assert m.names == ["alpha"]

    def test_all_disabled_raises_clear_error(self, fake_registry):
        # Regression (M1): when every sub-source is disabled the friendly
        # "all sub-datasets are disabled" error must surface, not an opaque
        # ThreadPoolExecutor "max_workers must be greater than 0".
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "alpha": {
                    "type": "robocoin",
                    "enabled": False,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "beta": {
                    "type": "egodex",
                    "enabled": False,
                    "weight": 1.0,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(RuntimeError, match="disabled"):
            MixtureDataset.from_config(cfg, split="train")


class TestFromConfigListDatasets:
    def test_list_style_derives_names_from_type(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": [
                {
                    "type": "robotwin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 10,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                {
                    "type": "robotwin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 20,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            ],
        }
        m = MixtureDataset.from_config(cfg, split="train")
        # Duplicate types get numeric suffix.
        assert m.names == ["robotwin", "robotwin_1"]


class TestWeightStrategyValidation:
    def test_unknown_strategy_raises(self, fake_registry):
        cfg = {
            "weight_strategy": "garbage",
            "datasets": {
                "x": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(ValueError, match="unknown weight_strategy"):
            MixtureDataset.from_config(cfg)

    def test_legacy_token_name_in_error_message(self, fake_registry):
        cfg = {
            "weight_strategy": "token",  # renamed to inverse_size
            "datasets": {
                "x": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(ValueError, match="inverse_size"):
            MixtureDataset.from_config(cfg)


class TestShapeSanity:
    def test_num_frames_mismatch_raises(self, fake_registry):
        cfg = {
            "weight_strategy": "manual",
            "datasets": {
                "a": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
                "b": {
                    "type": "egodex",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 49,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        with pytest.raises(ValueError, match="same 'num_frames'"):
            MixtureDataset.from_config(cfg)


class TestActionDimStrictMode:
    def test_no_action_dim_override_field_uses_strict(self, fake_registry):
        # No action_dim_override field => strict_action_dim=True path.
        # All fake datasets have action_dim=20, should pass.
        cfg = {
            "weight_strategy": "proportional",
            "datasets": {
                "a": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg)
        assert m.action_dim == 20

    def test_action_dim_override_field_present_uses_legacy(self, fake_registry):
        cfg = {
            "weight_strategy": "proportional",
            "action_dim_override": None,  # presence opts into legacy path
            "datasets": {
                "a": {
                    "type": "robocoin",
                    "enabled": True,
                    "weight": 1.0,
                    "n": 5,
                    "num_frames": 33,
                    "video_stride": 4,
                    "height": 384,
                    "width": 320,
                },
            },
        }
        m = MixtureDataset.from_config(cfg)
        assert m.action_dim == 20
