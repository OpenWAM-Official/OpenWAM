"""Shared dataloader helpers.

This package gathers small, stateless utility code reused across the
dataloader readers (RoboCOIN, EgoDex, MixtureDataset, ...). It is
deliberately dependency-light so any reader can import it without
triggering heavy module side-effects.

Members:
  - ``get_cfg``           — config-getter used by every ``from_config``
  - ``utils.lerobotv3``   — LeRobot v3 per-bucket helpers (info.json,
                            episodes parquet, offsets, splits, video
                            sampling validation)
"""

from __future__ import annotations

from typing import Any


def get_cfg(config: Any, key: str, default: Any = None) -> Any:
    """Get ``config[key]`` (dict, DictConfig, or attribute-bearing namespace).

    Tries attribute access first (works for OmegaConf DictConfig with non-None
    values), then ``cfg.get(key, default)`` (works for plain dict and DictConfig
    when the attribute lookup returned None despite the key being present).
    Returns ``default`` when neither path resolves the key.
    """
    v = getattr(config, key, None)
    if v is not None:
        return v
    if hasattr(config, "get"):
        return config.get(key, default)
    return default
