"""Bridge encoder temporal contract from model yaml to dataloader cfg.

The encoder's temporal contract (``temporal_compression`` /
``causal_temporal``) lives on the model side of the config tree — it is
cross-checked against the constructed encoder spec inside
:meth:`BaseWAMArchitecture._init_video_backbone`. The dataloader's
``num_video_frames`` divisibility rule needs the same contract, so before
``build_dataset`` runs the launcher forwards both fields onto
``cfg.dataloader`` via :func:`apply_temporal_contract_bridge`.

Defaults reproduce the Wan VAE contract (``tc=4, causal=True``) bit-for-bit
so legacy yamls without the new fields keep working.

Kept in :mod:`openwam.train.utils` (rather than inside ``scripts/train.py``)
so unit tests can exercise the helper without importing the Hydra launcher.
"""

from __future__ import annotations

import logging

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def apply_temporal_contract_bridge(cfg) -> None:
    """Forward ``cfg.model.video_backbone.temporal_compression`` and
    ``causal_temporal`` onto ``cfg.dataloader``.

    Idempotent: if the dataloader already carries an out-of-sync value, we
    overwrite it but log a warning so the user notices the conflict instead
    of silently winning.
    """
    model_tc = cfg.model.video_backbone.get("temporal_compression", 4)
    model_causal = cfg.model.video_backbone.get("causal_temporal", True)

    def _read(key):
        if isinstance(cfg.dataloader, dict):
            return cfg.dataloader.get(key, None)
        return getattr(cfg.dataloader, key, None)

    existing_tc = _read("temporal_compression")
    if existing_tc is not None and existing_tc != model_tc:
        logger.warning(
            "Overriding cfg.dataloader.temporal_compression=%s with model-side value %s. "
            "This field is internal — set cfg.model.video_backbone.temporal_compression instead.",
            existing_tc,
            model_tc,
        )
    existing_causal = _read("causal_temporal")
    if existing_causal is not None and bool(existing_causal) != bool(model_causal):
        logger.warning(
            "Overriding cfg.dataloader.causal_temporal=%s with model-side value %s. "
            "This field is internal — set cfg.model.video_backbone.causal_temporal instead.",
            existing_causal,
            model_causal,
        )
    OmegaConf.update(cfg.dataloader, "temporal_compression", model_tc, force_add=True)
    OmegaConf.update(cfg.dataloader, "causal_temporal", model_causal, force_add=True)


__all__ = ["apply_temporal_contract_bridge"]
