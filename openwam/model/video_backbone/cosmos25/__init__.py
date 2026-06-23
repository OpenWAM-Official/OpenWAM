"""Cosmos-Predict2.5 video backbone helper package.

The public backbone class :class:`Cosmos25VideoBackbone` lives one level up in
``openwam/model/video_backbone/cosmos25_backbone.py`` (mirroring
``wan_backbone.py``). This package only holds its auxiliary "src" modules
(scheduler, pipeline wrapper/builder, text encoder, block split, deploy specs,
VAE utils). Nothing here is part of the externally-visible backbone interface;
the modules are imported by ``cosmos25_backbone.py`` (and each other) only.

The heavy ``cosmos_predict2`` import is deferred to
:func:`pipeline_builder.build_cosmos25_pipeline` so importing this package stays
CPU-only-CI safe.
"""

from openwam.model.video_backbone.cosmos25.scheduler import CosmosFlowSchedulerAdapter

__all__ = ["CosmosFlowSchedulerAdapter"]
