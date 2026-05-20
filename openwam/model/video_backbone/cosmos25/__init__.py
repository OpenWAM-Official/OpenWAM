"""Cosmos-Predict2.5 video backbone for OpenWAM.

Wraps the NVIDIA Cosmos-Predict2.5 world-DiT (2B / 14B) behind the
:class:`openwam.model.video_backbone.VideoBackbone` contract so it plugs
into the existing WAM architectures without further changes.

The Cosmos source is taken as a runtime dependency (installed via the
``[cosmos25]`` optional extra in ``pyproject.toml``). The import is done
lazily inside :meth:`Cosmos25VideoBackbone.from_pretrained` so that
CPU-only CI keeps passing without the heavy dependency installed.
"""

from openwam.model.video_backbone.cosmos25.adapter import Cosmos25VideoBackbone
from openwam.model.video_backbone.cosmos25.scheduler import CosmosFlowSchedulerAdapter

__all__ = ["Cosmos25VideoBackbone", "CosmosFlowSchedulerAdapter"]
