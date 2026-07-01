"""VlmBackbone ABC: the contract a vision-language feature backbone exposes to
the host architecture (tri_system).

A VLM backbone is a frozen feature extractor — it turns ``(prompt, image)`` pairs
into hidden states that the architecture's understanding expert projects into
trimodal joint attention. The package ``__init__`` re-exports this alongside the
registry + factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class VlmBackbone(ABC, nn.Module):
    """Swappable vision-language feature backbone.

    Subclasses MUST implement ``hidden_size`` / ``prepare_vlm_inputs`` /
    ``batch_vlm_inputs`` / ``extract_features``. Construction loads the frozen
    pretrained VLM in ``__init__`` (no two-stage ``from_pretrained``), so the
    registry forwards yaml fields straight to the subclass constructor.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Channel dim of the hidden states returned by :meth:`extract_features`."""

    @abstractmethod
    def prepare_vlm_inputs(self, prompts: list[str], images: list[Any]) -> dict[str, Tensor]:
        """``(prompts, first-frame images)`` -> one processor tensor dict."""

    @abstractmethod
    def batch_vlm_inputs(self, vlm_inputs: dict | list[dict]) -> dict[str, Tensor]:
        """Collate one dict (passthrough of tensor entries) or a list of
        per-sample dicts (pad + concat) into a single batched tensor dict.

        Public because the host architecture collates per-sample dicts here
        before :meth:`extract_features` (tri_system batches the dataloader's
        list-of-dicts)."""

    @abstractmethod
    def extract_features(self, vlm_inputs: dict | list[dict]) -> Tensor:
        """``vlm_inputs`` -> ``(B, L, hidden_size)`` frozen hidden states.

        Freeze / no_grad policy is owned centrally by
        ``BaseWAMArchitecture.freeze_modules`` (it wraps frozen sub-trees'
        ``forward`` in ``torch.no_grad``); this method does no freeze inspection.
        """

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Match the lightweight backbone interface used by BaseWAMArchitecture."""
        self.dtype = dtype
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Copy non-weight deploy artifacts into ``output_dir`` so deploy is
        self-contained. Orchestrated by the architecture before ``config.yaml`` is
        written, same hook as video backbones. Default no-op."""
