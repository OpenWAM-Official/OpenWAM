"""Qwen3-VL-2B-Instruct backbone for tri_system architecture."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from openwam.model.vlm_backbone.base import VlmBackbone

logger = logging.getLogger(__name__)


class Qwen3VLBackbone(VlmBackbone):
    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype = torch.bfloat16,
        load_pretrained: bool = True,
        max_length: int = 512,
    ):
        super().__init__()
        self._max_length = max_length
        try:
            from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Qwen3VL requires transformers>=4.50 with Qwen3VLForConditionalGeneration. "
                "Install via `pip install -U transformers`."
            ) from e

        self.dtype = dtype
        self._checkpoint_path = checkpoint_path
        self.processor = None
        if checkpoint_path:
            try:
                self.processor = AutoProcessor.from_pretrained(checkpoint_path)
            except Exception:
                if load_pretrained:
                    raise
        # ``trust_remote_code`` is intentionally NOT set: ``Qwen3VLForConditionalGeneration``
        # is imported directly from upstream ``transformers``, so no remote
        # ``modeling_*.py`` is needed. Leaving ``trust_remote_code=True`` would
        # let a config/model directory under ``checkpoint_path`` execute
        # arbitrary Python at load time — an unnecessary code-execution
        # surface in training and deployment.
        if load_pretrained:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint_path, dtype=dtype)
        else:
            # ``_from_config`` is a transformers private API whose signature
            # has shifted across 4.5x versions (some accept ``torch_dtype``,
            # earlier ones do not). Use the public PreTrainedModel
            # constructor + ``.to(dtype=...)`` instead — equivalent
            # semantically (random init from cfg, no weights downloaded)
            # and stable across releases.
            cfg = AutoConfig.from_pretrained(checkpoint_path)
            self.vlm_model = Qwen3VLForConditionalGeneration(cfg).to(dtype=dtype)

    @property
    def hidden_size(self) -> int:
        return int(self.vlm_model.config.text_config.hidden_size)

    def get_submodule(self, name: str) -> nn.Module | None:
        return getattr(self, name, None)

    def _pad_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        config_pad = getattr(getattr(self.vlm_model, "config", None), "pad_token_id", None)
        if config_pad is not None:
            return int(config_pad)
        raise ValueError(
            "Qwen3VLBackbone: cannot determine pad_token_id from processor.tokenizer "
            "or model.config. Pass a processor whose tokenizer has pad_token_id set, "
            "or set pad_token_id on the model config."
        )

    def format_prompt(self, prompt: str) -> str:
        if self.processor is None:
            return prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def prepare_vlm_inputs(self, prompts: list[str], images: list[Any]) -> dict[str, torch.Tensor]:
        """Build Qwen3-VL processor inputs from OpenWAM prompt + first-frame images."""
        if self.processor is None:
            raise RuntimeError("Qwen3VLBackbone.prepare_vlm_inputs requires a loaded AutoProcessor.")
        texts = [self.format_prompt(prompt) for prompt in prompts]
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=self._max_length
        )
        return {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    @property
    def device(self) -> torch.device:
        try:
            return next(self.vlm_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def batch_vlm_inputs(self, vlm_inputs: dict | list[dict]) -> dict[str, torch.Tensor]:
        if isinstance(vlm_inputs, dict):
            # Only tensor entries are forwarded to the VLM; non-tensor metadata (e.g.
            # processor attributes, image grids stored as lists) is dropped. The list
            # input path below is stricter — it validates required tensors explicitly.
            return {
                key: value for key, value in vlm_inputs.items() if isinstance(value, torch.Tensor) and value is not None
            }

        if not isinstance(vlm_inputs, list) or not vlm_inputs:
            raise ValueError("vlm_inputs must be a non-empty dict or list of dicts.")

        input_ids_list = [item["input_ids"] for item in vlm_inputs]
        attention_mask_list = [
            item.get("attention_mask", torch.ones_like(item["input_ids"], dtype=torch.long)) for item in vlm_inputs
        ]
        for idx, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            if ids.ndim != 2:
                raise ValueError(f"vlm_inputs[{idx}]['input_ids'] must be 2D [B, L], got shape {tuple(ids.shape)}")
            if mask.shape != ids.shape:
                raise ValueError(
                    f"vlm_inputs[{idx}]['attention_mask'] must match input_ids shape "
                    f"{tuple(ids.shape)}, got {tuple(mask.shape)}"
                )
        max_seq_len = max(ids.shape[1] for ids in input_ids_list)
        pad_token_id = self._pad_token_id()

        padded_ids = []
        padded_masks = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad = max_seq_len - ids.shape[1]
            if pad > 0:
                padded_ids.append(F.pad(ids, (0, pad), value=pad_token_id))
                padded_masks.append(F.pad(mask, (0, pad), value=0))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)

        batched = {
            "input_ids": torch.cat(padded_ids, dim=0),
            "attention_mask": torch.cat(padded_masks, dim=0),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [item.get(key) for item in vlm_inputs]
            if any(value is not None for value in values):
                if not all(value is not None for value in values):
                    raise ValueError(f"Mixed missing/non-missing {key} in vlm_inputs list.")
                batched[key] = torch.cat(values, dim=0)
        return batched

    def extract_features(self, vlm_inputs: dict | list[dict]) -> torch.Tensor:
        """Extract Qwen3-VL hidden states for tri-system understanding tokens.

        Mirrors Motus's VLM path but uses the upstream Qwen3-VL model wrapper
        for image embedding insertion, DeepStack routing, and RoPE index
        construction. Freeze/no_grad policy is controlled centrally by
        ``BaseWAMArchitecture.freeze_modules``, which wraps frozen sub-trees'
        ``forward`` in ``torch.no_grad`` — this method does no freeze
        inspection of its own.

        **Alignment assumption** (Qwen2-VL family convention): the Qwen3-VL
        processor expands ``<image>`` placeholders in ``input_ids`` to one
        token per visual patch before the model forward, so
        ``hidden_states.shape[1] == input_ids.shape[1] == attention_mask.shape[1]``.
        This lets us derive ``UnderstandingState.und_mask`` directly from
        ``vlm_inputs["attention_mask"]``.
        ``UnderstandingExpert.prepare_state`` validates this contract at
        runtime and raises with a clear error if it ever breaks (e.g. if a
        future upstream version inserts visual patches at the embedding stage
        without growing ``input_ids``).
        """
        batch = self.batch_vlm_inputs(vlm_inputs)
        device = self.device
        model_inputs = {}
        for key, value in batch.items():
            if key in {"pixel_values", "pixel_values_videos"}:
                value = value.to(device=device, dtype=self.dtype)
            else:
                value = value.to(device=device)
            model_inputs[key] = value

        outputs = self.vlm_model.model(
            **model_inputs,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Copy the VLM checkpoint into the deploy dir so deploy is self-contained
        (no dependency on the training-time checkpoint_path). Idempotent. Invoked by
        the architecture's save_assets_for_deployment, same hook as video backbones."""
        import os
        import shutil

        if not self._checkpoint_path:
            return
        dest = os.path.join(output_dir, "vlm_backbone")
        if os.path.exists(dest):
            return
        shutil.copytree(self._checkpoint_path, dest)
        logger.info("Copied VLM checkpoint to %s", dest)
