"""Live Cosmos-Reason1-7B text encoder for the CosmosPredict25 video backbone.

Constructed once at training start; produces **pre-projection**
``(B, L=512, 100352) bf16`` tensors. The backbone's preprocess detects the
``crossattn_proj_in_channels=100352`` match and applies the DiT-owned
``net.crossattn_proj`` Linear(100352→1024)+GELU to land at the post-projection
context the DiT consumes.

Why a plain Python class (NOT an ``nn.Module``)?

- ``CosmosPredict25VideoBackbone`` stores us as ``self.text_encoder = <instance>``.
  Mirroring the ``Wan2pt1VAEInterface`` precedent, the wrapper reaches in for
  our ``self.model`` (the inner ``Qwen2_5_VLForConditionalGeneration``
  ``nn.Module``) and registers it as ``reason1``. Result: our 16 GB of
  weights DO flow into the unified state_dict / safetensors via that single
  registered child — but the wrapper class itself stays a plain attribute so
  the tokenizer + dtype/device tracking don't trip ``__setattr__`` or
  ``state_dict()``.
- ``CosmosPredict25VideoBackbone.get_submodule('text_encoder')`` already filters with
  ``isinstance(attr, nn.Module)``, so returning ``None`` for us is the
  existing ABC-compliant behaviour (§12.3.2 ``base.py:417-424``).
- DeepSpeed ZeRO-3 partitioning of the 16 GB frozen Qwen weights is avoided
  by the trainer's ``zero.Init(enabled=False)`` guard
  (``openwam_trainer.py:74-75``) — params constructed inside that scope stay
  replicated; the inner Qwen module sits there because the architecture is
  built inside the disabled scope.

Device/dtype movement goes through ``adapter.py::_move_cosmos_reason1``
(mirrors ``_move_cosmos_vae``) instead of ``nn.Module.to(...)``.

Deploy-time construction: the ``Reason1LiveTextEncoder.from_empty``
classmethod builds the inner Qwen2.5-VL on the meta device using
``accelerate.init_empty_weights``, reading only ``config.json`` +
``tokenizer.json`` from ``<ckpt_dir>/reason1/``.
``BaseWAMArchitecture.load_checkpoint`` then populates the meta tensors from
the unified safetensors. This is the symmetric path to the empty-DiT and
empty-VAE shells in ``pipeline_builder.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Copied verbatim from upstream
# cosmos_predict2/_src/predict2/text_encoders/text_encoder.py (via the
# third_party/cosmos-predict2.5 submodule) so we don't import third_party
# at runtime.
_COSMOS_REASON1_SYSTEM_PROMPT = "You are a helpful assistant who will provide prompts to an image generator."

# Upstream pad/truncation target.
_NUM_EMBEDDING_PADDING_TOKENS = 512

# Reason1 / Qwen2.5-VL-7B geometry: 28 transformer layers × hidden_size=3584
# → full_concat produces 100352 channels (matches Cosmos `crossattn_proj_in_channels`).
_REASON1_NUM_TRANSFORMER_LAYERS = 28
_REASON1_HIDDEN_SIZE = 3584
_REASON1_FULL_CONCAT_DIM = _REASON1_NUM_TRANSFORMER_LAYERS * _REASON1_HIDDEN_SIZE  # 100352


def _tokenize_with_chat_template(tokenizer, prompt: str):
    """Match upstream's chat-template wrap: system prompt + user content."""
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": _COSMOS_REASON1_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]
    # HF chat-template returns the formatted *string*; tokenize after.
    chat_string = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(
        chat_string,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    input_ids = enc["input_ids"][0].tolist()

    pad_id = int(tokenizer.pad_token_id)
    if len(input_ids) < _NUM_EMBEDDING_PADDING_TOKENS:
        pad_len = _NUM_EMBEDDING_PADDING_TOKENS - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
    else:
        input_ids = input_ids[:_NUM_EMBEDDING_PADDING_TOKENS]
    return input_ids


def _mean_normalize_along_last(hs):
    """Per-token, per-layer normalize: (x - mean) / (std + 1e-8) along channel dim."""
    return (hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8)


class Reason1LiveTextEncoder:
    """Live Cosmos-Reason1-7B text encoder. Plain Python class — NOT an ``nn.Module``.

    Returns pre-projection ``(B, L=512, 100352) bf16`` embeddings; the
    backbone's preprocess applies the DiT-owned ``net.crossattn_proj``
    Linear(100352→1024)+GELU on the way into the block loop.
    """

    L_PAD = _NUM_EMBEDDING_PADDING_TOKENS  # 512
    OUT_DIM = _REASON1_FULL_CONCAT_DIM  # 100352
    NUM_LAYERS = _REASON1_NUM_TRANSFORMER_LAYERS  # 28
    HIDDEN = _REASON1_HIDDEN_SIZE  # 3584

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: Any = None,
    ) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_dir():
            raise FileNotFoundError(
                f"Cosmos-Reason1 weights not found at {ckpt_path}. "
                "Download via `huggingface-cli download nvidia/Cosmos-Reason1-7B` "
                "or point `video_backbone.text_encoder_path` at the bundle root."
            )

        # Lazy upstream import — keeps this module importable on CPU CI without
        # `transformers` installed for the heavy Qwen-VL classes.
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Always load to a single device. ``device_map="auto"`` would shard the
        # 16 GB Qwen2.5-VL across every visible GPU via accelerate, but the
        # rest of the architecture (DiT, VAE) loads on a single device and
        # ``set_dtype_device`` / ``adapter._move_cosmos_reason1`` only moves a
        # single device worth — leaving stragglers on cuda:1+ that then trip
        # "Expected all tensors to be on the same device" at the forward call.
        # When the caller doesn't pin a device, stage on CPU and let
        # ``set_dtype_device`` move us to the right GPU explicitly.
        target = device if device is not None else "cpu"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(ckpt_path),
            torch_dtype=dtype,
            device_map={"": target},
        ).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self._validate_geometry(model)

        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        # Track the device explicitly so `to()` can stay in sync without
        # touching every parameter on each invocation.
        first_param = next(model.parameters(), None)
        if first_param is not None:
            self.device = first_param.device
        elif device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        logger.info(
            "Reason1LiveTextEncoder ready (ckpt=%s, dtype=%s, device=%s)",
            ckpt_path,
            self.dtype,
            self.device,
        )

    @classmethod
    def _validate_geometry(cls, model: Any) -> None:
        """Fail fast on Reason1 variants that don't match the expected geometry."""
        cfg = getattr(model, "config", None)
        # transformers ≥5 ``Qwen2_5_VLConfig`` exposes the language-model
        # geometry under ``config.text_config``. ``or cfg`` is a defensive
        # fallback for the (unsupported) case where ``text_config`` is missing
        # or ``None``.
        text_cfg = getattr(cfg, "text_config", None) or cfg
        hidden = getattr(text_cfg, "hidden_size", None)
        layers = getattr(text_cfg, "num_hidden_layers", None)
        if hidden != cls.HIDDEN:
            raise ValueError(
                f"Reason1 hidden_size={hidden} != expected {cls.HIDDEN}. "
                "Reason1LiveTextEncoder targets Cosmos-Reason1-7B / Qwen2.5-VL-7B only."
            )
        if layers != cls.NUM_LAYERS:
            raise ValueError(
                f"Reason1 num_hidden_layers={layers} != expected {cls.NUM_LAYERS}. "
                "Reason1LiveTextEncoder targets Cosmos-Reason1-7B / Qwen2.5-VL-7B only."
            )

    def __call__(self, prompts: Union[str, List[str]]) -> Tensor:
        """Encode prompts → ``(B, 512, 100352) bf16`` pre-projection on ``self.device``."""
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("Reason1LiveTextEncoder received an empty prompts list.")

        batch_ids = [_tokenize_with_chat_template(self.tokenizer, str(p)) for p in prompts]
        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states  # tuple len NUM_LAYERS + 1
        expected = self.NUM_LAYERS + 1
        if len(hidden_states) != expected:
            raise RuntimeError(
                f"Reason1 hidden_states has {len(hidden_states)} entries; "
                f"expected {expected} (1 embed + {self.NUM_LAYERS} transformer)."
            )

        normalized = [_mean_normalize_along_last(h) for h in hidden_states[1:]]
        full_concat = torch.cat(normalized, dim=-1).to(dtype=self.dtype)
        if full_concat.shape[-1] != self.OUT_DIM:
            raise RuntimeError(
                f"Reason1 full_concat dim mismatch: got {full_concat.shape[-1]}, expected {self.OUT_DIM}."
            )
        return full_concat

    @classmethod
    def from_empty(
        cls,
        artifact_dir: Union[str, Path],
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: Any = None,
    ) -> "Reason1LiveTextEncoder":
        """Construct a shell with meta-device weights for deploy-time loading.

        Reads only the small structural files (``config.json``,
        ``tokenizer.json``, etc.) from ``artifact_dir``; the inner Qwen2.5-VL
        ``nn.Module`` is built under ``accelerate.init_empty_weights`` so its
        parameters live on the ``meta`` device until
        :meth:`BaseWAMArchitecture.load_checkpoint` materialises them from the
        unified safetensors.

        ``artifact_dir`` is typically ``<ckpt_dir>/reason1/`` — populated by
        :func:`openwam.model.video_backbone.cosmos_predict25.component_specs.copy_cosmos_predict25_artifacts`
        at training save time. Use the regular constructor (which loads
        weights from the full Cosmos-Reason1 bundle) on the training path.
        """
        artifact_dir = Path(artifact_dir)
        if not artifact_dir.is_dir():
            raise FileNotFoundError(
                f"Reason1 artifact dir not found at {artifact_dir}. Deploy "
                "needs `<ckpt_dir>/reason1/` populated with config.json + "
                "tokenizer.json (see copy_cosmos_predict25_artifacts). Either save a "
                "ckpt with this change applied, or point `text_encoder_path` "
                "at a full Cosmos-Reason1-7B bundle."
            )

        # Lazy upstream imports — match the eager constructor's pattern so
        # CPU CI doesn't pay the heavy transformers import on the cache path.
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir), trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        config = AutoConfig.from_pretrained(str(artifact_dir), trust_remote_code=True)
        config.torch_dtype = dtype
        with init_empty_weights():
            model = Qwen2_5_VLForConditionalGeneration._from_config(config, torch_dtype=dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        cls._validate_geometry(model)

        # Build an instance without going through ``__init__`` (which would
        # try to load weights from disk). The wrapper's attribute contract
        # is: self.model, self.tokenizer, self.dtype, self.device.
        self = cls.__new__(cls)
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        # On meta device until load_checkpoint moves us. The trainer/deploy
        # path then calls ``adapter._move_cosmos_reason1`` via
        # ``set_dtype_device`` to materialise on the real device.
        if device is not None:
            self.device = torch.device(device) if not isinstance(device, torch.device) else device
        else:
            self.device = torch.device("meta")

        logger.info(
            "Reason1LiveTextEncoder.from_empty ready (artifact_dir=%s, dtype=%s, device=%s)",
            artifact_dir,
            self.dtype,
            self.device,
        )
        return self

    def to(self, *, dtype: torch.dtype = None, device: Any = None) -> "Reason1LiveTextEncoder":
        """Move the inner ``nn.Module`` + update cached dtype/device.

        Called from ``CosmosPredict25VideoBackbone.set_dtype_device`` via
        ``adapter._move_cosmos_reason1``. Returns ``self`` to mirror
        ``nn.Module.to`` semantics.
        """
        kwargs: dict = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
            self.dtype = dtype
        if device is not None:
            kwargs["device"] = device
            self.device = torch.device(device) if not isinstance(device, torch.device) else device
        if kwargs:
            self.model.to(**kwargs)
        return self


__all__ = ["Reason1LiveTextEncoder"]
