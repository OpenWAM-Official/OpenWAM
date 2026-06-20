"""Action-stream backbone ABCs.

Two independent roots, because the two architecture families drive the action
stream through genuinely different contracts and share no common base:

- :class:`SharedActionBackbone` — for SharedBackbone (vanilla / MoE): action
  tokens ride the video DiT sequence, so the contract is ``encode`` /
  ``encode_state`` / ``decode``. Holds the shared action I/O modules. No latent
  decoder.
- :class:`ActionDiTBackbone` — for the DualSystem / tri-system ActionDiT: a
  standalone action transformer driven either by bridge cross-attention
  (``forward``) or by the MoT joint-attention loop
  (``prepare_state`` / ``pre_attn_at_layer`` / ``post_attn_at_layer`` /
  ``extract_prediction``). Declares that contract as abstract. May own a
  latent->action decoder.

Both roots inherit ``nn.Module, ABC`` directly — concrete subclasses inherit so
the state_dict lives at ``action_backbone.<param>`` with no wrapping prefix. The
common members (scheduler, shift-action, set_dtype_device, action_mean/std
buffer contract) are written once in each root; the duplication is the price of
the two families being independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from openwam.model.action_backbone.components import (
    DEFAULT_ACTION_DECODER_HIDDEN_DIM,
    ActionEncoder,
    ActionOutputMLP,
    StateEncoder,
)
from openwam.model.action_backbone.scheduler import ActionScheduler


class SharedActionBackbone(nn.Module, ABC):
    """ABC for SharedBackbone action backbones (vanilla / MoE).

    Action tokens project directly into ``video_dim`` and are concatenated into
    the video DiT sequence; the architecture's forward runs the video block loop
    and calls these helpers at the right moments. The contract is:

        encode(noisy_actions, timestep) -> tokens (+ extras, MoE only)
        encode_state(proprio)     -> state token | None
        decode(action_tail)             -> action_prediction

    Shared action I/O (``input_proj`` / ``state_encoder`` / ``action_output_head``
    / normalization buffers) is created by two ordered helpers so subclasses can
    interleave their own modules (e.g. MoE's expert blocks) without perturbing
    parameter-init order.

    Buffer contract: every concrete backbone registers the persistent
    denormalization buffers ``action_mean`` / ``action_std`` (shape
    ``(action_dim,)``) — ``nn.Module.__getattr__`` exposes them as attributes,
    so the architecture reads ``action_backbone.action_mean`` / ``.action_std``
    through this contract rather than reaching into a subclass.
    """

    def __init__(
        self,
        action_dim: int,
        video_dim: int,
        *,
        max_action_len: int,
        action_decoder_hidden_dim: Optional[int] = None,
        use_proprioception: bool = False,
        state_dim: int = 0,
    ):
        super().__init__()
        self.scheduler = ActionScheduler()
        self._shift_action = None
        self._action_dim = int(action_dim)
        self._video_dim = int(video_dim)
        self._max_action_len = int(max_action_len)
        self._action_decoder_hidden_dim = int(action_decoder_hidden_dim or DEFAULT_ACTION_DECODER_HIDDEN_DIM)
        self._use_proprioception = bool(use_proprioception)
        self.state_dim = int(state_dim or 0)
        if self._use_proprioception and self.state_dim <= 0:
            raise ValueError("use_proprioception=True requires state_dim > 0 for SharedBackbone state tokens.")

    def _init_action_input(self) -> None:
        """Create action input projection + optional state encoder (first init step)."""
        self.input_proj = ActionEncoder(self._action_dim, self._video_dim)
        self.state_encoder = StateEncoder(self.state_dim, self._video_dim) if self._use_proprioception else None

    def _init_action_output(self) -> None:
        """Create the action output head + normalization buffers (last init step)."""
        self.action_output_head = ActionOutputMLP(self._video_dim, self._action_decoder_hidden_dim, self._action_dim)
        self.register_buffer("action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(self._action_dim), persistent=True)

    @property
    def shift_action(self):
        """Optional α-shift for the action scheduler — single source of truth read by
        the architecture for both training and inference. ``None`` (the default) falls
        back to the scheduler template. Symmetric to ``VideoBackbone.shift_video``."""
        return self._shift_action

    @property
    def bridge_layers(self) -> tuple[int, ...]:
        """Video DiT blocks where the action stream couples to the video stream.
        Default none. SharedMoE overrides with its expert-FFN layers — the
        architecture iterates the video block loop and applies the expert FFN at
        these block ids."""
        return getattr(self, "_bridge_layers", ())

    def apply_expert(self, layer_id: int, x_action, t_mod):
        """Apply the per-layer SharedMoE expert FFN to action tokens. Default
        raises — the architecture only calls this for layer_id in bridge_layers,
        so vanilla (no expert FFN) is never reached."""
        raise NotImplementedError(
            f"{type(self).__name__} has no expert FFN (bridge_layers carries no expert correction)."
        )

    def set_dtype_device(self, dtype, device) -> None:
        """Move action backbone params/buffers to (dtype, device)."""
        self.to(dtype=dtype, device=device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def uses_proprioception(self) -> bool:
        return self._use_proprioception

    def encode_state(self, proprio: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._use_proprioception:
            return None
        if proprio is None:
            raise ValueError("SharedBackbone use_proprioception=True requires `proprio`.")
        assert self.state_encoder is not None
        return self.state_encoder(proprio)

    def decode(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, video_dim) action tail -> (B, T, action_dim) prediction."""
        return self.action_output_head(action_tokens)

    @abstractmethod
    def encode(self, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        """Project noisy actions into ``video_dim`` space.

        Vanilla returns ``(B, T, video_dim)`` tokens; MoE returns
        ``(tokens, t_mod)`` since it also builds the expert-FFN AdaLN modulation.
        """
        raise NotImplementedError


class ActionDiTBackbone(nn.Module, ABC):
    """ABC for standalone DualSystem / tri-system action transformers (ActionDiT).

    The action stream is a separate transformer that couples to the video stream
    either by bridge cross-attention (``forward``) or by the MoT joint
    self-attention loop. Both coupling paths plus the geometry the MoT driver
    validates against the video backbone are declared abstract here.

    Buffer contract: registers ``action_mean`` / ``action_std`` (shape
    ``(action_dim,)``) — same denormalization contract as SharedActionBackbone.
    """

    def __init__(self):
        super().__init__()
        self.scheduler = ActionScheduler()
        self._shift_action = None

    @property
    def shift_action(self):
        """Optional α-shift for the action scheduler — single source of truth read by
        the architecture for both training and inference. ``None`` (the default) falls
        back to the scheduler template; concrete backbones may set ``self._shift_action``
        in ``__init__``. Symmetric to ``VideoBackbone.shift_video``."""
        return self._shift_action

    @property
    def bridge_layers(self) -> tuple[int, ...]:
        """Video DiT blocks where the action stream couples to the video stream.
        Default none. ActionDiT overrides with its bridge cross-attn layers — the
        architecture iterates the video block loop and cross-attends at these
        block ids."""
        return getattr(self, "_bridge_layers", ())

    def apply_expert(self, layer_id: int, x_action, t_mod):
        """ActionDiT couples by cross-attention, not expert FFN — never called.
        Kept for a uniform coupling contract; raises if reached."""
        raise NotImplementedError(f"{type(self).__name__} has no expert FFN (couples by bridge cross-attention).")

    def set_dtype_device(self, dtype, device) -> None:
        """Move action backbone params/buffers to (dtype, device)."""
        self.to(dtype=dtype, device=device)

    @property
    def uses_proprioception(self) -> bool:
        """Whether this backbone consumes a ``proprio`` input. Default False."""
        return False

    @property
    def has_latent_decoder(self) -> bool:
        """Whether this backbone owns a latent->action decoder. Default False.

        The architecture is decoder-agnostic: it only asks the action backbone
        whether one exists (for fail-fast) and calls ``decode_latent_to_action``.
        """
        return False

    def decode_latent_to_action(self, latent, proprio=None):
        """Decode predicted latent action into real action, or None if no decoder.

        ``latent`` is the clean latent reconstructed from the backbone's own
        velocity prediction (carries its gradient). ``proprio`` is the optional
        normalized current state, used only when the decoder conditions on it."""
        return None

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """joint_cross_attn coupling: standalone action prediction from per-layer bridges."""
        raise NotImplementedError

    @abstractmethod
    def prepare_state(self, *args, **kwargs):
        """MoT coupling: build the per-layer action state consumed by the driver."""
        raise NotImplementedError

    @abstractmethod
    def pre_attn_at_layer(self, *args, **kwargs):
        """MoT coupling: first half of an action block (norm + AdaLN + Q/K/V + RoPE)."""
        raise NotImplementedError

    @abstractmethod
    def post_attn_at_layer(self, *args, **kwargs):
        """MoT coupling: second half of an action block (gate + context cross-attn + FFN)."""
        raise NotImplementedError

    @abstractmethod
    def extract_prediction(self, *args, **kwargs) -> torch.Tensor:
        """MoT coupling: decode the final action hidden state into a prediction."""
        raise NotImplementedError


__all__ = ["ActionDiTBackbone", "SharedActionBackbone"]
