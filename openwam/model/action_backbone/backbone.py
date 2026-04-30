"""Self-contained action stream backbone ABC.

Mirrors VideoBackbone in role: owns its own scheduler and module(s), and knows
how to drive itself given the video DiT block loop. The architecture only
orchestrates calls — it does NOT reach into action internals.

Action computation is fundamentally interleaved with video DiT execution
(unlike video which has a clean prepare/run/finalize), so the interface here
is a 5-method block-loop adapter:

    prepare_state(noisy_actions, timestep, ...)            -> ActionState
    before_loop(vb, vstate, astate)                        -> (vstate, astate)
    run_block(block_id, vb, vstate, astate)                -> (vstate, astate)
    after_loop(vb, vstate, astate)                         -> (vstate, astate)
    extract_prediction(astate)                             -> Tensor

The architecture's single ``forward()`` drives this contract end-to-end and is
identical for ALL variants (cross_attn / self_attn / vanilla / moe).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

from torch import Tensor, nn

from openwam.model.action_backbone.scheduler import ActionScheduler

if TYPE_CHECKING:
    from openwam.model.base import ActionState, ExecutionPlan
    from openwam.model.video_backbone.adapter import BlockLoopState, VideoBackbone


class ActionBackbone(nn.Module, ABC):
    """Abstract base class for the action stream backbone.

    Concrete subclasses (ActionDiT, MoEExpertDiT, SharedVanillaActionBackbone)
    inherit from this directly — there is no wrapping layer. The state_dict
    therefore lives at ``action_backbone.<param-name>`` with no extra prefix.
    """

    def __init__(self):
        super().__init__()
        self.scheduler = ActionScheduler()

    # --- properties (each subclass must declare) ---

    @property
    @abstractmethod
    def execution_plan(self) -> "ExecutionPlan":
        """Canonical execution plan describing how this backbone interleaves
        with the video DiT block loop."""

    # Subclasses are required to expose the following as either attributes
    # or properties. They are not declared @abstractmethod here because
    # Python's ABC machinery cannot reconcile abstract properties with
    # instance attributes set in __init__ (e.g. registered buffers like
    # ``action_mean``). Subclass contract is documented:
    #
    #   action_dim:    int
    #   bridge_layers: tuple[int, ...]
    #     For BRIDGE_COLLECTION / SPLIT_SELF_ATTENTION: indices where bridge
    #     features are captured. For SPLIT_FFN: expert layer indices.
    #     For INTERLEAVED_WHOLE_BLOCK: empty tuple.
    #   action_mean:   Tensor — per-dim mean for denormalization
    #   action_std:    Tensor — per-dim std for denormalization

    @property
    def uses_proprioception(self) -> bool:
        """Whether this backbone consumes a ``proprio_state`` input."""
        return False

    # --- block-loop adapter ---

    @abstractmethod
    def prepare_state(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> "ActionState":
        """Initialize action state before the video DiT block loop.

        Padding handling: following FastWAM's MoT design, padding is NOT
        applied at the attention layer. Padded action / video tokens still
        participate in joint attention; ``action_is_pad`` / ``video_is_pad``
        are consumed only at loss time.
        """

    def before_loop(
        self,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Runs once before the per-block loop. Default: no-op.

        Subclasses that prepend/append action tokens onto the video sequence
        (vanilla, MoE) override this to call ``vb.inject_action_tokens(...)``.
        """
        return vstate, astate

    @abstractmethod
    def run_block(
        self,
        block_id: int,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Run one video DiT block plus this backbone's per-block work.

        The implementation MUST call ``vb.run_block(block_id, vstate)`` exactly
        once. Action-side work (bridge capture / expert apply / joint attn /
        residual write-back) happens before or after that call as appropriate.
        """

    def after_loop(
        self,
        vb: "VideoBackbone",
        vstate: "BlockLoopState",
        astate: "ActionState",
    ) -> Tuple["BlockLoopState", "ActionState"]:
        """Runs once after the per-block loop. Default: no-op.

        Subclasses that inject tokens in ``before_loop`` typically extract
        them here via ``vb.extract_action_tokens(...)``.
        """
        return vstate, astate

    @abstractmethod
    def extract_prediction(self, astate: "ActionState") -> Tensor:
        """Produce the final action noise prediction from accumulated state."""


__all__ = ["ActionBackbone"]
