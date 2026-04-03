"""Continuous action representation (identity transform).

This is the default representation: raw continuous action vectors pass
through unchanged.  Normalization is handled separately by the ActionDiT's
action_mean/action_std buffers — this module does not normalize.

Using this representation is fully backward compatible with the existing
training and inference pipelines.
"""

from torch import Tensor

from open_wam.models.action_repr.base import BaseActionRepresentation


class ContinuousActionRepresentation(BaseActionRepresentation):
    """Identity representation for continuous actions.

    encode/decode are no-ops. latent_dim == native_dim.

    Args:
        action_dim: Raw action vector dimension (e.g. 7 for single arm,
            14 for bimanual).
    """

    def __init__(self, action_dim: int = 14):
        super().__init__()
        self._action_dim = action_dim

    def encode(self, actions: Tensor) -> Tensor:
        return actions

    def decode(self, latents: Tensor) -> Tensor:
        return latents

    @property
    def latent_dim(self) -> int:
        return self._action_dim

    @property
    def native_dim(self) -> int:
        return self._action_dim
