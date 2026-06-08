"""Abstract base class for every dataset reader in this package."""

from abc import ABC, abstractmethod

import torch


class BaseDataset(ABC, torch.utils.data.Dataset):
    """Abstract root for openwam's dataset readers.

    The contract is intentionally minimal: subclasses implement
    ``__getitem__`` and ``__len__``; that's it. Per-source metadata such
    as the action dimensionality (``action_dim``) or normalization stats
    (``normalization_stats``) is NOT part of the base contract — the
    project happens to train action-conditioned models today, but the
    dataset abstraction shouldn't bake that in. Consumers that need such
    metadata (``MixtureDataset``, the trainer's normalizer loading)
    discover it via duck-typed ``getattr`` with sensible defaults, so
    future pure-video or pure-prompt subclasses stay clean subclasses
    without lying about action fields.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        Conventional keys when the sample carries action supervision (used
        by every reader currently in this package, but not required by
        the base contract):

            video:           List[PIL.Image]
            action:          Tensor (T, action_dim)
            action_mask:     Tensor (T,) bool
            video_mask:      Tensor (num_video_frames,) bool
            proprio:         Tensor (1, state_dim)
            proprio_mask:    Tensor (1,) bool
            prompt:          str
            first_frame_image: List[PIL.Image]
            vace_video:      Optional[List[PIL.Image]]

        ``video`` is returned as PIL Images because the legacy
        WanVideoPipeline.preprocess_video handles cropping, resizing, and
        VAE encoding internally — converting to Tensor prematurely would
        bypass that step.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...
