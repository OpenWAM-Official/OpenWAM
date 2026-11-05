"""Training-batch collation in the DataLoader workers (``OPENWAM_FAST_INPUT=1``).

The dataset yields samples whose ``video`` is a list of PIL RGB frames. The
default ``collate_fn=list`` leaves the frames to the training process, which
converts every frame to a float tensor (``wan.preprocess.preprocess_image``:
``np.array(img, float32)`` -> ``dtype`` on the GPU -> ``* 2/255 - 1``) with one
pageable host-to-device copy per frame.

``openwam_collate`` keeps the batch a list of the same sample dicts but replaces
each RGB frame list by one ``uint8`` tensor ``(T, H, W, 3)`` holding the same
pixels (the float32 values converted by the default path are these integers).
``wan.encode.preprocess_video`` then converts a list of such clips on the GPU
with the same ops in the same order, so the model input is bitwise the same
tensor. Samples whose frames are not same-size RGB PIL images are left as they
are (default path).
"""

from __future__ import annotations

import numpy as np
import torch


def _rgb_frames_to_uint8(frames):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        return None
    size = None
    for f in frames:
        if not isinstance(f, Image.Image) or f.mode != "RGB":
            return None
        if size is None:
            size = f.size
        elif f.size != size:
            return None
    arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)
    return torch.from_numpy(np.ascontiguousarray(arr))


def openwam_collate(samples):
    """``collate_fn`` for the training DataLoader: a list of sample dicts (as ``list``),
    with RGB frame lists turned into ``uint8`` ``(T, H, W, 3)`` tensors."""
    out = []
    for s in samples:
        if isinstance(s, dict) and "video" in s:
            u8 = _rgb_frames_to_uint8(s["video"])
            if u8 is not None:
                s = dict(s)
                s["video"] = u8
        out.append(s)
    return out
