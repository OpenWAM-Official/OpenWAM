"""Helpers for the V-JEPA 2.1 encoder (the ``encoder/vjepa21.py`` subclass).

``loader`` builds the ViT from a manifest; ``vision_transformer`` + ``modules`` +
``patch_embed`` + ``masks_utils`` + ``tensors`` are MIT-licensed ViT code adapted
from facebookresearch/vjepa2 at commit
``ce64921e94f0ffdc330c00fc62618157894b74be``, so this subsystem needs no
``third_party/vjepa2`` submodule. See
the upstream MIT license (Copyright Meta Platforms, Inc. and affiliates).
"""
