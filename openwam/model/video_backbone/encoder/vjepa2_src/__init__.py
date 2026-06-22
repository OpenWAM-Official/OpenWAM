"""Helpers for the V-JEPA 2.1 encoder (the ``encoder/vjepa2_1.py`` subclass).

``loader`` builds the ViT from a manifest; ``vision_transformer`` + ``modules`` +
``patch_embed`` + ``masks_utils`` + ``tensors`` are the Apache-2.0 ViT code lifted
from facebookresearch/vjepa2 (branch ``vjepa2_1``) so this subsystem needs no
``third_party/vjepa2`` submodule. See ``APACHE-LICENSE``.
"""
