"""Vendored V-JEPA 2.1 ViT — Apache-2.0, from facebookresearch/vjepa2 @ ``vjepa2_1``.

Encoder-only lift (``vision_transformer`` + ``modules`` + ``patch_embed`` plus
the two ``src`` helpers ``masks_utils`` / ``tensors``); the JEPA ``predictor`` is
omitted. Carried in-tree so the encoder subsystem needs no ``third_party/vjepa2``
submodule. Only the 4 cross-package imports in ``vision_transformer.py`` were
rewritten to package-relative — behaviour is byte-for-byte unchanged. The RoPE
dtype fix stays an external monkey-patch in ``loader.py`` (this code is kept
verbatim). See ``APACHE-LICENSE``.
"""
