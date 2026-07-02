"""Offline stats / embedding precompute tools for dataloaders.

CLI modules that scan a dataset once and write the artifacts the readers
require at train time:

  - ``robotwin_stats_computation`` / ``robocoin_stats_computation`` /
    ``oxe_stats_computation`` — normalization stats.
  - ``reason1_embedding_computation`` — Cosmos-Reason1 text-embedding cache.

Run any of them as ``python -m openwam.dataloader.utils.stats_computation.<module>``.
"""
