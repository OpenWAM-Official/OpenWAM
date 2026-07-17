"""Offline stats precompute tools for dataloaders.

CLI modules that scan a dataset once and write the normalization stats the
readers require at train time (``robotwin_stats_computation`` /
``robocoin_stats_computation`` / ``oxe_stats_computation``).

Run any of them as ``python -m openwam.dataloader.utils.stats_computation.<module>``.
"""
