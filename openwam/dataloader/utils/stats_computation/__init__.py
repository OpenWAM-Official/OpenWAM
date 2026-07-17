"""Offline stats precompute tools for dataloaders.

CLI modules that scan a dataset once and write the normalization stats the
readers require at train time — one ``<dataset>_stats_computation`` module per
dataset family.

Run any of them as ``python -m openwam.dataloader.utils.stats_computation.<module>``.
"""
