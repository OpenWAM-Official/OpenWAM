"""Flow-matching scheduler adapter for the Cosmos3-Edge backbone.

Reuses :class:`CosmosFlowSchedulerAdapter` verbatim — the α-shift sigma grid and
``timesteps = sigmas × num_train_timesteps`` linear relation are hard constraints
(``generate()`` / ``deploy/denoise_schedule.py`` invert ``σ = t / num_train_ts``
outside the adapter), and Cosmos3's rectified-flow training uses the same
``(1−σ)x₀ + σε`` interpolant with velocity target ``ε − x₀``.

Why not UniPC (the bundle's scheduler): NVIDIA's intended Edge inference grid is
the *native linear flow ramp + flow_shift* (train-time shift is resolution-keyed
{256: 3, 480: 5, 720: 10}); the bundle's karras ``sigma_max=200`` fields are
bypassed when that path works. As of diffusers @6ad35739 the native ramp is
silently ignored by base UniPC under ``use_karras_sigmas=True`` (fixed in open
PR huggingface/diffusers#14272), so the reference numbers from current main
reflect a grid NVIDIA did not intend. OpenWAM's shifted-flow Euler grid (this
adapter, default shift 5.0 = the 480p train value) matches the intended
semantics; multi-step UniPC correction is a possible later deploy enhancement.
"""

from __future__ import annotations

from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter


class Cosmos3FlowSchedulerAdapter(CosmosFlowSchedulerAdapter):
    """Identical math to the predict2.5 adapter; separate class so the two
    backbones can diverge (e.g. a UniPC deploy grid) without cross-family edits."""
