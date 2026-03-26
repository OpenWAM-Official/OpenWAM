"""Classifier-Free Guidance parallelism for accelerated inference.

Two strategies for reducing CFG overhead:

1. **Batch merge** (single GPU): Concatenate conditional and unconditional
   inputs along the batch dimension, run a single DiT forward pass, then
   split and apply the CFG formula. Already supported by WanVideoPipeline's
   ``cfg_merge`` flag.

2. **Multi-GPU parallel** (2+ GPUs): Run the conditional pass on GPU 0
   and unconditional pass on GPU 1 simultaneously, then gather results
   on the main device. Near 2x speedup for the DiT forward pass portion.
"""

import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor


class CFGBatchMerger:
    """Merge conditional/unconditional passes into a single batched forward.

    Instead of two sequential DiT calls, concatenates inputs along batch
    dimension, runs once, then splits. This halves the number of kernel
    launches and improves GPU utilization on a single device.

    Usage:
        merger = CFGBatchMerger(cfg_scale=5.0)
        noise_pred = merger.forward(model_fn, models, inputs_shared,
                                     inputs_posi, inputs_nega, timestep)
    """

    def __init__(self, cfg_scale: float = 5.0):
        self.cfg_scale = cfg_scale

    def forward(
        self,
        model_fn: Callable,
        models: dict,
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
        timestep: Tensor,
        **extra_kwargs,
    ) -> Tensor:
        """Run batched CFG forward pass.

        Concatenates positive and negative conditioning along batch dim,
        runs a single model_fn call, splits, and applies CFG formula.
        """
        if self.cfg_scale == 1.0:
            return model_fn(**models, **inputs_shared, **inputs_posi,
                          timestep=timestep, **extra_kwargs)

        # Batch positive and negative
        batched_inputs = {}
        for key in inputs_shared:
            v = inputs_shared[key]
            if isinstance(v, Tensor):
                batched_inputs[key] = torch.cat([v, v], dim=0)
            else:
                batched_inputs[key] = v

        # Merge posi/nega conditioning
        cond_inputs = {}
        all_keys = set(list(inputs_posi.keys()) + list(inputs_nega.keys()))
        for key in all_keys:
            v_p = inputs_posi.get(key)
            v_n = inputs_nega.get(key)
            if isinstance(v_p, Tensor) and isinstance(v_n, Tensor):
                cond_inputs[key] = torch.cat([v_p, v_n], dim=0)
            else:
                cond_inputs[key] = v_p  # fallback to positive

        # Double the timestep
        t_batched = torch.cat([timestep, timestep], dim=0) if isinstance(timestep, Tensor) else timestep

        # Single forward pass
        noise_pred_batched = model_fn(
            **models, **batched_inputs, **cond_inputs,
            timestep=t_batched, cfg_merge=True, **extra_kwargs,
        )

        # Split and apply CFG
        B = noise_pred_batched.shape[0] // 2
        noise_pred_posi = noise_pred_batched[:B]
        noise_pred_nega = noise_pred_batched[B:]

        return noise_pred_nega + self.cfg_scale * (noise_pred_posi - noise_pred_nega)


class CFGParallelExecutor:
    """Run CFG conditional and unconditional passes on separate GPUs.

    Distributes the two DiT forward passes across two devices and gathers
    results for CFG combination. Achieves near 2x speedup on multi-GPU
    systems for the model forward pass portion.

    Args:
        devices: List of torch devices for parallel execution.
            First device runs positive (conditional) pass,
            second runs negative (unconditional) pass.
        cfg_scale: Classifier-free guidance scale.
    """

    def __init__(
        self,
        devices: List[str] = ["cuda:0", "cuda:1"],
        cfg_scale: float = 5.0,
    ):
        self.devices = [torch.device(d) for d in devices]
        self.cfg_scale = cfg_scale
        self._executor = ThreadPoolExecutor(max_workers=2)

    def forward(
        self,
        model_fn: Callable,
        models: dict,
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
        timestep: Tensor,
        **extra_kwargs,
    ) -> Tensor:
        """Run parallel CFG forward pass across two GPUs.

        Both passes execute concurrently via thread pool. Results are
        gathered on the first device for CFG combination.
        """
        if self.cfg_scale == 1.0:
            return model_fn(**models, **inputs_shared, **inputs_posi,
                          timestep=timestep, **extra_kwargs)

        main_device = self.devices[0]
        neg_device = self.devices[1]

        def _run_positive():
            return model_fn(
                **models, **inputs_shared, **inputs_posi,
                timestep=timestep, **extra_kwargs,
            )

        def _run_negative():
            # Move inputs to second device
            neg_shared = {k: v.to(neg_device) if isinstance(v, Tensor) else v
                         for k, v in inputs_shared.items()}
            neg_inputs = {k: v.to(neg_device) if isinstance(v, Tensor) else v
                         for k, v in inputs_nega.items()}
            neg_t = timestep.to(neg_device) if isinstance(timestep, Tensor) else timestep
            return model_fn(
                **models, **neg_shared, **neg_inputs,
                timestep=neg_t, **extra_kwargs,
            )

        # Submit both passes concurrently
        future_posi = self._executor.submit(_run_positive)
        future_nega = self._executor.submit(_run_negative)

        noise_pred_posi = future_posi.result()
        noise_pred_nega = future_nega.result().to(main_device)

        return noise_pred_nega + self.cfg_scale * (noise_pred_posi - noise_pred_nega)

    def shutdown(self):
        """Clean up the thread pool."""
        self._executor.shutdown(wait=False)
