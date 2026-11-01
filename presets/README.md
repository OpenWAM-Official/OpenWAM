# Training speed presets

Every speedup is an environment variable switch that is off by default; with none set, training runs exactly as before. A
preset contains a set of speedup switches that `scripts/train.sh` loads when `OPENWAM_PRESET` points to it:

```bash
OPENWAM_PRESET=presets/fast_80gb.env bash scripts/train.sh dataloader=libero \
  training.finetune_ckpt_path=<foundation_ckpt_dir_path> training.batch_size=32 training.num_epochs=10
```

Both presets maintain a per-GPU batch size of 32 on 8 GPUs, matching the global batch size of the training recipe of the released OpenWAM-Alpha-Sim-LIBERO checkpoint:

| preset | peak memory per GPU (8 GPUs, 32 per GPU) | Speed-up |
|---|---|---|
| `fast.env` | 120.4 GiB | 3.13× |
| `fast_80gb.env` | 69.1 GiB | 2.41× |


## Switches

The following table lists the speedup switches in the presets, their effect on training, and whether they are on in each preset. The `results` column shows how a switch affects the training numbers:
- **exact**: training is bit-for-bit identical to the default code (checked in deterministic mode).
- **rounding**: the same computation done with other (fused, compiled or cuDNN) kernels, so results differ only by floating-point rounding.

| switch | what it does | results | fast | fast_80gb |
|---|---|---|---|---|
| `OPENWAM_TEXT_CACHE=1` | bounded per-process cache of the frozen umT5 output rows (`OPENWAM_TEXT_CACHE_MAX_ENTRIES`, default 256) | exact | on | on |
| `OPENWAM_TEXT_MISS_FAST=1` | cache misses: encode only the distinct missing prompts, on a 64-token-multiple prefix | rounding | on | on |
| `OPENWAM_CONTEXT_TRUNC=1` | drop the all-padding tail of the text context (needs the text cache) | rounding | on | on |
| `OPENWAM_FAST_INPUT=1` | uint8 frames collated in the DataLoader workers, batched GPU preprocessing, non-blocking pinned copies, memoized RoPE table and VAE statistics | exact | on | on |
| `OPENWAM_VAE_FIRST=1` | queue the VAE encode before the text encode | exact | on | on |
| `OPENWAM_SDPA_CUDNN=1` | masked attention prefers cuDNN (deterministic mode keeps the default kernel) | rounding | on | on |
| `OPENWAM_FUSED_ROPE=1` | Triton RoPE forward/backward; `OPENWAM_ROPE_FMA_MODE` (0/1/2, default 1) selects the FMA pattern that matches PyTorch | exact | on | on |
| `OPENWAM_FAST_CAT=1` | joint [video, action] q/k/v concatenation and output split as Triton row copies | exact | on | on |
| `OPENWAM_COMPILE_BLOCKS=1` | torch.compile the memory-bound chains of each MoT layer (eager in deterministic mode) | rounding | on | on |
| `OPENWAM_VAE_FULLCLIP=1` | Wan2.2 VAE encode over the whole clip in one pass | rounding | on | on |
| `OPENWAM_COMPILE_VAE=1` | torch.compile that encode | rounding | on | on |
| `OPENWAM_VAE_CL=1` | its 3-D convolutions channels-last | rounding | on | on |
| `OPENWAM_CKPT_LAYERS=k` | with gradient checkpointing, recompute only the first k MoT layers (unset: all) | exact | 0 | 16 |
| `OPENWAM_MIXED_ATTN_CKPT=0/1` | override `mot_checkpoint_mixed_attn` (inner checkpoint of the joint attention) | exact | 0 | 0 |
| `OPENWAM_TEXT_ENCODER_OFFLOAD=1` | umT5 weights in pinned host memory between text-cache misses (−10.6 GiB per GPU) | exact | off | on |
| `OPENWAM_DS_TIMER_NOSYNC=1` | DeepSpeed throughput timer without device syncs | exact | on | on |
| `OPENWAM_DS_OVERLAP_COMM=1` | ZeRO-2 `overlap_comm` | exact | on | on |
| `OPENWAM_GC_FREEZE=1` | `gc.collect()` + `gc.freeze()` after steps 1, 2 and 4 | exact | on | on |
| `OPENWAM_NUMA_BIND=1` | each rank and its DataLoader workers on its GPU's NUMA node | exact | on | on |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | PyTorch allocator setting | exact | on | on |
