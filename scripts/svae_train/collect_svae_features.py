"""Offline feature collector for training the S-VAE reducer.

Builds the same encoder + dataloader production training uses (via the
``configs/model/video_backbone/encoder/svae/collect.yaml`` Hydra chain), then iterates the
FULL dataset exactly once — the same single-epoch traversal training does — and
runs every clip through ``encoder.batch_encode_pooled_for_svae_training`` to
harvest the POST-POOL, pre-feature_norm features the S-VAE compresses. Each clip
yields a ``(raw_dim, T_lat, h, w)`` tensor — 1 un-pooled cond latent + the
mean-pooled target latents, i.e. both sub-populations the reducer sees at
inference.

Traversal matches training: the dataset, ``collate_fn=list``, ``num_workers``
and the ``cfg.project.seed`` worker-seeding (so the train-time augmentation in
``openwam/dataloader/transforms/video.py`` is reproduced) are all inherited from
``train.yaml``. Across ranks the dataset is partitioned by a *non-padding*
strided sampler over one shared shuffled permutation, so the union is the full
dataset with no duplicates (unlike ``DistributedSampler``, which pads).

Features are written in bounded-size chunks (``features_rank{r}_part{p}.pt``) so
a full-dataset collection never has to fit in host RAM; a global per-channel
``stats.pt`` (mean/std for the S-VAE input standardisation) is accumulated
online. ``scripts/svae_train/train_svae.py`` memory-maps the shards to fit the S-VAE over
multiple epochs.

Requires ``model.video_backbone.encoder.svae_path`` unset (collect on a RAW
encoder) — also enforced by ``batch_encode_pooled_for_svae_training``.

Usage (single node, auto-detect GPUs):

    cd /path/to/workspace/openwam/openwam-feat-encoder-svae
    bash scripts/svae_train/collect_svae_features.sh \
        training.batch_size=4 \
        model.video_backbone.encoder.name=vjepa21 \
        model.video_backbone.encoder.model_path=/path/to/weights/vjepa21
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

# Target on-disk size per feature shard part. Bounds the collector's resident
# memory (one part is buffered before each flush) and the trainer's per-file
# mmap granularity, independent of the total dataset size.
_PART_BYTES_TARGET = 1 << 30  # 1 GiB


def _setup_distributed() -> tuple[int, int, int]:
    """Init the process group under torchrun; degenerate 1-rank world otherwise."""
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29502")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _all_reduce_sum(t: torch.Tensor) -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


def _resolve_raw_dim(encoder) -> int:
    """Raw post-backbone channel dim. V-JEPA 2/2.1 expose ``_raw_embed_dim``.
    Falls back to ``spec.z_dim`` (== raw when the reducer is disabled, the
    only legal state for collection)."""
    for attr in ("_raw_z_dim", "_raw_embed_dim"):
        val = getattr(encoder, attr, None)
        if val is not None:
            return int(val)
    return int(encoder.spec.z_dim)


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="model/video_backbone/encoder/svae/collect",
)
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size = _setup_distributed()
    is_main = rank == 0

    if is_main:
        print("=" * 60)
        print("OpenWAM S-VAE offline feature collection")
        print(f"world_size={world_size}, rank={rank}, local_rank={local_rank}")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)

    output_root = Path(str(cfg.svae_collect.output_dir))
    # Shuffle seed must be IDENTICAL on every rank so the strided partition below
    # is a clean disjoint cover. Reuse the production training seed (cfg.project.seed)
    # when set; fall back to 0 otherwise (still rank-agreed). Augmentation worker
    # seeding is enabled only when project.seed is set, mirroring training.
    proj_seed = cfg.get("project", {}).get("seed", None) if "project" in cfg else None
    shuffle_seed = int(proj_seed) if proj_seed is not None else 0

    encoder_name = cfg.model.video_backbone.encoder.name
    if encoder_name == "wan22_vae":
        raise ValueError(
            "S-VAE feature collection does not apply to wan22_vae (the native VAE has no "
            "high-dim raw feature to reduce). Pick vjepa21 (or another external encoder) "
            "via FIELD overrides on the CLI, e.g. "
            "model.video_backbone.encoder.name=vjepa21 "
            "model.video_backbone.encoder.model_path=<weights dir> "
            "(the 'model/video_backbone/encoder=...' group-select idiom does not resolve from this "
            "nested entry config)."
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{encoder_name}_{timestamp}"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        logger.info("S-VAE feature output directory: %s", output_dir)
    # Broadcast the rank-0 timestamped dir to all ranks so every rank writes its
    # shard into the SAME directory (datetime.now() differs per rank otherwise).
    if dist.is_initialized() and world_size > 1:
        obj = [str(output_dir)]
        dist.broadcast_object_list(obj, src=0)
        output_dir = Path(obj[0])
        dist.barrier()

    sys.path.insert(0, str(PROJECT_ROOT))
    from openwam.dataloader.registry import build_dataset
    from openwam.model.video_backbone.encoder import build_video_encoder
    from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator

    dataset = build_dataset(cfg.dataloader, split="train")

    encoder_cfg = cfg.model.video_backbone.encoder
    if encoder_cfg.get("svae_path") is not None:
        raise ValueError(
            "S-VAE feature collection refuses to run on an S-VAE-enabled encoder. "
            "Set 'model.video_backbone.encoder.svae_path=null' on the CLI."
        )
    encoder = build_video_encoder(encoder_cfg)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder.to(device=device, dtype=dtype)
    encoder.eval()
    raw_dim = _resolve_raw_dim(encoder)

    if is_main:
        logger.info(
            "encoder=%s raw_dim=%d world_size=%d dataset_len=%d", encoder_name, raw_dim, world_size, len(dataset)
        )

    from torch.utils.data import DataLoader

    bs = int(cfg.training.batch_size)
    nw = int(cfg.training.dataset_num_workers)

    # Full-dataset, single-epoch traversal. One shared shuffled permutation
    # (same seed on every rank) sliced by a strided non-padding sampler ->
    # disjoint per-rank shards whose union is the whole dataset, no duplicates.
    gen = torch.Generator()
    gen.manual_seed(shuffle_seed)
    perm = torch.randperm(len(dataset), generator=gen).tolist()
    my_indices = perm[rank::world_size]

    loader_kwargs: dict = dict(
        batch_size=bs,
        sampler=my_indices,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        collate_fn=list,
    )
    if proj_seed is not None:
        # Match training: seed each worker's augmentation RNG reproducibly.
        loader_kwargs["generator"] = make_dataloader_generator(int(proj_seed), rank=rank)
        loader_kwargs["worker_init_fn"] = dataloader_worker_init_fn
    loader = DataLoader(dataset, **loader_kwargs)

    # Per-channel stats accumulators (fp64) for the S-VAE input standardisation.
    sum_acc = torch.zeros(raw_dim, dtype=torch.float64, device=device)
    sumsq_acc = torch.zeros(raw_dim, dtype=torch.float64, device=device)
    n_tokens_acc = torch.zeros(1, dtype=torch.float64, device=device)

    # Bounded-size chunked writing: buffer clips until ~_PART_BYTES_TARGET, then
    # flush a part file and clear, so the collector's RAM stays flat regardless
    # of dataset size.
    feat_buf: list[torch.Tensor] = []  # (b, raw_dim, T_lat, h, w) cpu fp16
    buf_clips = 0
    part_idx = 0
    flush_every = None  # clips per part, set from the first batch's clip size

    def _flush() -> None:
        nonlocal feat_buf, buf_clips, part_idx
        if not feat_buf:
            return
        features = torch.cat(feat_buf, dim=0)
        shard_path = output_dir / f"features_rank{rank}_part{part_idx:05d}.pt"
        torch.save({"features": features, "raw_dim": int(raw_dim), "encoder_name": encoder_name}, str(shard_path))
        logger.info("[rank %d] wrote part %d: %d clips -> %s", rank, part_idx, features.shape[0], shard_path)
        part_idx += 1
        feat_buf = []
        buf_clips = 0

    t0 = time.time()
    total_clips = 0
    with torch.no_grad():
        for batch_idx, samples in enumerate(loader):
            tensors = [encoder.preprocess_video(s["video"]) for s in samples]
            batch = torch.cat(tensors, dim=0)
            raw = encoder.batch_encode_pooled_for_svae_training(batch)  # (B, raw_dim, T_lat, h, w)
            flat = raw.permute(0, 2, 3, 4, 1).reshape(-1, raw_dim).to(torch.float64)
            sum_acc += flat.sum(dim=0)
            sumsq_acc += (flat * flat).sum(dim=0)
            n_tokens_acc += float(flat.shape[0])

            if flush_every is None:
                clip_bytes = max(1, raw[0].numel() * 2)  # fp16 stored on disk
                flush_every = max(1, _PART_BYTES_TARGET // clip_bytes)
            feat_buf.append(raw.detach().to("cpu", torch.float16))
            buf_clips += raw.shape[0]
            total_clips += raw.shape[0]
            if buf_clips >= flush_every:
                _flush()

            if is_main and (batch_idx + 1) % 50 == 0:
                logger.info(
                    "[rank %d] %d batches | %d clips | %.0f tokens | %.1fs",
                    rank,
                    batch_idx + 1,
                    total_clips,
                    float(n_tokens_acc.item()),
                    time.time() - t0,
                )
    _flush()  # remaining buffered clips
    logger.info("[rank %d] done: %d clips across %d parts", rank, total_clips, part_idx)

    _all_reduce_sum(sum_acc)
    _all_reduce_sum(sumsq_acc)
    _all_reduce_sum(n_tokens_acc)

    # All ranks compute n_total (n_tokens_acc was all-reduced) and raise
    # symmetrically on the degenerate 0-token path. A rank-0-only raise here —
    # after the all-reduces, before the barrier below — would leave the other
    # ranks hanging in that barrier until the NCCL timeout.
    n_total = int(n_tokens_acc.item())
    if n_total < 1:
        raise RuntimeError("S-VAE collection accumulated 0 tokens. Check the dataloader pipeline / dataset path.")

    if is_main:
        mean = sum_acc / float(n_total)
        var = (sumsq_acc / float(n_total) - mean * mean).clamp_min(0.0)
        std = var.sqrt()
        stats_path = output_dir / "stats.pt"
        torch.save(
            {
                "mean": mean.cpu().to(torch.float32),
                "std": std.cpu().to(torch.float32),
                "raw_dim": int(raw_dim),
                "num_tokens": n_total,
                "encoder_name": encoder_name,
                "world_size": world_size,
                "seed": shuffle_seed,
                "wallclock_seconds": time.time() - t0,
            },
            str(stats_path),
        )
        logger.info("S-VAE stats saved: %s (raw_dim=%d, tokens=%d)", stats_path, raw_dim, n_total)
        print()
        print(f"S-VAE features written under: {output_dir}")
        print(f"Summary: encoder={encoder_name} raw_dim={raw_dim} tokens={n_total} world_size={world_size}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
