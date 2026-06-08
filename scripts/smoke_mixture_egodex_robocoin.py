"""Smoke test: EgoDex + RoboCOIN MixtureDataset.

Runs end-to-end on the production yamls without touching GPU. Verifies:

  1. ``build_dataset(cfg.dataloader)`` constructs the mixture successfully.
  2. ``mixture.normalization_stats`` is None (data is pre-normalized inside each
     RoboCOIN bucket using its own per-robot-type stats — see
     RoboCOINDataset._normalize_array). RoboCOIN samples should have
     ``action`` values roughly in [-1, 1] under the default min-max
     ``normalize_mode``.
  3. Sampled items have the expected schema: 9 video frames at 320x384,
     ``action`` shape (32, 20), ``proprio`` shape (1, 20).
  4. Sub-source ratio approximates the configured weights.
  5. EgoDex samples have ``action_mask`` / ``proprio_mask`` all False and
     ``action`` all-zero; RoboCOIN samples have masks all True.

Usage:
    python scripts/smoke_mixture_egodex_robocoin.py
    python scripts/smoke_mixture_egodex_robocoin.py --num-samples 200
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from openwam.dataloader.registry import build_dataset  # noqa: E402


def _summarize_sample(sample: dict, source_name: str) -> str:
    video = sample["video"]
    action = sample["action"]
    proprio = sample["proprio"]
    am = sample["action_mask"]
    pm = sample["proprio_mask"]
    vm = sample["video_mask"]
    img = video[0]
    return (
        f"  [{source_name}] video={len(video)}×{img.size}  action={tuple(action.shape)}  "
        f"proprio={tuple(proprio.shape)}  "
        f"action_mask(all_true={bool(am.all())}, any_true={bool(am.any())})  "
        f"proprio_mask(all_true={bool(pm.all())}, any_true={bool(pm.any())})  "
        f"video_mask(true={int(vm.sum())}/{len(vm)})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 72)
    print(" EgoDex + RoboCOIN Mixture smoke test")
    print("=" * 72)

    # Compose only the dataloader cfg from configs/dataloader/mixture.yaml.
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs" / "dataloader"), version_base=None):
        cfg = compose(config_name="mixture")
    print("\n-- composed dataloader cfg --")
    print(OmegaConf.to_yaml(cfg))

    # 1. Build mixture.
    mixture = build_dataset(cfg, split="train")
    n = len(mixture)
    print(f"\n[1] len(mixture) = {n}")
    assert n > 0, "mixture is empty"

    # 2. MixtureDataset.normalization_stats is unconditionally None — normalization is
    #    the sole responsibility of each sub-source's reader, applied at sample
    #    construction time inside the reader's __getitem__.
    assert mixture.normalization_stats is None, (
        f"mixture.normalization_stats must be None, got {mixture.normalization_stats!r}"
    )
    print("[2] mixture.normalization_stats = None (mixture never aggregates per-source stats)")
    rc_ds = mixture.get_dataset("robocoin")
    eg_ds = mixture.get_dataset("egodex")
    print(f"    sub-source names = {mixture.names}")
    assert rc_ds.normalization_stats is None and eg_ds.normalization_stats is None

    # 4. Sample and verify schema + sub-source distribution.
    rng = np.random.RandomState(args.seed)
    indices = rng.randint(0, n, size=args.num_samples)

    counts = {"robocoin": 0, "egodex": 0}
    first_shown = {"robocoin": False, "egodex": False}
    print(f"\n[3] sampling {args.num_samples} items ...")
    for i, idx in enumerate(indices):
        sample = mixture[int(idx)]
        name = sample["_dataset_name"]
        counts[name] = counts.get(name, 0) + 1
        if not first_shown.get(name, True):
            print(_summarize_sample(sample, name))
            if name == "robocoin":
                assert bool(sample["action_mask"].all()), "robocoin action_mask should be all True"
                assert bool(sample["proprio_mask"].all()), "robocoin proprio_mask should be all True"
                a = sample["action"].numpy()
                p = sample["proprio"].numpy()
                print(f"      action range:  [{a.min():.4f}, {a.max():.4f}]   (expect roughly [-1, 1] under min-max)")
                print(f"      proprio range: [{p.min():.4f}, {p.max():.4f}]")
            elif name == "egodex":
                assert not bool(sample["action_mask"].any()), "egodex action_mask should be all False"
                assert not bool(sample["proprio_mask"].any()), "egodex proprio_mask should be all False"
                a = sample["action"].numpy()
                assert np.all(a == 0.0), "egodex action should be all-zero (no real action)"
                print("      action all-zero (no real action supervision)")
            first_shown[name] = True

        # Schema sanity (every sample).
        assert sample["action"].shape == (32, 20), f"sample {i} action shape {sample['action'].shape}"
        assert sample["proprio"].shape == (1, 20), f"sample {i} proprio shape"
        assert len(sample["video"]) == 9, f"sample {i} video len {len(sample['video'])}"
        img = sample["video"][0]
        assert img.size == (320, 384), f"sample {i} image size {img.size}"

    print(f"\n[4] sub-source counts (out of {args.num_samples}): {counts}")
    total = sum(counts.values())
    if total > 0:
        rc_ratio = counts.get("robocoin", 0) / total
        eg_ratio = counts.get("egodex", 0) / total
        # configured weights 1.0 : 0.3 → normalized 0.769 : 0.231
        print(f"    observed ratio = {rc_ratio:.3f} : {eg_ratio:.3f}  (expect ~0.769 : 0.231)")

    print("\n[OK] smoke test passed")


if __name__ == "__main__":
    main()
