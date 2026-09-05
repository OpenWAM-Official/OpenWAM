#!/usr/bin/env python3
"""Interactive downloader for released OpenWAM checkpoints.

Fetches a checkpoint from the public OpenWAM collections on HuggingFace —
either an OpenWAM-Alpha release (pretrained foundation model + finetuned
variants) or one of the OpenWAM-Study ablation checkpoints — into the assets
checkpoint directory. Every checkpoint directory is self-contained
(config.yaml + weights + tokenizer + normalization stats), so after the
download it deploys directly:

    bash scripts/deploy.sh <download_dir>

The one exception is the pretrain foundation model, which is meant as a
finetuning start (training.finetune_ckpt_path in configs/train.yaml), not
for direct deployment.

Usage:
    python scripts/download_assets/download_openwam_checkpoints.py
"""

from __future__ import annotations

import os

# HuggingFace downloads render their own stack of progress bars (including the
# xet backend's, which repaint on interpreter exit and smear over our status
# lines). Silence them all — this script draws a single progress bar itself.
# Must be set before huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ASSETS_ROOT = Path.cwd() / "assets" / "openwam_ckpt"

# ── terminal colors ──────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def cyan(t: str) -> str:
    return _c("36", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


# ── checkpoint registry (mirrors the OpenWAM collections on HuggingFace) ─────
@dataclass(frozen=True)
class Ckpt:
    name: str  # repo name under the OpenWAM org == download subdir
    approx_gb: float  # fallback size when the live metadata query fails
    note: str | None = None  # extra label shown in the menu, e.g. the backbone
    finetune_only: bool = False  # pretrain checkpoint: finetune it, don't deploy it

    @property
    def repo_id(self) -> str:
        return f"OpenWAM/{self.name}"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.note})" if self.note else self.name


@dataclass(frozen=True)
class Group:
    title: str  # collection name shown in the menu
    subdir: str  # path under assets/openwam_ckpt/
    ckpts: tuple[Ckpt, ...]


ALPHA = Group(
    "OpenWAM_Alpha",
    "openwam_alpha",
    (
        Ckpt("OpenWAM-Alpha-Pretrain-Foundation-Model", 24.8, finetune_only=True),
        Ckpt("OpenWAM-Alpha-Real-Dexterous-Hand-Wuji", 24.8),
        Ckpt("OpenWAM-Alpha-Real-RoboDojo-ARX-X5", 24.8),
        Ckpt("OpenWAM-Alpha-Real-RoboDojo-Piper", 24.8),
        Ckpt("OpenWAM-Alpha-Real-RoboDojo-Piper-X", 24.8),
        Ckpt("OpenWAM-Alpha-Real-Single-Arm-Franka", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-EBench", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-LIBERO", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-RoboCasa-GR1", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-RoboCasa365", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-RoboDojo", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-RoboTwin-Clean2Random", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-RoboTwin-Full", 24.8),
        Ckpt("OpenWAM-Alpha-Sim-VLABench", 24.8),
    ),
)

STUDY_GROUPS: tuple[Group, ...] = (
    Group(
        "OpenWAM_Study_Architecture",
        "openwam_study/architecture",
        (
            Ckpt("robotwin_dual_system_idm", 24.8),
            Ckpt("robotwin_dual_system_joint_cross_attention", 24.8),
            Ckpt("robotwin_dual_system_joint_cross_attention_detach", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention", 24.8),
            Ckpt("robotwin_single_system_moe", 24.8),
            Ckpt("robotwin_single_system_vanilla", 24.8),
            Ckpt("robotwin_tri_system_joint_self_attention", 29.3),
        ),
    ),
    Group(
        "OpenWAM_Study_Pretrain",
        "openwam_study/pretrain",
        (
            Ckpt("pretrain_ego_robot_cotrain_action_sees_video", 24.8),
            Ckpt("pretrain_ego_robot_cotrain_mutual", 24.8),
            Ckpt("pretrain_robot_only", 24.8),
            Ckpt("pretrain_two_stage_ego", 24.8),
            Ckpt("pretrain_two_stage_robot", 24.8),
            Ckpt("sft_ego_robot_cotrain_robotwin_clean_action_sees_video", 24.8),
            Ckpt("sft_ego_robot_cotrain_robotwin_clean_mutual", 24.8),
            Ckpt("sft_ego_robot_cotrain_robotwin_full_action_sees_video", 24.8),
            Ckpt("sft_ego_robot_cotrain_robotwin_full_mutual", 24.8),
            Ckpt("sft_from_scratch_robotwin_clean", 24.8),
            Ckpt("sft_robot_only_robotwin_clean", 24.8),
            Ckpt("sft_two_stage_robotwin_clean", 24.8),
        ),
    ),
    Group(
        "OpenWAM_Study_Visual_Encoder",
        "openwam_study/visual_encoder",
        (
            Ckpt("robotwin_dual_system_joint_self_attention_dinov3", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention_dinov3_svae", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention_flux2", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention_vjepa21", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention_vjepa21_svae", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention_wan22_vae", 24.8),
        ),
    ),
    Group(
        "OpenWAM_Study_Video_Backbone",
        "openwam_study/video_backbone",
        (
            Ckpt("robotwin_dual_system_joint_self_attention", 24.8, note="wan22_ti2v_5b"),
            Ckpt("robotwin_dual_system_joint_self_attention_cosmos25", 22.4),
            Ckpt("robotwin_dual_system_joint_self_attention_cosmos3", 25.6),
            Ckpt("robotwin_dual_system_joint_self_attention_wan21_i2v_14b", 49.7),
            Ckpt("robotwin_dual_system_joint_self_attention_wan21_vace_1_3b", 17.2),
        ),
    ),
    Group(
        "OpenWAM_Study_Attention_Mask",
        "openwam_study/attention_mask",
        (
            Ckpt("robotwin_dual_system_joint_self_attention", 24.8, note="action sees video"),
            Ckpt("robotwin_dual_system_joint_self_attention_isolated", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention_mutual", 24.8),
            Ckpt("robotwin_dual_system_joint_self_attention_video_sees_action", 24.8),
        ),
    ),
)


# ── interaction helpers ──────────────────────────────────────────────────────
def ask(prompt: str) -> str:
    try:
        return input(cyan(prompt)).strip()
    except EOFError:
        print()
        sys.exit(1)


def ask_choice(title: str, options: list[str], prompt: str) -> str:
    print()
    print(bold(title))
    for line in options:
        print(f"  {line}")
    while True:
        answer = ask(prompt)
        if answer in {str(i) for i in range(1, len(options) + 1)}:
            return answer
        print(red(f"Invalid choice {answer!r} — enter a number between 1 and {len(options)}."))


def ask_yes_no(prompt: str, default_yes: bool) -> bool:
    suffix = "[Y/n] " if default_yes else "[y/N] "
    answer = ask(prompt + suffix).lower()
    if not answer:
        return default_yes
    return answer in {"y", "yes"}


# ── steps ────────────────────────────────────────────────────────────────────
def choose_group() -> Group:
    key = ask_choice(
        "Select the checkpoint family",
        [
            "(1) OpenWAM_Alpha — pretrained foundation model + finetuned variants",
            "(2) OpenWAM_Study — ablation checkpoints from the design study",
        ],
        "Family number: ",
    )
    if key == "1":
        return ALPHA
    type_key = ask_choice(
        "Select the study type",
        [f"({i}) {g.title}" for i, g in enumerate(STUDY_GROUPS, 1)],
        "Type number: ",
    )
    return STUDY_GROUPS[int(type_key) - 1]


def choose_storage_root(group: Group) -> Path:
    default_root = ASSETS_ROOT / group.subdir
    print()
    print(bold("Storage location"))
    print(f"  Default: {default_root}")
    answer = ask("Storage path (press Enter for the default): ")
    if not answer:
        root = default_root
        if not root.is_dir():
            root.mkdir(parents=True, exist_ok=True)
            print(yellow(f"Created default directory {root}"))
        return root
    root = Path(answer).expanduser()
    if not root.is_dir():
        print(red(f"Error: {root} does not exist. Create it first, then re-run this script."))
        sys.exit(1)
    return root


def choose_ckpt(group: Group) -> Ckpt:
    key = ask_choice(
        f"Select the checkpoint to download from {group.title}",
        [f"({i}) {c.label}" for i, c in enumerate(group.ckpts, 1)],
        "Checkpoint number: ",
    )
    return group.ckpts[int(key) - 1]


def query_download_size(ckpt: Ckpt) -> tuple[int, bool]:
    """Byte count for the repo: live HuggingFace metadata, or registry fallback."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(ckpt.repo_id, files_metadata=True)
        return sum(f.size or 0 for f in info.siblings), True
    except Exception:
        return int(ckpt.approx_gb * 1e9), False


# ── progress bar (self-drawn; HF native bars are disabled at the top) ────────
def _dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


_BAR_WIDTH = 28


def _draw_bar(done: int, expected: int, speed: float | None, final: bool) -> None:
    frac = min(done / expected, 1.0) if expected > 0 else 0.0
    filled = int(frac * _BAR_WIDTH)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    tail = f"{speed / 1e6:6.1f} MB/s" if speed is not None else " " * 11
    line = f"\r  [{bar}] {frac * 100:3.0f}%  {done / 1e9:6.2f} / {expected / 1e9:.2f} GB  {tail}"
    sys.stdout.write(line + ("\n" if final else ""))
    sys.stdout.flush()


def _watch_progress(target: Path, expected: int, stop: threading.Event) -> None:
    # Session-average speed: xet downloads land bytes in bursts (chunks are
    # fetched to a cache, then reconstructed into files), so an instantaneous
    # rate flaps between 0 and spikes.
    start_bytes = _dir_bytes(target)
    start_time = time.monotonic()
    while not stop.wait(1.0):
        done = _dir_bytes(target)
        speed = max(0.0, (done - start_bytes) / max(time.monotonic() - start_time, 1e-6))
        _draw_bar(done, expected, speed, final=False)


def download(ckpt: Ckpt, root: Path, expected: int) -> Path:
    target = root / ckpt.name
    if target.is_dir() and any(target.iterdir()):
        print(yellow(f"{target} already has content — resuming/skipping finished files."))
    print(bold(f"Downloading {ckpt.repo_id} -> {target}"))

    watch = sys.stdout.isatty()
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_progress, args=(target, expected, stop), daemon=True)
    started = time.monotonic()
    if watch:
        watcher.start()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=ckpt.repo_id, local_dir=str(target))
    finally:
        stop.set()
        if watch:
            watcher.join()
    if watch:
        elapsed = time.monotonic() - started
        _draw_bar(expected, expected, expected / max(elapsed, 1e-6), final=True)
    return target.resolve()


def final_hint(ckpt: Ckpt, target: Path) -> list[str]:
    if ckpt.finetune_only:
        return [
            "This is the pretrain foundation model — it is not meant for direct deployment.",
            f"Finetune from it by setting {bold(f'training.finetune_ckpt_path: {target}')} in "
            "configs/train.yaml, then run bash scripts/train.sh.",
        ]
    return [f"Deploy it with: {bold(f'bash scripts/deploy.sh {target}')}"]


def main() -> None:
    print(bold("OpenWAM released-checkpoint downloader"))
    print()
    group = choose_group()
    root = choose_storage_root(group)
    ckpt = choose_ckpt(group)

    expected, live = query_download_size(ckpt)
    origin = "" if live else " (approximate)"
    print()
    print(yellow(f"{ckpt.label} needs about {expected / 1e9:.1f} GB{origin} under {root}."))
    if not ask_yes_no("Start the download? ", default_yes=True):
        print(cyan("Aborted — nothing downloaded."))
        sys.exit(0)

    target = download(ckpt, root, expected)

    print()
    print(green(f"Done. {ckpt.label} is saved under:"))
    print(green(f"  {target}"))
    for line in final_hint(ckpt, target):
        print(line)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(yellow("Interrupted — partial downloads resume on the next run."))
        sys.exit(130)
