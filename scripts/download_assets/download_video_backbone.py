#!/usr/bin/env python3
"""Interactive downloader for video-backbone base checkpoints.

Fetches the pretrained weights a from-scratch OpenWAM training run needs
(Wan2.x / Cosmos families) into the assets checkpoint directory, then points
the matching config under configs/model/video_backbone/ at the download so
training picks it up without manual editing.

Usage:
    python scripts/download_assets/download_video_backbone.py
"""

from __future__ import annotations

import os

# HuggingFace downloads render their own stack of progress bars (including the
# xet backend's, which repaint on interpreter exit and smear over our status
# lines). Silence them all — this script draws a single progress bar itself.
# Must be set before huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "model" / "video_backbone"
DEFAULT_ROOT = Path.cwd() / "assets" / "video_backbone_ckpt"

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


# ── model registry ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Download:
    repo_id: str  # HuggingFace repo id (also the ModelScope id unless ms_id is set)
    subdir: str  # directory name under the storage root
    approx_gb: float  # fallback size when the live metadata query fails
    allow_patterns: tuple[str, ...] | None = None  # HF-only partial download
    ms_id: str | None = None


@dataclass(frozen=True)
class Model:
    name: str
    config: str  # yaml under configs/model/video_backbone/
    downloads: tuple[Download, ...]
    config_fields: tuple[str, ...]  # yaml field per download, positionally matched
    hf_only: bool = False


MODELS: dict[str, Model] = {
    "1": Model(
        name="Wan2.2-TI2V-5B",
        config="wan22_ti2v_5b.yaml",
        downloads=(Download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2-TI2V-5B", approx_gb=34.2),),
        config_fields=("model_path",),
    ),
    "2": Model(
        name="Wan2.1-VACE-1.3B",
        config="wan21_vace_1_3b.yaml",
        downloads=(Download("Wan-AI/Wan2.1-VACE-1.3B", "Wan2.1-VACE-1.3B", approx_gb=19.0),),
        config_fields=("model_path",),
    ),
    "3": Model(
        name="Wan2.1-I2V-14B-480P",
        config="wan21_i2v_14b_480p.yaml",
        downloads=(Download("Wan-AI/Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-480P", approx_gb=82.3),),
        config_fields=("model_path",),
    ),
    "4": Model(
        name="Cosmos-Predict2.5-2B",
        config="cosmos_predict25_2b.yaml",
        downloads=(
            Download(
                "nvidia/Cosmos-Predict2.5-2B",
                "Cosmos-Predict2.5-2B",
                approx_gb=4.6,
                allow_patterns=("base/post-trained/*", "tokenizer.pth"),
            ),
            Download("nvidia/Cosmos-Reason1-7B", "Cosmos-Reason1-7B", approx_gb=16.6),
        ),
        config_fields=("model_path", "text_encoder_path"),
        hf_only=True,
    ),
    "5": Model(
        name="Cosmos3-Edge",
        config="cosmos3_edge.yaml",
        downloads=(Download("nvidia/Cosmos3-Edge", "Cosmos3-Edge", approx_gb=9.2, ms_id="nv-community/Cosmos3-Edge"),),
        config_fields=("model_path",),
    ),
}


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
def choose_storage_root() -> Path:
    print(bold("Storage location"))
    print(f"  Default: {DEFAULT_ROOT}")
    answer = ask("Storage path (press Enter for the default): ")
    if not answer:
        root = DEFAULT_ROOT
        if not root.is_dir():
            root.mkdir(parents=True, exist_ok=True)
            print(yellow(f"Created default directory {root}"))
        return root
    root = Path(answer).expanduser()
    if not root.is_dir():
        print(red(f"Error: {root} does not exist. Create it first, then re-run this script."))
        sys.exit(1)
    return root


def choose_model() -> Model:
    key = ask_choice(
        "Select the model to download",
        [f"({k}) {m.name}" for k, m in MODELS.items()],
        "Model number: ",
    )
    return MODELS[key]


def choose_source(model: Model) -> str:
    key = ask_choice(
        "Select the download source",
        ["(1) huggingface", "(2) modelscope"],
        "Source number: ",
    )
    source = "huggingface" if key == "1" else "modelscope"
    if source == "modelscope" and model.hf_only:
        print(yellow(f"{model.name} is not available on ModelScope; only HuggingFace hosts it."))
        if ask_yes_no("Download via HuggingFace instead? ", default_yes=True):
            return "huggingface"
        print(cyan("Aborted — nothing downloaded."))
        sys.exit(0)
    return source


def query_download_sizes(model: Model, source: str) -> tuple[list[int], bool]:
    """Per-download byte counts: live HuggingFace metadata, or registry fallback."""
    if source == "huggingface":
        try:
            from fnmatch import fnmatch

            from huggingface_hub import HfApi

            api = HfApi()
            sizes = []
            for dl in model.downloads:
                info = api.model_info(dl.repo_id, files_metadata=True)
                total = 0
                for f in info.siblings:
                    if dl.allow_patterns and not any(fnmatch(f.rfilename, p) for p in dl.allow_patterns):
                        continue
                    total += f.size or 0
                sizes.append(total)
            return sizes, True
        except Exception:
            pass
    return [int(dl.approx_gb * 1e9) for dl in model.downloads], False


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


def download(model: Model, source: str, root: Path, sizes: list[int]) -> list[Path]:
    targets = []
    for i, (dl, expected) in enumerate(zip(model.downloads, sizes), 1):
        target = root / dl.subdir
        stage = f"[{i}/{len(model.downloads)}] " if len(model.downloads) > 1 else ""
        if target.is_dir() and any(target.iterdir()):
            print(yellow(f"{stage}{target} already has content — resuming/skipping finished files."))
        print(bold(f"{stage}Downloading {dl.repo_id} -> {target}"))

        watch = sys.stdout.isatty() and source == "huggingface"
        stop = threading.Event()
        watcher = threading.Thread(target=_watch_progress, args=(target, expected, stop), daemon=True)
        started = time.monotonic()
        if watch:
            watcher.start()
        try:
            if source == "huggingface":
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=dl.repo_id,
                    local_dir=str(target),
                    allow_patterns=list(dl.allow_patterns) if dl.allow_patterns else None,
                )
            else:
                from modelscope.hub.snapshot_download import snapshot_download

                snapshot_download(dl.ms_id or dl.repo_id, local_dir=str(target))
        finally:
            stop.set()
            if watch:
                watcher.join()
        if watch:
            elapsed = time.monotonic() - started
            _draw_bar(expected, expected, expected / max(elapsed, 1e-6), final=True)
        targets.append(target.resolve())
    return targets


def write_config(model: Model, targets: list[Path]) -> None:
    config_path = CONFIG_DIR / model.config
    text = config_path.read_text(encoding="utf-8")
    for field, target in zip(model.config_fields, targets):
        pattern = re.compile(rf"^({re.escape(field)}:\s*)\S+([^\n]*)$", re.MULTILINE)
        if not pattern.search(text):
            print(red(f"Warning: field {field!r} not found in {config_path}; update it manually."))
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}{target}{m.group(2)}", text, count=1)
        rel = config_path.relative_to(REPO_ROOT)
        print(green(f"Updated {rel}: {field} -> {target}"))
    config_path.write_text(text, encoding="utf-8")


def main() -> None:
    print(bold("OpenWAM video-backbone checkpoint downloader"))
    print()
    root = choose_storage_root()
    model = choose_model()
    source = choose_source(model)

    sizes, live = query_download_sizes(model, source)
    origin = "" if live else " (approximate)"
    print()
    print(yellow(f"{model.name} needs about {sum(sizes) / 1e9:.1f} GB{origin} under {root}."))
    if not ask_yes_no("Start the download? ", default_yes=True):
        print(cyan("Aborted — nothing downloaded."))
        sys.exit(0)

    targets = download(model, source, root, sizes)

    print()
    print(green(f"Done. {model.name} is saved under:"))
    for target in targets:
        print(green(f"  {target}"))
    write_config(model, targets)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(yellow("Interrupted — partial downloads resume on the next run."))
        sys.exit(130)
