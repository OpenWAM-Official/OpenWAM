#!/usr/bin/env python3
"""Interactive downloader for external visual-encoder checkpoints.

Fetches the pretrained weights the pluggable video-encoder subsystem needs
(DINOv3 / V-JEPA 2.1 / Wan2.2 VAE / FLUX.2 VAE) into the assets checkpoint
directory, then points the matching config under
configs/model/video_backbone/encoder/ at the download so training picks it up
without manual editing.

Usage:
    python scripts/download_assets/download_visual_encoder.py
"""

from __future__ import annotations

import os

# HuggingFace downloads render their own stack of progress bars (including the
# xet backend's, which repaint on interpreter exit and smear over our status
# lines). Silence them all — this script draws a single progress bar itself.
# Must be set before huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import json
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "model" / "video_backbone" / "encoder"
DEFAULT_ROOT = Path.cwd() / "assets" / "visual_encoder_ckpt"

VJEPA_MANIFEST = {
    "arch_name": "vit_giant_xformers_rope",
    "embed_dim": 1408,
    "variant": "vitg-rope-384",
    "patch": 16,
    "img_size": 384,
    "training_num_frames": 64,
    "tubelet": 2,
    "use_rope": True,
    "img_temporal_dim_size": 1,
    "interpolate_rope": True,
    "checkpoint_file": "vjepa2_1_vitg_384.pt",
    "checkpoint_key": "target_encoder",
}

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
class Model:
    name: str
    config: str  # yaml under configs/model/video_backbone/encoder/
    subdir: str  # directory name under the storage root
    approx_gb: float  # fallback size when the live query fails
    repo_id: str | None = None  # HuggingFace repo id (None = direct URL model)
    ms_id: str | None = None  # ModelScope id (defaults to repo_id)
    allow_patterns: tuple[str, ...] | None = None  # partial hub download
    direct_url: str | None = None  # direct-download models (no HF/MS hosting)
    manifest: dict | None = None  # written as <subdir>/manifest.json after download
    config_subpath: str = ""  # appended to the target dir in the yaml path
    hf_gated: str | None = None  # gate note shown when huggingface is selected


MODELS: dict[str, Model] = {
    "1": Model(
        name="dinov3-vitb16-pretrain-lvd1689m",
        config="dinov3.yaml",
        subdir="dinov3-vitb16-pretrain-lvd1689m",
        approx_gb=0.4,
        repo_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_gated="gated (manual approval): request access on the model page and `hf auth login` first",
    ),
    "2": Model(
        name="vjepa2_1_vitg_384",
        config="vjepa21.yaml",
        subdir="vjepa2_1_vitg_384",
        approx_gb=16.9,
        direct_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
        manifest=VJEPA_MANIFEST,
    ),
    "3": Model(
        name="wan2.2-vae",
        config="wan22_vae.yaml",
        subdir="Wan2.2-VAE",
        approx_gb=2.9,
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        allow_patterns=("Wan2.2_VAE.pth",),
    ),
    "4": Model(
        name="flux.2-vae",
        config="flux2_vae.yaml",
        subdir="FLUX.2-dev-VAE",
        approx_gb=0.4,
        repo_id="black-forest-labs/FLUX.2-dev",
        allow_patterns=("vae/*",),
        config_subpath="vae",
        hf_gated="gated (auto approval): accept the license on the model page and `hf auth login` first",
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
    if model.direct_url:
        print()
        print(
            yellow(
                f"{model.name} is not hosted on HuggingFace/ModelScope; it is downloaded "
                f"from the official release URL:\n  {model.direct_url}"
            )
        )
        return "direct"
    key = ask_choice(
        "Select the download source",
        ["(1) huggingface", "(2) modelscope"],
        "Source number: ",
    )
    source = "huggingface" if key == "1" else "modelscope"
    if source == "huggingface" and model.hf_gated:
        print(yellow(f"Note: on HuggingFace this repo is {model.hf_gated}."))
        print(yellow("ModelScope hosts the same files without authentication."))
        if not ask_yes_no("Continue with HuggingFace? ", default_yes=True):
            return "modelscope" if ask_yes_no("Use ModelScope instead? ", default_yes=True) else sys.exit(0)
    return source


def query_size_bytes(model: Model, source: str) -> tuple[int, bool]:
    """Total byte count: live metadata when possible, registry fallback otherwise."""
    try:
        if source == "direct":
            import urllib.request

            req = urllib.request.Request(model.direct_url, method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as resp:
                length = int(resp.headers.get("Content-Length", 0))
            if length > 0:
                return length, True
        elif source == "huggingface":
            from fnmatch import fnmatch

            from huggingface_hub import HfApi

            info = HfApi().model_info(model.repo_id, files_metadata=True)
            total = 0
            for f in info.siblings:
                if model.allow_patterns and not any(fnmatch(f.rfilename, p) for p in model.allow_patterns):
                    continue
                total += f.size or 0
            if total > 0:
                return total, True
    except Exception:
        pass
    return int(model.approx_gb * 1e9), False


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
    # Session-average speed: hub downloads land bytes in bursts, so an
    # instantaneous rate flaps between 0 and spikes.
    start_bytes = _dir_bytes(target)
    start_time = time.monotonic()
    while not stop.wait(1.0):
        done = _dir_bytes(target)
        speed = max(0.0, (done - start_bytes) / max(time.monotonic() - start_time, 1e-6))
        _draw_bar(done, expected, speed, final=False)


def _download_direct(url: str, dest: Path) -> None:
    """Stream a direct URL to dest with simple Range-based resume."""
    import urllib.request

    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)
    if pos:
        req.add_header("Range", f"bytes={pos}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if pos and resp.status != 206:  # server ignored the Range header
            pos = 0
        with open(part, "ab" if pos else "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    part.rename(dest)


def download(model: Model, source: str, root: Path, expected: int) -> Path:
    target = root / model.subdir
    if target.is_dir() and any(target.iterdir()):
        print(yellow(f"{target} already has content — resuming/skipping finished files."))
    origin = model.direct_url if source == "direct" else model.repo_id
    print(bold(f"Downloading {origin} -> {target}"))

    watch = sys.stdout.isatty() and source in ("huggingface", "direct")
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_progress, args=(target, expected, stop), daemon=True)
    started = time.monotonic()
    if watch:
        watcher.start()
    try:
        if source == "direct":
            target.mkdir(parents=True, exist_ok=True)
            dest = target / Path(model.direct_url).name
            if not dest.is_file():
                _download_direct(model.direct_url, dest)
        elif source == "huggingface":
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model.repo_id,
                local_dir=str(target),
                allow_patterns=list(model.allow_patterns) if model.allow_patterns else None,
            )
        else:
            from modelscope.hub.snapshot_download import snapshot_download

            snapshot_download(
                model.ms_id or model.repo_id,
                local_dir=str(target),
                allow_file_pattern=list(model.allow_patterns) if model.allow_patterns else None,
            )
    finally:
        stop.set()
        if watch:
            watcher.join()
    if watch:
        elapsed = time.monotonic() - started
        _draw_bar(expected, expected, expected / max(elapsed, 1e-6), final=True)

    if model.manifest is not None:
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(model.manifest, indent=2) + "\n", encoding="utf-8")
        print(green(f"Wrote {manifest_path}"))
    return target.resolve()


def write_config(model: Model, target: Path) -> None:
    config_path = CONFIG_DIR / model.config
    value = target / model.config_subpath if model.config_subpath else target
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(model_path:\s*)\S+([^\n]*)$", re.MULTILINE)
    if not pattern.search(text):
        print(red(f"Warning: field 'model_path' not found in {config_path}; update it manually."))
        return
    text = pattern.sub(lambda m: f"{m.group(1)}{value}{m.group(2)}", text, count=1)
    config_path.write_text(text, encoding="utf-8")
    rel = config_path.relative_to(REPO_ROOT)
    print(green(f"Updated {rel}: model_path -> {value}"))


def main() -> None:
    print(bold("OpenWAM visual-encoder checkpoint downloader"))
    print()
    root = choose_storage_root()
    model = choose_model()
    source = choose_source(model)

    expected, live = query_size_bytes(model, source)
    origin = "" if live else " (approximate)"
    print()
    print(yellow(f"{model.name} needs about {expected / 1e9:.1f} GB{origin} under {root}."))
    if not ask_yes_no("Start the download? ", default_yes=True):
        print(cyan("Aborted — nothing downloaded."))
        sys.exit(0)

    target = download(model, source, root, expected)

    print()
    print(green(f"Done. {model.name} is saved under:"))
    print(green(f"  {target}"))
    write_config(model, target)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(yellow("Interrupted — partial downloads resume on the next run."))
        sys.exit(130)
