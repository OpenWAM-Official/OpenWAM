#!/usr/bin/env python3
"""Interactive downloader for OpenWAM benchmark datasets.

Fetches a benchmark's training data from HuggingFace into the assets data
directory (folder name = benchmark name), then makes sure the in-dataset
normalization stats exist — computing them with the matching script under
openwam/dataloader/utils/stats_computation/ when they are missing.

Usage:
    python scripts/download_assets/download_benchmark_data.py
"""

from __future__ import annotations

import os

# HuggingFace downloads render their own stack of progress bars (including the
# xet backend's, which repaint on interpreter exit and smear over our status
# lines). Silence them all — this script draws a single progress bar itself.
# Must be set before huggingface_hub is imported anywhere.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "dataloader"
DEFAULT_ROOT = Path.cwd() / "assets" / "benchmark_data"

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


# ── benchmark registry ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Benchmark:
    name: str  # display name
    repo_id: str  # HuggingFace dataset repo
    subdir: str  # folder under the storage root = benchmark name
    approx_gb: float  # fallback size when the live metadata query fails
    config: str  # yaml under configs/dataloader/ pointed at the download
    allow_patterns: tuple[str, ...] | None = None  # partial download
    ignore_patterns: tuple[str, ...] | None = None
    strip_prefix: str | None = None  # repo subpath relocated to the folder root
    dataset_subpath: str | None = None  # dataset_dir points below the folder root
    config_note: str | None = None  # extra fields the user must still set


BENCHMARKS: dict[str, Benchmark] = {
    "1": Benchmark(
        "RoboTwin2.0",
        "TianxingChen/RoboTwin2.0",
        "robotwin2.0",
        approx_gb=415.0,
        config="robotwin.yaml",
        dataset_subpath="dataset",
        # Official upstream zips: 100 large archives transfer far better than
        # the ~110k extracted files. They are unpacked (and removed) right
        # after the download; the stats step then computes the normalization
        # stats from the extracted corpus.
        allow_patterns=("dataset/*aloha-agilex*.zip",),
    ),
    "2": Benchmark(
        "RoboDojo",
        "RoboDojo-Benchmark/RoboDojo",
        "robodojo",
        approx_gb=488.0,
        config="robodojo.yaml",
        allow_patterns=("data/RoboDojo/**",),
        strip_prefix="data/RoboDojo",
        config_note="robodojo.yaml serves both corpora — keep variant: sim for this one.",
    ),
    "3": Benchmark(
        "RoboDojo-Real",
        "RoboDojo-Benchmark/RoboDojo",
        "robodojo-real",
        approx_gb=255.0,
        config="robodojo.yaml",
        allow_patterns=("data/RoboDojo_real/**",),
        strip_prefix="data/RoboDojo_real",
        config_note="robodojo.yaml serves both corpora — also set variant: real (and the embodiment) for this one.",
    ),
    "4": Benchmark("LIBERO", "OpenWAM/LIBERO", "libero", approx_gb=1.9, config="libero.yaml"),
    "5": Benchmark(
        "VLABench",
        "OpenWAM/VLABench",
        "vlabench",
        approx_gb=13.2,
        config="vlabench.yaml",
        # Mirror of VLABench/vlabench_primitive_ft_lerobot_video repacked into
        # 16 large tars (the upstream repo's ~19k small files download at
        # request-latency, not bandwidth); unpacked automatically.
    ),
    "6": Benchmark(
        "EBench",
        "InternRobotics/EBench-Dataset",
        "ebench",
        approx_gb=305.0,
        config="ebench.yaml",
        ignore_patterns=(".ipynb_checkpoints/**", "**/.ipynb_checkpoints/**"),
    ),
    "7": Benchmark("RoboCasa365", "OpenWAM/RoboCasa365", "robocasa365", approx_gb=78.0, config="robocasa365.yaml"),
    "8": Benchmark(
        "RoboCasa_GR1",
        "OpenWAM/RoboCasa_GR1",
        "robocasa-gr1",
        approx_gb=43.9,
        config="robocasa_gr1.yaml",
        # 24 per-task tars + loose meta/ (the flat layout's ~48k small files
        # download at request-latency, not bandwidth); unpacked automatically.
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


def choose_benchmark() -> Benchmark:
    key = ask_choice(
        "Select the benchmark to download",
        [f"({k}) {b.name}" for k, b in BENCHMARKS.items()],
        "Benchmark number: ",
    )
    return BENCHMARKS[key]


def query_download_size(bench: Benchmark) -> tuple[int, bool]:
    """Byte count from live HuggingFace metadata, or the registry fallback."""
    try:
        from fnmatch import fnmatch

        from huggingface_hub import HfApi

        info = HfApi().dataset_info(bench.repo_id, files_metadata=True)
        total = 0
        for f in info.siblings:
            if bench.allow_patterns and not any(fnmatch(f.rfilename, p) for p in bench.allow_patterns):
                continue
            if bench.ignore_patterns and any(fnmatch(f.rfilename, p) for p in bench.ignore_patterns):
                continue
            total += f.size or 0
        return total, True
    except Exception:
        return int(bench.approx_gb * 1e9), False


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


def _extracted_zip_ignores(target: Path) -> list[str]:
    """Ignore patterns for archives an earlier run already unpacked and removed."""
    dataset_dir = target / "dataset"
    if not dataset_dir.is_dir():
        return []
    return [f"dataset/{d.parent.name}/{d.name}.zip" for d in sorted(dataset_dir.glob("*/aloha-agilex*")) if d.is_dir()]


def _extract_zips(bench: Benchmark, target: Path) -> None:
    """Unpack every downloaded archive next to itself, then drop the archive.

    Archives are processed one at a time so the transient disk overhead stays
    bounded by the largest single zip. Each carries its own top-level
    directory (e.g. aloha-agilex_clean_50/), so extraction lands the
    canonical layout directly.
    """
    import zipfile

    zips = sorted((target / "dataset").glob("*/aloha-agilex*.zip"))
    if not zips:
        return
    print(bold(f"Extracting {len(zips)} archives"))
    for i, zip_path in enumerate(zips, 1):
        print(f"  [{i}/{len(zips)}] {zip_path.relative_to(target)}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
        zip_path.unlink()


def _extracted_tar_ignores(target: Path) -> list[str]:
    """Ignore patterns for root tars an earlier run already unpacked and removed.

    Tar names encode their extraction path: ``videos__image__chunk-000.tar``
    unpacks to ``videos/image/chunk-000``. Any directory that exists therefore
    maps back to the archive that produced it (patterns for directories that
    never had an archive are harmless no-ops).
    """
    if not target.is_dir():
        return []
    ignores = []
    for d1 in target.iterdir():
        if not d1.is_dir() or d1.name in {".cache", ".extract_tmp", "meta"}:
            continue
        ignores.append(f"{d1.name}.tar")
        for d2 in d1.iterdir():
            if not d2.is_dir():
                continue
            ignores.append(f"{d1.name}__{d2.name}.tar")
            for d3 in d2.iterdir():
                if d3.is_dir():
                    ignores.append(f"{d1.name}__{d2.name}__{d3.name}.tar")
    return ignores


def _extract_tars(bench: Benchmark, target: Path) -> None:
    """Unpack every root ``<a>__<b>.tar`` to ``<a>/<b>``, then drop the archive.

    Extraction goes through a temp dir and is moved into place afterwards, so
    an interrupted unpack never leaves a half-filled directory that the resume
    logic would mistake for a finished one.
    """
    import shutil as _shutil
    import tarfile

    tmp_root = target / ".extract_tmp"
    _shutil.rmtree(tmp_root, ignore_errors=True)
    tars = sorted(target.glob("*.tar"))
    if not tars:
        return
    print(bold(f"Extracting {len(tars)} archives"))
    for i, tar_path in enumerate(tars, 1):
        rel = tar_path.name[: -len(".tar")].replace("__", "/")
        print(f"  [{i}/{len(tars)}] {tar_path.name} -> {rel}/")
        _shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir()
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp_root)
        src = tmp_root / rel
        if not src.is_dir():
            raise RuntimeError(f"{tar_path.name} did not contain the expected member {rel}/")
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        _shutil.rmtree(dest, ignore_errors=True)
        _shutil.move(str(src), str(dest))
        tar_path.unlink()
    _shutil.rmtree(tmp_root, ignore_errors=True)


_POST_DOWNLOAD = {
    "robotwin2.0": _extract_zips,
    "vlabench": _extract_tars,
    "robocasa-gr1": _extract_tars,
}


def _already_relocated(bench: Benchmark, target: Path) -> bool:
    """A strip_prefix benchmark whose content already sits at the folder root."""
    if bench.strip_prefix is None or not target.is_dir():
        return False
    children = {c.name for c in target.iterdir()} - {".cache", "data"}
    return bool(children)


def _relocate(bench: Benchmark, target: Path) -> None:
    """Move <target>/<strip_prefix>/* up to <target>/ after the download."""
    if bench.strip_prefix is None:
        return
    src = target / bench.strip_prefix
    if not src.is_dir():
        return
    for child in src.iterdir():
        dest = target / child.name
        if dest.exists():
            print(yellow(f"  {dest} already exists — keeping it, skipping the fresh copy."))
            continue
        shutil.move(str(child), str(dest))
    # Drop the now-empty scaffold (data/RoboDojo[/…]).
    top = target / bench.strip_prefix.split("/")[0]
    shutil.rmtree(top, ignore_errors=True)


def download(bench: Benchmark, root: Path, expected: int) -> Path:
    target = root / bench.subdir
    if _already_relocated(bench, target):
        print(yellow(f"{target} already has content — skipping the download."))
        return target.resolve()
    if target.is_dir() and any(target.iterdir()):
        print(yellow(f"{target} already has content — resuming/skipping finished files."))
    print(bold(f"Downloading {bench.repo_id} -> {target}"))
    print(
        yellow(
            "The repo file list is fetched first — for repos with many files "
            "the bar can sit at 0% for minutes before bytes start landing."
        )
    )

    # Archives already unpacked (and removed) by an earlier run must not be
    # re-downloaded just because the archive itself is gone.
    ignore = list(bench.ignore_patterns) if bench.ignore_patterns else []
    ignore += _extracted_zip_ignores(target)
    if _POST_DOWNLOAD.get(bench.subdir) is _extract_tars:
        ignore += _extracted_tar_ignores(target)

    watch = sys.stdout.isatty()
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_progress, args=(target, expected, stop), daemon=True)
    started = time.monotonic()
    if watch:
        watcher.start()
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=bench.repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=list(bench.allow_patterns) if bench.allow_patterns else None,
            ignore_patterns=ignore or None,
            max_workers=32,  # these corpora are dominated by many small files
        )
    finally:
        stop.set()
        if watch:
            watcher.join()
    if watch:
        elapsed = time.monotonic() - started
        _draw_bar(expected, expected, expected / max(elapsed, 1e-6), final=True)
    _relocate(bench, target)
    return target.resolve()


# ── normalization stats ──────────────────────────────────────────────────────
def _ensure_openwam_importable() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _report(path: Path) -> bool:
    if path.is_file():
        print(green(f"  found   {path}"))
        return True
    print(yellow(f"  missing {path} — computing..."))
    return False


def _stats_robotwin(target: Path) -> None:
    dataset_dir = target / "dataset"
    for variant in ("clean_50", "both"):
        path = dataset_dir / "meta" / f"robotwin_{variant}_normalization_stats.npy"
        if _report(path):
            continue
        subprocess.run(
            [
                sys.executable,
                "-m",
                "openwam.dataloader.utils.stats_computation.robotwin_stats_computation",
                "--dataset_dir",
                str(dataset_dir),
                "--variant",
                variant,
                "--embodiment",
                "aloha-agilex",
            ],
            cwd=REPO_ROOT,
            check=True,
        )


def _stats_robodojo(target: Path) -> None:
    from openwam.dataloader.robodojo import ensure_robodojo_stats

    path = target / "meta" / "robodojo_normalization_stats.npy"
    if _report(path):
        return
    ensure_robodojo_stats(target, variant="sim", embodiment="arx_x5")


def _stats_robodojo_real(target: Path) -> None:
    from openwam.dataloader.robodojo import ensure_robodojo_stats
    from openwam.dataloader.robodojo_contract import ROBODOJO_REAL_EMBODIMENTS

    present: set[str] = set()
    for task in target.iterdir():
        if task.is_dir() and task.name not in {"meta", ".cache"}:
            present.update(c.name for c in task.iterdir() if c.is_dir())
    embodiments = [e for e in ROBODOJO_REAL_EMBODIMENTS if e in present]
    if not embodiments:
        print(red(f"  no known embodiment directories found under {target}"))
        return
    for embodiment in embodiments:
        path = target / "meta" / f"robodojo_real_{embodiment}_normalization_stats.npy"
        if _report(path):
            continue
        ensure_robodojo_stats(target, variant="real", embodiment=embodiment)


def _stats_libero(target: Path) -> None:
    path = target / "meta" / "libero_normalization_stats.npy"
    if _report(path):
        return
    from openwam.dataloader.utils.stats_computation.libero_stats_computation import (
        build_and_save_libero_stats,
    )

    build_and_save_libero_stats(target, output=path)


def _stats_vlabench(target: Path) -> None:
    path = target / "meta" / "vlabench_normalization_stats.npy"
    if _report(path):
        return
    from openwam.dataloader.utils.stats_computation.vlabench_stats_computation import (
        build_and_save_vlabench_stats,
    )

    build_and_save_vlabench_stats(target, output=path)


def _stats_ebench(target: Path) -> None:
    path = target / "meta" / "ebench_normalization_stats.npy"
    if _report(path):
        return
    from openwam.dataloader.utils.stats_computation.ebench_stats_computation import (
        build_and_save_ebench_stats,
    )

    build_and_save_ebench_stats(str(target), output=str(path))


def _stats_robocasa365(target: Path) -> None:
    path = target / "meta" / "robocasa365_normalization_stats.npy"
    if _report(path):
        return
    from openwam.dataloader.utils.stats_computation.robocasa365_stats_computation import (
        build_and_save_robocasa365_stats,
    )

    roots = sorted(str(child) for child in target.iterdir() if (child / "meta" / "info.json").is_file())
    if not roots:
        raise FileNotFoundError(f"no compact repos (dirs with meta/info.json) under {target}")
    build_and_save_robocasa365_stats(roots, path)


def _stats_robocasa_gr1(target: Path) -> None:
    path = target / "meta" / "robocasa_gr1_normalization_stats.npy"
    if _report(path):
        return
    from omegaconf import OmegaConf

    from openwam.dataloader.robocasa_gr1 import RoboCasaGR1Dataset
    from openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation import (
        build_and_save_robocasa_gr1_stats,
    )

    cfg = OmegaConf.load(REPO_ROOT / "configs" / "dataloader" / "robocasa_gr1.yaml")
    cfg.dataset_dir = str(target)
    cfg.normalize_mode = None
    dataset = RoboCasaGR1Dataset.from_config(cfg, split=str(OmegaConf.select(cfg, "split", default="train")))
    build_and_save_robocasa_gr1_stats(dataset, path)


_STATS_STEPS = {
    "robotwin2.0": _stats_robotwin,
    "robodojo": _stats_robodojo,
    "robodojo-real": _stats_robodojo_real,
    "libero": _stats_libero,
    "vlabench": _stats_vlabench,
    "ebench": _stats_ebench,
    "robocasa365": _stats_robocasa365,
    "robocasa-gr1": _stats_robocasa_gr1,
}


def ensure_stats(bench: Benchmark, target: Path) -> None:
    print()
    print(bold(f"Checking normalization stats for {bench.name}"))
    _ensure_openwam_importable()
    _STATS_STEPS[bench.subdir](target)


def write_config(bench: Benchmark, target: Path) -> None:
    """Point the benchmark's dataloader yaml at the download."""
    config_path = CONFIG_DIR / bench.config
    dataset_dir = target / bench.dataset_subpath if bench.dataset_subpath else target
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(dataset_dir:\s*)\S+([^\n]*)$", re.MULTILINE)
    if not pattern.search(text):
        print(red(f"Warning: field 'dataset_dir' not found in {config_path}; update it manually."))
        return
    text = pattern.sub(lambda m: f"{m.group(1)}{dataset_dir}{m.group(2)}", text, count=1)
    config_path.write_text(text, encoding="utf-8")
    rel = config_path.relative_to(REPO_ROOT) if config_path.is_relative_to(REPO_ROOT) else config_path
    print(green(f"Updated {rel}: dataset_dir -> {dataset_dir}"))
    if bench.config_note:
        print(yellow(f"Note: {bench.config_note}"))


def main() -> None:
    print(bold("OpenWAM benchmark data downloader"))
    print()
    root = choose_storage_root()
    bench = choose_benchmark()

    expected, live = query_download_size(bench)
    origin = "" if live else " (approximate)"
    print()
    print(yellow(f"{bench.name} needs about {expected / 1e9:.1f} GB{origin} under {root}."))
    if not ask_yes_no("Start the download? ", default_yes=True):
        print(cyan("Aborted — nothing downloaded."))
        sys.exit(0)

    target = download(bench, root, expected)
    post = _POST_DOWNLOAD.get(bench.subdir)
    if post:
        post(bench, target)
    ensure_stats(bench, target)

    print()
    print(green(f"Done. {bench.name} is ready under:"))
    print(green(f"  {target}"))
    write_config(bench, target)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(yellow("Interrupted — partial downloads resume on the next run."))
        sys.exit(130)
