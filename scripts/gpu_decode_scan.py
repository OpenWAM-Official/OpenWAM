#!/usr/bin/env python3
"""GPU (NVDEC) full decode sweep for OpenWAM dataloaders — fast integrity scan.

Why
---
``scripts/scan_dataset.py`` verifies the *exact* training decode path, but that
path is pure **CPU PyAV** (``video_io.decode_video_frames`` forces
``hwaccel="none"`` by design). It is decode-bound and leaves GPU decode engines
idle, so a full every-window sweep of a large source can be slow.

This tool decodes **every video end-to-end on the GPU NVDEC engines**
(``ffmpeg -hwaccel cuda``, using the dedicated decode silicon that is separate
from the CUDA cores), sharded across all GPUs. It accelerates detection of the
class that crashes training — **truncated / undecodable clips**:

  * ``ffmpeg -xerror`` non-zero exit / decoder error  → the clip is broken.
  * decoded ``frame=N`` < the episode's expected length (single-episode files
    only, to avoid false positives on concatenated multi-episode files) →
    silently truncated.

Reuse
-----
It writes failure records in **the exact JSON format** of
``scripts/scan_dataset.py`` (``failures.shard*.jsonl`` under
``{out_dir}/{config}/``), so you union them into the reader-consumed exclusion
files with the existing merge step::

  python scripts/scan_dataset.py emit --config <cfg> --out-dir <same out_dir>

Caveat
------
NVDEC verifies the video is a **decodable stream** (the crash class); it is NOT
byte-identical to the CPU PyAV training decode. Treat this as the fast bulk
integrity pass, then run ``scan_dataset.py`` (CPU) to confirm the exact training
path is clean now that broken clips are excluded.

Examples
--------
  # Full GPU sweep of RoboCOIN across all GPUs (writes failures jsonl)
  python scripts/gpu_decode_scan.py --config robocoin --out-dir scan_out

  # Then union failures into the reader blacklist (reuses scan_dataset)
  python scripts/scan_dataset.py emit --config robocoin --out-dir scan_out

  # Shard across processes/nodes (disjoint file ranges), 6 concurrent decodes/GPU
  python scripts/gpu_decode_scan.py --config robocoin --shard 0 --num-shards 4 \
      --procs-per-gpu 6 --out-dir scan_out

Sharding caveat
---------------
The per-shard work list is ``jobs[shard::num_shards]`` over the files of the
freshly built dataset, so the shard slicing depends on which files are present.
Do NOT run ``scan_dataset.py emit`` (which writes the exclusion file the reader
then honors) until **all** shards have finished: emitting mid-run drops the
now-excluded files from a later shard's dataset build, re-strides ``jobs[shard::N]``,
and can leave some not-yet-scanned files covered by no shard — silently treated
as clean. Complete every shard, then emit once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# NOTE: ``scan_dataset`` (config build + leaf enumeration + exclusion targets, so
# the two tools agree on datasets, keys, and output format) and ``openwam`` are
# imported lazily inside the functions below — the sys.path insert above must run
# first, and deferring keeps ruff's import ordering happy (same pattern as
# scan_dataset.py's own openwam imports).


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------


def _detect_num_gpus() -> int:
    """Number of visible NVIDIA GPUs (respects CUDA_VISIBLE_DEVICES)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        # Explicitly empty means "no GPU visible" — NVDEC decode would then fail
        # for EVERY file. Falling back to an nvidia-smi count (the old behavior)
        # would let a GPU-less run silently record the whole dataset as corrupt.
        # Fail fast instead of manufacturing dataset-wide bogus failures.
        raise SystemExit(
            "[gpu-scan] CUDA_VISIBLE_DEVICES is set but empty — no GPU visible for NVDEC. "
            "Unset it or set real device ids before scanning."
        )
    if cvd is not None and cvd.strip() != "":
        return max(1, len([x for x in cvd.split(",") if x.strip() != ""]))
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        n = len([ln for ln in out.stdout.splitlines() if ln.strip().startswith("GPU ")])
        return max(1, n)
    except Exception:  # noqa: BLE001
        return 1


def _check_ffmpeg_cuda() -> None:
    """Abort loudly if this ffmpeg has no CUDA/NVDEC hwaccel.

    Without it EVERY file errors ``Unrecognized hwaccel: cuda`` and would be
    recorded as a decode failure — an environmental problem that must not be
    laundered into a dataset-wide exclusion list. Catch it once, up front.
    """
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[gpu-scan] cannot run ffmpeg ({type(e).__name__}: {e}); is it installed and on PATH?")
    if "cuda" not in (out.stdout + out.stderr).lower():
        raise SystemExit(
            "[gpu-scan] this ffmpeg reports no 'cuda' hwaccel — refusing to scan, because every file "
            "would falsely record as corrupt. Install an ffmpeg built with NVDEC (--enable-cuda/--enable-cuvid) "
            f"and CUDA on PATH. `ffmpeg -hwaccels` returned: {' '.join(out.stdout.split())[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Work list: unique video file -> {kind, target, expected_frames, keys}
# ---------------------------------------------------------------------------


def _worklist_lerobot(leaf, kind: str, target: str, files: Dict[str, dict]) -> None:
    """LeRobotV3Reader (e.g. RoboCOIN): head video path per episode from ``_eps_df``.

    Head cam = ``leaf._head_camera`` (only the head view is critical; wrist decode
    failures are tolerated by the reader). Concatenated files back several episodes
    → frame check skipped, exit-code only.
    """
    dd = Path(leaf._dataset_dir)
    cam = leaf._head_camera
    tmpl = leaf._video_path_template
    eps = leaf._eps_df
    chunk_col = f"videos/{cam}/chunk_index"
    file_col = f"videos/{cam}/file_index"
    if chunk_col not in eps.columns or file_col not in eps.columns:
        raise KeyError(
            f"gpu_decode_scan: leaf {getattr(leaf, '_dataset_id', '?')} missing head-cam columns for {cam!r}"
        )
    for _, row in eps.iterrows():
        ep = int(row["episode_index"])
        length = int(row["length"])
        path = str(dd / tmpl.format(video_key=cam, chunk_index=int(row[chunk_col]), file_index=int(row[file_col])))
        # LeRobot v3 mp4s pack multiple episodes → never trust the frame-count
        # cross-check (a filtered/subsampled _eps_df can leave one surviving
        # episode whose length is far below the file's real frame count).
        job = files.setdefault(
            path, {"path": path, "kind": kind, "target": target, "expected": 0, "keys": [], "single_ep_file": False}
        )
        job["keys"].append(ep)
        job["expected"] += length


def _build_worklist(ds) -> List[dict]:
    from scan_dataset import _enumerate_leaves, _leaf_info

    from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader

    files: Dict[str, dict] = {}
    for leaf in _enumerate_leaves(ds):
        kind, target = _leaf_info(leaf)
        if isinstance(leaf, LeRobotV3Reader):
            _worklist_lerobot(leaf, kind, target, files)
        else:
            raise TypeError(f"gpu_decode_scan: unsupported leaf type {type(leaf).__name__}")
    return list(files.values())


# ---------------------------------------------------------------------------
# Decode one file on a GPU (NVDEC)
# ---------------------------------------------------------------------------


def _decode_file(job: dict, gpu: int, timeout: int) -> dict:
    """Full NVDEC decode of ``job['path']``. Returns ``{ok, err?}``.

    ``ok=False`` on a decoder error / non-zero exit, or (single-episode files
    only) when the decoded frame count is short of the expected episode length.
    """
    path = job["path"]
    if not os.path.exists(path):
        return {"ok": False, "err": f"missing_file: {path}"}
    # -hwaccel_output_format cuda keeps decoded frames on the GPU (no per-frame
    # device→host copy), which removes host-memory/PCIe contention when many
    # streams decode concurrently. -xerror still trips on any decoder error and
    # -progress still reports frame=N regardless of output format.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-hwaccel_device",
        str(gpu),
        "-i",
        path,
        "-an",
        "-f",
        "null",
        "-progress",
        "pipe:1",
        "-",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": f"timeout: NVDEC decode exceeded {timeout}s"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}
    if p.returncode != 0:
        err = " ".join((p.stderr or "").split())[:300] or f"ffmpeg_exit_{p.returncode}"
        return {"ok": False, "err": f"decode_error: {err}"}
    frames: Optional[int] = None
    for line in p.stdout.splitlines():
        if line.startswith("frame="):
            try:
                frames = int(line.split("=", 1)[1])
            except ValueError:
                pass
    # Keep an opt-in frame-count check for future public readers that can prove
    # one episode owns the complete file. Current LeRobot v3 jobs leave it off
    # because one file may contain several episodes.
    if job.get("single_ep_file") and frames is not None and frames < int(job["expected"]):
        return {"ok": False, "err": f"truncated: decoded {frames} < expected {job['expected']} frames"}
    return {"ok": True}


def _is_transient(err: str) -> bool:
    """True for failures that are environmental / contention-related rather than
    genuine corruption, so they are worth one retry before being condemned.

    A per-file timeout or an NVDEC session-creation failure under high
    ``--procs-per-gpu`` concurrency is transient; a clean decoder error (``-xerror``
    tripped), a missing file, or a frame-count truncation is a real defect of the
    clip and must NOT be retried away.
    """
    return not err.startswith(("decode_error:", "missing_file:", "truncated:"))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="active dataloader config group (for example robocoin)")
    ap.add_argument("--out-dir", default="scan_out")
    ap.add_argument("--num-gpus", type=int, default=None, help="default: auto-detect")
    ap.add_argument("--procs-per-gpu", type=int, default=6, help="concurrent NVDEC decodes per GPU")
    ap.add_argument("--timeout", type=int, default=600, help="per-file ffmpeg timeout (s)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="decode at most N files (smoke test)")
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    from scan_dataset import _atomic_write_text, _build_dataset, _validate_shard

    # Same 0-based shard contract as the CPU scanner: a 1-based misuse would give
    # every shard non-empty work and exit 0 while jobs[0] etc. are covered by NO
    # shard — silently treated as clean (the module docstring's own warning).
    _validate_shard(args)
    _check_ffmpeg_cuda()  # fail before a non-NVDEC ffmpeg manufactures dataset-wide bogus failures
    num_gpus = args.num_gpus or _detect_num_gpus()
    n_workers = max(1, num_gpus * max(1, args.procs_per_gpu))

    print(f"[gpu-scan {args.config}] building dataset + work list…", flush=True)
    ds = _build_dataset(args.config)
    jobs = _build_worklist(ds)
    jobs.sort(key=lambda j: j["path"])  # stable order across runs/shards
    n_all_files = len(jobs)
    # NOTE: this stride assumes the full file set is present. Don't run `emit`
    # before every shard finishes (see the "Sharding caveat" in the module
    # docstring) — a mid-run emit shrinks the dataset and re-strides these shards.
    # (No max(1, …) fallback: _validate_shard already rejected num_shards < 1 —
    # a fallback here would only mask that class of invalid input.)
    jobs = jobs[args.shard :: args.num_shards]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    n_files = len(jobs)
    n_eps = sum(len(j["keys"]) for j in jobs)
    print(
        f"[gpu-scan {args.config}] unique_files={n_all_files} this_shard={n_files} (episodes={n_eps}) "
        f"gpus={num_gpus} procs_per_gpu={args.procs_per_gpu} workers={n_workers} "
        f"shard={args.shard}/{args.num_shards}",
        flush=True,
    )
    if n_files == 0:
        print("[gpu-scan] nothing to scan for this shard.", flush=True)
        return 0

    out_dir = Path(args.out_dir) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    # Name matches scan_dataset's glob `failures.shard*.jsonl` so `emit` picks it up.
    fail_path = out_dir / f"failures.shard{args.shard}-of-{args.num_shards}.gpu.jsonl"
    prog_path = out_dir / f"progress.gpu.shard{args.shard}-of-{args.num_shards}.json"
    fail_f = open(fail_path, "a", buffering=1, encoding="utf-8")

    t0 = time.time()
    scanned = 0
    n_bad_files = 0
    n_bad_eps = 0

    def _run(job_gpu):
        job, gpu = job_gpu
        res = _decode_file(job, gpu, args.timeout)
        if not res["ok"] and _is_transient(res["err"]):
            # One retry on a different GPU: don't condemn (and later exclude) a
            # healthy clip for a transient timeout / NVDEC session hiccup.
            res = _decode_file(job, (gpu + 1) % num_gpus, args.timeout)
        return job, res

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run, (job, i % num_gpus)) for i, job in enumerate(jobs)]
            for fut in as_completed(futures):
                job, res = fut.result()
                scanned += 1
                if not res["ok"]:
                    n_bad_files += 1
                    for key in job["keys"]:
                        n_bad_eps += 1
                        rec = {
                            "kind": job["kind"],
                            "target": job["target"],
                            "key": key,
                            "local": -1,  # not a window index (file-level failure)
                            "err": res["err"],
                            "path": job["path"],
                        }
                        fail_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if scanned % args.log_every == 0 or scanned == n_files:
                    dt = time.time() - t0
                    rate = scanned / max(dt, 1e-6)
                    eta = (n_files - scanned) / max(rate, 1e-6)
                    _atomic_write_text(
                        prog_path,
                        json.dumps(
                            {
                                "scanned_files": scanned,
                                "total_files": n_files,
                                "bad_files": n_bad_files,
                                "bad_episodes": n_bad_eps,
                                "rate_files_per_s": round(rate, 1),
                                "eta_s": int(eta),
                                "elapsed_s": int(dt),
                            }
                        ),
                    )
                    print(
                        f"[gpu-scan {args.config}] {scanned}/{n_files} files ({100 * scanned / n_files:.1f}%) "
                        f"bad_files={n_bad_files} bad_eps={n_bad_eps} rate={rate:.0f} files/s "
                        f"eta={eta / 3600:.2f}h",
                        flush=True,
                    )
    finally:
        fail_f.close()

    dt = time.time() - t0
    (out_dir / f"done.gpu.shard{args.shard}-of-{args.num_shards}.json").write_text(
        json.dumps(
            {"scanned_files": scanned, "bad_files": n_bad_files, "bad_episodes": n_bad_eps, "elapsed_s": int(dt)}
        )
    )
    status = "CLEAN ✅" if n_bad_files == 0 else f"{n_bad_files} BAD FILES ({n_bad_eps} eps) ❌"
    print(
        f"[gpu-scan {args.config}] DONE shard {args.shard}/{args.num_shards}: {status} "
        f"scanned {scanned} files in {dt / 3600:.2f}h. "
        f"Union exclusions: python scripts/scan_dataset.py emit --config {args.config} --out-dir {args.out_dir}",
        flush=True,
    )
    return 0 if n_bad_files == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
