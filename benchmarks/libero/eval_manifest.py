#!/usr/bin/env python3
"""Generate sealed LIBERO trial manifests without running policy inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils.eval_manifest import seal_manifest, write_manifest  # noqa: E402
from benchmarks.utils.rng_domain import GlobalRngDomain, encode_numpy_state  # noqa: E402
from benchmarks.utils.task_progress import write_progress  # noqa: E402

BENCHMARK = "libero"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def task_assets(task) -> tuple[Path, Path]:
    from libero.libero import get_libero_path

    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return init_path, bddl_path


def build_manifest(
    *,
    config_path: Path,
    suite: str,
    task_id: int,
    total_trials: int,
    progress_file: Path | None = None,
) -> dict[str, Any]:
    """Build the complete reset-only RNG manifest for one LIBERO task."""

    if total_trials <= 0:
        raise ValueError("total_trials must be positive")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if cfg.get("rng_mode") != "environment" or cfg.get("reseed_each_trial") is not False:
        raise ValueError("LIBERO manifests require the continuous environment RNG protocol")

    from libero.libero import benchmark

    mapping = benchmark.get_benchmark_dict()
    task_suite = mapping[suite]()
    task = task_suite.get_task(task_id)
    init_states = task_suite.get_task_init_states(task_id)
    init_path, bddl_path = task_assets(task)
    entries: list[dict[str, Any]] = []
    # Import lazily so manifest schema and cache validation remain testable without
    # a LIBERO installation.
    from single_eval import _make_env_with_randomization_retries

    env = _make_env_with_randomization_retries(task, cfg)
    try:
        env.seed(int(cfg.get("seed", 42)))
        domain = GlobalRngDomain.from_numpy_state(np.random.get_state())
        for trial in range(total_trials):
            entries.append(
                {
                    "trial": trial,
                    "init_state_index": trial % len(init_states),
                    "pre_reset_numpy_state": encode_numpy_state(domain.numpy_state),
                }
            )
            with domain.activate():
                env.reset()
            if progress_file is not None:
                write_progress(
                    progress_file,
                    {
                        "completed": trial + 1,
                        "total": total_trials,
                        "trial": trial,
                    },
                )
    finally:
        env.close()
    return seal_manifest(
        {
            "benchmark": BENCHMARK,
            "suite": suite,
            "task_id": task_id,
            "seed": int(cfg.get("seed", 42)),
            "init_state_asset": {"name": task.init_states_file, "sha256": file_sha256(init_path)},
            "bddl_asset": {"name": task.bddl_file, "sha256": file_sha256(bddl_path)},
            "entries": entries,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--total-trials", type=int, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        config_path=args.config.resolve(),
        suite=args.suite,
        task_id=args.task_id,
        total_trials=args.total_trials,
        progress_file=args.progress_file,
    )
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.result_file, manifest)
    print(
        json.dumps(
            {
                "operation": "build_manifest",
                "suite": args.suite,
                "task_id": args.task_id,
                "manifest_hash": manifest["manifest_hash"],
                "entries": len(manifest["entries"]),
                "path": str(args.result_file),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
