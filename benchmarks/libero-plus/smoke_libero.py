#!/usr/bin/env python3
"""Smoke checks for a LIBERO-plus installation."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from glob import glob
from pathlib import Path

import yaml


def _repo_root() -> Path:
    raw_root = os.environ.get("LIBERO_PLUS_PATH", "")
    if not raw_root:
        raise SystemExit("LIBERO_PLUS_PATH is not set")
    root = Path(raw_root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"LIBERO_PLUS_PATH does not exist: {root}")
    return root.resolve()


def _benchmark_root(repo_root: Path) -> Path:
    root = repo_root / "libero" / "libero"
    if not root.is_dir():
        raise SystemExit(f"LIBERO package root not found: {root}")
    return root


def _config_root() -> Path:
    default = Path.home() / ".libero-openwam-plus"
    raw_root = os.environ.get("LIBERO_PLUS_CONFIG_ROOT", os.environ.get("LIBERO_CONFIG_PATH", str(default)))
    return Path(raw_root).expanduser().resolve()


def write_config() -> dict[str, str]:
    repo_root = _repo_root()
    benchmark_root = _benchmark_root(repo_root)
    config_root = _config_root()
    config_root.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(repo_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with (config_root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return config


def _load_task_map() -> dict[str, list[str]]:
    task_map_path = _benchmark_root(_repo_root()) / "benchmark" / "libero_suite_task_map.py"
    spec = importlib.util.spec_from_file_location("libero_plus_suite_task_map", task_map_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load task map: {task_map_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.libero_task_map


def import_smoke() -> None:
    config = write_config()
    os.environ["LIBERO_CONFIG_PATH"] = str(_config_root())

    import libero  # noqa: PLC0415
    import libero.libero as libero_core  # noqa: PLC0415
    from libero.libero import get_libero_path  # noqa: PLC0415

    task_map = _load_task_map()
    package_file = getattr(libero, "__file__", None) or getattr(libero_core, "__file__", "")
    print(f"libero_package={Path(package_file).resolve()}")
    print(f"benchmark_root={get_libero_path('benchmark_root')}")
    print(f"assets={config['assets']}")
    print(f"suites={','.join(sorted(task_map))}")


def _language_from_task_name(task_name: str) -> str:
    if task_name.startswith(("KITCHEN_SCENE", "LIVING_ROOM_SCENE", "STUDY_SCENE")):
        match = re.search(r"SCENE\d+_(.*)", task_name)
        if match:
            return match.group(1).replace("_", " ")
    return re.sub(
        r"_(?:table|tb|light|language|view|add|level\d+|noise|robot|background)_?\d*$",
        "",
        task_name,
    ).replace("_", " ")


def task_smoke(suite: str, task_id: int) -> None:
    import_smoke()

    from libero.libero import get_libero_path  # noqa: PLC0415

    task_map = _load_task_map()
    if suite not in task_map:
        raise SystemExit(f"Unknown suite '{suite}'. Available suites: {', '.join(sorted(task_map))}")
    tasks = task_map[suite]
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"task_id {task_id} out of range for {suite}: 0..{len(tasks) - 1}")

    task_name = tasks[task_id]
    bddl_file = Path(get_libero_path("bddl_files")) / suite / f"{task_name}.bddl"
    if not bddl_file.is_file():
        raise SystemExit(f"Task BDDL not found: {bddl_file}")
    print(f"task_suite={suite}")
    print(f"task_id={task_id}")
    print(f"task_name={task_name}")
    print(f"task_language={_language_from_task_name(task_name)}")
    print(f"task_bddl={bddl_file}")


def _check_egl_runtime() -> None:
    if os.environ.get("MUJOCO_GL", "egl").lower() != "egl":
        return
    candidates = [Path(path) for path in glob("/usr/lib*/**/libEGL_nvidia.so*", recursive=True)]
    if candidates and not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
        joined = ", ".join(str(path) for path in candidates)
        raise SystemExit(
            f"EGL runtime is not usable: NVIDIA EGL libraries are empty placeholders ({joined}). "
            "Install a working NVIDIA EGL stack or set MUJOCO_GL=osmesa."
        )


def env_smoke(suite: str, task_id: int, camera_size: int, steps: int) -> None:
    task_smoke(suite, task_id)
    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    assets_dir = Path(get_libero_path("assets"))
    if not assets_dir.is_dir():
        raise SystemExit(f"LIBERO assets are missing: {assets_dir}")
    _check_egl_runtime()

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    try:
        env.seed(0)
        env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        if len(init_states) > 0:
            env.set_init_state(init_states[0])
        action = [0.0] * 7
        for _ in range(steps):
            obs, reward, done, _ = env.step(action)
        print(f"env_smoke=ok reward={reward} done={done} obs_keys={','.join(sorted(obs))}")
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["import", "task", "env"], default="import")
    parser.add_argument("--suite", default=os.environ.get("LIBERO_SMOKE_SUITE", "libero_spatial"))
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("LIBERO_SMOKE_TASK_ID", "0")))
    parser.add_argument("--camera-size", type=int, default=int(os.environ.get("LIBERO_SMOKE_CAMERA_SIZE", "128")))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("LIBERO_SMOKE_STEPS", "1")))
    args = parser.parse_args(argv)

    if args.mode == "import":
        import_smoke()
    elif args.mode == "task":
        task_smoke(args.suite, args.task_id)
    else:
        env_smoke(args.suite, args.task_id, args.camera_size, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
