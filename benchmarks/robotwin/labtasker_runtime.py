"""RoboTwin inputs, native Labtasker operations and local evaluation resources."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import labtasker

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchmarks.utils.eval_manifest import (  # noqa: E402
    load_manifest,
    require_manifests,  # noqa: E402
    select_entries,
    verify_manifest,
)
from benchmarks.utils.eval_sharding import split_range  # noqa: E402
from benchmarks.utils.labtasker_utils import (  # noqa: E402
    count_task_statuses,
    create_client,
    list_submission_tasks,
    print_submission_id,
    print_summary_table,
    project_context,
    submit_tasks,
    validate_submission_coverage,
)
from benchmarks.utils.labtasker_utils import new_submission_id as new_submission_id  # noqa: E402, F401
from benchmarks.utils.owned_processes import ProcessRegistry, spawn_resource  # noqa: E402, F401
from benchmarks.utils.task_policy import load_policy_config, resolve_policy_model_spec  # noqa: E402

ROBOTWIN_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def normalize_paths(args: argparse.Namespace) -> None:
    for key in (
        "output_dir",
        "policy_config",
        "robotwin_path",
        "robotwin_python",
        "server_python",
        "ckpt_dir",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, Path(value).expanduser().absolute())


def task_names() -> list[str]:
    return list(ROBOTWIN_TASKS)


def manifest_cache_key(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "robotwin",
        "task": inputs["task"],
        "mode": inputs["mode"],
        "seed": 0,
    }


def parse_tasks(value: str) -> list[str]:
    if value == "all":
        return task_names()
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in ROBOTWIN_TASKS]
    if not selected or unknown:
        raise argparse.ArgumentTypeError(f"unknown RoboTwin task(s): {', '.join(unknown) or value}")
    return list(dict.fromkeys(selected))


def build_task_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    normalize_paths(args)
    args.manifest_dir = args.manifest_dir.expanduser().resolve()
    if args.mode is None:
        raise ValueError("--mode is required")
    if args.operation == "build_manifest" and args.deploy_args:
        raise ValueError("deploy arguments after -- are only accepted for run_eval")
    if args.operation == "run_eval" and args.rebuild:
        raise ValueError("--rebuild is only accepted for build_manifest")
    config = load_policy_config(args.policy_config)
    manifests = [
        dict(
            operation="build_manifest",
            task=task,
            mode=args.mode,
            total_episodes=args.episodes,
            policy_config=config,
            manifest_dir=str(args.manifest_dir),
            rebuild=args.rebuild,
        )
        for task in args.tasks
    ]
    if args.operation == "build_manifest":
        return manifests
    task_selector = "all" if args.tasks == task_names() else ",".join(args.tasks)
    cached = require_manifests(
        args.manifest_dir,
        [manifest_cache_key(item) for item in manifests],
        f"python benchmarks/robotwin/labtasker_submit.py --operation build_manifest --mode {args.mode} "
        f"--tasks {task_selector} --episodes {args.episodes} --manifest-dir {str(args.manifest_dir)!r}",
    )
    model = resolve_policy_model_spec(args.deploy_args)
    inputs = []
    for item, (path, manifest) in zip(manifests, cached, strict=True):
        validate_manifest(manifest, item, minimum_entries=item["total_episodes"])
        print(f"Using cached manifest {path}")
        for batch in split_range(0, args.episodes, args.episode_batch_size):
            inputs.append(
                {key: item[key] for key in ("task", "mode", "total_episodes", "policy_config")}
                | dict(
                    operation="run_eval",
                    episode_start=batch.start,
                    num_episodes=batch.count,
                    manifest_path=str(path),
                    manifest_hash=manifest["manifest_hash"],
                    instruction_type=args.instruction_type,
                    model=model,
                )
            )
    return inputs


def validate_manifest(manifest: dict[str, Any], inputs: dict[str, Any], *, minimum_entries: int) -> None:
    entries = manifest["entries"]
    seeds = [entry.get("seed") for entry in entries]
    if (
        len(entries) < minimum_entries
        or manifest.get("benchmark") != "robotwin"
        or manifest.get("task") != inputs["task"]
        or manifest["mode"] != inputs["mode"]
        or not all(type(seed) is int and seed >= 100000 for seed in seeds)
        or seeds != sorted(set(seeds))
        or any(entry.get("episode") != index for index, entry in enumerate(entries))
        or not all(
            isinstance(entry.get("instructions"), dict)
            and all(
                isinstance(entry["instructions"].get(kind), str) and entry["instructions"][kind]
                for kind in ("seen", "unseen")
            )
            for entry in entries
        )
    ):
        raise ValueError("invalid RoboTwin manifest; rebuild the cache with --operation build_manifest --rebuild")


def list_robotwin_submission_tasks(
    client: labtasker.Client, submission_id: str, operation: str | None = None
) -> list[labtasker.Task]:
    return list_submission_tasks(client, "robotwin", submission_id, operation)


def submit(
    inputs: Iterable[dict[str, Any]],
    *,
    submission_id: str,
    route: str,
    queue: str | None = None,
    max_attempts: int = 3,
    priority: int = 0,
    note: str | None = None,
    auto_start_local_server: bool = False,
) -> list[str]:
    """Create missing Tasks and reuse exact Tasks from the same submission."""

    inputs = list(inputs)
    print(
        f"[submission] id={submission_id} operation={inputs[0]['operation'] if inputs else '-'} "
        f"queue={queue or 'default'} route={route}",
        flush=True,
    )
    print_submission_id(submission_id)
    grouped_counts: dict[tuple[str, str, str], int] = {}
    for item in inputs:
        if item["operation"] == "run_eval":
            key = (item["mode"], item["task"], item["instruction_type"])
            grouped_counts[key] = grouped_counts.get(key, 0) + 1
    grouped_indices: dict[tuple[str, str, str], int] = {}
    names: list[str] = []
    for item in inputs:
        if item["operation"] == "build_manifest":
            display = f"robotwin({item['mode']}):{item['task']}_build_manifest"
        else:
            key = (item["mode"], item["task"], item["instruction_type"])
            grouped_indices[key] = grouped_indices.get(key, 0) + 1
            total = grouped_counts[key]
            width = max(2, len(str(total)))
            display = (
                f"robotwin({item['mode']}):{item['task']}_{item['instruction_type']}_"
                f"{grouped_indices[key]:0{width}d}_{total:0{width}d}"
            )
        names.append(display + (f"_{note}" if note else ""))
    with project_context(), create_client(queue=queue, auto_start_local_server=auto_start_local_server) as client:
        task_ids = submit_tasks(
            client,
            inputs,
            benchmark="robotwin",
            submission_id=submission_id,
            route=route,
            names=names,
            max_attempts=max_attempts,
            priority=priority,
        )
    print(f"[submitted] id={submission_id} confirmed={len(task_ids)}/{len(inputs)}", flush=True)
    print_submission_id(submission_id)
    return task_ids


def validated_task_result(task: labtasker.Task) -> dict[str, Any] | None:
    result = task.result
    if task.status != "succeeded" or not isinstance(result, dict) or task.args.get("operation") != "run_eval":
        return None
    args = task.args
    if (
        type(args.get("episode_start")) is not int
        or type(args.get("num_episodes")) is not int
        or type(args.get("total_episodes")) is not int
        or not all(isinstance(args.get(key), str) for key in ("manifest_hash", "manifest_path"))
    ):
        return None
    episodes = result.get("episodes")
    expected_indices = set(range(args["episode_start"], args["episode_start"] + args["num_episodes"]))
    valid = (
        result.get("operation") == "run_eval"
        and all(type(result.get(key)) is int for key in ("successes", "num_episodes", "total_episodes"))
        and 0 <= result["successes"] <= result["num_episodes"]
        and result["num_episodes"] == args["num_episodes"]
        and result["total_episodes"] == args["total_episodes"]
        and result.get("episode_start") == args["episode_start"]
        and all(result.get(key) == args.get(key) for key in ("task", "mode", "instruction_type"))
        and isinstance(episodes, list)
        and len(episodes) == result["num_episodes"]
        and all(
            type(row.get("episode")) is int and type(row.get("seed")) is int and type(row.get("success")) is bool
            for row in episodes
        )
        and {row["episode"] for row in episodes} == expected_indices
        and result.get("manifest_hash") == args["manifest_hash"]
        and result["successes"] == sum(row["success"] for row in episodes)
    )
    return result if valid else None


def summarize(
    submission_id: str,
    output: str | Path,
    *,
    queue: str | None = None,
    auto_start_local_server: bool = False,
    skip_coverage_check: bool = False,
) -> int:
    """Write one task-level view from authoritative Labtasker state."""

    with project_context(), create_client(queue=queue, auto_start_local_server=auto_start_local_server) as client:
        tasks = list_robotwin_submission_tasks(client, submission_id, "run_eval")
    if not tasks:
        raise ValueError(f"RoboTwin eval submission not found: {submission_id}")
    validate_submission_coverage(tasks, skip_coverage_check=skip_coverage_check)

    grouped: dict[tuple[str, str, str], list[labtasker.Task]] = {}
    for task in tasks:
        key = (task.args.get("mode"), task.args.get("task"), task.args.get("instruction_type"))
        if (
            key[0] not in ("demo_clean", "demo_randomized")
            or key[1] not in task_names()
            or key[2] not in ("seen", "unseen")
        ):
            raise ValueError(f"invalid RoboTwin eval Task: {task.id}")
        grouped.setdefault(key, []).append(task)

    rows = []
    for (mode, task_name, instruction_type), group in sorted(grouped.items()):
        results = []
        for task in group:
            if task.status == "succeeded":
                result = validated_task_result(task)
                if result is None:
                    raise ValueError(f"invalid RoboTwin result: {task.id}")
                results.append(result)
        successes = sum(result["successes"] for result in results)
        episodes = sum(result["num_episodes"] for result in results)
        rows.append(
            {
                "mode": mode,
                "task": task_name,
                "instruction_type": instruction_type,
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes if episodes else None,
                **count_task_statuses(group),
            }
        )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = (
        "mode",
        "task",
        "instruction_type",
        "successes",
        "episodes",
        "success_rate",
        "succeeded",
        "pending",
        "running",
        "failed",
        "expected",
    )
    path = output / "task_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    benchmark_rows = []
    for mode, kind in sorted({(row["mode"], row["instruction_type"]) for row in rows}):
        selected = [row for row in rows if (row["mode"], row["instruction_type"]) == (mode, kind)]
        totals = {
            key: sum(row[key] for row in selected)
            for key in ("successes", "episodes", "succeeded", "pending", "running", "failed", "expected")
        }
        benchmark_rows.append(
            dict(
                mode=mode,
                instruction_type=kind,
                **totals,
                success_rate=totals["successes"] / totals["episodes"] if totals["episodes"] else None,
            )
        )
    with (output / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in fields if key != "task"])
        writer.writeheader()
        writer.writerows(benchmark_rows)

    def rate(row: dict[str, Any]) -> str:
        return "-" if row["success_rate"] is None else f"{100 * row['success_rate']:.2f}%"

    print_summary_table(
        "Benchmark summary",
        (
            "Mode",
            "Instructions",
            "Success/Episodes",
            "Success rate",
            "Done/Total",
            "Pending",
            "Running",
            "Failed",
        ),
        [
            (
                row["mode"],
                row["instruction_type"],
                f"{row['successes']}/{row['episodes']}",
                rate(row),
                f"{row['succeeded']}/{row['expected']}",
                row["pending"],
                row["running"],
                row["failed"],
            )
            for row in benchmark_rows
        ],
    )
    print()
    print_summary_table(
        "Task summary",
        (
            "Mode",
            "Instructions",
            "Task",
            "Success/Episodes",
            "Success rate",
            "Done/Total",
            "Pending",
            "Running",
            "Failed",
        ),
        [
            (
                row["mode"],
                row["instruction_type"],
                row["task"],
                f"{row['successes']}/{row['episodes']}",
                rate(row),
                f"{row['succeeded']}/{row['expected']}",
                row["pending"],
                row["running"],
                row["failed"],
            )
            for row in rows
        ],
    )
    print("\nOutput files")
    print(f"  Benchmark summary: {output / 'benchmark_summary.csv'}")
    print(f"  Task summary:      {path}")
    return 0


def build_client_environment(args: argparse.Namespace, inputs: dict[str, Any], attempt_dir: Path) -> dict[str, str]:
    env = dict(
        os.environ,
        ROBOTWIN_PATH=str(args.robotwin_path),
        ROBOTWIN_RUNTIME_ROOT=str(attempt_dir / "runtime"),
        ROBOTWIN_TEST_NUM=str(
            inputs["total_episodes"] if inputs["operation"] == "build_manifest" else inputs["num_episodes"]
        ),
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONUNBUFFERED="1",
    )
    env["PYTHONPATH"] = f"{args.robotwin_path}:{HERE}:{ROOT}:{env.get('PYTHONPATH', '')}"
    egl = args.robotwin_python.parent.parent / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    if not env.get("__EGL_VENDOR_LIBRARY_FILENAMES") and not env.get("__EGL_VENDOR_LIBRARY_DIRS") and egl.is_file():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
    env.setdefault("MPLCONFIGDIR", str(attempt_dir / "matplotlib"))
    return env


def read_client_result(path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """Read and validate the structured output from exactly this attempt."""

    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid RoboTwin client result: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"RoboTwin client result must be an object: {path}")
    if inputs["operation"] == "build_manifest":
        try:
            checked = verify_manifest(payload)
            validate_manifest(checked, inputs, minimum_entries=inputs["total_episodes"])
        except ValueError:
            checked = None
        valid = checked is not None
    else:
        checked = None
        expected_indices = set(range(inputs["episode_start"], inputs["episode_start"] + inputs["num_episodes"]))
        episodes = payload.get("episodes")
        valid = (
            payload.get("operation") == "run_eval"
            and payload.get("task") == inputs["task"]
            and payload.get("mode") == inputs["mode"]
            and payload.get("instruction_type") == inputs["instruction_type"]
            and payload.get("episode_start") == inputs["episode_start"]
            and payload.get("num_episodes") == inputs["num_episodes"]
            and payload.get("total_episodes") == inputs["total_episodes"]
            and isinstance(episodes, list)
            and len(episodes) == inputs["num_episodes"]
            and all(
                isinstance(row, dict)
                and type(row.get("episode")) is int
                and type(row.get("seed")) is int
                and type(row.get("success")) is bool
                for row in episodes
            )
            and {row["episode"] for row in episodes} == expected_indices
            and payload.get("successes") == sum(row["success"] for row in episodes)
            and payload.get("manifest_hash") == inputs["manifest_hash"]
        )
        if valid:
            try:
                manifest = load_manifest(
                    Path(inputs["manifest_path"]),
                    expected_manifest_hash=inputs["manifest_hash"],
                )
                validate_manifest(manifest, inputs, minimum_entries=inputs["total_episodes"])
                entries = select_entries(manifest, inputs["episode_start"], inputs["num_episodes"])
                ordered = sorted(episodes, key=lambda row: row["episode"])
                valid = all(
                    row["seed"] == entry["seed"]
                    and row.get("instruction") == entry["instructions"][inputs["instruction_type"]]
                    for row, entry in zip(ordered, entries, strict=True)
                )
            except ValueError:
                valid = False
    if not valid:
        raise ValueError(f"RoboTwin client result is inconsistent: {path}")
    return checked if inputs["operation"] == "build_manifest" else payload


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value: str) -> float:
    number = float(value)
    if number < 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def parser(mode: str = "submit") -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    worker = mode == "worker"
    result.add_argument("--route", help="Task route (default: openwam-robotwin-<operation>)")
    result.add_argument("--queue")
    result.add_argument(
        "--auto-start-local-server",
        action="store_true",
        help="explicitly authorize startup of the project-local Labtasker Server",
    )
    result.add_argument("--operation", choices=("build_manifest", "run_eval"), default="run_eval")
    if not worker:
        result.add_argument("--mode", choices=("demo_clean", "demo_randomized"), required=True)
        result.add_argument("--tasks", type=parse_tasks, default=task_names())
        result.add_argument("--episodes", type=positive, default=100)
        result.add_argument("--episode-batch-size", type=positive, default=5)
        result.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
        result.add_argument("--policy-config", type=Path, default=HERE / "policy_config.yml")
        result.add_argument("--manifest-dir", type=Path, default=ROOT / "outputs/manifest-cache/robotwin")
        result.add_argument("--rebuild", action="store_true")
        result.add_argument("--submission-id")
        result.add_argument("--max-attempts", type=positive, default=3)
        result.add_argument("--priority", type=int, default=0)
        result.add_argument("--dry-run", action="store_true")
        result.add_argument("--note")
        result.set_defaults(deploy_args=[])
        result.epilog = "Use -- followed by native OpenWAM deploy arguments (run_eval only)."
        return result
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--robotwin-path", type=Path, default=os.environ.get("ROBOTWIN_PATH"))
    result.add_argument("--robotwin-python", type=Path, default=os.environ.get("ROBOTWIN_PYTHON"))
    result.add_argument("--render-device")
    result.add_argument("--idle-timeout", type=nonnegative, default=300)
    result.add_argument("--max-consecutive-failures", type=positive, default=5)
    result.add_argument("--gpu", type=int, required=True)
    if mode == "worker":
        result.add_argument("--server-python", type=Path, default=Path(sys.executable))
        result.add_argument("--host", default="127.0.0.1")
        result.add_argument("--port", type=int, help="Policy port (default: 8848 + GPU ID)")
        result.add_argument("--server-start-timeout", type=positive, default=1200)
    return result


def parse_submit_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse submission options strictly and forward only arguments after --."""
    argv = list(sys.argv[1:] if argv is None else argv)
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser("submit").parse_args(argv[:separator])
    args.deploy_args = argv[separator + 1 :]
    return args
