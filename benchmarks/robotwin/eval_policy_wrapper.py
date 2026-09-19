#!/usr/bin/env python3
"""Run RoboTwin evaluation for traditional scripts or a Labtasker episode operation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.robotwin.eval_manifest import (  # noqa: E402
    BENCHMARK,
    choose_instructions,
)
from benchmarks.utils.eval_manifest import (  # noqa: E402
    load_manifest,
    seal_manifest,
    select_entries,
)
from benchmarks.utils.rng_domain import GlobalRngDomain  # noqa: E402
from benchmarks.utils.task_progress import write_progress  # noqa: E402

# RoboTwin commit this adapter was verified against (README "Verified versions").
VERIFIED_ROBOTWIN_COMMIT = "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"
# RoboTwin paths the adapter depends on; local edits there invalidate the check.
_ROBOTWIN_WATCHED_PATHS = ("script", "envs", "task_config", "policy")


def _canonicalize_torch_cuda(seed: int) -> None:
    """Apply RoboTwin's candidate seed after lazy CUDA initialization.

    RoboTwin calls ``torch.manual_seed(candidate)`` during setup, but a fresh
    process can initialize the CUDA generator later while a reused process has
    already initialized it. Repeating the same intended CUDA seed after setup
    removes that cold-versus-warm history without changing the created scene.
    """

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _robotwin_git(robotwin_path: str, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", robotwin_path, *args], capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _warn_on_unverified_robotwin(robotwin_path: str) -> None:
    """Log RoboTwin provenance; warn when it differs from the verified state.

    Non-fatal. Checks the HEAD commit and local modifications under the paths
    the adapter loads from, and always prints the hash of the
    ``script/eval_policy.py`` actually loaded. ``ROBOTWIN_SKIP_VERSION_CHECK=1``
    disables the git checks (e.g. for a non-git RoboTwin copy).
    """
    script_path = os.path.join(robotwin_path, "script", "eval_policy.py")
    script_sha = _file_sha256(script_path)[:16] if os.path.isfile(script_path) else "missing"
    if os.environ.get("ROBOTWIN_SKIP_VERSION_CHECK", "") == "1":
        print(f"[eval_policy_wrapper] RoboTwin version check skipped; eval_policy.py sha256={script_sha}")
        return
    head = _robotwin_git(robotwin_path, "rev-parse", "HEAD")
    if not head:
        print(
            "[eval_policy_wrapper] WARNING: could not read the RoboTwin git commit (not a git checkout?); "
            f"verified commit is {VERIFIED_ROBOTWIN_COMMIT[:12]}; eval_policy.py sha256={script_sha}"
        )
        return
    modified = _robotwin_git(
        robotwin_path, "status", "--porcelain", "--untracked-files=no", "--", *_ROBOTWIN_WATCHED_PATHS
    ).splitlines()
    print(f"[eval_policy_wrapper] RoboTwin commit={head[:12]} eval_policy.py sha256={script_sha}")
    if head != VERIFIED_ROBOTWIN_COMMIT:
        print(
            f"[eval_policy_wrapper] WARNING: RoboTwin at {head[:12]} differs from the verified commit "
            f"{VERIFIED_ROBOTWIN_COMMIT[:12]}; if eval breaks, check out the verified commit first "
            "(ROBOTWIN_SKIP_VERSION_CHECK=1 silences this)."
        )
    if modified:
        shown = ", ".join(line.split(maxsplit=1)[-1] for line in modified[:5]) + (" …" if len(modified) > 5 else "")
        print(
            f"[eval_policy_wrapper] WARNING: RoboTwin has {len(modified)} locally modified file(s) under "
            f"{'/'.join(_ROBOTWIN_WATCHED_PATHS)}: {shown}. Results may not match the verified stack."
        )


def _load_robotwin_eval_module(robotwin_path: str):
    script_path = os.path.join(robotwin_path, "script", "eval_policy.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"RoboTwin eval script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("robotwin_eval_policy", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load RoboTwin eval module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_runtime_root(robotwin_path: str) -> str:
    runtime_root = os.environ.get("ROBOTWIN_RUNTIME_ROOT", "")
    if runtime_root:
        runtime_root = os.path.abspath(runtime_root)
        os.makedirs(runtime_root, exist_ok=True)
    elif os.access(robotwin_path, os.W_OK):
        return robotwin_path
    else:
        runtime_root = tempfile.mkdtemp(prefix="robotwin_runtime.", dir=os.environ.get("TMPDIR", "/tmp"))

    for name in os.listdir(robotwin_path):
        if name == "eval_result":
            continue
        src = os.path.join(robotwin_path, name)
        dst = os.path.join(runtime_root, name)
        if os.path.lexists(dst):
            continue
        os.symlink(src, dst)

    os.makedirs(os.path.join(runtime_root, "eval_result"), exist_ok=True)
    print(f"[eval_policy_wrapper] runtime_root={runtime_root}")
    return runtime_root


def _prewarm_cuda_for_curobo() -> None:
    """Initialize CUDA/Curobo before RoboTwin imports SAPIEN.

    On this container, importing ``sapien`` first can poison CUDA discovery
    for the rest of the process (torch then reports Error 304). Prewarming
    torch.cuda and importing curobo up front keeps the later RoboTwin import
    chain on the healthy path.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            print("[eval_policy_wrapper] torch.cuda.is_available() is false before RoboTwin import")
            return

        _ = torch.cuda.device_count()
        _ = torch.zeros(1, device="cuda")

        from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401

        print("[eval_policy_wrapper] prewarmed CUDA/Curobo before SAPIEN import")
    except Exception as exc:
        print(f"[eval_policy_wrapper] CUDA/Curobo prewarm failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def _patch_warp_torch_namespace() -> None:
    """Provide the legacy ``warp.torch`` namespace expected by this cuRobo fork.

    Newer Warp versions expose Torch interop as top-level functions
    (``warp.device_from_torch`` etc.) but no longer ship a ``warp.torch``
    submodule. This RoboTwin-pinned cuRobo still calls ``wp.torch.*`` once in
    ``world_mesh.py``. Recreate that namespace as a thin compatibility alias.
    """
    try:
        import warp as wp
    except Exception:
        return

    if hasattr(wp, "torch"):
        return

    interop = types.SimpleNamespace(
        from_torch=getattr(wp, "from_torch", None),
        to_torch=getattr(wp, "to_torch", None),
        dtype_from_torch=getattr(wp, "dtype_from_torch", None),
        dtype_to_torch=getattr(wp, "dtype_to_torch", None),
        device_from_torch=getattr(wp, "device_from_torch", None),
        device_to_torch=getattr(wp, "device_to_torch", None),
        stream_from_torch=getattr(wp, "stream_from_torch", None),
        stream_to_torch=getattr(wp, "stream_to_torch", None),
    )
    if interop.device_from_torch is not None:
        wp.torch = interop
        print("[eval_policy_wrapper] patched warp.torch compatibility namespace")


def _install_env_trace_hooks(module) -> None:
    orig_class_decorator = module.class_decorator
    unstable_error = getattr(module, "UnStableError", None)

    def _wrap_method(env, method_name: str) -> None:
        method = getattr(env, method_name, None)
        if not callable(method):
            return

        def wrapped(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - diagnostic path
                if unstable_error is not None and isinstance(exc, unstable_error):
                    raise
                print(f"[eval_policy_wrapper] exception in {env.__class__.__name__}.{method_name}")
                traceback.print_exc()
                raise

        setattr(env, method_name, wrapped)

    def traced_class_decorator(task_name):
        env = orig_class_decorator(task_name)
        for method_name in ("setup_demo", "play_once"):
            _wrap_method(env, method_name)
        return env

    module.class_decorator = traced_class_decorator


def _install_test_num_override(module) -> None:
    value = os.environ.get("ROBOTWIN_TEST_NUM", "").strip()
    if not value:
        return
    try:
        test_num = int(value)
    except ValueError as exc:
        raise ValueError(f"ROBOTWIN_TEST_NUM must be an integer, got {value!r}") from exc
    if test_num <= 0:
        raise ValueError(f"ROBOTWIN_TEST_NUM must be > 0, got {test_num}")

    orig_eval_policy = module.eval_policy

    def capped_eval_policy(*args, **kwargs):
        kwargs["test_num"] = test_num
        print(f"[eval_policy_wrapper] overriding RoboTwin eval test_num={test_num}")
        return orig_eval_policy(*args, **kwargs)

    module.eval_policy = capped_eval_policy


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_robot_planner_fallbacks(robotwin_path: str) -> None:
    import envs

    robot_dir = os.path.join(robotwin_path, "envs", "robot")
    planner_path = os.path.join(robot_dir, "planner.py")
    robot_path = os.path.join(robot_dir, "robot.py")

    robot_pkg = types.ModuleType("envs.robot")
    robot_pkg.__file__ = os.path.join(robot_dir, "__init__.py")
    robot_pkg.__package__ = "envs.robot"
    robot_pkg.__path__ = [robot_dir]
    sys.modules["envs.robot"] = robot_pkg
    setattr(envs, "robot", robot_pkg)

    planner_mod = _load_module("envs.robot.planner", planner_path)
    if not hasattr(planner_mod, "CuroboPlanner"):
        planner_mod.CuroboPlanner = type("CuroboPlanner", (), {})
    curobo_cls = planner_mod.CuroboPlanner
    mplib_cls = getattr(planner_mod, "MplibPlanner", None)
    if mplib_cls is None:
        return

    if not hasattr(mplib_cls, "plan_batch"):

        def plan_batch(self, now_qpos, target_pose_list, constraint_pose=None, arms_tag=None):
            statuses = []
            positions = []
            velocities = []
            for pose in target_pose_list:
                result = self.plan_path(now_qpos, pose, arms_tag=arms_tag, log=False)
                success = result.get("status") == "Success"
                statuses.append("Success" if success else "Failure")
                positions.append(result.get("position") if success else None)
                velocities.append(result.get("velocity") if success else None)
            return {"status": statuses, "position": positions, "velocity": velocities}

        mplib_cls.plan_batch = plan_batch

    robot_mod = _load_module("envs.robot._robot_impl", robot_path)

    orig_init_robot = robot_mod.Robot._init_robot_
    orig_set_planner = robot_mod.Robot.set_planner

    def safe_init_robot(self, scene, need_topp=False, **kwargs):
        self.left_planner = None
        self.right_planner = None
        self.left_conn = None
        self.right_conn = None
        self.left_proc = None
        self.right_proc = None
        self.communication_flag = False
        return orig_init_robot(self, scene, need_topp, **kwargs)

    def safe_set_planner(self, scene=None):
        try:
            return orig_set_planner(self, scene=scene)
        except Exception as exc:
            print(f"[eval_policy_wrapper] planner fallback engaged: {type(exc).__name__}: {exc}")
            traceback.print_exc()

            self.communication_flag = False
            self.left_conn = None
            self.right_conn = None
            self.left_proc = None
            self.right_proc = None
            self.left_planner = mplib_cls(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type if self.left_planner_type != "curobo" else "mplib_RRT",
                scene,
            )
            self.right_planner = mplib_cls(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type if self.right_planner_type != "curobo" else "mplib_RRT",
                scene,
            )
            if self.need_topp:
                self.left_mplib_planner = self.left_planner
                self.right_mplib_planner = self.right_planner

    def safe_reset(self, scene, need_topp=False, **kwargs):
        self._init_robot_(scene, need_topp, **kwargs)

        if getattr(self, "communication_flag", False):
            if getattr(self, "left_conn", None):
                self.left_conn.send({"cmd": "reset"})
                _ = self.left_conn.recv()
            if getattr(self, "right_conn", None):
                self.right_conn.send({"cmd": "reset"})
                _ = self.right_conn.recv()
        else:
            left_planner = getattr(self, "left_planner", None)
            right_planner = getattr(self, "right_planner", None)
            curobo_ready = (
                left_planner is not None
                and right_planner is not None
                and isinstance(left_planner, curobo_cls)
                and isinstance(right_planner, curobo_cls)
            )
            if not curobo_ready:
                self.set_planner(scene=scene)

        self.init_joints()

    robot_mod.Robot._init_robot_ = safe_init_robot
    robot_mod.Robot.set_planner = safe_set_planner
    robot_mod.Robot.reset = safe_reset

    robot_pkg.Robot = robot_mod.Robot
    robot_pkg.CuroboPlanner = planner_mod.CuroboPlanner
    robot_pkg.MplibPlanner = planner_mod.MplibPlanner
    robot_pkg.planner = planner_mod
    robot_pkg.robot = robot_mod


def bootstrap_robotwin_module(*, install_trace_hooks: bool = True):
    """Prepare the RoboTwin runtime and return its loaded ``eval_policy`` module.

    Does everything both entrypoints share: resolve ``ROBOTWIN_PATH``, set up a
    writable runtime root + chdir + sys.path, prewarm CUDA/Curobo before SAPIEN,
    patch the legacy ``warp.torch`` namespace, load ``script/eval_policy.py``,
    install the optional planner fallback (``ROBOTWIN_ENABLE_PLANNER_FALLBACK=1``)
    and the per-env exception trace hooks. Callers (such as ``main`` here for
    whole-task eval) then monkeypatch / drive the module as they need.
    """
    robotwin_path = os.environ.get("ROBOTWIN_PATH")
    if not robotwin_path:
        raise SystemExit("ROBOTWIN_PATH must be set")

    _warn_on_unverified_robotwin(robotwin_path)
    runtime_root = _prepare_runtime_root(robotwin_path)
    os.chdir(runtime_root)
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)

    _prewarm_cuda_for_curobo()
    _patch_warp_torch_namespace()
    module = _load_robotwin_eval_module(robotwin_path)
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK", "") == "1":
        _install_robot_planner_fallbacks(robotwin_path)
    if install_trace_hooks:
        _install_env_trace_hooks(module)
    return module


def _write_labtasker_result(payload: dict) -> None:
    result_path = os.environ.get("ROBOTWIN_LABTASKER_RESULT")
    if not result_path:
        raise RuntimeError("ROBOTWIN_LABTASKER_RESULT is required for sharded evaluation")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _initialize_eval_task_state(task_name: str, task_env: Any) -> None:
    """Recreate task fields that verified RoboTwin sets only in ``play_once``.

    Manifest construction runs the expert ``play_once`` in another process.
    Cached evaluation starts from ``setup_demo`` and therefore must initialize
    the scene-derived fields that these tasks read from ``check_success``.
    """

    if task_name not in ("open_laptop", "place_object_scale", "put_object_cabinet"):
        return

    task_module = sys.modules[type(task_env).__module__]
    arm_tag_type = task_module.ArmTag
    if task_name == "open_laptop":
        face_prod = task_module.get_face_prod(task_env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        task_env.arm_tag = arm_tag_type("left" if face_prod > 0 else "right")
        return

    object_pose = task_env.object.get_pose().p
    task_env.arm_tag = arm_tag_type("right" if object_pose[0] > 0 else "left")
    if task_name == "put_object_cabinet":
        task_env.origin_z = object_pose[2]


def _install_manifest(module) -> None:
    """Replace policy rollout with a contiguous expert-feasibility scan."""

    original_decorator = module.eval_function_decorator

    def decorator(policy_name, model_name):
        if model_name == "get_model":
            return lambda _args: None
        return original_decorator(policy_name, model_name)

    def build_manifest(task_name, task_env, args, model, st_seed, test_num=100, video_size=None, instruction_type=None):
        del model, video_size
        accepted = []
        now_seed = st_seed
        progress_path = Path(os.environ["ROBOTWIN_PROGRESS_FILE"])

        args["eval_mode"] = True
        task_env.suc = 0
        task_env.test_num = 0
        while len(accepted) < test_num:
            render_freq = args["render_freq"]
            args["render_freq"] = 0
            domain = GlobalRngDomain.from_seed(now_seed)
            episode_info = None
            try:
                with domain.activate():
                    task_env.setup_demo(now_ep_num=len(accepted), seed=now_seed, is_test=True, **args)
                    episode_info = task_env.play_once()
                    eligible = bool(task_env.plan_success and task_env.check_success())
            except Exception as exc:
                unstable = getattr(module, "UnStableError", ())
                if not unstable or not isinstance(exc, unstable):
                    print(
                        f"[eval_policy_wrapper] expert check skipped seed={now_seed}: {type(exc).__name__}",
                        flush=True,
                    )
                eligible = False
            finally:
                with domain.activate():
                    task_env.close_env()
                args["render_freq"] = render_freq
            if eligible:
                info = episode_info["info"]
                instructions = choose_instructions(
                    module.generate_episode_descriptions,
                    task=task_name,
                    candidate_seed=now_seed,
                    episode_info=info,
                )
                accepted.append(
                    {
                        "episode": len(accepted),
                        "seed": now_seed,
                        "instructions": instructions,
                    }
                )
            now_seed += 1
            write_progress(
                progress_path,
                {
                    "completed": len(accepted),
                    "total": test_num,
                    "candidate_seed": now_seed - 1,
                },
            )
        sealed = seal_manifest(
            {
                "benchmark": BENCHMARK,
                "task": task_name,
                "mode": args["task_config"],
                "entries": accepted,
            }
        )
        _write_labtasker_result(sealed)
        return now_seed, 0

    module.eval_function_decorator = decorator
    module.eval_policy = build_manifest


def _install_eval(module) -> None:
    """Evaluate exact sealed manifest entries without repeating the expert scan."""

    manifest_hash = os.environ["ROBOTWIN_MANIFEST_HASH"]
    manifest = load_manifest(
        Path(os.environ["ROBOTWIN_EPISODE_MANIFEST"]),
        expected_manifest_hash=manifest_hash,
    )
    episode_start = int(os.environ["ROBOTWIN_EPISODE_START"])
    total_episodes = int(os.environ["ROBOTWIN_TOTAL_EPISODES"])
    count = int(os.environ["ROBOTWIN_EPISODE_COUNT"])
    selected_type = os.environ["ROBOTWIN_INSTRUCTION_TYPE"]
    entries = select_entries(manifest, episode_start, count)
    progress_path = Path(os.environ["ROBOTWIN_PROGRESS_FILE"])
    if manifest.get("benchmark") != BENCHMARK or len(manifest["entries"]) < total_episodes:
        raise ValueError("RoboTwin manifest does not cover the requested evaluation")

    def evaluate(task_name, task_env, args, model, st_seed, test_num=100, video_size=None, instruction_type=None):
        del st_seed, instruction_type
        if task_name != manifest["task"] or args["task_config"] != manifest["mode"]:
            raise ValueError("RoboTwin evaluator task/mode does not match sealed manifest")
        if test_num != len(entries):
            raise ValueError(f"batch expected {len(entries)} episodes, evaluator requested {test_num}")
        policy_name = args["policy_name"]
        eval_func = module.eval_function_decorator(policy_name, "eval")
        reset_func = module.eval_function_decorator(policy_name, "reset_model")
        policy_module = __import__(policy_name)
        results = []
        task_env.suc = 0
        task_env.test_num = 0
        args["eval_mode"] = True
        try:
            for entry in entries:
                episode = entry["episode"]
                seed = entry["seed"]
                domain = GlobalRngDomain.from_seed(seed)
                if hasattr(policy_module, "set_benchmark_rng_domain"):
                    policy_module.set_benchmark_rng_domain(domain)
                with domain.activate():
                    task_env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **args)
                    _initialize_eval_task_state(task_name, task_env)
                    task_env.set_instruction(instruction=entry["instructions"][selected_type])
                    _canonicalize_torch_cuda(seed)

                ffmpeg = None
                if task_env.eval_video_path is not None:
                    ffmpeg = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            video_size,
                            "-framerate",
                            "10",
                            "-i",
                            "-",
                            "-pix_fmt",
                            "yuv420p",
                            "-vcodec",
                            "libx264",
                            "-crf",
                            "23",
                            f"{task_env.eval_video_path}/episode{episode}.mp4",
                        ],
                        stdin=subprocess.PIPE,
                    )
                    task_env._set_eval_video_ffmpeg(ffmpeg)

                reset_func(model)
                success = False
                while task_env.take_action_cnt < task_env.step_lim:
                    with domain.activate():
                        observation = task_env.get_obs()
                    eval_func(task_env, model, observation)
                    if task_env.eval_success:
                        success = True
                        break
                if task_env.eval_video_path is not None:
                    task_env._del_eval_video_ffmpeg()
                if success:
                    task_env.suc += 1
                task_env.test_num += 1
                with domain.activate():
                    task_env.close_env(clear_cache=((episode + 2) % args["clear_cache_freq"] == 0))
                    if task_env.render_freq:
                        task_env.viewer.close()
                results.append(
                    {
                        "episode": episode,
                        "seed": seed,
                        "instruction": entry["instructions"][selected_type],
                        "success": success,
                    }
                )
                write_progress(
                    progress_path,
                    {
                        "completed": len(results),
                        "total": len(entries),
                        "episode": episode,
                        "successes": sum(row["success"] for row in results),
                    },
                )
                print(
                    f"[RESULT] episode={episode} seed={seed} success={success} manifest={manifest_hash}",
                    flush=True,
                )
        finally:
            if hasattr(policy_module, "set_benchmark_rng_domain"):
                policy_module.set_benchmark_rng_domain(None)
        successes = sum(row["success"] for row in results)
        _write_labtasker_result(
            {
                "operation": "run_eval",
                "instruction_type": selected_type,
                "task": task_name,
                "mode": args["task_config"],
                "episode_start": episode_start,
                "num_episodes": len(results),
                "total_episodes": total_episodes,
                "manifest_hash": manifest_hash,
                "successes": successes,
                "episodes": results,
            }
        )
        return entries[-1]["seed"] + 1, successes

    module.eval_policy = evaluate


def _configure_labtasker(argv: list[str]) -> argparse.Namespace:
    """Validate one sharded operation and translate it to RoboTwin's evaluator config."""

    parser = argparse.ArgumentParser(description="Run one Labtasker RoboTwin operation")
    for key in ("task", "mode", "policy-config"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--operation", choices=("build_manifest", "run_eval"), required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--total-episodes", type=int, required=True)
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--manifest-hash")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--render-device")
    args = parser.parse_args(argv)
    if args.total_episodes <= 0:
        parser.error("--total-episodes must be positive")
    if args.operation == "run_eval":
        if args.host is None or args.port is None:
            parser.error("eval requires --host and --port")
        if (
            args.episode_start is None
            or args.episode_start < 0
            or args.num_episodes is None
            or args.num_episodes <= 0
            or args.manifest_file is None
            or not args.manifest_hash
        ):
            parser.error("eval requires a manifest/hash and a positive logical range")
        if args.episode_start + args.num_episodes > args.total_episodes:
            parser.error("episode batch exceeds --total-episodes")
        manifest = load_manifest(
            args.manifest_file,
            expected_manifest_hash=args.manifest_hash,
        )
        if (
            manifest.get("benchmark") != BENCHMARK
            or manifest.get("task") != args.task
            or manifest.get("mode") != args.mode
            or len(manifest["entries"]) < args.total_episodes
        ):
            parser.error("sealed manifest does not cover the requested task evaluation")
        select_entries(manifest, args.episode_start, args.num_episodes)
    policy = yaml.safe_load(Path(args.policy_config).read_text())
    policy.update(
        task_name=args.task,
        task_config=args.mode,
        ckpt_setting="openwam",
        seed=0,
        policy_name="openwam2robotwin_interface",
    )
    if args.operation == "run_eval":
        policy.update(host=args.host, port=args.port)
    config = Path(os.environ["ROBOTWIN_RUNTIME_ROOT"]).parent / "client-policy.yml"
    config.write_text(yaml.safe_dump(policy))
    os.environ["ROBOTWIN_INSTRUCTION_TYPE"] = args.instruction_type
    os.environ["ROBOTWIN_LABTASKER_OPERATION"] = args.operation
    os.environ["ROBOTWIN_LABTASKER_RESULT"] = str(args.result_file)
    os.environ["ROBOTWIN_PROGRESS_FILE"] = str(args.progress_file)
    os.environ["ROBOTWIN_TEST_NUM"] = str(
        args.total_episodes if args.operation == "build_manifest" else args.num_episodes
    )
    if args.operation == "run_eval":
        os.environ["ROBOTWIN_EPISODE_START"] = str(args.episode_start)
        os.environ["ROBOTWIN_EPISODE_COUNT"] = str(args.num_episodes)
        os.environ["ROBOTWIN_TOTAL_EPISODES"] = str(args.total_episodes)
        os.environ["ROBOTWIN_EPISODE_MANIFEST"] = str(args.manifest_file)
        os.environ["ROBOTWIN_MANIFEST_HASH"] = args.manifest_hash
    sys.argv = [sys.argv[0], "--config", str(config)]
    return args


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    labtasker_args = None
    if raw_args[:1] == ["labtasker"]:
        labtasker_args = _configure_labtasker(raw_args[1:])
        if labtasker_args.render_device is not None:
            # Explicit renderer selection must follow the original CUDA-before-SAPIEN ordering.
            os.chdir(_prepare_runtime_root(os.environ["ROBOTWIN_PATH"]))
            _prewarm_cuda_for_curobo()
            import sapien.render

            renderer = sapien.render.RenderSystem(device=labtasker_args.render_device)
            print(
                f"[render] requested={labtasker_args.render_device} selected={renderer.device}",
                flush=True,
            )
    elif argv is not None:
        sys.argv = [sys.argv[0], *raw_args]
    module = bootstrap_robotwin_module()
    operation = os.environ.get("ROBOTWIN_LABTASKER_OPERATION")
    if operation == "build_manifest":
        _install_manifest(module)
    elif operation == "run_eval":
        _install_eval(module)
    elif operation:
        raise ValueError(f"unknown ROBOTWIN_LABTASKER_OPERATION: {operation}")
    _install_test_num_override(module)
    usr_args = module.parse_args_and_config()
    module.main(usr_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
