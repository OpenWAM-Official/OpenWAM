#!/usr/bin/env python3
"""Run RoboTwin's eval_policy.py without its fragile render self-test."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import traceback
import types


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


def main() -> int:
    module = bootstrap_robotwin_module()
    _install_test_num_override(module)
    usr_args = module.parse_args_and_config()
    module.main(usr_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
