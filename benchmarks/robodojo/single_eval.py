#!/usr/bin/env python3
"""Run RoboDojo's native single-env evaluator against an OpenWAM server.

Isaac Lab and RoboDojo imports stay inside :func:`run_eval`, after configuration
validation and runtime path setup.  Importing this module is safe in unit tests.
"""

from __future__ import annotations

import argparse
import importlib
import ipaddress
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, MutableMapping

import yaml

from benchmarks.robodojo.openwam_model_client import OpenWAMRoboDojoModelClient

_OPENWAM_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = Path(__file__).with_name("policy_config.yml")
_NATIVE_KIT_EXTENSIONS = (
    "isaacsim.replicator.behavior",
    "isaacsim.sensors.camera",
)
_MAX_INPROCESS_RESTARTS = 3
_ISAACLAB_SOURCE_PACKAGES = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_mimic",
    "isaaclab_rl",
    "isaaclab_tasks",
)
_OPENWAM_IMPORT_ROOTS = {
    "benchmarks": "",
    "openwam": "",
}
_ROBODOJO_IMPORT_ROOTS = {
    "env": "",
    "task": "",
    "utils": "",
    "XPolicyLab": "",
    "client_server": "XPolicyLab",
    "src": "",
    "isaaclab": "third_party/IsaacLab/source/isaaclab",
    "isaaclab_assets": "third_party/IsaacLab/source/isaaclab_assets",
    "isaaclab_tasks": "third_party/IsaacLab/source/isaaclab_tasks",
    "curobo": "third_party/curobo",
}
_REMOVABLE_EDITABLE_NAMESPACE_HOOKS = (
    "__editable__.nvidia_curobo-",
    "__editable__.xpolicylab-",
)


class _DisabledPhysXBrokenError(Exception):
    pass


class _DisabledPhysXFatalError(Exception):
    pass


@dataclass(frozen=True)
class PhysXRuntime:
    enabled: bool
    monitor: Any | None
    broken_error: type[BaseException]
    fatal_error: type[BaseException]


class PhysXRestartRequired(RuntimeError):
    """Signal a persisted fatal PhysX failure after normal resource cleanup."""

    def __init__(
        self,
        message: str,
        *,
        restart_count: int,
        shell_restart: bool,
    ):
        super().__init__(message)
        self.restart_count = restart_count
        self.shell_restart = shell_restart


class NoNetworkModelClient:
    """Construction-only stand-in for RoboDojo's MsgPack WebSocket client."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.closed = False

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> None:
        _ = (func_name, obs, kwargs)
        return None

    def close(self) -> None:
        self.closed = True


def construct_with_no_network_client(
    symbol_owner: Any,
    constructor,
):
    """Patch ``symbol_owner.WsModelClient`` only while ``constructor`` runs.

    Every placeholder instance is closed, the original symbol is restored in a
    ``finally`` block, and a successful object's ``model_client`` reference is
    cleared before it is returned.
    """

    original = symbol_owner.WsModelClient
    instances: list[NoNetworkModelClient] = []

    class TrackedNoNetworkModelClient(NoNetworkModelClient):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            instances.append(self)

    constructed = None
    try:
        symbol_owner.WsModelClient = TrackedNoNetworkModelClient
        constructed = constructor()
    finally:
        try:
            for instance in instances:
                instance.close()
        finally:
            symbol_owner.WsModelClient = original

    if getattr(constructed, "model_client", None) in instances:
        constructed.model_client = None
    return constructed


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")


def _positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be positive, got {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")
    return parsed


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from error
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return parsed


def _positive_integer(value: Any, field_name: str) -> int:
    message = f"{field_name} must be a positive integer, got {value!r}"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if isinstance(value, str):
        if re.fullmatch(r"[+-]?\d+", value) is None:
            raise ValueError(message)
    else:
        try:
            if value != parsed:
                raise ValueError(message)
        except (TypeError, ValueError):
            raise ValueError(message) from None
    if parsed <= 0:
        raise ValueError(message)
    return parsed


def validate_server_endpoint(
    host: Any,
    port: Any,
    timeout: Any,
) -> tuple[str, int, float]:
    """Validate a hostname/IP and OpenWAM transport bounds."""

    if not isinstance(host, str) or not host or host != host.strip():
        raise ValueError(f"host must be a non-empty hostname or IP, got {host!r}")
    candidate = host
    bracketed_ipv6 = candidate.startswith("[") and candidate.endswith("]")
    ip_candidate = candidate[1:-1] if bracketed_ipv6 else candidate
    try:
        address = ipaddress.ip_address(ip_candidate)
    except ValueError:
        labels = candidate.split(".")
        valid_hostname = (
            len(candidate) <= 253
            and all(
                label
                and len(label) <= 63
                and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                for label in labels
            )
        )
        if not valid_hostname:
            raise ValueError(
                f"host must be a hostname or IP without scheme/path, got {host!r}"
            ) from None
    else:
        candidate = f"[{address}]" if address.version == 6 else str(address)

    parsed_port = _integer(port, "port")
    if not 1 <= parsed_port <= 65535:
        raise ValueError(f"port must be in [1, 65535], got {parsed_port}")
    parsed_timeout = _positive_float(timeout, "timeout")
    return candidate, parsed_port, parsed_timeout


def validate_runner_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize every option before importing or launching Isaac."""

    if not isinstance(config, dict):
        raise TypeError("RoboDojo runner config must be a mapping")
    normalized = dict(config)

    root_value = normalized.get("robodojo_root")
    if not isinstance(root_value, (str, os.PathLike)) or not str(root_value).strip():
        raise ValueError("robodojo_root must be a non-empty path")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboDojo root does not exist: {root}")
    normalized["robodojo_root"] = str(root)

    env_config = normalized.get("env_config", "arx_x5")
    if env_config != "arx_x5":
        raise ValueError(
            f"OpenWAM RoboDojo evaluation requires env_config 'arx_x5', got {env_config!r}"
        )
    normalized["env_config"] = env_config

    num_envs = _integer(normalized.get("num_envs", 1), "num_envs")
    if num_envs != 1:
        raise ValueError(
            f"OpenWAM RoboDojo evaluation supports exactly one environment, got {num_envs}"
        )
    normalized["num_envs"] = num_envs

    eval_batch = _as_bool(normalized.get("eval_batch", False), "eval_batch")
    if eval_batch:
        raise ValueError("eval_batch must be false for OpenWAM's single-env adapter")
    normalized["eval_batch"] = False

    device_id = _integer(normalized.get("device_id", 0), "device_id")
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id}")
    normalized["device_id"] = device_id

    host, port, timeout = validate_server_endpoint(
        normalized.get("host", "127.0.0.1"),
        normalized.get("port", 8848),
        normalized.get("timeout", 300.0),
    )
    normalized["host"] = host
    normalized["port"] = port
    normalized["timeout"] = timeout

    eval_count = _integer(normalized.get("eval_count", 1), "eval_count")
    if eval_count <= 0:
        raise ValueError(f"eval_count must be positive, got {eval_count}")
    normalized["eval_count"] = eval_count

    max_steps = normalized.get("max_steps")
    normalized["max_steps"] = (
        None if max_steps is None else _positive_integer(max_steps, "max_steps")
    )

    task = normalized.get("task")
    if (
        not isinstance(task, str)
        or not task.strip()
        or task != task.strip()
        or "/" in task
        or "\\" in task
    ):
        raise ValueError(
            f"task must be a non-empty RoboDojo task name, got {task!r}"
        )
    normalized["task"] = task
    normalized["seed"] = _integer(normalized.get("seed", 0), "seed")
    normalized["headless"] = _as_bool(
        normalized.get("headless", True), "headless"
    )
    normalized["debug"] = _as_bool(normalized.get("debug", False), "debug")
    normalized["additional_info"] = str(
        normalized.get("additional_info", "openwam")
    )

    calibration_output = normalized.get("calibration_output")
    if isinstance(calibration_output, str) and calibration_output.strip().lower() in {
        "",
        "none",
        "null",
    }:
        calibration_output = None
    if calibration_output is not None:
        calibration_output = str(Path(calibration_output).expanduser().resolve())
    normalized["calibration_output"] = calibration_output
    return normalized


def configure_runtime_import_paths(
    openwam_root: str | os.PathLike[str],
    robodojo_root: str | os.PathLike[str],
) -> None:
    """Prioritize configured checkouts and all vendored Python source roots."""

    openwam_path = Path(openwam_root).expanduser().resolve()
    robodojo_path = Path(robodojo_root).expanduser().resolve()
    configured_paths = [
        openwam_path,
        robodojo_path,
        robodojo_path / "XPolicyLab",
        *(
            robodojo_path / "third_party" / "IsaacLab" / "source" / package
            for package in _ISAACLAB_SOURCE_PACKAGES
        ),
        robodojo_path / "third_party" / "curobo",
    ]
    configured = {str(path.resolve()) for path in configured_paths}

    def removable_editable_hook(entry: Any) -> bool:
        value = str(entry)
        return value.endswith(".finder.__path_hook__") and value.startswith(
            _REMOVABLE_EDITABLE_NAMESPACE_HOOKS
        )

    retained = [
        entry
        for entry in sys.path
        if not removable_editable_hook(entry)
        and str(Path(entry or os.curdir).resolve()) not in configured
    ]
    sys.path[:] = [*(str(path.resolve()) for path in configured_paths), *retained]


def _is_editable_namespace_hook(location: Any) -> bool:
    value = str(location)
    return value.endswith(".finder.__path_hook__") and value.startswith(
        _REMOVABLE_EDITABLE_NAMESPACE_HOOKS
    )


def _module_locations(module_or_spec: Any) -> list[Path]:
    locations: list[Path] = []
    origin = getattr(module_or_spec, "__file__", None)
    if origin is None:
        origin = getattr(module_or_spec, "origin", None)
    if origin not in {None, "built-in", "frozen", "namespace"}:
        locations.append(Path(origin).resolve())

    search_locations = getattr(module_or_spec, "__path__", None)
    if search_locations is None:
        search_locations = getattr(module_or_spec, "submodule_search_locations", None)
    if search_locations is not None:
        locations.extend(
            Path(location).resolve()
            for location in search_locations
            if not _is_editable_namespace_hook(location)
        )
    return list(dict.fromkeys(locations))


def _verify_module_locations(
    name: str,
    locations: list[Path],
    expected_root: Path,
    *,
    source: str,
    checkout_name: str,
) -> str:
    if not locations:
        raise RuntimeError(
            f"{name} has no inspectable {source} origin; expected it under "
            f"configured {checkout_name} checkout {expected_root}"
        )
    inside = [
        location for location in locations if location.is_relative_to(expected_root)
    ]
    outside = [
        location for location in locations if not location.is_relative_to(expected_root)
    ]
    if inside and outside:
        rendered = ", ".join(str(location) for location in outside)
        raise RuntimeError(
            f"{name} has mixed {source} filesystem origins outside the configured "
            f"{checkout_name} checkout {expected_root}: {rendered}"
        )
    if outside:
        rendered = ", ".join(str(location) for location in outside)
        raise RuntimeError(
            f"{name} {source} origin is outside the configured {checkout_name} "
            f"checkout {expected_root}: {rendered}"
        )
    return str(inside[0])


def verify_runtime_import_provenance(
    openwam_root: str | os.PathLike[str],
    robodojo_root: str | os.PathLike[str],
) -> dict[str, str]:
    """Fail before live imports if a top-level package resolves elsewhere."""

    openwam = Path(openwam_root).expanduser().resolve()
    robodojo = Path(robodojo_root).expanduser().resolve()
    if not sys.path or Path(sys.path[0] or os.curdir).resolve() != openwam:
        raise RuntimeError(
            f"OpenWAM checkout must be first on sys.path, expected {openwam}"
        )

    verified: dict[str, str] = {}
    groups = (
        ("OpenWAM", openwam, _OPENWAM_IMPORT_ROOTS),
        ("RoboDojo", robodojo, _ROBODOJO_IMPORT_ROOTS),
    )
    for checkout_name, checkout_root, modules in groups:
        for name, relative_root in modules.items():
            expected = (checkout_root / relative_root).resolve()
            spec = PathFinder.find_spec(name, sys.path)
            if spec is None:
                raise RuntimeError(
                    f"cannot resolve {name} from configured {checkout_name} "
                    f"checkout {expected}"
                )
            verified[name] = _verify_module_locations(
                name,
                _module_locations(spec),
                expected,
                source="resolved",
                checkout_name=checkout_name,
            )
            loaded = sys.modules.get(name)
            if loaded is not None:
                _verify_module_locations(
                    name,
                    _module_locations(loaded),
                    expected,
                    source="loaded",
                    checkout_name=checkout_name,
                )
    return verified


def prepare_launcher_runtime(
    config: dict[str, Any],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply native GPU masking and return visible-device launcher kwargs."""

    environment = os.environ if environ is None else environ
    device_id = _integer(config.get("device_id"), "device_id")
    if device_id < 0:
        raise ValueError(f"device_id must be non-negative, got {device_id}")
    headless = _as_bool(config.get("headless", True), "headless")
    environment["CUDA_VISIBLE_DEVICES"] = str(device_id)
    kit_args = " ".join(
        f"--enable {extension}" for extension in _NATIVE_KIT_EXTENSIONS
    )
    return {
        "headless": headless,
        "enable_cameras": True,
        "device": "cuda:0",
        "kit_args": kit_args,
    }


def isolate_launcher_argv(
    argv: list[str] | None = None,
) -> list[str]:
    """Keep only ``argv[0]`` so Isaac Kit does not inherit OpenWAM CLI flags.

    ``AppLauncher`` forwards the process ``sys.argv`` into Kit. Our runner parses
    its own flags first; leaving ``--config`` / ``--calibration-output`` etc. in
    ``sys.argv`` makes Kit treat them as carb options and can abort the app
    before the native eval body runs.
    """

    current = list(sys.argv if argv is None else argv)
    program = current[0] if current else "benchmarks.robodojo.single_eval"
    sys.argv = [program]
    return current


class _PreservedStdioFDs:
    def __init__(
        self,
        *,
        dup_fn: Callable[[int], int],
        dup2_fn: Callable[[int, int], Any],
        close_fn: Callable[[int], Any],
    ):
        self._dup = dup_fn
        self._dup2 = dup2_fn
        self._close = close_fn
        self._saved: list[tuple[int, int]] = []

    def __enter__(self) -> _PreservedStdioFDs:
        for target in (1, 2):
            try:
                saved = self._dup(target)
            except Exception as error:
                for _saved_target, saved_fd in self._saved:
                    try:
                        self._close(saved_fd)
                    except Exception:
                        pass
                self._saved.clear()
                raise RuntimeError(
                    f"failed to preserve stdio fd {target}: {error}"
                ) from error
            self._saved.append((target, saved))
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        failures: list[tuple[str, int, Exception]] = []
        for target, saved in self._saved:
            try:
                self._dup2(saved, target)
            except Exception as error:
                failures.append(("restore", target, error))
            finally:
                try:
                    self._close(saved)
                except Exception as error:
                    failures.append(("close saved copy for", target, error))
        self._saved.clear()
        if failures and exc_type is None:
            operation, target, error = failures[0]
            raise RuntimeError(
                f"failed to {operation} stdio fd {target}: {error}"
            ) from error
        return False


def preserve_stdio_fds(
    *,
    dup_fn: Callable[[int], int] = os.dup,
    dup2_fn: Callable[[int, int], Any] = os.dup2,
    close_fn: Callable[[int], Any] = os.close,
) -> _PreservedStdioFDs:
    """Preserve stdout/stderr across monitor fd redirection and cleanup."""

    return _PreservedStdioFDs(
        dup_fn=dup_fn,
        dup2_fn=dup2_fn,
        close_fn=close_fn,
    )


def physx_monitor_needed(task_config_path: str | os.PathLike[str]) -> bool:
    """Mirror upstream's lightweight Articulation check, failing safe."""

    try:
        config = _load_yaml(Path(task_config_path))
    except Exception:
        return True
    return bool(config.get("Articulation"))


def start_physx_monitor(
    task_config_path: str | os.PathLike[str],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
) -> PhysXRuntime:
    """Start the monitor before AppLauncher only for articulation tasks."""

    if not physx_monitor_needed(task_config_path):
        return PhysXRuntime(
            enabled=False,
            monitor=None,
            broken_error=_DisabledPhysXBrokenError,
            fatal_error=_DisabledPhysXFatalError,
        )
    module = module_loader("src.eval_client.physx_warning_monitor")
    monitor = module.get_monitor()
    try:
        monitor.start(enabled=True)
    except BaseException:
        try:
            monitor.shutdown()
        except Exception:
            pass
        raise
    return PhysXRuntime(
        enabled=True,
        monitor=monitor,
        broken_error=module.PhysXBrokenError,
        fatal_error=module.PhysXFatalError,
    )


@contextmanager
def _physx_runtime_session(
    task_config_path: str | os.PathLike[str],
) -> Iterator[PhysXRuntime]:
    """Keep original stdio alive until monitor and live-body cleanup finish."""

    cleanup_error: Exception | None = None
    try:
        with preserve_stdio_fds():
            runtime = start_physx_monitor(task_config_path)
            try:
                yield runtime
            finally:
                if runtime.monitor is not None:
                    try:
                        runtime.monitor.shutdown()
                    except Exception as error:
                        cleanup_error = error
    except BaseException:
        if cleanup_error is not None:
            try:
                print(
                    "[OpenWAM RoboDojo] PhysX monitor cleanup failed: "
                    f"{cleanup_error}"
                )
            except Exception:
                pass
        raise
    if cleanup_error is not None:
        try:
            print(
                f"[OpenWAM RoboDojo] PhysX monitor cleanup failed: {cleanup_error}"
            )
        except Exception:
            pass


def build_eval_config_overrides(
    config: Mapping[str, Any],
    *,
    physx_monitor_enabled: bool,
) -> dict[str, Any]:
    """Build upstream-compatible evaluator overrides in one testable place."""

    return {
        "task_name": config["task"],
        "num_envs": 1,
        "device_id": config["device_id"],
        "eval_batch": False,
        "policy_name": "openwam",
        "additional_info": str(config.get("additional_info", "openwam")),
        "seed": config.get("seed", 0),
        "physx_monitor_enabled": bool(physx_monitor_enabled),
    }


def apply_max_steps_override(env_cfg: Any, max_steps: Any) -> Any:
    """Override the processed native limit only when explicitly requested."""

    if max_steps is None:
        return env_cfg
    from omegaconf import OmegaConf

    OmegaConf.update(
        env_cfg,
        "eval_cfg.max_steps",
        _positive_integer(max_steps, "max_steps"),
        force_add=True,
    )
    return env_cfg


@contextmanager
def runtime_working_directory(
    path: str | os.PathLike[str],
) -> Iterator[Path]:
    """Run native relative-path code under RoboDojo and always restore cwd."""

    previous = Path.cwd()
    target = Path(path).expanduser().resolve()
    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(previous)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return value


def load_runner_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _load_yaml(Path(path).expanduser().resolve())


def resume_manifest_path(
    eval_cfg: Mapping[str, Any] | Any,
    run_id: str,
    *,
    benchmark: str,
) -> Path:
    """Mirror current EvalEnv.resume_manifest_path before env construction."""

    return Path(
        "eval_result",
        benchmark,
        str(eval_cfg["task_name"]),
        str(eval_cfg["policy_name"]),
        str(eval_cfg["config_name"]),
        f"{eval_cfg.get('seed', 0)}_{eval_cfg.get('additional_info', '')}",
        f"_resume_{run_id}.json",
    )


def load_resume_manifest(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Load a prior fatal-restart manifest, ignoring absent/corrupt files."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("resume manifest must contain a JSON object")
    except Exception as error:
        print(
            f"[OpenWAM RoboDojo] failed to load resume manifest "
            f"{manifest_path}: {error}; ignoring"
        )
        return None
    print(
        f"[OpenWAM RoboDojo] resuming from {manifest_path} "
        f"(success={data.get('success_nums')} fail={data.get('fail_nums')} "
        f"abandoned={len(data.get('abandoned_layout_ids') or [])})"
    )
    return data


def delete_resume_manifest(env: Any) -> bool:
    """Best-effort manifest deletion after requested completion only."""

    try:
        path = Path(env.resume_manifest_path())
    except Exception:
        return False
    try:
        if not path.exists():
            return False
        path.unlink()
        print(f"[OpenWAM RoboDojo] removed completed resume manifest {path}")
        return True
    except Exception as error:
        print(
            f"[OpenWAM RoboDojo] failed to remove resume manifest {path}: {error}"
        )
        return False


def construct_eval_env_with_resume(
    eval_env_module: Any,
    env_cfg: Any,
    simulation_app: Any,
    *,
    run_id: str,
    benchmark: str,
) -> Any:
    """Load resume state, then construct under the no-network client patch."""

    manifest = resume_manifest_path(
        env_cfg.eval_cfg,
        run_id,
        benchmark=benchmark,
    )
    resume_state = load_resume_manifest(manifest)
    return construct_with_no_network_client(
        eval_env_module,
        lambda: eval_env_module.create_eval_env(
            env_cfg,
            simulation_app,
            resume_state=resume_state,
        ),
    )


def _task_runtime_metadata(task_name: str) -> tuple[str, Path]:
    """Resolve benchmark and task YAML through RoboDojo's safe registries."""

    from env.global_configs import BENCHMARK, ROOT_DIR

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    config_root = Path(ROOT_DIR) / "task" / BENCHMARK / "config"
    return BENCHMARK, Path(
        task_registry.task_config_path(str(config_root), task_name)
    ).resolve()


def _assemble_env_config(
    config: dict[str, Any],
    *,
    physx_monitor_enabled: bool,
):
    """Recreate RoboDojo ``src/eval_client/main.py`` config assembly."""

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    env_config_path = Path(ENV_CONFIG_PATH)
    eval_cfg = load_yaml(
        str(env_config_path / f"{config['env_config']}.yml")
    )
    eval_cfg.update(
        build_eval_config_overrides(
            config,
            physx_monitor_enabled=physx_monitor_enabled,
        )
    )
    deploy_cfg = {
        "policy_name": "openwam",
        "port": config["port"],
        "host": config["host"],
        "protocol": "ws",
        "policy_server_url": f"ws://{config['host']}:{config['port']}",
        "evaluation_id": os.environ["ROBODOJO_RUN_ID"],
        "trial_id": f"{config['task']}-{os.environ['ROBODOJO_RUN_ID']}",
        "action_case_id": f"{config['task']}_case",
        "repeat_index": None,
    }
    benchmark_path = Path(ROOT_DIR) / "task" / BENCHMARK

    def configured_yaml(section: str) -> dict[str, Any]:
        name = eval_cfg["config"][section]
        return load_yaml(str(env_config_path / section / f"{name}.yml"))

    env_cfg = OmegaConf.create(
        {
            "sim": configured_yaml("sim"),
            "scene": configured_yaml("scene"),
            "camera": configured_yaml("camera"),
            "robot": configured_yaml("robot"),
            "task_env": load_yaml(
                task_registry.task_config_path(
                    str(benchmark_path / "config"), config["task"]
                )
            ),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _native_eval_count = process_config(
        env_cfg, task_name=config["task"]
    )
    apply_max_steps_override(env_cfg, config.get("max_steps"))
    OmegaConf.update(
        env_cfg, "eval_cfg.eval_num", config["eval_count"], force_add=True
    )
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg.get("observation", {}).get("collect_freq", 0),
        force_add=True,
    )
    env_cfg.sim.seed = [0]
    return env_cfg


def _install_openwam_policy_alias() -> None:
    """Alias RoboDojo's native demo rollout under a recognizable policy name."""

    deploy = importlib.import_module("XPolicyLab.policy.demo_policy.deploy")
    policy_package = importlib.import_module("XPolicyLab.policy")
    package_name = "XPolicyLab.policy.openwam"
    package = ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    package.deploy = deploy
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.deploy"] = deploy
    setattr(policy_package, "openwam", package)


def _abandon_current_physx_seed(
    env: Any,
    seeds: Any,
    error: BaseException,
    *,
    broken_envs: set[int] | None = None,
) -> None:
    broken_envs = (
        set(getattr(error, "broken_envs", {0}))
        if broken_envs is None
        else set(broken_envs)
    ) or {0}
    abandoned: set[int] = set()
    get_seeds = getattr(env, "get_seeds_for_envs", None)
    if callable(get_seeds):
        abandoned.update(int(seed) for seed in get_seeds(broken_envs))
    if not abandoned and seeds is not None and len(seeds) > 0:
        abandoned.add(int(seeds[0]))
    env.abandoned_seeds.update(abandoned)
    print(
        f"[OpenWAM RoboDojo] PhysX broke single env; abandoning "
        f"seed(s) {sorted(abandoned)}"
    )


def _raise_physx_restart(
    env: Any,
    error: BaseException,
    *,
    monitor: Any | None,
    restart_count: int,
    message: str | None = None,
) -> None:
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as persist_error:
        print(
            "[OpenWAM RoboDojo] failed to persist fatal resume manifest: "
            f"{persist_error}"
        )
    shell_restart = bool(
        monitor is not None and monitor.requires_shell_restart()
    )
    raise PhysXRestartRequired(
        str(error) if message is None else message,
        restart_count=restart_count,
        shell_restart=shell_restart,
    ) from error


def _recover_generic_physx_exception(
    env: Any,
    seeds: Any,
    error: Exception,
    *,
    monitor: Any | None,
    restart_count: int,
) -> bool:
    """Mirror upstream's monitor backstop for otherwise generic failures."""

    if monitor is None:
        return False
    if monitor.is_fatal():
        _raise_physx_restart(
            env,
            error,
            monitor=monitor,
            restart_count=restart_count,
            message=monitor.get_fatal_message() or str(error),
        )

    num_envs = int(getattr(env, "num_envs", 1))
    broken_envs = {
        int(index)
        for index in monitor.get_broken_envs()
        if 0 <= int(index) < num_envs
    }
    if not broken_envs:
        return False
    _abandon_current_physx_seed(
        env,
        seeds,
        error,
        broken_envs=broken_envs,
    )
    env.close()
    return True


def _run_native_episodes(
    env: Any,
    eval_count: int,
    *,
    unstable_error: type[BaseException],
    physx_broken_error: type[BaseException] = _DisabledPhysXBrokenError,
    physx_fatal_error: type[BaseException] = _DisabledPhysXFatalError,
    monitor: Any | None = None,
    fatal_restart_count: int = 1,
    calibration_callback: Callable[[Any], None] | None = None,
) -> None:
    target_completed = eval_count
    calibration_complete = calibration_callback is None
    while int(env.success_nums) + int(env.fail_nums) < target_completed:
        remaining = target_completed - int(env.success_nums) - int(env.fail_nums)
        seeds = env.seed_manager.get_seeds(max_count=remaining)
        if seeds is None:
            raise RuntimeError(
                "RoboDojo seed manager exhausted before the requested "
                f"{eval_count} evaluation episode(s) completed"
            )
        env.env_seeds = seeds
        if monitor is not None:
            monitor.reset()
        try:
            env.reset(seed=seeds)
        except physx_fatal_error as error:
            _raise_physx_restart(
                env,
                error,
                monitor=monitor,
                restart_count=fatal_restart_count,
            )
        except physx_broken_error as error:
            _abandon_current_physx_seed(env, seeds, error)
            env.close()
            continue
        except unstable_error:
            env.seed_manager.eval_step()
            env.close()
            continue
        except Exception as error:
            if _recover_generic_physx_exception(
                env,
                seeds,
                error,
                monitor=monitor,
                restart_count=fatal_restart_count,
            ):
                continue
            raise

        if not calibration_complete:
            calibration_callback(env)
            calibration_complete = True

        try:
            env.run_eval()
        except physx_fatal_error as error:
            _raise_physx_restart(
                env,
                error,
                monitor=monitor,
                restart_count=fatal_restart_count,
            )
        except physx_broken_error as error:
            _abandon_current_physx_seed(env, seeds, error)
            env.close()
            continue
        except unstable_error:
            env.seed_manager.eval_step()
            env.close()
            continue
        except Exception as error:
            if _recover_generic_physx_exception(
                env,
                seeds,
                error,
                monitor=monitor,
                restart_count=fatal_restart_count,
            ):
                continue
            raise

        env.seed_manager.eval_step()
        completed = int(env.success_nums) + int(env.fail_nums)
        print(
            f"[OpenWAM RoboDojo] completed={completed}/{eval_count} "
            f"success={env.success_nums} fail={env.fail_nums}"
        )
        if completed < target_completed:
            env.close()


def _next_fatal_restart_count(
    environ: Mapping[str, str] | None = None,
) -> int:
    environment = os.environ if environ is None else environ
    try:
        current = int(environment.get("ROBODOJO_FATAL_RESTART_COUNT", "0"))
    except (TypeError, ValueError):
        current = 0
    return max(0, current) + 1


def handle_physx_restart(
    restart: PhysXRestartRequired,
    argv: list[str],
    *,
    environ: MutableMapping[str, str] | None = None,
    executable: str | None = None,
    execv: Callable[[str, list[str]], Any] = os.execv,
    max_inprocess_restarts: int = _MAX_INPROCESS_RESTARTS,
) -> int:
    """Re-exec below the cap, otherwise clearly request shell restart."""

    environment = os.environ if environ is None else environ
    if restart.shell_restart or restart.restart_count > max_inprocess_restarts:
        print(
            "[OpenWAM RoboDojo] fatal PhysX restart requires a fresh shell "
            f"process (count={restart.restart_count}); exiting 99"
        )
        return 99

    environment["ROBODOJO_FATAL_RESTART_COUNT"] = str(restart.restart_count)
    python = sys.executable if executable is None else executable
    command = [
        python,
        "-m",
        "benchmarks.robodojo.single_eval",
        *argv,
    ]
    print(
        f"[OpenWAM RoboDojo] re-executing after fatal PhysX failure "
        f"({restart.restart_count}/{max_inprocess_restarts}, "
        f"run_id={environment.get('ROBODOJO_RUN_ID')})"
    )
    try:
        execv(python, command)
    except OSError as error:
        print(f"[OpenWAM RoboDojo] re-exec failed: {error}; exiting 99")
    return 99


def run_eval(config: dict[str, Any]) -> int:
    """Validate, launch Isaac, build the native env, inject OpenWAM, and roll out."""

    config = validate_runner_config(config)
    configure_runtime_import_paths(_OPENWAM_ROOT, config["robodojo_root"])
    verify_runtime_import_provenance(_OPENWAM_ROOT, config["robodojo_root"])
    launcher_kwargs = prepare_launcher_runtime(config)
    os.environ.setdefault(
        "ROBODOJO_RUN_ID", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    run_id = os.environ["ROBODOJO_RUN_ID"]

    with runtime_working_directory(config["robodojo_root"]):
        benchmark, task_config_path = _task_runtime_metadata(config["task"])
        with _physx_runtime_session(task_config_path) as physx_runtime:
            # Heavy imports begin only after validation, provenance checks, GPU
            # masking, monitor startup, and native cwd are established.
            from isaaclab.app import AppLauncher

            isolate_launcher_argv()
            stage_log = Path(
                os.environ.get(
                    "ROBODOJO_STAGE_LOG",
                    "/tmp/openwam-robodojo-eval-stages.log",
                )
            )

            def _stage(message: str) -> None:
                line = f"[OpenWAM RoboDojo] {message}"
                try:
                    with stage_log.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                        handle.flush()
                except Exception:
                    pass
                print(line, flush=True)

            _stage(
                "launching Isaac AppLauncher "
                f"(sys.argv={sys.argv!r}, calibration_output="
                f"{config.get('calibration_output')!r})"
            )
            app_launcher = AppLauncher(**launcher_kwargs)
            simulation_app = app_launcher.app
            _stage("SimulationApp ready")
            try:
                try:
                    from utils.cluttered_generator import UnStableError

                    _stage("assembling env config")
                    env_cfg = _assemble_env_config(
                        config,
                        physx_monitor_enabled=physx_runtime.enabled,
                    )
                    _stage("importing eval_env")
                    eval_env_module = importlib.import_module(
                        "src.eval_client.eval_env"
                    )
                    _stage("constructing EvalEnv")
                    env = construct_eval_env_with_resume(
                        eval_env_module,
                        env_cfg,
                        simulation_app,
                        run_id=run_id,
                        benchmark=benchmark,
                    )
                    _stage("EvalEnv constructed")
                    try:
                        policy_client = OpenWAMRoboDojoModelClient(
                            task_env=env,
                            host=config["host"],
                            port=config["port"],
                            timeout=config["timeout"],
                            num_envs=1,
                            env_config=config["env_config"],
                            robot_action_dim_info=env.robot_action_dim_info,
                            debug=config["debug"],
                        )
                        env.model_client = policy_client
                        try:
                            calibration_callback = None
                            if config["calibration_output"] is not None:

                                def calibration_callback(
                                    current_env: Any,
                                ) -> None:
                                    from benchmarks.robodojo.calibrate_frames import (
                                        calibrate_and_save,
                                    )

                                    calibrate_and_save(
                                        current_env,
                                        config["calibration_output"],
                                        env_idx=0,
                                    )
                                    _stage(
                                        "wrote live calibration to "
                                        f"{config['calibration_output']}"
                                    )

                            _install_openwam_policy_alias()
                            _stage(
                                f"starting native episodes "
                                f"eval_count={config['eval_count']}"
                            )
                            _run_native_episodes(
                                env,
                                config["eval_count"],
                                unstable_error=UnStableError,
                                physx_broken_error=physx_runtime.broken_error,
                                physx_fatal_error=physx_runtime.fatal_error,
                                monitor=physx_runtime.monitor,
                                fatal_restart_count=_next_fatal_restart_count(),
                                calibration_callback=calibration_callback,
                            )
                            delete_resume_manifest(env)
                            _stage("native episodes finished")
                        finally:
                            try:
                                policy_client.close()
                            except Exception as error:
                                _stage(
                                    "policy client cleanup failed: "
                                    f"{error}"
                                )
                            finally:
                                if (
                                    getattr(env, "model_client", None)
                                    is policy_client
                                ):
                                    env.model_client = None
                    finally:
                        try:
                            env.close()
                        except Exception as error:
                            _stage(f"environment cleanup failed: {error}")
                except BaseException as error:
                    import traceback

                    _stage(
                        "eval body failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    try:
                        with stage_log.open("a", encoding="utf-8") as handle:
                            traceback.print_exc(file=handle)
                    except Exception:
                        pass
                    raise
            finally:
                try:
                    simulation_app.close()
                except Exception as error:
                    _stage(f"SimulationApp cleanup failed: {error}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--robodojo-root")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--task")
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--eval-count", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--calibration-output")
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(effective_argv)

    config = load_runner_config(args.config)
    for key, value in (
        ("robodojo_root", args.robodojo_root),
        ("host", args.host),
        ("port", args.port),
        ("task", args.task),
        ("device_id", args.device_id),
        ("eval_count", args.eval_count),
        ("max_steps", args.max_steps),
        ("seed", args.seed),
        ("timeout", args.timeout),
        ("calibration_output", args.calibration_output),
    ):
        if value is not None:
            config[key] = value
    try:
        return run_eval(config)
    except PhysXRestartRequired as restart:
        return handle_physx_restart(restart, effective_argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NoNetworkModelClient",
    "PhysXRestartRequired",
    "PhysXRuntime",
    "apply_max_steps_override",
    "build_eval_config_overrides",
    "configure_runtime_import_paths",
    "construct_eval_env_with_resume",
    "construct_with_no_network_client",
    "delete_resume_manifest",
    "handle_physx_restart",
    "isolate_launcher_argv",
    "load_resume_manifest",
    "load_runner_config",
    "main",
    "physx_monitor_needed",
    "preserve_stdio_fds",
    "prepare_launcher_runtime",
    "resume_manifest_path",
    "run_eval",
    "runtime_working_directory",
    "start_physx_monitor",
    "validate_runner_config",
    "validate_server_endpoint",
    "verify_runtime_import_provenance",
]
