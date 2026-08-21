#!/usr/bin/env python3
"""Run one LIBERO task against an already-running OpenWAM policy server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from openwam2libero_interface import OpenWAMLiberoPolicy


def _repo_root() -> Path:
    env_name = "LIBERO_PATH"
    raw_root = os.environ.get(env_name, "")
    if not raw_root:
        raise SystemExit(f"{env_name} is not set")
    root = Path(raw_root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{env_name} does not point to a LIBERO repo: {root}")
    return root.resolve()


def _config_root() -> Path:
    default = Path.home() / (".libero-openwam")
    env_name = "LIBERO_CONFIG_ROOT"
    return Path(os.environ.get(env_name, default)).expanduser().resolve()


def _write_libero_config() -> None:
    repo_root = _repo_root()
    benchmark_root = repo_root / "libero" / "libero"
    config_root = _config_root()
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(repo_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with (config_root / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a YAML boolean, got {value!r}")
    return value


def _parse_optional_int(value, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a YAML integer or null, got {value!r}")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive or null, got {value!r}")
    return parsed


def _parse_optional_float(value, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a YAML number or null, got {value!r}")
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive or null, got {value!r}")
    return parsed


def _resolve_trial_range(cfg: dict) -> tuple[int, int]:
    """Return the contiguous ``[trial_start, trial_stop)`` evaluation range.

    Trial indices are global within the task and select the matching LIBERO
    init state. The full-suite launcher uses one 0..49 range per task so the
    environment RNG advances continuously across all 50 trials.
    """
    trial_start = cfg.get("trial_start", 0)
    num_trials = cfg.get("num_trials", 1)
    if isinstance(trial_start, bool) or not isinstance(trial_start, int):
        raise TypeError(f"trial_start must be a YAML integer, got {trial_start!r}")
    if isinstance(num_trials, bool) or not isinstance(num_trials, int):
        raise TypeError(f"num_trials must be a YAML integer, got {num_trials!r}")
    if trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {trial_start!r}")
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials!r}")
    return trial_start, trial_start + num_trials


def _resolve_max_steps(cfg: dict, suite_name: str) -> int:
    """Resolve a per-suite horizon, falling back to the legacy scalar field."""
    by_suite = cfg.get("max_steps_by_suite") or {}
    if not isinstance(by_suite, dict):
        raise TypeError(f"max_steps_by_suite must be a YAML mapping, got {by_suite!r}")
    value = by_suite.get(suite_name, cfg.get("max_steps", 600))
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"max_steps for {suite_name} must be a YAML integer, got {value!r}")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"max_steps for {suite_name} must be positive, got {value!r}")
    return parsed


def _parse_settle_action(value, action_dim: int) -> np.ndarray:
    """Parse the benchmark no-op action; legacy configs default to all zeros."""
    if value is None:
        return np.zeros(action_dim, dtype=np.float32)
    if not isinstance(value, (list, tuple)) or len(value) != action_dim:
        raise ValueError(f"settle_action must contain exactly {action_dim} values, got {value!r}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"settle_action values must be YAML numbers, got {value!r}")
    action = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(action)):
        raise ValueError(f"settle_action values must be finite, got {value!r}")
    return action




def _resolve_rng_mode(cfg: dict) -> str:
    mode = cfg.get("rng_mode", "environment")
    if mode != "environment":
        raise ValueError(f"rng_mode must be 'environment', got {mode!r}")
    return mode


def _transform_video_frame(obs: dict, key: str, image_transform: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"LIBERO obs missing camera key while recording video: {key}")
    frame = np.asarray(obs[key])
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Camera '{key}' must be HxWx3 while recording, got {frame.shape}")
    frame = frame.astype(np.uint8, copy=False)
    if image_transform == "rotate_180":
        frame = frame[::-1, ::-1]
    elif image_transform != "none":
        raise ValueError(f"Unsupported image_transform while recording: {image_transform!r}")
    return np.ascontiguousarray(frame)


def _assemble_recording_layout(
    head: np.ndarray,
    left_wrist: np.ndarray | None,
    right_wrist: np.ndarray | None,
    *,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reproduce the checkpoint's L-shaped multi-view layout for review video."""
    from PIL import Image

    top_h = int(round(out_h * (2.0 / 3.0)))
    bottom_h = out_h - top_h
    left_w = out_w // 2
    right_w = out_w - left_w
    canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    canvas.paste(Image.fromarray(head).resize((out_w, top_h), Image.BILINEAR), (0, 0))
    if left_wrist is not None:
        canvas.paste(Image.fromarray(left_wrist).resize((left_w, bottom_h), Image.BILINEAR), (0, top_h))
    if right_wrist is not None:
        canvas.paste(
            Image.fromarray(right_wrist).resize((right_w, bottom_h), Image.BILINEAR),
            (left_w, top_h),
        )
    return np.asarray(canvas, dtype=np.uint8)


class _TrialVideoRecorder:
    """Stream every simulator frame to per-camera and model-layout MP4 files."""

    def __init__(self, run_dir: Path, trial: int, cfg: dict):
        import imageio.v2 as imageio

        self._cfg = cfg
        self._imageio = imageio
        self._trial = int(trial)
        self._frame_count = 0
        self._writers = {}
        self.paths = {}
        fps = int(cfg.get("video_fps", 20))
        if fps <= 0:
            raise ValueError(f"video_fps must be positive, got {fps}")
        run_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"trial_{trial:03d}"
        streams = ["agentview", "model_layout"]
        if cfg.get("left_wrist_camera_key"):
            streams.append("left_wrist")
        if cfg.get("right_wrist_camera_key"):
            streams.append("right_wrist")
        try:
            for stream in streams:
                path = run_dir / f"{prefix}_{stream}.mp4"
                self.paths[stream] = str(path)
                self._writers[stream] = imageio.get_writer(
                    str(path),
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=None,
                )
        except Exception:
            self.close()
            raise

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def append(self, obs: dict) -> None:
        transform = self._cfg.get("image_transform", "rotate_180")
        head = _transform_video_frame(obs, self._cfg.get("head_camera_key", "agentview_image"), transform)
        left_key = self._cfg.get("left_wrist_camera_key")
        right_key = self._cfg.get("right_wrist_camera_key")
        left = _transform_video_frame(obs, left_key, transform) if left_key else None
        right = _transform_video_frame(obs, right_key, transform) if right_key else None
        self._writers["agentview"].append_data(head)
        if left is not None:
            self._writers["left_wrist"].append_data(left)
        if right is not None:
            self._writers["right_wrist"].append_data(right)
        layout = _assemble_recording_layout(
            head,
            left,
            right,
            out_h=int(self._cfg.get("video_model_height", 384)),
            out_w=int(self._cfg.get("video_model_width", 320)),
        )
        self._writers["model_layout"].append_data(layout)
        self._frame_count += 1

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def _make_env(task, cfg: dict):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    assets_dir = Path(get_libero_path("assets"))
    if not assets_dir.is_dir():
        raise SystemExit(f"LIBERO assets are missing: {assets_dir}")
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=int(cfg.get("camera_height", 128)),
        camera_widths=int(cfg.get("camera_width", 128)),
    )




def run_eval(cfg: dict) -> int:
    _write_libero_config()

    seed = int(cfg.get("seed", 42))
    rng_mode = _resolve_rng_mode(cfg)

    from libero.libero import benchmark

    suite_name = cfg.get("suite", "libero_spatial")
    task_id = int(cfg.get("task_id", 0))
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise SystemExit(f"Unknown LIBERO suite: {suite_name}. Available: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    print(f"[libero-eval] suite={suite_name} task_id={task_id} task={task.name}")
    print(f"[libero-eval] instruction={task.language}")

    trial_start, trial_stop = _resolve_trial_range(cfg)
    num_trials = trial_stop - trial_start
    max_steps = _resolve_max_steps(cfg, suite_name)
    settle_steps = int(cfg.get("settle_steps", 10))
    action_dim = int(cfg.get("action_dim", 7))
    settle_action = _parse_settle_action(cfg.get("settle_action"), action_dim)
    reseed_each_trial = _require_bool(cfg.get("reseed_each_trial", False), "reseed_each_trial")
    fail_on_incomplete = _require_bool(cfg.get("fail_on_incomplete", False), "fail_on_incomplete")
    save_videos = _require_bool(cfg.get("save_videos", False), "save_videos")
    run_dir = None
    if save_videos:
        output_root = Path(cfg.get("video_dir", "./evaluate_results/libero")).expanduser().resolve()
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / (
            f"{suite_name}_task{task_id:02d}_trials{trial_start:03d}-{trial_stop - 1:03d}_{run_stamp}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        print(f"[libero-eval] video/output directory={run_dir}")
    init_states = task_suite.get_task_init_states(task_id)
    env = _make_env(task, cfg)
    if rng_mode == "environment" and not reseed_each_trial:
        # FastWAM seeds the environment once before the episode loop. Repeated
        # reset()/set_init_state() calls then advance the same simulator RNG.
        env.seed(seed)
    print(
        f"[libero-eval] trials={trial_start}:{trial_stop} count={num_trials} max_steps={max_steps} "
        f"settle_steps={settle_steps} seed={seed} "
        f"rng_mode={rng_mode} reseed_each_trial={reseed_each_trial}"
    )

    policy = OpenWAMLiberoPolicy(
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8848)),
        request_timeout=int(cfg.get("request_timeout", 300)),
        action_mode=cfg.get("action_mode", "eef"),
        head_camera_key=cfg.get("head_camera_key", "agentview_image"),
        left_wrist_camera_key=cfg.get("left_wrist_camera_key"),
        right_wrist_camera_key=cfg.get("right_wrist_camera_key"),
        image_transform=cfg.get("image_transform", "rotate_180"),
        send_state=_require_bool(cfg.get("send_state", True), "send_state"),
        state_keys=list(cfg.get("state_keys") or []),
        state_dim=_parse_optional_int(cfg.get("state_dim"), "state_dim"),
        action_dim=action_dim,
        action_indices=cfg.get("action_indices"),
        action_clip=_parse_optional_float(cfg.get("action_clip"), "action_clip"),
        osc_pos_scale=_parse_optional_float(cfg.get("osc_pos_scale"), "osc_pos_scale"),
        osc_rot_scale=_parse_optional_float(cfg.get("osc_rot_scale"), "osc_rot_scale"),
        env=env,
        debug=_require_bool(cfg.get("debug", False), "debug"),
        debug_dir=cfg.get("debug_dir", "./debug_libero"),
    )
    successes = 0
    trial_results = []
    try:
        for trial in range(trial_start, trial_stop):
            recorder = _TrialVideoRecorder(run_dir, trial, cfg) if run_dir is not None else None
            trial_result = {
                "trial": trial,
                "success": False,
                "policy_steps": 0,
                "settle_steps": settle_steps,
                "video_paths": {} if recorder is None else recorder.paths,
            }
            if reseed_each_trial:
                env.seed(seed + trial)
            try:
                obs = env.reset()
                if len(init_states) > 0:
                    obs = env.set_init_state(init_states[trial % len(init_states)])
                if recorder is not None:
                    recorder.append(obs)
                for _ in range(settle_steps):
                    obs, _, _, _ = env.step(settle_action)
                    if recorder is not None:
                        recorder.append(obs)
                policy.reset()

                done = False
                for step in range(max_steps):
                    action = policy.act(obs, task.language)
                    obs, reward, done, info = env.step(action)
                    trial_result["policy_steps"] = step + 1
                    trial_result["last_reward"] = float(reward)
                    if recorder is not None:
                        recorder.append(obs)
                    if done:
                        successes += 1
                        trial_result["success"] = True
                        print(f"[RESULT] trial={trial} success step={step + 1} reward={reward}")
                        break
                if not done:
                    print(f"[RESULT] trial={trial} failed max_steps={max_steps}")
            finally:
                if recorder is not None:
                    trial_result["video_frames"] = recorder.frame_count
                    recorder.close()
                trial_results.append(trial_result)
    finally:
        env.close()
        policy.close()

    rate = successes / max(num_trials, 1)
    print(f"Success rate: {successes}/{num_trials} => {rate * 100:.1f}%")
    if run_dir is not None:
        result = {
            "suite": suite_name,
            "task_id": task_id,
            "task": task.name,
            "instruction": task.language,
            "num_trials": num_trials,
            "trial_start": trial_start,
            "trial_stop": trial_stop,
            "successes": successes,
            "success_rate": rate,
            "max_steps": max_steps,
            "settle_steps": settle_steps,
            "seed": seed,
            "rng_mode": rng_mode,
            "reseed_each_trial": reseed_each_trial,
            "policy_config_sha256": cfg.get("policy_config_sha256"),
            "camera_height": int(cfg.get("camera_height", 128)),
            "camera_width": int(cfg.get("camera_width", 128)),
            "trials": trial_results,
        }
        result_path = run_dir / "results.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[libero-eval] results={result_path}")
    return 1 if fail_on_incomplete and successes != num_trials else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--suite")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--trial-start", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    cfg["policy_config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    for key in [
        "host",
        "port",
        
        "suite",
        "task_id",
        "num_trials",
        "trial_start",
        "seed",
        "video_dir",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return run_eval(cfg)


if __name__ == "__main__":
    sys.exit(main())
