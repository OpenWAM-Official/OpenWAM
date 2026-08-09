#!/usr/bin/env python
"""Drive a VLABench evaluation against a running OpenWAM policy server.

Runs inside the VLABench conda environment (it imports VLABench's ``Evaluator``)
and is launched by ``single_eval.sh``, which puts the VLABench repo on
``PYTHONPATH`` and sets ``VLABENCH_ROOT`` / ``MUJOCO_GL``.

This exists instead of patching VLABench's own ``scripts/evaluate_policy.py``
so the OpenWAM integration stays entirely inside this repo — VLABench needs no
modification and any checkout works.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from openwam2vlabench_interface import OpenWAMVLABenchPolicy  # noqa: E402

# The dimension VLABench leaves open. Its README lists six tracks but ships only
# five configs: track 5 has no fixed episode set because the train/eval task
# split is the user's to choose ("kept open in this setting, allowing users to
# choose training tasks and evaluation tasks according to their needs"). So it
# runs on seeded episodes and the caller must name the held-out tasks.
OPEN_TRACK = "track_5_cross_task"

TRACKS = (
    "track_1_in_distribution",
    "track_2_cross_category",
    "track_3_common_sense",
    "track_4_semantic_instruction",
    OPEN_TRACK,
    "track_6_unseen_texture",
)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def patch_intention_score_keyerror(env_cls) -> bool:
    """Stop an upstream metric bug from discarding otherwise-valid episodes.

    ``Evaluator.evaluate_single_episode`` calls ``env.get_intention_score()``
    AFTER the rollout finishes but BEFORE it records ``info["success"]``. For
    tasks whose target is resolved positionally (the ``*_spatial`` family, and
    ``select_ingredient``) that call raises ``KeyError``:
    ``reset_intention_distance`` only keys ``self.entities``
    (VLABench/tasks/dm_task.py:353), so a positionally-resolved target name is
    absent. The exception unwinds to the per-episode ``except`` in
    ``Evaluator.evaluate``, which drops the entire episode — a completed, often
    successful rollout is thrown away. Requesting only ``success_rate`` does not
    help: the call site is unconditional.

    Returning NaN rather than 0.0 is deliberate. ``compute_metric`` averages
    each metric independently over the same ``infos`` list, so NaN confines the
    damage to ``intention_score`` (which reads as visibly unavailable) while
    ``success_rate`` and ``progress_score`` are recovered intact. A 0.0 would
    silently deflate the intention_score average instead.

    Only ``KeyError`` is caught, so unrelated failures still surface. Returns
    True if the patch was applied, False if it was already in place.
    """
    original = env_cls.get_intention_score
    if getattr(original, "_openwam_keyerror_safe", False):
        return False

    state = {"warned": False}

    def get_intention_score(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except KeyError as exc:
            if not state["warned"]:
                print(
                    f"[vlabench] NOTE: upstream get_intention_score raised KeyError({exc}); "
                    "reporting intention_score=nan so the episode's success/progress survive. "
                    "See patch_intention_score_keyerror() for why."
                )
                state["warned"] = True
            return float("nan")

    get_intention_score._openwam_keyerror_safe = True
    env_cls.get_intention_score = get_intention_score
    return True


def _resolve_episodes(cfg: dict, vlabench_root: Path):
    """Return ``(tasks, episode_config)`` for the configured track / task list.

    ``episode_config`` is ``None`` for seeded episodes — the evaluator then
    builds each scene from ``seed=42+i`` instead of a frozen config.
    """
    track = cfg.get("eval_track")
    episode_config = None
    tasks = cfg.get("tasks")
    if track:
        if track not in TRACKS:
            raise SystemExit(f"unknown eval_track {track!r}; choose from {TRACKS} or null")
        track_file = vlabench_root / "configs" / "evaluation" / "tracks" / f"{track}.json"
        if track == OPEN_TRACK:
            if track_file.is_file():
                # A future VLABench release, or a config the user froze themselves.
                with open(track_file) as f:
                    episode_config = json.load(f)
            elif not tasks:
                raise SystemExit(
                    f"{OPEN_TRACK} ships no episode config — the held-out task split is yours to "
                    "choose. Pass --tasks (or set `tasks:` in the policy config) to name them."
                )
        elif not track_file.is_file():
            raise SystemExit(f"track config not found: {track_file}")
        else:
            with open(track_file) as f:
                episode_config = json.load(f)
        if episode_config is not None:
            tasks = list(tasks) if tasks else list(episode_config.keys())
            missing = [t for t in tasks if t not in episode_config]
            if missing:
                raise SystemExit(f"task(s) {missing} are not part of track {track}")
    if not tasks:
        raise SystemExit(
            "no tasks to evaluate: set `tasks:` in the policy config, or an `eval_track:` to take "
            "the track's full task list"
        )
    return tasks, episode_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="policy_config.yml path")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--eval-track", default=None, help="override the config's eval_track ('' = none)")
    parser.add_argument("--tasks", nargs="+", default=None, help="override the config's task list")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--visualization", action="store_true", default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    if args.host is not None:
        cfg["host"] = args.host
    if args.port is not None:
        cfg["port"] = args.port
    if args.eval_track is not None:
        cfg["eval_track"] = args.eval_track or None
    if args.tasks is not None:
        cfg["tasks"] = args.tasks
    if args.n_episodes is not None:
        cfg["n_episodes"] = args.n_episodes
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.visualization:
        cfg["visualization"] = True

    vlabench_root = Path(os.environ.get("VLABENCH_ROOT", ""))
    if not vlabench_root.is_dir():
        raise SystemExit("VLABENCH_ROOT must point at the VLABench repo (single_eval.sh sets it)")

    # Imported here so a config error surfaces before MuJoCo/dm_control load.
    # Importing the robots / tasks packages is what registers the classes the
    # evaluator resolves by name (`import *` is a SyntaxError inside a function,
    # and unnecessary — the registration happens on module execution).
    import VLABench.robots  # noqa: F401
    import VLABench.tasks  # noqa: F401
    from VLABench.envs.dm_env import LM4ManipDMEnv
    from VLABench.evaluation.evaluator import Evaluator

    patch_intention_score_keyerror(LM4ManipDMEnv)

    tasks, episode_config = _resolve_episodes(cfg, vlabench_root)
    n_episodes = int(cfg.get("n_episodes", 1))
    save_dir = Path(cfg.get("save_dir", "benchmarks/vlabench/_eval_out"))
    if cfg.get("eval_track"):
        save_dir = save_dir / cfg["eval_track"]
    save_dir.mkdir(parents=True, exist_ok=True)

    episodes_from = "frozen track config" if episode_config is not None else "seeds 42..42+n"
    print(f"[vlabench] track   : {cfg.get('eval_track') or '(no track)'}  [{episodes_from}]")
    print(f"[vlabench] tasks   : {tasks}")
    print(f"[vlabench] episodes: {n_episodes}")
    print(f"[vlabench] server  : ws://{cfg.get('host')}:{cfg.get('port')}")
    print(f"[vlabench] save_dir: {save_dir}")

    policy = OpenWAMVLABenchPolicy(
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8848)),
        request_timeout=int(cfg.get("request_timeout", 300)),
        head_camera_index=int(cfg.get("head_camera_index", 2)),
        left_wrist_camera_index=cfg.get("left_wrist_camera_index", 3),
        right_wrist_camera_index=cfg.get("right_wrist_camera_index", 0),
        send_state=bool(cfg.get("send_state", True)),
        state_dim=cfg.get("state_dim", 10),
        action_dim=int(cfg.get("action_dim", 10)),
        gripper_open_threshold=float(cfg.get("gripper_open_threshold", 0.5)),
        gripper_open_width=float(cfg.get("gripper_open_width", 0.04)),
        robot_base_fallback=tuple(cfg.get("robot_base_fallback", (0.0, -0.4, 0.78))),
        debug=bool(cfg.get("debug", False)),
        debug_dir=str(cfg.get("debug_dir", "./debug_vlabench")),
    )

    # One Evaluator per task, each with its own episode budget. Tracks are not
    # uniform — track_2_cross_category ships 50 episodes for most tasks but only
    # 10 for insert_flower — and Evaluator asserts
    # `len(episode_config[task]) >= n_episodes`. A single global clamp would drag
    # every task down to the smallest track entry, so clamp per task and say so.
    result = {}
    try:
        for task in tasks:
            budget = n_episodes
            if episode_config is not None:
                available = len(episode_config[task])
                if budget > available:
                    print(
                        f"[vlabench] NOTE: {task} has only {available} episodes in this track; "
                        f"evaluating {available} instead of the requested {n_episodes}"
                    )
                    budget = available
            evaluator = Evaluator(
                tasks=[task],
                n_episodes=budget,
                episode_config=episode_config,
                max_substeps=1,
                save_dir=str(save_dir),
                visulization=bool(cfg.get("visualization", False)),
                metrics=list(cfg.get("metrics", ["success_rate"])),
            )
            task_result = evaluator.evaluate(policy)
            for name, metrics in task_result.items():
                result[name] = {**metrics, "n_episodes": budget}
    finally:
        policy.close()

    out_dir = save_dir / "openwam"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_result.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[vlabench] result -> {out_file}")
    for task, metrics in result.items():
        rendered = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {task:28s} {rendered}")


if __name__ == "__main__":
    main()
