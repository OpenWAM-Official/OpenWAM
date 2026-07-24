#!/usr/bin/env python3
"""Episode-level RoboTwin eval worker driven by the central dispatcher.

One worker process == one ``(task, mode)`` assignment lifetime:

    connect -> hello -> request_task
      -> exit  : dispatcher has no more work for this slot  (exit code 3)
      -> run T : boot RoboTwin env for T once, then loop pulling seeds:
           request_seed -> expert-check(seed)
             -> invalid : report_probe, pull next seed
             -> valid   : request_commit
                  -> commit:false : target met -> stop  (exit code 0, relaunch)
                  -> commit:true  : policy rollout -> report_result -> next seed
           ...until drain (exit code 0, supervisor launches a fresh worker)

Why reuse RoboTwin's ``main()`` instead of re-implementing setup:
``script/eval_policy.py``'s ``main()`` does a lot of version-sensitive argument
assembly (task_config yaml, embodiment/camera config, save_dir, video sizing)
and *then* calls ``eval_policy(...)``. We keep all of that by monkeypatching
``module.eval_policy`` with the dispatcher-driven loop below — which mirrors the
upstream loop body exactly (the two ``setup_demo`` phases, the expert
``plan_success and check_success`` gate, ``generate_episode_descriptions`` +
``np.random.choice`` instruction, ``reset_model`` before rollout, the
``take_action_cnt < step_lim`` rollout, and per-episode ``close_env``) — and
only swaps the ``while succ_seed < test_num`` driver for the dispatcher stream.

This module needs the RoboTwin environment to run (numpy + RoboTwin +
``benchmarks.utils``); it can be import/syntax-checked anywhere.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dispatcher as D  # noqa: E402  (colocated module; stdlib-only)
from eval_policy_wrapper import bootstrap_robotwin_module  # noqa: E402


def _make_dispatcher_eval_policy(module, client: "D.DispatcherClient"):
    """Build a drop-in replacement for ``module.eval_policy`` that pulls seeds
    from ``client`` instead of running the ``while succ_seed < test_num`` loop.

    Signature matches how ``module.main`` calls it:
        eval_policy(task_name, TASK_ENV, args, model, st_seed,
                    test_num=..., video_size=..., instruction_type=...)
    """
    import subprocess

    import numpy as np

    UnStableError = module.UnStableError
    eval_function_decorator = module.eval_function_decorator
    generate_episode_descriptions = module.generate_episode_descriptions

    def eval_policy(task_name, TASK_ENV, args, model, st_seed, test_num=100, video_size=None, instruction_type=None):
        print(f"\033[34mTask Name: {args['task_name']}\033[0m")
        print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

        policy_name = args["policy_name"]
        eval_func = eval_function_decorator(policy_name, "eval")
        reset_func = eval_function_decorator(policy_name, "reset_model")
        clear_cache_freq = args["clear_cache_freq"]
        args["eval_mode"] = True

        TASK_ENV.suc = 0
        TASK_ENV.test_num = 0
        now_id = 0

        while True:
            r = client.request_seed()
            if r.get("drain"):
                break
            now_seed = int(r["seed"])

            # -- (1) expert check: cheap-ish gate; only valid scenes are testable --
            render_freq = args["render_freq"]
            args["render_freq"] = 0
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                client.report_probe(now_seed, valid=False)
                continue
            except Exception:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                print("error occurs !")
                traceback.print_exc()
                client.report_probe(now_seed, valid=False)
                continue

            if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
                args["render_freq"] = render_freq
                client.report_probe(now_seed, valid=False)
                continue
            args["render_freq"] = render_freq

            # -- commit handshake: exactly test_num episodes, no overshoot --
            commit = client.request_commit(now_seed)
            if not commit.get("commit"):
                break  # target already met elsewhere; discard this valid scene

            # -- (2) rollout with the policy under test (same seed => same scene) --
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            results = generate_episode_descriptions(args["task_name"], [episode_info["info"]], test_num)
            instruction = np.random.choice(results[0][instruction_type])
            TASK_ENV.set_instruction(instruction=instruction)

            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", video_size,
                        "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                        "-vcodec", "libx264", "-crf", "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

            succ = False
            reset_func(model)
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break
            # A failed episode that ran out of steps (hit step_lim) vs. a real
            # terminal failure — dispatcher records this so a tight step_lim is
            # distinguishable from genuine model error.
            step_limit_hit = (not succ) and (TASK_ENV.take_action_cnt >= TASK_ENV.step_lim)

            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()

            if succ:
                TASK_ENV.suc += 1
                print("\033[92mSuccess!\033[0m")
            else:
                print("\033[91mFail!\033[0m")

            now_id += 1
            TASK_ENV.close_env(clear_cache=(now_id % clear_cache_freq == 0))
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()
            TASK_ENV.test_num += 1

            print(
                f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
                f"\033[92m{args['task_config']}\033[0m\n"
                f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
                f"\033[95m{round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\033[0m, "
                f"current seed: \033[90m{now_seed}\033[0m\n"
            )
            client.report_result(now_seed, success=bool(succ), step_limit_hit=bool(step_limit_hit))

        return st_seed, TASK_ENV.suc

    return eval_policy


def _load_usr_args(config_path: str, overrides: dict) -> dict:
    import yaml  # lazy: dry-run has no RoboTwin/yaml dependency

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _run_dry(client: "D.DispatcherClient", args) -> int:
    """Simulate one job's episode stream without RoboTwin: sleep for the expert
    check + rollout, fake validity/success. Exercises the full protocol +
    supervisor contract end-to-end (used by --dry-run and DLC dry-run smoke)."""
    import random
    import time

    rng = random.Random(1000 * args.node + args.worker)
    print(f"[episode_worker:dry] n{args.node}/w{args.worker} simulating task stream")
    while True:
        r = client.request_seed()
        if r.get("drain"):
            break
        seed = int(r["seed"])
        time.sleep(args.dry_run_expert)
        if rng.random() >= args.dry_run_valid_prob:
            client.report_probe(seed, valid=False)
            continue
        if not client.request_commit(seed).get("commit"):
            break
        time.sleep(args.dry_run_sleep)
        client.report_result(seed, success=(rng.random() < args.dry_run_success_prob))
    return D.EXIT_RELAUNCH


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Episode-level RoboTwin eval worker")
    ap.add_argument("--config", default=None, help="policy_config.yml (host/port/action_type/...)")
    ap.add_argument("--dispatcher", required=True, help="dispatcher host:port")
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--server-host", default=None, help="policy server host (overrides config)")
    ap.add_argument("--server-port", type=int, default=None, help="policy server port (overrides config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-setting", default="openwam")
    ap.add_argument("--policy-name", default=None)
    ap.add_argument("--instruction-type", default=None)
    ap.add_argument("--connect-timeout", type=float, default=120.0)
    # Dry-run: simulate episodes without RoboTwin (protocol/scheduling smoke).
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-run-sleep", type=float, default=0.2, help="simulated rollout seconds")
    ap.add_argument("--dry-run-expert", type=float, default=0.05, help="simulated expert-check seconds")
    ap.add_argument("--dry-run-valid-prob", type=float, default=0.8)
    ap.add_argument("--dry-run-success-prob", type=float, default=0.5)
    args = ap.parse_args(argv)
    if not args.dry_run and not args.config:
        ap.error("--config is required unless --dry-run")

    import time

    d_host, _, d_port = args.dispatcher.rpartition(":")
    if not d_host or not d_port.isdigit():
        print(f"[episode_worker] bad --dispatcher {args.dispatcher!r}", file=sys.stderr)
        return D.EXIT_ERROR

    # Connect + claim a task BEFORE the expensive RoboTwin bootstrap, so a slot
    # that has no more work exits instantly without paying import/CUDA cost.
    # A persistent connect failure means the dispatcher already finished and
    # shut down (the launcher confirmed it was up before starting us), so treat
    # it as "no more work" rather than an error — avoids spurious error noise on
    # the normal end-of-run reconnect race.
    client = None
    for attempt in range(6):
        try:
            client = D.DispatcherClient(d_host, int(d_port), timeout=args.connect_timeout)
            break
        except OSError:
            time.sleep(0.5)
    if client is None:
        print(f"[episode_worker] dispatcher {args.dispatcher} unreachable; assuming run complete")
        return D.EXIT_NO_MORE_WORK

    try:
        client.hello(node=args.node, worker=args.worker, gpu=args.gpu, port=args.server_port or -1)
        task_resp = client.request_task()
        if task_resp.get("action") != "run":
            print(f"[episode_worker] no more work (n{args.node}/w{args.worker}); exiting")
            client.close()
            return D.EXIT_NO_MORE_WORK

        task, mode = task_resp["task"], task_resp["mode"]
        print(f"[episode_worker] n{args.node}/w{args.worker} booting task={task} mode={mode}")

        if args.dry_run:
            rc = _run_dry(client, args)
            client.close()
            return rc

        module = bootstrap_robotwin_module()

        overrides = {
            "task_name": task,
            "task_config": mode,
            # Namespace save_dir/videos per slot so concurrent workers on the
            # same task never collide on eval_result/.../episodeN.mp4.
            "ckpt_setting": f"{args.ckpt_setting}_n{args.node}w{args.worker}",
            "seed": args.seed,
            "policy_name": args.policy_name or "openwam2robotwin_interface",
        }
        if args.instruction_type is not None:
            overrides["instruction_type"] = args.instruction_type
        if args.server_host is not None:
            overrides["host"] = args.server_host
        if args.server_port is not None:
            overrides["port"] = args.server_port
        usr_args = _load_usr_args(args.config, overrides)

        # Swap the batch loop for the dispatcher-driven one, then let RoboTwin's
        # own main() build TASK_ENV/model and call into it.
        module.eval_policy = _make_dispatcher_eval_policy(module, client)
        module.main(usr_args)
        client.close()
        return D.EXIT_RELAUNCH
    except ConnectionError:
        # Dispatcher went away (normally: it completed and shut down). Nothing
        # more this slot can do; rank0's completeness check is the source of
        # truth for whether the whole run actually finished.
        print("[episode_worker] dispatcher connection closed; assuming run complete")
        return D.EXIT_NO_MORE_WORK
    except Exception:
        traceback.print_exc()
        try:
            client.close()
        except OSError:
            pass
        return D.EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
