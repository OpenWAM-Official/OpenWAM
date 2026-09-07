# EBench (GenManip) Evaluation

Two processes on the OpenWAM side: the **policy server** (this repo's env) and a thin **bridge client**. The bridge polls the GenManip eval server over HTTP (north) and queries the policy server over WebSocket (south, [wire protocol](../README.md)); the Isaac Sim evaluation server itself is the [GenManip](https://github.com/InternRobotics/GenManip) part of the [EBench](https://github.com/InternRobotics/EBench) project and may run on a different machine.

Commands below assume the bridge env's python at `/path/to/bridge-env/bin/python` — substitute your actual paths.

## 1. Environment Setup

**Sim server** (per the EBench repo, possibly another machine): follow the [EBench environment guide](https://internrobotics.github.io/EBench-doc/getting-started/environment/) — Isaac Sim 4.1.0 (CUDA 12.1) + cuRobo from the [GenManip](https://github.com/InternRobotics/GenManip) repo, plus the `EBench-Assets` dataset (~34 GB, `huggingface-cli download InternRobotics/EBench-Assets --repo-type dataset --local-dir saved`). Isaac Sim 4.1.0 does not support Blackwell GPUs.

**Bridge env** (what OpenWAM's launch scripts run — no torch, no OpenWAM install):

```bash
git clone --recursive https://github.com/InternRobotics/EBench && cd EBench
pip install -e third_party/genmanip-client          # the only EBench piece the bridge imports
# genmanip-client declares only `requests`; its EvalClient additionally imports these at runtime:
pip install numpy Pillow "websockets>=15" PyYAML opencv-python-headless "PyTurboJPEG<2" filelock
```

Point `EBENCH_PYTHON` at this env's python. `PyTurboJPEG>=2` needs libjpeg-turbo 3.x on the system (`libturbojpeg.so`); on Ubuntu ≤ 22.04 keep the `<2` pin. For a no-Isaac sanity check see the mock flow in section 3.

> Hugging Face unreachable? Export `HF_ENDPOINT=https://hf-mirror.com` before running any downloader in this guide.

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-EBench
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench` (or use a checkpoint you trained yourself). Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench   # keep running in its own terminal
```

WebSocket port 8848 by default; the server is ready once it logs `WebSocket server started`. For N parallel workers: `NUM_GPUS=N bash scripts/deploy.sh <ckpt_dir>` → one server per GPU on ports 8848…8848+N-1 (per-GPU logs in `logs/deploy_gpu<i>.log`).

**Paper settings.** The numbers in section 5 were produced with `bash scripts/deploy.sh <ckpt_dir> --compile-enabled false optimization.dit_cache.enabled=false` — the DiT velocity cache shifts the commanded EE pose by ~2 mm and is kept off for benchmark runs; compile is only a speed knob. Everything else is the `configs/deploy.yaml` default (`denoise_steps: 10`, sync denoising, sync executor, `inference_horizon: null` = execute the full 32-step chunk before re-planning).

## 3. Run the Evaluation

On the sim machine (GenManip repo): `python ray_eval_server.py --host 0.0.0.0 --port 8087 --no_save_process`, then submit the split you want, e.g. the held-out split used for the paper numbers: `gmp submit ebench/generalist/test_mini --run_id <run_id>` (`val_train` / `val_unseen` are the open tuning splits; `ebench/mobile_manip/<split>` and `ebench/table_top_manip/<split>` are the specialist tracks). Single worker from this repo:

```bash
EBENCH_PYTHON=/path/to/bridge-env/bin/python \
bash benchmarks/ebench/single_eval.sh \
    --url http://<sim-host>:8087 --run-id <run_id> \
    --ckpt-config assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench/config.yaml
```

All bridge flags can also come from a YAML: `--config benchmarks/ebench/policy_config.yml` (explicit CLI flags win).

Parallel run — one policy server per worker:

```bash
NUM_GPUS=4 bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench   # terminal 1; blocks until Ctrl+C
# terminal 2, once all servers log "WebSocket server started":
NUM_WORKERS=4 EBENCH_PYTHON=/path/to/bridge-env/bin/python \
bash benchmarks/ebench/multi_eval.sh \
    --url http://<sim-host>:8087 --run-id <run_id> \
    --ckpt-config assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench/config.yaml
```

Scores are written by the GenManip server under `saved/eval_results/<task>/<run_id>/` (`gmp status` to see the resolved path); the bridge's `EvalClient` also mirrors per-episode `episode_result.json` files into `client_results/` under the directory it runs from (override with `GENMANIP_RESULT_DIR=/some/dir`).

Offline sanity check without Isaac Sim — the mock replays real EBench-Dataset episodes over the exact wire format (verifies bridge + conversion; produces no scores). It runs in the **training env** (needs `pandas`, `pyarrow`, `av`), needs only one dataset bucket, and serves exactly one bridge per mock instance (use `single_eval.sh`, not `multi_eval.sh`):

```bash
# one bucket is enough (the full EBench-Dataset is ~330 GB; the interactive
# scripts/download_assets/download_benchmark_data.py fetches all of it)
huggingface-cli download InternRobotics/EBench-Dataset --repo-type dataset \
    --local-dir assets/benchmark_data/ebench --include 'simple_pnp/task1/*'
python benchmarks/ebench/mock_genmanip_server.py \
    --dataset-dir assets/benchmark_data/ebench --bucket simple_pnp/task1 \
    --episodes 2 --steps-per-episode 8 --port 8087
# in another shell: single_eval.sh as above with --url http://127.0.0.1:8087 (no --run-id needed)
```

A clean mock run ends with the bridge logging `run complete: 2 episodes, 16 steps bridged` and the mock logging `16 actions, 0 violations`. Any action-contract violation makes the mock return HTTP 500 for subsequent reset, reset-result, and step requests (`ACTION CONTRACT VIOLATION` in the mock log), so the bridge's reconnect attempts burn out and it exits non-zero instead of silently restarting the replay. The mock writes every accepted action to `./ebench_mock_actions.jsonl` (`--log-file`).

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Always pass `--ckpt-config` when the checkpoint dir is reachable: it hard-verifies the contract (`dataloader.type=ebench`, `action_mode=eef`, `unify_action=true`); without it you only get an UNVERIFIED warning. Checkpoints trained with the pre-release codebase carry `action_mode: ebench` and are rejected by this check — retrain, or edit their `config.yaml` (and the `normalization_stats.npy` key) to `eef` if the raw-23 layout is unchanged.
- One policy server per worker (the server-side executor is stateful); `multi_eval.sh` maps worker `i` → south port `SOUTH_PORT_BASE+i` (default 8848+i), matching `deploy.sh`'s `PORT_BASE+i`.
- The bridge waits up to 300 s for the policy server's first ping (checkpoint load; the server only binds its port once the model is on the GPU) — not a hang. With `compile.enabled` left on, the compile warm-up is paid on each server's first `obs` request and is bounded by `--request-timeout` (300 s); the paper runs used `--compile-enabled false`. `--no-send-state` only for non-proprio checkpoints.
- The bridge rebuilds its `EvalClient` and restarts the episode on any sim-side step failure; after `--max-reconnects` (20) failures in one run it exits non-zero instead of looping on a deterministic rejection.
- `RuntimeError: Timed out waiting for reset result after 3000s` at start-up means the Isaac worker behind that worker id never finished loading its scene (the GenManip server replaces such workers itself and requeues their episodes). Relaunch that one bridge with the same `--worker-id` / `--run-id` — the server hands a re-registered worker the next unclaimed episode while the other workers keep running — and if the same worker id times out again, leave it out (`NUM_WORKERS` minus one); the run still completes on the remaining workers. For long runs wrap each bridge in a restart loop.
- Behind a corporate proxy, unset `http_proxy` / `https_proxy` / `all_proxy` in the bridge shell (or put the sim host in `no_proxy`): the EvalClient's HTTP calls to the sim server honour them, while the WebSocket side to the policy server already bypasses proxies.
- Arms are absolute EE poses; GenManip solves IK server-side and silently holds joints on IK failure — validate in local sim before online submissions. The EE-pose path depends on GenManip's IK stabilisation fixes (commit `fbf7acb`, 2026-08-26, or later): older GenManip checkouts terminate many long-horizon episodes with `arm_state_jump_too_large`.
- Official online eval: `gmp online submit --base_url https://internrobotics.shlab.org.cn/eval --token $TOK --benchmark_set ebench_generalist --model_name ... --model_type WAM --submitter_name ... --submitter_homepage ... --is_public 0` → endpoint + `task_id`; then `single_eval.sh --url "$ENDPOINT" --token "$TOK" --run-id "$TASK_ID"`. ≤16 workers per run, 10-min inactivity disconnect (warm up the policy server first); failed runs resume with the same `task_id`. See the [Challenge guide](https://internrobotics.github.io/EBench-doc/challenge/).
- Budget: `test_mini` is 510 episodes with per-task step limits of 600–5000 sim steps (worst case ≈ 1.15 M steps, one policy call each); with 4 workers on 8× RTX 4090 the full split took about a day. The three public splits together (`val_train` 130 + `val_unseen` 154 + `test_mini` 510) are the "794 task instances" quoted by EBench.

</details>

## 4. Evaluation Protocol

What the numbers in section 5 were measured on:

| | |
|---|---|
| Benchmark | EBench **generalist** track, split **`test_mini`** (`gmp submit ebench/generalist/test_mini`) — the held-out split the public leaderboard uses. GenManip task configs v0.1.0, commit `fbf7acb`. |
| Tasks / episodes | 26 tasks × 20 episodes (15 for `make_sandwich` and `microwave`) = **510 episodes**, fixed per-seed initial layouts. Families: **TableTop** 7 (`teleop_tasks`: collect_coffee_beans, flip_cup_collect_cookies, frame_against_pen_holder, install_gear, peg_in_hole, put_glass_in_glassbox, tighten_nut) · **PnP** 10 (`simple_pnp`) · **LongHorizon** 9 (`long_horizon`: bottle, detergent, dish, dishwasher, fruit, make_sandwich, microwave, pen, shop). Step budgets 600–1000 (PnP), 3000–5000 (LongHorizon), 1500–3500 (TableTop). |
| Robot / sim | lift2 dual-arm mobile manipulator (R5a arms), Isaac Sim 4.1.0 at 30 Hz physics, one policy call per sim step (`EvalClient.step`, no `/step_chunk`). |
| Metrics | **SR** = episode success (goal condition met within the budget); **Score** = GenManip's partial-credit task score in [0, 1] (sub-goal progress). Family numbers are unweighted means over the family's tasks; **Overall** is the unweighted mean over all 26 tasks (the leaderboard averages over episodes instead, which differs only through the two 15-episode tasks). |
| Observations | `video.overlook_camera_view` → head slot, `video.left/right_camera_view` → wrist slots (480×640 RGB, resized to 320×256 / 160×128 and composed into the 384×320 L-shape the checkpoint was trained on); prompt = `instruction` wrapped by `prompt_template.py`; proprio = raw-23 `[L xyz, L rot6d, L gripper, R xyz, R rot6d, R gripper, base Δx, Δy, Δyaw°]` from `state.ee_pose` / `state.gripper` (mean of the two fingers) / `state.base` differenced against the previous step. |
| Actions | raw-23 in the same layout, sent as absolute `ee_pose` targets (`is_rel=False`; GenManip runs cuRobo IK per arm), gripper duplicated to both fingers and clipped to [0, 0.044] m, base as `base_motion=[dx_m, dy_m, dyaw_deg]` with `base_is_rel=True` (GenManip clips each step to ±0.015 m / ±1°). |
| Checkpoint | `OpenWAM-Alpha-Sim-EBench`: dual_system / joint_self_attn, mutual attention mask, Wan2.2-TI2V-5B backbone, fine-tuned for 100 k steps from the OpenWAM-Alpha foundation model on all 26 EBench-Dataset buckets (`configs/dataloader/ebench.yaml`: 33 frames, video stride 4, min-max normalisation, `action_mode: eef` scattered into the unified 80-D space). |
| Inference | `denoise_steps: 10`, sync denoising, sync executor with `inference_horizon: null` (32 actions per chunk, re-plan when the chunk is consumed), **`optimization.dit_cache.enabled=false`**, `--compile-enabled false`; 4 workers, one policy server each. |

## 5. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

| Method | Type | TableTop SR | TableTop Score | PnP SR | PnP Score | LongHorizon SR | LongHorizon Score | Overall SR | Overall Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StarVLA-OFT | VLA | - | - | - | - | - | - | 0.0 | 0.2 |
| π₀ | VLA | 15.7 | 30.0 | 35.0 | 39.0 | 17.0 | 41.0 | 23.6 | 37.0 |
| X-VLA | VLA | 8.6 | 24.0 | 50.0 | 54.0 | 6.2 | 25.0 | 23.7 | 36.0 |
| InternVLA-A1 | VLA | 4.3 | 11.0 | 43.0 | 47.0 | 17.9 | 46.0 | 23.9 | 36.0 |
| π₀.₅ | VLA | 12.9 | 32.0 | 45.0 | 50.0 | 18.1 | 39.0 | 27.1 | 41.0 |
| GigaBrain-0.7 | VLA | - | - | - | - | - | - | 33.3 | 46.0 |
| Qwen-RobotManip | VLA | **50.0** | **70.0** | <u>56.5</u> | <u>60.0</u> | <u>29.9</u> | <u>55.0</u> | <u>45.6</u> | <u>60.0</u> |
| Fast-WAM | WAM | - | - | - | - | - | - | 4.7 | 7.6 |
| **OpenWAM-α** | WAM | <u>30.0</u> | <u>44.2</u> | **67.5** | **72.0** | **44.3** | **72.6** | **49.4** | **64.7** |
