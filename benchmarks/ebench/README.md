# EBench (GenManip) Evaluation

The bridge connects [GenManip](https://github.com/InternRobotics/GenManip)'s `EvalClient` to one stateful OpenWAM policy server. It sends observations over HTTP and receives policy actions over WebSocket; the GenManip server may run on another machine.

Commands below assume the bridge env's python at `/path/to/bridge-env/bin/python` — substitute your actual paths.

## 1. Environment Setup

**Sim server** (possibly another machine): follow the [EBench environment guide](https://internrobotics.github.io/EBench-doc/getting-started/environment/) for Isaac Sim 4.1.0 (CUDA 12.1), cuRobo, and the `EBench-Assets` dataset (~34 GB). Isaac Sim 4.1.0 does not support Blackwell GPUs.

**Bridge env** (what OpenWAM's launch scripts run — no torch, no OpenWAM install):

```bash
git clone --recursive https://github.com/InternRobotics/EBench && cd EBench
pip install -e third_party/genmanip-client
pip install numpy Pillow "websockets>=15" PyYAML opencv-python-headless "PyTurboJPEG<2" filelock
```

Point `EBENCH_PYTHON` at this env's python. On Ubuntu ≤ 22.04, keep the `PyTurboJPEG<2` pin unless libjpeg-turbo 3.x is installed. For a no-Isaac sanity check see section 3.

> Hugging Face unreachable? Export `HF_ENDPOINT=https://hf-mirror.com` before running any downloader in this guide.

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
```

Select `OpenWAM_Alpha → OpenWAM-Alpha-Sim-EBench` in the menu. Run from the repo root; the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench` (or use a checkpoint you trained yourself). Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench
```

The default WebSocket port is 8848; wait for `WebSocket server started`. For N workers, `NUM_GPUS=N bash scripts/deploy.sh <ckpt_dir>` starts one server per GPU on ports 8848…8848+N-1.

For the reported results, use `--compile-enabled false optimization.dit_cache.enabled=false`; other settings use `configs/deploy.yaml` defaults.

## 3. Run the Evaluation

On the sim machine (GenManip), start `python ray_eval_server.py --host 0.0.0.0 --port 8087 --no_save_process` and submit a split, for example `gmp submit ebench/generalist/test_mini --run_id <run_id>`. Then run one worker from this repo:

```bash
EBENCH_PYTHON=/path/to/bridge-env/bin/python \
bash benchmarks/ebench/single_eval.sh \
    --url http://<sim-host>:8087 --run-id <run_id> \
    --ckpt-config assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench/config.yaml
```

All bridge flags can also come from a YAML: `--config benchmarks/ebench/policy_config.yml` (explicit CLI flags win).

Parallel run — one policy server per worker:

```bash
NUM_GPUS=4 bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench
NUM_WORKERS=4 EBENCH_PYTHON=/path/to/bridge-env/bin/python \
bash benchmarks/ebench/multi_eval.sh \
    --url http://<sim-host>:8087 --run-id <run_id> \
    --ckpt-config assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench/config.yaml
```

GenManip writes scores under `saved/eval_results/<task>/<run_id>/`; `gmp status` shows the resolved path. The bridge mirrors per-episode `episode_result.json` files into `client_results/` (override with `GENMANIP_RESULT_DIR`).

Offline sanity check without Isaac Sim: the mock replays one EBench-Dataset bucket over the EvalClient wire format. Run it in the **training env** (`pandas`, `pyarrow`, and `av` required) with one `single_eval.sh` worker:

```bash
huggingface-cli download InternRobotics/EBench-Dataset --repo-type dataset \
    --local-dir assets/benchmark_data/ebench --include 'simple_pnp/task1/*'
python benchmarks/ebench/mock_genmanip_server.py \
    --dataset-dir assets/benchmark_data/ebench --bucket simple_pnp/task1 \
    --episodes 2 --steps-per-episode 8 --port 8087
```

In another shell, run the single-worker command above with `--url http://127.0.0.1:8087` and omit `--run-id`.

A successful mock run logs `run complete: 2 episodes, 16 steps bridged` and `16 actions, 0 violations`. Contract violations return HTTP 500 for later reset, reset-result, and step requests, and the bridge exits non-zero. Accepted actions are written to `./ebench_mock_actions.jsonl` (`--log-file`).

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Pass `--ckpt-config` when available; it verifies `dataloader.type=ebench`, `action_mode=eef`, and `unify_action=true`.
- Use one policy server per worker. The bridge may wait up to 300 s for startup; `--compile-enabled false` avoids compile warm-up during evaluation.
- A failed sim step resets the episode and retries up to `--max-reconnects` (20), then exits non-zero.
- If reset times out, restart the affected worker with the same `--worker-id` and `--run-id`; reduce `NUM_WORKERS` if it continues to fail.
- Arms are absolute EE poses and GenManip performs IK server-side; validate the checkpoint in local simulation first.
- Online runs allow at most 16 workers and disconnect after 10 minutes of inactivity; failed runs can resume with the same `task_id`. See the [Challenge guide](https://internrobotics.github.io/EBench-doc/challenge/).
- `test_mini` contains 510 episodes; use parallel workers for long runs.

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
