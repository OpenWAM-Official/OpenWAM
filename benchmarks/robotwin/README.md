# RoboTwin 2.0 Evaluation

Two processes: the **OpenWAM policy server** (this repo's env, holds the model) and the **RoboTwin client** (its own env). They talk over WebSocket ([wire protocol](../README.md)), so the two environments never interfere.

Commands below assume conda at `/path/to/miniconda3` and the RoboTwin checkout at `/path/to/RoboTwin` — substitute your actual paths.

## 1. Environment Setup

Build the RoboTwin env separately from OpenWAM's (do not mix them):

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
# follow RoboTwin's official installation guide → conda env "robotwin" + task assets

/path/to/miniconda3/envs/robotwin/bin/pip install websockets pyyaml   # client deps OpenWAM needs
```

- `ROBOTWIN_PATH` → the checkout; `ROBOTWIN_PYTHON` → `/path/to/miniconda3/envs/robotwin/bin/python`. Every eval command needs both.
- Headless node? Prefix commands with `xvfb-run -a` (SAPIEN needs a display).

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-RoboTwin-Full
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full`. (`...-Clean2Random` is the OOD variant trained on clean data only; any checkpoint you trained yourself works the same way.) Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full
```

WebSocket port 8848 by default (`--port` to change). Keep it running.

## 3. Run the Evaluation

Single task — args are `task_name task_config ckpt_setting gpu_id [port] [host]`:

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/miniconda3/envs/robotwin/bin/python \
bash benchmarks/robotwin/single_eval.sh adjust_bottle demo_clean openwam 0
```

All 50 tasks (`-m` mode, `-n` run name, `-d` checkpoint dir; tasks = names, comma lists, `all`, or a file):

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/miniconda3/envs/robotwin/bin/python \
bash benchmarks/robotwin/multi_eval.sh -m demo_clean -n run1 \
  -d assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full all
```

`benchmarks/robotwin/policy_config.yml` must match the checkpoint: `action_type: ee` + `state_dim: 20` (end-effector, the released checkpoints) or `action_type: qpos` + `state_dim: 14` (joint-space).

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Start the server first; the client health-checks it for up to 300 s, then aborts. Remote server: pass `[host]`/`[port]` (single) or `--host`/`--port` (multi); defaults `127.0.0.1:8848`.
- `task_config` ∈ {`demo_clean`, `demo_randomized`}; `ckpt_setting` is only a label in RoboTwin's result filenames; `-d` is used for log placement — weights never load client-side.
- `ROBOTWIN_TEST_NUM=5` caps episodes per task for a quick smoke run (default 100/task; the full `all` run takes many GPU-hours per mode).
- Startup CUDA/cuRobo prewarm chatter from `eval_policy_wrapper.py` is expected; don't call RoboTwin's `eval_policy.py` directly. If cuRobo planning fails, set `ROBOTWIN_ENABLE_PLANNER_FALLBACK=1`.
- Read-only checkout? Set `ROBOTWIN_RUNTIME_ROOT` to a writable dir. Per-task step limits: `benchmarks/robotwin/step_limits.yml` (all commented out by default; read once at startup).
- Results: RoboTwin's native `eval_result/` inside the checkout (or runtime root); `multi_eval.sh` also tees per-task logs under `<ckpt_dir>/robotwin_eval_logs/…` and prints each task's `Success rate`.

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

**RoboTwin2.0-Clean2Random** (fine-tune on clean only; OOD probe):

| Method | Type | Clean | Randomized | Avg |
|---|---|---:|---:|---:|
| StarVLA | VLA | 46.5 | 3.2 | 24.9 |
| GR00T-N1.7 | VLA | 43.6 | 20.7 | 32.2 |
| X-VLA | VLA | 68.0 | 20.9 | 44.5 |
| Spatial Forcing | VLA | 77.2 | 26.7 | 52.0 |
| ABot-M0 | VLA | 70.7 | 36.0 | 53.4 |
| π₀.₅ | VLA | 70.7 | 46.0 | 58.4 |
| GigaBrain-0.7 | VLA | 66.8 | <u>67.9</u> | 67.4 |
| Qwen-RobotManip | VLA | <u>84.7</u> | **69.4** | **77.1** |
| AHA-WAM | WAM | 64.3 | 3.2 | 33.8 |
| Fast-WAM | WAM | 77.8 | 1.9 | 39.9 |
| X-WAM | WAM | 70.0 | 25.8 | 47.9 |
| 4D-WAM | WAM | 81.5 | 41.8 | 61.7 |
| **OpenWAM-α** | WAM | **89.4** | 48.7 | <u>69.0</u> |

**RoboTwin2.0-Full** (fine-tune on clean + randomized; ID probe):

| Method | Type | Clean | Randomized | Avg |
|---|---|---:|---:|---:|
| X-VLA | VLA | 72.80 | 72.84 | 72.82 |
| π₀.₅ | VLA | 82.70 | 76.80 | 79.75 |
| ABot-M0 | VLA | 86.06 | 85.08 | 85.57 |
| Qwen-VLA | VLA | 86.10 | 87.20 | 86.65 |
| StarVLA | VLA | 88.18 | 88.32 | 88.25 |
| Galaxea G0.5 | VLA | 93.70 | 92.80 | 93.25 |
| Qwen-RobotManip | VLA | 93.70 | <u>94.00</u> | <u>93.85</u> |
| Motus | WAM | 88.66 | 87.02 | 87.84 |
| Fast-WAM | WAM | 91.90 | 91.80 | 91.85 |
| LingBot-VA | WAM | 92.93 | 91.55 | 92.24 |
| ImageWAM | WAM | 93.20 | 93.56 | 93.38 |
| LingBot-VA 2.0 | WAM | <u>93.80</u> | 93.40 | 93.60 |
| ABot-M0.5 | WAM | **94.00** | **94.20** | **94.10** |
| **OpenWAM-α** | WAM | 93.74 | 93.46 | 93.60 |
