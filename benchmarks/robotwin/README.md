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

- `ROBOTWIN_PATH` → the checkout; `ROBOTWIN_PYTHON` → `/path/to/miniconda3/envs/robotwin/bin/python`. Every eval command needs both (`single_eval.sh` refuses to start without them).
- Headless nodes normally work as-is: SAPIEN renders offscreen through its bundled Vulkan loader, no X server needed. Only if you hit a display/EGL error at SAPIEN import, prefix the command with `xvfb-run -a`.

**Verified versions.** The adapter (`eval_policy_wrapper.py`) loads RoboTwin's `script/eval_policy.py` by path and patches it, so it is tied to the upstream layout. The combination below is what the released checkpoints were evaluated with; other versions may work but are unverified (the wrapper prints a warning when the RoboTwin commit differs, `ROBOTWIN_SKIP_VERSION_CHECK=1` silences it).

| Component | Version |
|---|---|
| RoboTwin | commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` (2026-04-19) |
| Python (RoboTwin env) | 3.10 |
| SAPIEN | 3.0.0b1 |
| cuRobo | v0.7.8, built into `<RoboTwin>/envs/curobo` by RoboTwin's `script/_install.sh` |
| warp-lang | 1.13.0 |
| mplib | 0.2.1 |
| torch (RoboTwin env) | 2.4.1 (the policy server uses OpenWAM's own env) |

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
- Don't call RoboTwin's `eval_policy.py` directly — go through `eval_policy_wrapper.py`. Its startup messages are expected; each one works around a known crash in the verified stack:
  - `prewarmed CUDA/Curobo before SAPIEN import` — cuRobo must initialise CUDA before SAPIEN's Vulkan context exists, otherwise the first plan segfaults.
  - `patched warp.torch compatibility namespace` — cuRobo v0.7.8 still calls `warp.torch.*`, which warp ≥ 1.x no longer ships; the wrapper aliases the new top-level functions.
  - `SAPIEN EGL ICD: …` (from `single_eval.sh`) — SAPIEN's EGL probe crashes on images without `/usr/share/glvnd/egl_vendor.d`; the script points it at SAPIEN's bundled ICD instead.
  - `RoboTwin commit=… eval_policy.py sha256=…` — provenance of the RoboTwin code actually loaded; a `WARNING` follows if the commit differs from the verified one or files under `script/ envs/ task_config/ policy/` are locally modified. Check out the verified commit before debugging anything else.
- If cuRobo planning itself fails, set `ROBOTWIN_ENABLE_PLANNER_FALLBACK=1` to plan with `mplib_RRT` instead (slower, results not comparable to the tables below).
- Read-only checkout? Set `ROBOTWIN_RUNTIME_ROOT` to a writable dir. Per-task step limits: `benchmarks/robotwin/step_limits.yml` (all commented out by default; read once at startup).
- Results: RoboTwin's native `eval_result/` inside the checkout (or runtime root); `multi_eval.sh` also tees per-task logs under `<ckpt_dir>/robotwin_eval_logs/…`, records the run there (`run.env` — shell-quoted, `source`-able — plus verbatim copies of the `policy_config.yml` and `step_limits.yml` used), and prints each task's `Success rate`.

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

Evaluation protocol for the OpenWAM-α rows (reproduce with the commands in §3): released checkpoint served with `scripts/deploy.sh` defaults from `configs/deploy.yaml` — `denoise_steps: 10`, `denoise_mode: sync`, DiT velocity cache **on** (`cosine_threshold: 0.99`, `max_skips: 3`), `torch.compile` on — 100 episodes per task at seed 0, RoboTwin's upstream per-task step limits (`step_limits.yml` empty), and the versions in §1. Changing any of these (in particular disabling the DiT cache) changes the numbers.

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
