# RoboCasa365 Evaluation

Two processes: the **OpenWAM policy server** (this repo's env, holds the model) and the **RoboCasa365 client** (its own env). They talk over WebSocket ([wire protocol](../README.md)), so the two environments never interfere.

Commands below assume conda at `/path/to/miniconda3` and the external checkouts at `/path/to/robocasa`, `/path/to/robosuite` — substitute your actual paths.

## 1. Environment Setup

One command builds everything (clones + pins [robocasa](https://github.com/robocasa/robocasa) and [robosuite](https://github.com/ARISE-Initiative/robosuite), creates the `robocasa365` conda env, downloads the ~10 GB kitchen assets, asserts every version):

```bash
CONDA_BIN=/path/to/miniconda3/bin/conda \
ROBOCASA365_ENV_PREFIX=/path/to/miniconda3/envs/robocasa365 \
ROBOCASA365_PATH=/path/to/robocasa \
ROBOSUITE_PATH=/path/to/robosuite \
bash benchmarks/robocasa365/setup_env.sh
```

Verify (`roundtrip` also needs the server from section 2):

```bash
ROBOCASA365_PYTHON=/path/to/miniconda3/envs/robocasa365/bin/python \
bash benchmarks/robocasa365/run_smoke.sh env OpenDrawer    # also: import | roundtrip
```

<details>
<summary><b>What the script pins (details)</b></summary>

- robocasa `a07e365c958c4216cd6bbd5f30b47f09a65c6f00` (v1.0.1, includes the official 1.5× horizon update); robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. Setup errors on any mismatch.
- Python 3.11, `mujoco==3.3.1`, `numpy==2.2.5`, `websockets==15.0.1`; asserts a registered `robocasa/*` gym env.
- `ROBOCASA365_DOWNLOAD_ASSETS=0` skips the asset download (run robocasa's downloader later).
- EGL env vars are exported by the launch wrappers as overridable defaults.

</details>

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-RoboCasa365
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa365` (or use a checkpoint you trained yourself). Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa365
```

WebSocket port 8848 by default (`--port` to change). Keep it running.

## 3. Run the Evaluation

**Split and trials.**

- `pretrain` (default) — the leaderboard / multi-task protocol: the 50 target tasks
  evaluated in training-distribution kitchens (paper §4.1). All §4 numbers use this.
- `target` — the foundation track: evaluate in the 10 held-out kitchens after
  finetuning on target demos (§4.2). Pass it explicitly.
- In `multi_eval.sh target`, that `target` is the task-list token, not the split.
- The commands below run `num_trials: 5` per task (smoke scale, from
  `policy_config.yml`); benchmark numbers use 50 rollouts/task
  ([protocol](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html)) —
  copy `policy_config.yml`, set `num_trials: 50`, and point `ROBOCASA365_POLICY_CONFIG` at it.

Single task (args: task, split, port, host):

```bash
ROBOCASA365_PYTHON=/path/to/miniconda3/envs/robocasa365/bin/python \
bash benchmarks/robocasa365/single_eval.sh OpenDrawer pretrain 8848 127.0.0.1
```

Official 50-task target list (18 atomic + 32 composite) with a CSV summary:

```bash
ROBOCASA365_PYTHON=/path/to/miniconda3/envs/robocasa365/bin/python \
bash benchmarks/robocasa365/multi_eval.sh --out ./results_robocasa365 target
```

Or fully managed (starts the server, runs the tasks, stops the server; requires `setsid`):

```bash
SERVER_PYTHON=/path/to/miniconda3/envs/openwam/bin/python \
ROBOCASA365_PYTHON=/path/to/miniconda3/envs/robocasa365/bin/python \
bash benchmarks/robocasa365/run_eval.sh \
  assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa365 checkpoint_step_10000.safetensors target
```

<details>
<summary><b>Notes & troubleshooting</b></summary>

- The server's ping must advertise representation `robocasa365` — the client hard-fails otherwise. The released checkpoint matches; your own must be trained on the robocasa365 conversion.
- Rollout horizons come from robocasa's official task registry at runtime; leave `max_steps_override: null` in `policy_config.yml` for benchmark runs.
- Client defaults live in `benchmarks/robocasa365/policy_config.yml` (5 trials/task — smoke scale, see *Split and trials* above — and `pretrain` split). Splits: `pretrain` = training-distribution kitchens (layouts/styles 11-60), the leaderboard/multi-task protocol; `target` = the 10 held-out kitchens of the foundation-model (finetune) track; `all` = everything; pass the split as arg 2 / `--split` / `SPLIT=` to switch. CLI flags and `ROBOCASA365_PORT` / `ROBOCASA365_POLICY_HOST` override; `ROBOCASA365_POLICY_CONFIG` points at a custom file.
- `multi_eval.sh` tasks: literal names, `target`/`all` (expands `target_tasks.txt`), or a file; per-task success rates aggregate into `<out>/summary_<split>.csv`.
- `run_eval.sh` requires an explicit checkpoint filename (substitute your actual `checkpoint_step_*.safetensors`) and writes `server.log` / `client.log` / `tasks/summary_<split>.csv` under `outputs/robocasa365/<timestamp>` (override with `OUTPUT_DIR`); `scripts/deploy.sh` alone may omit `--ckpt-name` (picks the latest).

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

| Method | Type | Atomic | Comp.-Seen | Comp.-Unseen | Avg |
|---|---|---:|---:|---:|---:|
| Diffusion Policy | VLA | 15.7 | 0.2 | 1.3 | 6.1 |
| π₀ | VLA | 36.3 | 5.2 | 0.7 | 15.0 |
| π₀.₅ | VLA | 39.6 | 7.1 | 1.2 | 16.9 |
| GR00T-N1.5 | VLA | 50.7 | 14.8 | 2.7 | 23.9 |
| Qwen-RobotManip | VLA | 68.6 | 20.1 | <u>14.9</u> | 35.9 |
| RLDX-1 | VLA | 67.6 | 27.9 | 8.5 | 36.0 |
| Xiaomi-Robotics-1 | VLA | **80.2** | **57.1** | **32.1** | **57.4** |
| GigaWorld-Policy | WAM | 44.4 | 11.8 | 2.9 | 20.7 |
| ABot-M0.5 | WAM | <u>75.9</u> | <u>38.3</u> | 2.7 | <u>40.4</u> |
| **OpenWAM-α** | WAM | 69.7 | 32.1 | 8.9 | 38.2 |
