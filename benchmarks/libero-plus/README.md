# LIBERO-plus Evaluation

Two processes: the **OpenWAM policy server** (this repo's env, holds the model) and the **LIBERO-plus client** (its own env). They talk over WebSocket ([wire protocol](../README.md)), so the two environments never interfere. LIBERO-plus is the perturbation-robustness suite over LIBERO (camera, robot state, language, light, background, noise, layout); for standard LIBERO see [`benchmarks/libero/`](../libero/README.md).

Commands below assume conda at `/path/to/miniconda3` and the LIBERO-plus checkout at `/path/to/LIBERO-plus` — substitute your actual paths.

## 1. Environment Setup

One command builds everything (clones LIBERO-plus at the pinned commit, applies the compatibility patch, creates the conda env, downloads + SHA-256-verifies the assets):

```bash
CONDA_BIN=/path/to/miniconda3/bin/conda \
LIBERO_PLUS_ENV_PREFIX=/path/to/miniconda3/envs/libero-plus \
LIBERO_PLUS_PATH=/path/to/LIBERO-plus \
bash benchmarks/libero-plus/setup_env.sh
```

Verify (no model needed):

```bash
LIBERO_PLUS_PATH=/path/to/LIBERO-plus LIBERO_PLUS_PYTHON=/path/to/miniconda3/envs/libero-plus/bin/python \
bash benchmarks/libero-plus/run_smoke.sh env    # also: import | task
```

<details>
<summary><b>What the script pins (details)</b></summary>

- https://github.com/sylvestf/LIBERO-plus at commit `4976dc30028e805ff8094b55501d532c48fec182`; setup aborts on any mismatch.
- Python 3.10, `mujoco==3.3.2`, full upstream pip pins; `git`, `curl`, `unzip`, `sha256sum` required on PATH.
- `patches/libero-plus-compatibility.patch` applied automatically (editable-install package marker, `torch.load` fix, motion-blur decode fix).
- Benchmark assets (`assets.zip`, several GB) fetched from the `Sylvest/LIBERO-plus` HF dataset with checksum verification.
- Clients auto-generate `~/.libero-openwam-plus/config.yaml` (relocate with `LIBERO_PLUS_CONFIG_ROOT`); EGL env vars are exported by the launch scripts.

</details>

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-LIBERO
```

LIBERO-plus evaluates the **same LIBERO checkpoint** under perturbations — there is no plus-specific release. Run from the repo root; it lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO` (or use a checkpoint you trained yourself). Needed only for `single_eval.sh`; the managed `run_eval.sh` starts its own servers:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO
```

WebSocket port 8848 by default (`--port` to change). Keep it running.

## 3. Run the Evaluation

Full run (primary command) — the managed launcher starts its own servers, spreads the suites over the GPUs, and tears everything down:

```bash
SERVER_PYTHON=/path/to/miniconda3/envs/openwam/bin/python \
LIBERO_PLUS_PYTHON=/path/to/miniconda3/envs/libero-plus/bin/python \
LIBERO_PLUS_PATH=/path/to/LIBERO-plus \
GPUS=0,1 REPLICAS_PER_GPU=1 \
bash benchmarks/libero-plus/run_eval.sh \
  assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO checkpoint_step_10000.safetensors
```

Append `--smoke` first for a one-task end-to-end validation. Single suite against the running server from section 2 (args: suite, task id, port, host):

```bash
LIBERO_PLUS_PATH=/path/to/LIBERO-plus LIBERO_PLUS_PYTHON=/path/to/miniconda3/envs/libero-plus/bin/python \
bash benchmarks/libero-plus/single_eval.sh libero_spatial 0 8848 127.0.0.1
```

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Always pass the checkpoint dir + filename explicitly (in-code defaults are placeholders); the dir must contain `config.yaml` and `normalization_stats.npy`, and the checkpoint must serve the `eef` representation. Substitute your actual `checkpoint_step_*.safetensors` name.
- The official LIBERO-plus one-rollout protocol is pinned and validated at launch: **1 trial per task**, seed 10000, `rng_mode: official_global` (seeds the global RNGs, not the environment), settle 30, `max_steps` 600/700; deviations abort.
- Managed replicas use ports from 8920; `single_eval.sh` defaults to `127.0.0.1:8848`.
- Resume: add `--output-dir outputs/libero-plus/<run_name>` and rerun the same command.
- `summary.csv` / `summary.json` include per-task difficulty levels and per-perturbation-category breakdowns (the seven categories from `task_classification.json`).

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA. Columns are the seven perturbation categories.

| Method | Type | Camera | Robot | Language | Light | Background | Noise | Layout | Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| π₀ | VLA | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| OpenVLA-OFT | VLA | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| StarVLA | VLA | 52.5 | 49.8 | 88.5 | 95.7 | 95.7 | 73.0 | 76.9 | 74.1 |
| ABot-M0 | VLA | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 |
| π₀.₅ | VLA | 78.4 | 73.6 | 80.8 | 96.2 | 94.1 | 89.0 | 84.5 | 84.4 |
| ACoT-VLA | VLA | 72.6 | <u>82.6</u> | 87.5 | 97.7 | <u>96.5</u> | 87.8 | <u>88.1</u> | <u>86.6</u> |
| Qwen-RobotManip | VLA | **87.2** | 75.5 | 85.6 | 96.6 | **97.7** | **97.7** | 87.3 | **89.0** |
| Fast-WAM | WAM | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| Being-H0.7 | WAM | <u>82.0</u> | 59.0 | 82.8 | <u>97.8</u> | 90.0 | 93.5 | **88.5** | 82.1 |
| Cosmos-Policy | WAM | 75.8 | 63.3 | 81.7 | 96.5 | 88.9 | 92.7 | 82.2 | 82.2 |
| ImageWAM | WAM | 80.8 | 50.3 | **91.4** | **98.1** | 85.5 | <u>93.8</u> | 80.5 | 83.1 |
| ABot-M0.5 | WAM | 70.5 | **87.4** | <u>88.6</u> | 94.0 | 89.7 | 75.5 | 85.2 | 83.4 |
| **OpenWAM-α** | WAM | 33.8 | 76.1 | 88.0 | 97.0 | 87.1 | 39.8 | 77.5 | 69.2 |
