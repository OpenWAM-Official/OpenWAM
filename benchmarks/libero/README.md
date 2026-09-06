# LIBERO Evaluation

Two processes: the **OpenWAM policy server** (this repo's env, holds the model) and the **LIBERO client** (its own env). They talk over WebSocket ([wire protocol](../README.md)), so the two environments never interfere. For the LIBERO-plus perturbation suite see [`benchmarks/libero-plus/`](../libero-plus/README.md).

Commands below assume conda at `/path/to/miniconda3` and the LIBERO checkout at `/path/to/LIBERO` — substitute your actual paths.

## 1. Environment Setup

One command builds everything (clones LIBERO at the pinned commit, applies the PyTorch≥2.6 patch, creates the conda env, verifies every version):

```bash
CONDA_BIN=/path/to/miniconda3/bin/conda \
LIBERO_ENV_PREFIX=/path/to/miniconda3/envs/libero \
LIBERO_PATH=/path/to/LIBERO \
bash benchmarks/libero/setup_env.sh
```

Verify (no model needed):

```bash
LIBERO_PATH=/path/to/LIBERO LIBERO_PYTHON=/path/to/miniconda3/envs/libero/bin/python \
bash benchmarks/libero/run_smoke.sh env    # also: import | task
```

<details>
<summary><b>What the script pins (details)</b></summary>

- LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`; setup aborts on any mismatch.
- Python 3.10, `mujoco==3.3.2` (re-checked at every launch), full upstream pip pins.
- `patches/libero-pytorch-load.patch` applied automatically (`torch.load(..., weights_only=False)`).
- No separate asset download — assets ship inside the clone. Clients auto-generate `~/.libero-openwam/config.yaml` (relocate with `LIBERO_CONFIG_ROOT`).
- EGL rendering env vars are exported by the launch scripts; no manual setup.

</details>

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-LIBERO
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO` (or use a checkpoint you trained yourself). Needed only for `single_eval.sh`; the managed `run_eval.sh` below starts its own servers:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO
```

WebSocket port 8848 by default (`--port` to change). Keep it running.

## 3. Run the Evaluation

Full run (primary command) — the managed launcher starts its own servers, spreads all four suites over the GPUs, and tears everything down:

```bash
SERVER_PYTHON=/path/to/miniconda3/envs/openwam/bin/python \
LIBERO_PYTHON=/path/to/miniconda3/envs/libero/bin/python \
LIBERO_PATH=/path/to/LIBERO \
GPUS=0,1 REPLICAS_PER_GPU=1 \
bash benchmarks/libero/run_eval.sh \
  assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO checkpoint_step_10000.safetensors
```

Append `--smoke` first for a one-task end-to-end validation. Single suite against the running server from section 2 (args: suite, task id, port, host):

```bash
LIBERO_PATH=/path/to/LIBERO LIBERO_PYTHON=/path/to/miniconda3/envs/libero/bin/python \
bash benchmarks/libero/single_eval.sh libero_spatial 0 8848 127.0.0.1
```

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Always pass the checkpoint dir + filename explicitly (in-code defaults are placeholders); the dir must contain `config.yaml` and `normalization_stats.npy`, and the checkpoint must serve the `eef` representation — the client pings and refuses mismatches. Substitute your actual `checkpoint_step_*.safetensors` name.
- Protocol is pinned and validated at launch: 50 trials/task, seed 42, `max_steps` 600 (700 for `libero_10`); deviations abort.
- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` (aliases `spatial|object|goal|long`). Managed replicas use ports from 8920.
- Resume: add `--output-dir outputs/libero/<run_name>` and rerun the same command — completed tasks are skipped after a signature check.
- Results land in the output dir: `summary.csv`, `summary.json`, `manifest.json`, per-task `results.json`, client/server logs.

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

| Method | Type | Spatial | Object | Goal | Long | Avg |
|---|---|---:|---:|---:|---:|---:|
| OpenVLA | VLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | VLA | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| StarVLA | VLA | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| π₀.₅ | VLA | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.6 | VLA | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| OpenVLA-OFT | VLA | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| X-VLA | VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| ABot-M0 | VLA | 98.8 | <u>99.8</u> | 99.0 | 96.6 | 98.6 |
| Being-H0.5 | VLA | 99.2 | 99.6 | <u>99.4</u> | 97.4 | 98.9 |
| Qwen-RobotManip | VLA | - | - | - | - | 99.2 |
| Fast-WAM | WAM | 98.2 | **100.0** | 97.0 | 95.2 | 97.6 |
| Motus | WAM | 96.8 | <u>99.8</u> | 96.6 | 97.6 | 97.7 |
| ImageWAM | WAM | 97.2 | 99.2 | 98.8 | <u>98.4</u> | 98.4 |
| LingBot-VA | WAM | 98.5 | 99.6 | 97.2 | **98.5** | 98.5 |
| DiT4DiT | WAM | - | - | - | - | 98.6 |
| Being-H0.7 | WAM | - | - | - | - | 99.2 |
| ABot-M0.5 | WAM | **100.0** | <u>99.8</u> | <u>99.4</u> | <u>98.4</u> | **99.4** |
| **OpenWAM-α** | WAM | <u>99.6</u> | 99.6 | **99.8** | 98.2 | <u>99.3</u> |
