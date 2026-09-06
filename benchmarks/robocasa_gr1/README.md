# RoboCasa GR1 Tabletop Evaluation

Two processes: the **OpenWAM policy server** (this repo's env, holds the model) and the **RoboCasa GR1 client** (its own env). They talk over WebSocket ([wire protocol](../README.md)), so the two environments never interfere.

Commands below assume conda at `/path/to/miniconda3` and the benchmark checkout at `/path/to/robocasa-gr1-tabletop-tasks` — substitute your actual paths.

## 1. Environment Setup

No setup script ships here — build the env by hand (keep the clone **outside** the OpenWAM checkout):

```bash
/path/to/miniconda3/bin/conda create -c conda-forge -n robocasa-gr1 python=3.10 -y
conda activate robocasa-gr1

git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks /path/to/robocasa-gr1-tabletop-tasks
cd /path/to/robocasa-gr1-tabletop-tasks
pip install -e .
pip uninstall -y robosuite mujoco && pip install robosuite==1.5.1 mujoco==3.2.6   # asserted at import
pip install pyyaml numpy Pillow websockets                                        # OpenWAM client deps
python robocasa/scripts/download_tabletop_assets.py -y                            # simulator assets
```

Verify, from the OpenWAM repo root:

```bash
export ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks
export ROBOCASA_GR1_PYTHON=/path/to/miniconda3/envs/robocasa-gr1/bin/python
ROBOCASA_GR1_ENABLE_RENDER=1 bash benchmarks/robocasa_gr1/run_smoke.sh env   # also: import | task
```

Success prints `env_smoke=ok` with `video.ego_view_pad_res256_freq20` among the observation keys.

<details>
<summary><b>Constraints & troubleshooting (details)</b></summary>

- `robosuite==1.5.1` + `mujoco==3.2.6` exactly — the official repo asserts them at import, hence the uninstall-first step.
- `ROBOCASA_GR1_PATH` and `ROBOCASA_GR1_PYTHON` are needed by every command below.
- Rendering needs a working NVIDIA EGL runtime. Failure signature: `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` → fix the EGL/GLVND driver install or change node. `ROBOCASA_GR1_ENABLE_RENDER=0` runs a render-free reset/step check.
- Simulator assets must be downloaded before any env smoke or eval.

</details>

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-RoboCasa-GR1
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1` (or use a checkpoint you trained yourself). Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboCasa-GR1
```

WebSocket port 8848 by default (`--port` to change). Keep it running.

## 3. Run the Evaluation

Args are `[env_id] [port] [host]`, all optional:

```bash
ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
ROBOCASA_GR1_PYTHON=/path/to/miniconda3/envs/robocasa-gr1/bin/python \
ROBOCASA_GR1_GPU=0 \
bash benchmarks/robocasa_gr1/single_eval.sh gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env 8848 127.0.0.1
```

For a full sweep, invoke it once per `gr1_unified/*` env id (24 official tasks, named `gr1_unified/<Task>_GR1ArmsAndWaistFourierHands_Env`); one server handles them sequentially.

<details>
<summary><b>Notes & troubleshooting</b></summary>

- The checkpoint must be the EEF33 GR1 one: the client sends 33-dim state and requires a 33-dim action back, otherwise it raises.
- `env_id` is validated against `gr1_unified/*`; host/port must match the running server.
- Other knobs (`num_episodes`, `max_steps` (720), `seed`, `fail_on_incomplete`, …): copy `benchmarks/robocasa_gr1/policy_config.yml`, edit, pass `POLICY_CONFIG_PATH=/path/to/custom.yml`.
- `ROBOCASA_GR1_GPU` feeds `CUDA_VISIBLE_DEVICES` + `MUJOCO_EGL_DEVICE_ID`.
- Results print to stdout: per-episode `[RESULT]` lines and `Success rate: n/N => x%`. Exit code stays 0 below 100% unless `fail_on_incomplete: true`.

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

| Method | Type | SR (%) |
|---|---|---:|
| π₀ | VLA | 13.6 |
| π₀.₅ | VLA | 37.0 |
| GR00T-N1.5 | VLA | 48.0 |
| StarVLA | VLA | 48.8 |
| GR00T-N1.6 | VLA | 49.9 |
| VP-VLA | VLA | 53.8 |
| Being-H0.5 | VLA | 53.9 |
| Qwen-VLA-Instruct | VLA | 56.7 |
| RLDX-1 | VLA | 58.7 |
| PhysBrain 1.0 | VLA | **64.5** |
| UWM | WAM | 20.0 |
| Being-H0.7 | WAM | 49.2 |
| DiT4DiT | WAM | 50.8 |
| LDA-1B | WAM | 55.4 |
| **OpenWAM-α** | WAM | <u>60.5</u> |
