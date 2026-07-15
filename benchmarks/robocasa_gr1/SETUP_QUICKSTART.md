# RoboCasa GR1 Quick Setup

Use this page when moving the `robocasa-gr1-benchmark` branch to another
machine for render smoke or evaluation.

## 1. Sync This Branch

```bash
cd /path/to/openwam
git fetch origin
git switch robocasa-gr1-benchmark
git pull --ff-only
```

## 2. Install RoboCasa GR1 Env

Keep the official RoboCasa GR1 repo outside `openwam`:

```bash
mkdir -p /path/to/bench_deps
cd /path/to/bench_deps

conda create -c conda-forge -n robocasa-gr1 python=3.10 -y
conda activate robocasa-gr1

git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
cd robocasa-gr1-tabletop-tasks
pip install -e .

# RoboCasa asserts these versions at import time.
pip uninstall -y robosuite mujoco
pip install "robosuite==1.5.1" "mujoco==3.2.6"
```

## 3. Download Official Assets

```bash
cd /path/to/bench_deps/robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y
```

This downloads the simulator assets only. The demonstration dataset is optional
for benchmark smoke/eval.

Optional dataset download:

```bash
huggingface-cli download \
  --repo-type dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim \
  --local-dir /path/to/bench_deps/gr00t_teleop_sim
```

Dataset size:

- HDF5: about `14GB`
- LeRobot: about `39GB`
- Duration: about `81h`, `24k` trajectories at `20fps`

## 4. Export Paths

Run these from `openwam`:

```bash
export ROBOCASA_GR1_PATH=/path/to/bench_deps/robocasa-gr1-tabletop-tasks
export ROBOCASA_GR1_PYTHON=/path/to/miniconda/envs/robocasa-gr1/bin/python
export ROBOCASA_GR1_SMOKE_GPU=0
export ROBOCASA_GR1_RENDER_BACKEND=egl
```

If the conda env path is standard, `ROBOCASA_GR1_PYTHON` is usually:

```bash
export ROBOCASA_GR1_PYTHON="$CONDA_PREFIX/bin/python"
```

## 5. Run Smoke Tests

Basic checks:

```bash
bash benchmarks/robocasa_gr1/run_smoke.sh import
bash benchmarks/robocasa_gr1/run_smoke.sh task
```

No-render simulator reset/step:

```bash
ROBOCASA_GR1_ENABLE_RENDER=0 bash benchmarks/robocasa_gr1/run_smoke.sh env
```

Render-enabled visual smoke:

```bash
ROBOCASA_GR1_ENABLE_RENDER=1 bash benchmarks/robocasa_gr1/run_smoke.sh env
```

Success should include:

```text
env_smoke=ok ... obs_keys=...video.ego_view_pad_res256_freq20...
```

If render fails with:

```text
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

the machine does not expose a usable EGL runtime. Move to another GPU node or
fix NVIDIA EGL / GLVND before evaluating visual policies.

## 6. Run Evaluation

Start the OpenWAM server in the model environment first, for example:

```bash
bash scripts/deploy.sh --ckpt-dir /path/to/openwam_ckpt --port 8848
```

Then run the RoboCasa GR1 client in the RoboCasa env:

```bash
export ROBOCASA_GR1_GPU=0

bash benchmarks/robocasa_gr1/single_eval.sh \
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  8848 \
  127.0.0.1
```

## 7. Notes

- Default benchmark env:
  `gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env`
- Default render backend: `egl`
- Default max episode steps: `720`
- Default GR1 arms+waist Fourier-hands action dims:
  `left_hand=6`, `right_hand=6`, `left_arm=7`, `right_arm=7`, `waist=3`
- Do not open/merge the PR until render smoke passes on the target machine.
