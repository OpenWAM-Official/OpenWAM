# RoboCasa GR1 Rendering Handoff

This note captures the render-specific setup needed to finish RoboCasa GR1
visual smoke tests on a machine with working MuJoCo offscreen rendering.

## Current Status

Validated on the local H-card node:

- `import` smoke: passed
- `task` smoke: passed
- no-render `env` reset/step smoke: passed
- render-enabled `env` smoke: blocked by local EGL runtime

The render failure is:

```text
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

This happens before RoboCasa can return camera observations and usually means
the node does not expose a usable NVIDIA EGL / GLVND runtime. It is not a
RoboCasa benchmark adapter error.

## Environment Setup

Use the official RoboCasa GR1 repository and assets outside this repo:

```bash
conda create -c conda-forge -n robocasa-gr1 python=3.10 -y
conda activate robocasa-gr1

git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
cd robocasa-gr1-tabletop-tasks
pip install -e .

# The official repo asserts these versions at import time.
pip install "robosuite==1.5.1" "mujoco==3.2.6"

python robocasa/scripts/download_tabletop_assets.py -y
```

If a local clone of `robosuite` is installed editable from `main`, remove it
before the pin:

```bash
pip uninstall -y robosuite
pip install "robosuite==1.5.1" "mujoco==3.2.6"
```

## Render Smoke On The New Machine

From the OpenWAM repo checkout:

```bash
export ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks
export ROBOCASA_GR1_PYTHON=/path/to/miniconda/envs/robocasa-gr1/bin/python
export ROBOCASA_GR1_SMOKE_GPU=0
export ROBOCASA_GR1_RENDER_BACKEND=egl

bash benchmarks/robocasa_gr1/run_smoke.sh import
bash benchmarks/robocasa_gr1/run_smoke.sh task
ROBOCASA_GR1_ENABLE_RENDER=0 bash benchmarks/robocasa_gr1/run_smoke.sh env
ROBOCASA_GR1_ENABLE_RENDER=1 bash benchmarks/robocasa_gr1/run_smoke.sh env
```

Expected final render smoke output includes:

```text
env_smoke=ok ... obs_keys=...video.ego_view_pad_res256_freq20...
```

The default GR1 arms+waist Fourier-hands env reports these action dimensions:

```text
action.left_hand=6
action.right_hand=6
action.left_arm=7
action.right_arm=7
action.waist=3
```

## Useful Render Variables

- `ROBOCASA_GR1_RENDER_BACKEND`: copied into `MUJOCO_GL` when `MUJOCO_GL` is
  not already set. Default: `egl`.
- `ROBOCASA_GR1_SMOKE_GPU`: smoke-test GPU index. Default: `0`.
- `ROBOCASA_GR1_GPU`: evaluation GPU index. Default: `0`.
- `MUJOCO_EGL_DEVICE_ID`: EGL render device. Defaults to the corresponding
  smoke/eval GPU variable.
- `CUDA_VISIBLE_DEVICES`: defaults to the corresponding smoke/eval GPU
  variable, but can be set explicitly.
- `PYOPENGL_PLATFORM`: set to `egl` automatically when `MUJOCO_GL=egl`.

Example for GPU 1:

```bash
ROBOCASA_GR1_SMOKE_GPU=1 \
MUJOCO_EGL_DEVICE_ID=1 \
CUDA_VISIBLE_DEVICES=1 \
ROBOCASA_GR1_ENABLE_RENDER=1 \
bash benchmarks/robocasa_gr1/run_smoke.sh env
```

## Minimal EGL Checks

Run these before the render smoke if the new machine is suspicious:

```bash
nvidia-smi
python - <<'PY'
import mujoco
print("mujoco", mujoco.__version__)
from OpenGL import EGL
print("egl", EGL)
PY
```

If importing `OpenGL.EGL` raises `eglQueryString` / `NoneType`, the node still
lacks a usable EGL runtime. Move to a different GPU node or fix the driver /
GLVND installation before rerunning RoboCasa render smoke.

## Evaluation After Render Smoke

Start OpenWAM server in the model environment, then run:

```bash
export ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks
export ROBOCASA_GR1_PYTHON=/path/to/miniconda/envs/robocasa-gr1/bin/python
export ROBOCASA_GR1_GPU=0

bash benchmarks/robocasa_gr1/single_eval.sh \
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  8848 \
  127.0.0.1
```

Do not open the PR until render smoke passes on the target machine.
