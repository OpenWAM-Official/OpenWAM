# VLABench Benchmark Evaluation

Closed-loop evaluation on [VLABench](https://github.com/OpenMOSS/VLABench)
(ICCV 2025) against a running OpenWAM policy server.

VLABench is a MuJoCo/dm_control benchmark for language-conditioned manipulation
on a Franka Panda. This client drives its `Evaluator` and speaks the OpenWAM
WebSocket protocol; the model, checkpoint and action denormalization all stay
server-side.

**Nothing in the VLABench repo needs to be modified.** The adapter duck-types
VLABench's `Policy` interface (`name` / `control_mode` / `reset` / `predict`)
and imports nothing from it, so any checkout works.

## Files

| File | Description |
|---|---|
| `openwam2vlabench_interface.py` | `OpenWAMVLABenchPolicy` — WebSocket adapter from VLABench observations to OpenWAM server payloads. |
| `single_eval.py` | Run one task (or one whole track) against an OpenWAM policy server; drives VLABench's `Evaluator`. |
| `single_eval.sh` | Shell wrapper; patches host/port/task/track/episodes at runtime. |
| `multi_eval.sh` | Fan tasks x tracks across GPUs, then merge per-job results. |
| `smoke_vlabench.py` | Preflight: `env` / `loop` checks. No GPU, checkpoint or server needed. |
| `run_smoke.sh` | Smoke launcher. |
| `policy_config.yml` | Eval client config template. |

Conversions live in [`benchmarks/utils/action_conversion.py`](../utils/action_conversion.py)
(`vlabench_obs_to_eef10`, `eef10_to_vlabench_ee`, `rot6d_to_euler_xyz`) and are
covered by [`tests/benchmarks/test_vlabench_bridge.py`](../../tests/benchmarks/test_vlabench_bridge.py)
— pure numpy, no simulator needed.

## Setup

VLABench pins `numpy==1.25.0`, `mujoco==3.2.2`, `dm_control==1.0.22`, so it needs
its own environment, separate from OpenWAM's:

```bash
conda create -n vlabench python=3.10 && conda activate vlabench
git clone https://github.com/OpenMOSS/VLABench.git && cd VLABench
pip install -r requirements.txt && pip install -e .
python scripts/download_assets.py        # ~17 GB
```

The client needs only `numpy`, `Pillow` and `websockets` on top of that.

## Smoke Tests

Preflight the VLABench install and the adapter's observation contract without a
GPU, a checkpoint or a running server:

```bash
VLABENCH_PATH=/path/to/VLABench \
VLABENCH_PYTHON=/path/to/envs/vlabench/bin/python \
  bash benchmarks/vlabench/run_smoke.sh env select_fruit
```

| Mode | Checks |
|---|---|
| `env` (default) | Loads one task and asserts the observation contract the adapter depends on: camera count / order, `ee_state`, robot base frame, instruction. |
| `loop` | `env` plus a full closed loop against an in-process mock OpenWAM server that echoes proprio back as the action (hold-still policy). Pass a `host:port` as argument 4 to drive a real server instead — the cheapest preflight before committing GPUs to a track sweep. |

Smoke knobs:

| Variable | Default | Description |
|---|---:|---|
| `VLABENCH_SMOKE_MODE` | `env` | Mode, when not passed as argument 1. |
| `VLABENCH_SMOKE_TASK` | `select_fruit` | Task to instantiate, when not passed as argument 2. |
| argument 3 | `8` | Max steps for the `loop` mode rollout. |
| `VLABENCH_SMOKE_SERVER` | (unset) | Argument 4. `loop` mode only: `host:port` of a REAL server to drive instead of the mock — exercises the wire action width, server-side denormalization and latency. |

## Run

Start a server, then point the client at it:

```bash
# 1. server (OpenWAM env)
bash scripts/deploy.sh /path/to/vlabench_finetuned_ckpt --device cuda:0 --port 8848

# 2. eval (VLABench env)
VLABENCH_PATH=/path/to/VLABench \
VLABENCH_PYTHON=/path/to/envs/vlabench/bin/python \
  bash benchmarks/vlabench/single_eval.sh select_fruit track_1_in_distribution 50 8848
```

`all` in place of the task name runs every task in the track. For a full sweep:

```bash
VLABENCH_PATH=... VLABENCH_PYTHON=... \
  bash benchmarks/vlabench/multi_eval.sh all all 50 8848
```

A `single_eval.sh` run writes `<save_dir>/<track>/` — `metrics.json`,
per-task `detail_info.json`, and `openwam/evaluation_result.json`.

`multi_eval.sh` gives every job its own `_eval_out/by_task/<task>/<track>/`
and merges them into `_eval_out/combined_results.json` at the end, printing a
per-track mean. The isolation is required, not cosmetic: VLABench's `Evaluator`
updates `metrics.json` with a read-modify-write, so concurrent jobs sharing a
directory would silently drop each other's results. Per-job logs are under
`_eval_out/logs/<track>__<task>.log`.

> Budget the time: VLABench renders **every** camera's RGB + depth + segmentation
> each step and solves IK per step. Upstream quotes 30–60 min per task in a
> single process; 10 tasks x 5 tracks needs the parallel runner.

## Evaluation tracks

Pass the track as the second argument, or set `eval_track` in `policy_config.yml`.

| Track | Measures | Episode source |
|---|---|---|
| `track_1_in_distribution` | Task learning on in-domain episodes | frozen config |
| `track_2_cross_category` | Object category / instance generalization | frozen config |
| `track_3_common_sense` | Common-sense target description | frozen config |
| `track_4_semantic_instruction` | Semantically rich instructions | frozen config |
| `track_5_cross_task` | Skill transfer to held-out tasks | **seeds — see below** |
| `track_6_unseen_texture` | Unseen backgrounds and table textures | frozen config |

Tracks pin the episode set so runs compare across machines. Setting
`eval_track: null` falls back to seed-sampled episodes, which upstream advises
against — *"there is a risk of improperly initialized episodes. We recommend
using the 'evaluation_tracks' method."*

### track_5 is the open one

VLABench's README lists six dimensions but `configs/evaluation/tracks/` ships
only five JSONs. Track 5 has none by design: the train/eval task split is the
user's to choose (*"kept open in this setting, allowing users to choose training
tasks and evaluation tasks according to their needs"*), so there is nothing
upstream could freeze — and consequently no published baseline for it.

It therefore runs on seeded episodes (`seed=42+i`, reproducible for a given
VLABench commit) over a task list you supply. `multi_eval.sh` carries a default
split for a policy finetuned on the standard 10-task primitive dataset: each
trained family's `*_spatial` sibling (same objects and motor skill, an
instruction family never seen in training) plus `select_billiards` and
`select_ingredient` (unseen object categories). Override with
`VLABENCH_TRACK5_TASKS=a,b,c`.

`insert_bloom_flower`, `replace_wilted_flower`, `select_painting_by_style`,
`select_billiards_semantic` and `select_drink_spatial` are excluded because they
fail to instantiate in VLABench `main` (PhysicsError / `KeyError: 'task'` /
ConfigManager signature mismatch), not by choice.

Published Track 1 success rates for reference: Pi0-ft 47%, Pi-fast-ft
(delta chunk) 51.2%, Pi05-ft 40.6%, Pi-fast-ft (relative chunk) 29.1%.

## Contract

Checkpoints come from [`configs/dataloader/vlabench.yaml`](../../configs/dataloader/vlabench.yaml)
(`unify_action_map: ["0-9"]`), so the wire carries raw **EEF10**
`[xyz3, rot6d6, grip1]` in both directions — the server scatters the proprio
into the unified 80-D space and gathers the action back out.

### Cameras

`obs["rgb"]` is `(4, 480, 480, 3)`. Indices verified against the live MuJoCo
model (names read via `mj_id2name`) and VLABench's LeRobot converter:

| Index | MuJoCo camera | Dataset key | OpenWAM slot |
|---|---|---|---|
| 0 | `right` | `second_image` | `right_wrist_camera` |
| 1 | `left` | — | unused |
| 2 | `forward` | `image` | `head_camera` |
| 3 | `franka/Franka_wrist_cam` | `wrist_image` | `left_wrist_camera` |

### Frames

Training data is in the **robot base** frame — VLABench's converter subtracts
the base position from the recorded world pose. VLABench's evaluator runs IK on
an absolute **world** target. So the client subtracts on the way in and adds
back on the way out, reading the live base from `obs["robot_frame"]` rather
than assuming the converter's `[0, -0.4, 0.78]` fallback.

### Gripper: the polarity trap

State and action use **opposite** gripper polarity in the VLABench dataset
(measured correlation across the corpus: **-0.93**):

- **State `1 = closed`.** `ee_state[7]` comes from `robot.get_ee_open_state()`,
  which for the Franka returns `True` when the fingers are *closed* — an
  acknowledged upstream bug
  ([`franka.py:57`](https://github.com/OpenMOSS/VLABench/blob/main/VLABench/robots/single_arm/franka.py)
  carries the comment `# BUG: should be False`; the WidowX implementation has
  the correct polarity).
- **Action `1 = open`.** The converter binarizes the commanded finger width with
  `> 0.03` against a `0.04 m` open span.

The client forwards the proprio **verbatim** and thresholds the action with
`>= 0.5 -> open`. Do not "fix" the state side: the eval env reads the same buggy
accessor the training data was recorded through, so the inversion cancels, and
correcting it here would take the policy off-distribution. Both
`gripper_open_threshold` and `gripper_open_width` are config fields if a future
upstream fix changes this.

### Chunking

None client-side. The server buffers action chunks internally — one obs in, one
action out. (The upstream `openpi` and `lingbot_va` adapters keep their own
`action_plan` deque; this one does not need to.)

## Notes

- `WARNING:absl:Failed to converge after 99 steps` during rollout is expected,
  not a bug: it is VLABench's IK giving up when the model commands a pose
  outside the arm's working envelope. An undertrained checkpoint will produce a
  lot of it.
- `single_eval.py` runtime-patches `LM4ManipDMEnv.get_intention_score` to return
  NaN on `KeyError` (`patch_intention_score_keyerror`). Without it, tasks whose
  target is resolved positionally lose *entire episodes*: the evaluator calls
  `get_intention_score()` after the rollout but before recording
  `info["success"]`, `reset_intention_distance` never keyed the positionally-
  resolved target (`tasks/dm_task.py:353`), and the per-episode `except` in
  `Evaluator.evaluate` then discards a finished, possibly successful rollout.
  This wiped all 50 episodes of `select_poker_spatial` on the first attempt.
  Requesting only `success_rate` does not avoid it — the call is unconditional.
  NaN rather than 0.0 keeps the loss confined to `intention_score`, since
  `compute_metric` averages each metric independently.
- A small number of episodes still die to `PhysicsError: mjWARN_BADQACC`
  (~1% on the track_5 task set). That is upstream MuJoCo instability, flagged in
  VLABench's own source (`# BUG some episodes are unstable and lead to crash`),
  and is not recoverable client-side.
- VLABench's own `sh/evaluation/example_multi_gpu_eval.sh` parses `--track` /
  `--task` into `TRACK_OPT` / `TASK_OPT` but then loops over the never-defined
  `${TRACKS[@]}` / `${TASKS[@]}` (along with undefined `CKPT` and `job_idx`), so
  its first loop body never executes. `multi_eval.sh` here is the working
  equivalent.
- Headless rendering needs `MUJOCO_GL=egl` (the scripts set it) plus
  `libglu1-mesa` / `libgl1-mesa-dri` / `libgl1-mesa-glx`.
