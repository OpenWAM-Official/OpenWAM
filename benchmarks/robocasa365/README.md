# RoboCasa365 Benchmark Evaluation

These scripts connect the RoboCasa365 robosuite/MuJoCo simulator to an
already-running OpenWAM WebSocket policy server, mirroring the RoboTwin / LIBERO
benchmark pattern: the benchmark process owns the simulation and observations,
while the OpenWAM model, checkpoint, preprocessing and action denormalization
stay server-side.

RoboCasa365 is the single-arm **PandaOmron** (Franka arm + holonomic mobile base)
kitchen benchmark: a 12-D OSC action, a 16-D raw proprio state (converted client-side to
the model's 20-D EEF), and 3 cameras at
256×256 (2 third-person `agentview` + 1 `eye_in_hand` wrist). We send **two real
views** — `agentview_left` (3rd-person) → `head_camera` and `eye_in_hand` (the
arm's real wrist) → `left_wrist_camera`. The single arm has no 2nd wrist, so
`right_wrist_camera` is left empty and the server black-fills it.

> **Scope.** This directory is the eval **client + smoke**; the matching training dataloader
> (`openwam.dataloader.robocasa365`, registered `robocasa365`) is in the same PR — see
> "Training data + dataloader" below. The full train→deploy→eval chain is validated (plumbing);
> real success rates need a full training run (the smoke checkpoints are undertrained).
>
> **Action spaces (two layers).** The env (`RoboCasaGymEnv`) consumes a fixed **12-D**
> robosuite OSC + base action. The OpenWAM model trained by `RoboCasa365Dataset`
> predicts a **20-D absolute EEF pose** (repo-standard EEF schema, dual of robotwin) —
> NOT 12-D. With **mobile base** (`mobile_base=true`, the full-task default), the model
> ALSO emits a 5-D RoboCasa-native base command in the unified reserved slots, so the server
> returns a **25-D** `[arm20, base5]`; `act()` bridges the arm 20-D → OSC and passes `base5`
> ([x/y/yaw velocity, torso, control_mode]) through RAW into the env's `base_motion`+`control_mode`.
> A 20-D (arm-only) action still bridges with a zero base; a 12-D server action is passed through.
> The bridge needs the env's OSC scaling (`osc_pos_scale` / `osc_rot_scale`, from the OSC_POSE
> controller config). End-to-end correctness of those scalars must be confirmed with a trained
> checkpoint in the env.

## Files

| File | Description |
|---|---|
| `openwam2robocasa365_interface.py` | WS adapter: `RoboCasaGymEnv` obs → OpenWAM payload; server action → env 12-D action dict (20-D EEF bridged via `eef20d_to_robocasa12d`, 12-D passed through). |
| `single_eval.py` | Run one RoboCasa365 task against an OpenWAM server. |
| `single_eval.sh` | Shell wrapper; patches host/port/task/split at runtime. |
| `multi_eval.sh` | Evaluate a list of tasks / a task-file (e.g. `target_tasks.txt`); aggregates per-task success into a CSV. |
| `step_limits.yml` | Per-task eval horizon overrides (robotwin-style `ceil(avg/32)*32`); unlisted tasks fall back to the config `max_steps`. |
| `smoke_robocasa365.py` | Preflight: `import` / `env` / `roundtrip` checks. |
| `run_smoke.sh` | Smoke launcher. |
| `policy_config.yml` | Eval client config template. |
| `target_tasks.txt` | The official 50 eval target tasks (the multi-task leaderboard set). |

## Full task set + mobile base

This benchmark + dataloader cover the **full RoboCasa365** task set (all tasks — mobile +
fixed; 65 atomic + 235 composite have pretrain data). The model commands the holonomic base via the **mobile base** channel
(`mobile_base=true`), so tasks are no longer restricted to the fixed-base
(`moma_required=No`) subset. (Earlier revisions scoped to a fixed-base manifest and filled
`base_motion=0`; that filter — `fixed_base_tasks.json`, the `moma` root-mode drop, and the
`_assert_fixed_base` eval gate — has been removed.)

- **Training:** multi-task discovery keeps every task in the v3 repo(s) (by `source_prefix`) — see
  "Training data + dataloader" below for `dataset_dir` (single repo or the atomic+composite list).
- **Eval:** the official **50 target tasks** (the multi-task leaderboard set — 18 atomic + 16
  composite-seen + 16 composite-unseen), listed in [`target_tasks.txt`](target_tasks.txt). The 16
  composite-**unseen** tasks are held out of training (zero-shot). Run them with
  `multi_eval.sh ... target_tasks.txt`.

See the mobile design in `docs/plans/robocasa365-unify-raw-vector-refactor.md`: with `mobile_base` the
base command is folded INTO the raw vector → raw **25-D `[arm20, base5]`** (action & proprio share ONE
layout), scattered to the unified 80-D via ONE map `["0-9","34-43","68-72"]` (arm → `[0:10)`+`[34:44)`,
base5 → `[68:73)`) — like BEHAVIOR's RAW-27, base is not a bypass channel.

## Training data + dataloader

The trainer side is `openwam.dataloader.robocasa365.MultiTaskRoboCasa365Dataset`
(registered `robocasa365`), reading the RAW RoboCasa365 LeRobot **v3.0 aggregated** repo
directly (no conversion) — the HuggingFace mirrors
`ember-lab-berkeley/robocasa365-pretrain-{atomic,composite}`. `dataset_dir` points at the
repo exactly as downloaded (aggregated `data/chunk-*/file-*.parquet` + `videos/…` +
`meta/episodes/*.parquet`). Config + knobs: `configs/dataloader/robocasa365.yaml`.

v3 packs many tasks into ONE aggregated repo, tagged per episode by `source_prefix`:
- `task_name` set → single task (the repo filtered to that task).
- `task_name` null → multi-task: every task across the repo(s), one sub-dataset per task,
  concatenated, sharing ONE pooled stats file.

The full **300 train tasks** live in TWO separate repos (65 atomic + 235 composite), so pass
`dataset_dir` as a **list**. Multi-repo requires an explicit `normalization_stats_path` (there
is no single root to auto-place the pooled stats — the reader raises if it's unset):

```bash
# FULL 300-task multi-task training (atomic + composite):
scripts/train.sh dataloader=robocasa365 \
  'dataloader.dataset_dir=[/data/robocasa365-pretrain-atomic,/data/robocasa365-pretrain-composite]' \
  dataloader.normalization_stats_path=/data/robocasa365_multitask_eefbase_stats.npy

# Single repo (atomic-only, or a one-task smoke):
scripts/train.sh dataloader=robocasa365 \
  dataloader.dataset_dir=/data/robocasa365-pretrain-atomic   # [+ dataloader.task_name=OpenDrawer]
```

`unify_action` + `mobile_base` are ON by default. The arm action is the repo-standard **20-D
EEF** (full base-relative pose from `observation.state`, bridged to 12-D OSC at eval) in unified
slots `[0:10)`; the **5-D base command** (raw from the LeRobot `action` field: `[x_vel, y_vel,
yaw_vel, torso, control_mode]`, direct-to-env at eval) maps to `[68:73)`. **Proprio mirrors the
layout**: the arm current pose plus the **body-frame base velocity** in `[68:71)` (finite-diff of
the base pose, rescaled into the action command space so it shares the base stats — **A′**), with
torso + control_mode masked (no achieved value). ONE combined **25-D `eef_base`** stats block
normalizes the whole `[arm20, base5]` vector; deploy gathers 80→25 and un-normalizes with it (no
base special-casing — the eval client sends 25-D proprio and bridges arm→OSC, base5 direct).
`configs/model/dual_system.yaml` **defaults** `action_dim/state_dim=80`, so **no override is
needed** for the default model (only a model config that hardcodes 20 would need
`model.architecture.action_dim=80 model.architecture.state_dim=80`).

`mobile_base` and `unify_action` are **decoupled**: non-unify emits the raw 25-D head directly. A
fixed-base (arm-only) run sets `mobile_base=false` → 20-D `eef` stats, no base5. Changing
`mobile_base` changes the action/proprio definition + stats schema, so it takes effect only on a
fresh training run.

## Environment setup

RoboCasa / robosuite / MuJoCo conflict with the OpenWAM serving stack, so they
live in a **separate** env (like RoboTwin's `robotwin` env / LIBERO's
`LIBERO_PYTHON`). The client is torch-free (`websockets>=15` + `numpy` + `Pillow`).

> **`websockets>=15` is required** on the client. The transport disables keepalive
> pings via `connect(..., ping_interval=None)` (so slow first-call inference / compile
> warmup doesn't trip the 20s ping deadline), and `websockets.sync.client.connect`
> only accepts `ping_interval` from 15.0 onward — older versions raise
> `TypeError: ... unexpected keyword argument 'ping_interval'` at connect time. This is
> a transport-layer requirement shared by all benchmark clients, not RoboCasa-specific.

1. Create an isolated `robocasa365` env and install RoboCasa (pulls robosuite +
   MuJoCo):
   ```bash
   pip install -e /path/to/robocasa            # github.com/robocasa/robocasa (v1.0)
   # robosuite: install the branch RoboCasa pins (NOT pypi); mujoco==3.3.1
   python -m robocasa.scripts.download_kitchen_assets   # ~10GB
   ```
2. Point the scripts at that env's python:
   ```bash
   export ROBOCASA365_PYTHON=/path/to/robocasa365/env/bin/python
   ```
3. Match the checkpoint's action/state config in `policy_config.yml`:
   - `state_dim: null` — auto-derives the expected proprio width (20-D EEF, or **25-D** `[arm20,
     base5]` when `mobile_base: true`). Set an explicit int only to pin it; it's a fail-fast
     check against what the client sends. (The env's raw 16-D state is converted client-side.)
   - `mobile_base: true` **only** for a checkpoint trained with `dataloader.mobile_base=true` (the
     default; client then sends 25-D proprio `[arm20, base5]` where `base5 = [vx, vy, vyaw (A′
     command-space velocity), 0, 0]`); a mismatch fails fast at the server. Leave `false` for a
     fixed-base ckpt.
   - The model emits the repo-standard **20-D EEF** (or **25-D** `[arm20, base5]` for a mobile
     ckpt); the client bridges the arm 20-D → the env's **12-D OSC** (position → scaled OSC delta
     vs the current/last-target eef per `control_mode`; rot6d → axis-angle) and passes the base
     command through. Set `osc_pos_scale` / `osc_rot_scale` to the eval env's OSC_POSE
     `output_max` (metres / radians mapped to action 1.0) — unset → raises rather than emitting
     wrong-magnitude motions. A 12-D server action (legacy ckpt) is passed through unchanged.
4. Start the OpenWAM server (in the OpenWAM env) with a RoboCasa365 checkpoint:
   ```bash
   bash scripts/deploy.sh --ckpt-dir /path/to/robocasa365_ckpt --port 8848
   ```

## Smoke

```bash
# sim plumbing (no server, no policy):
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/run_smoke.sh import
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/run_smoke.sh env OpenDrawer
# client <-> server WS path (needs a running server, no sim):
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/run_smoke.sh roundtrip
```
`env` steps a zero action dict through the sim; `roundtrip` pings + predicts a
dummy obs and prints the returned action dim.

## Checking correctness (debug bundle)

Set `debug: true` in `policy_config.yml`, then run `single_eval`. The adapter
writes a per-step bundle under `{debug_dir}/ep{N}/step_{N}/` — the same layout as
robotwin's debug mode, plus a montage and pass/fail checks — so you can eyeball
the obs → payload → action mapping:

- `head.jpg` / `left.jpg` — the frames the client sends (`agentview_left` and the
  `eye_in_hand` wrist). The empty `right` slot writes a `{stem}_missing.txt` stub
  (the server black-fills it).
- `cameras.png` — the sent frame(s) decoded + labeled (written on the first step of
  each episode). Verify the frame is upright (the top of the scene stays on top —
  `image_transform: none` because `RoboCasaGymEnv` already flips them).
- `meta.json` — `episode` / `step` / `prompt` / `state` / `action` / `server_step`
  / `latency_ms` (robotwin fields) plus a per-key `state_breakdown`, the 12-D
  `action_sliced` into env keys, and a `checks` block (`state_dim_ok`,
  `head_and_wrist_present`, `action_dim_is_12`).

## Single-task evaluation

```bash
ROBOCASA365_PYTHON=/path/to/robocasa365/env/bin/python \
  bash benchmarks/robocasa365/single_eval.sh OpenDrawer target 8848 127.0.0.1
```
Args: `<task> <split> <port> <host>`. Use `POLICY_CONFIG_PATH=/path/to/custom.yml`
for a copied config. Headless rendering uses `MUJOCO_GL=egl`.

## Multi-task evaluation

```bash
# named tasks
ROBOCASA365_PYTHON=/path/to/env/bin/python \
  bash benchmarks/robocasa365/multi_eval.sh --split target --port 8848 OpenDrawer CloseDrawer
# every official eval target (the 50 in target_tasks.txt)
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/multi_eval.sh target
# from a task-list file (one task per line, `#` comments)
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/multi_eval.sh my_tasks.txt
```
Tasks run sequentially; per-task `Success rate` lines are aggregated into
`results_robocasa365/summary_<split>.csv`.

## Per-task step limits

`single_eval.py` resolves the rollout horizon from `step_limits.yml` (per-task override),
falling back to `max_steps` in the policy config (default 500) for unlisted tasks; the env's
own `done`/`truncated` still ends an episode early. Values follow robotwin's
`ceil(avg_episode_len / 32) * 32` — seed a new task by computing its mean episode length from the
v3 repo's `meta/episodes/*.parquet` (`length` column), filtered to that task's `source_prefix`.

## Known limitations

- **Plumbing is validated end-to-end** (train → deploy → real-sim, EXIT 0) on `OpenDrawer`
  and `NavigateKitchen` — including the mobile raw 25-D `[arm20, base5]` (base command in action
  `[68:73)`, A′-rescaled base velocity in proprio `[68:71)`, one map, one `eef_base` stats block).
  But these were 100-step smoke checkpoints, so **success rates are ~0 (undertrained)** — a real
  SR needs a full training run. Other tasks share the same obs/action contract but haven't been
  individually run.
- **No parallel / distributed eval** (cf. robotwin's `parallel_eval.sh` / DLC path) — tasks
  run sequentially.
- **OSC scales + gripper threshold** in `policy_config.yml` were measured on `OpenDrawer` /
  `default_pandaomron.json`; re-verify for other tasks/controllers.
- `step_limits.yml` is seeded only for tasks whose data is local; add the rest as needed.

## Contract notes

- **Two views (agentview + wrist)**: the client sends `agentview_left` → `head_camera`
  and the `eye_in_hand` wrist → `left_wrist_camera`. The single arm has no 2nd wrist,
  so `right_wrist_camera` stays `null` and the server black-fills that slot when it
  composes the multi-view layout. Mapping lives in `policy_config.yml`; training must use
  the same 2-view layout.
- **No client resize / flip**: frames go full resolution; `RoboCasaGymEnv` already
  flips them upright (`image_transform: none`).
- **State / action are raw physical units**: the server (de)normalizes. The client converts
  the env's raw 16-D state to the model's 20-D EEF proprio, which must match the checkpoint's
  training config. The action is 20-D EEF
  out of the model, bridged to the env's 12-D OSC client-side (see the action-spaces
  note above); the 12-D layout/order must match `RoboCasaGymEnv`.
