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

> **Scope.** This directory is the eval **client + smoke**. The matching training
> dataloader (`openwam.dataloader.robocasa365.RoboCasa365Dataset`) now lives in the
> same PR. Real success rates still need a RoboCasa365-trained OpenWAM checkpoint;
> there is no public one yet, so a smoke run against a mismatched checkpoint will
> fail the `state_dim` check or produce garbage actions (expected).
>
> **Action spaces (two layers).** The env (`RoboCasaGymEnv`) consumes a fixed **12-D**
> robosuite OSC + base action. The OpenWAM model trained by `RoboCasa365Dataset`
> predicts a **20-D absolute EEF pose** (repo-standard EEF schema, dual of robotwin) —
> NOT 12-D. So `act()` bridges 20-D → 12-D client-side via
> `benchmarks.utils.eef20d_to_robocasa12d` (dual of robotwin's `eef20d_to_ee16d`); a
> 12-D server action is passed through unchanged. The bridge needs the env's OSC
> scaling (`osc_pos_scale` / `osc_rot_scale`, from the OSC_POSE controller config) and
> fills `base_motion`=0 / `control_mode`=-1 (fixed-base constants). End-to-end
> correctness of those scalars must be confirmed with a trained checkpoint in the env.

## Files

| File | Description |
|---|---|
| `openwam2robocasa365_interface.py` | WS adapter: `RoboCasaGymEnv` obs → OpenWAM payload; server action → env 12-D action dict (20-D EEF bridged via `eef20d_to_robocasa12d`, 12-D passed through). |
| `single_eval.py` | Run one RoboCasa365 task against an OpenWAM server. |
| `single_eval.sh` | Shell wrapper; patches host/port/task/split at runtime. |
| `multi_eval.sh` | Evaluate a list of tasks / `all` (from `fixed_base_tasks.json`) / a task-file; aggregates per-task success into a CSV. |
| `step_limits.yml` | Per-task eval horizon overrides (robotwin-style `ceil(avg/32)*32`); unlisted tasks fall back to the config `max_steps`. |
| `smoke_robocasa365.py` | Preflight: `import` / `env` / `roundtrip` checks. |
| `run_smoke.sh` | Smoke launcher. |
| `policy_config.yml` | Eval client config template. |
| `fixed_base_tasks.json` | The fixed-base task subset this benchmark + dataloader target (see below). |

## Fixed-base task subset

RoboCasa365 ships 365 tasks; many need the holonomic **mobile base** (whole-body
manipulation, `moma_required=Yes`). This benchmark + the dataloader scope to the
**fixed-base** subset — `moma_required=No`, **112 tasks** — so a single-arm EEF policy
never has to command base motion (the eval bridge fills `base_motion`=0).

The authoritative list is [`fixed_base_tasks.json`](fixed_base_tasks.json) (with a
`_meta` block: source, per-split counts, notes). Split groups:

| split_group | meaning | atomic | composite | total |
|---|---|---|---|---|
| `train_only` | pretrain data only | 38 | 51 | 89 |
| `train_and_eval` | pretrain + target (**seen** eval) | 14 | 5 | 19 |
| `eval_only` | target only (**unseen**, zero-shot composite) | 0 | 4 | 4 |

So 108 tasks are trainable and 23 have eval targets (19 seen + 4 unseen). Per-task demo
counts aren't fixed — read each bucket's `meta/info.json` `total_episodes` after download.

## Training data + dataloader (Phase 2)

The trainer side is `openwam.dataloader.robocasa365.RoboCasa365Dataset` (registered
`robocasa365`), reading the RAW RoboCasa365 LeRobot **v2.1** download directly (no
conversion). Point `dataset_dir` at one task's `lerobot/` bucket, or at a root holding
many `.../<Task>/<date>/lerobot` buckets (multi-task). Config + knobs live in
`configs/dataloader/robocasa365.yaml`.

```bash
# single task
scripts/train.sh dataloader=robocasa365 \
  dataloader.dataset_dir=/path/to/robocasa365/.../OpenDrawer/<date>/lerobot \
  dataloader.task_name=OpenDrawer
# multi-task: point dataset_dir at the root of buckets, drop task_name
```

The model trains the repo-standard **20-D EEF** action (action + proprio are both the
absolute EEF pose from `observation.state`; see the action-spaces note above), bridged
back to the env's 12-D OSC at eval time.

## Environment setup

RoboCasa / robosuite / MuJoCo conflict with the OpenWAM serving stack, so they
live in a **separate** env (like RoboTwin's `robotwin` env / LIBERO's
`LIBERO_PYTHON`). The client is torch-free (`websockets` + `numpy` + `Pillow`).

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
   - `state_dim: 20` — the model's 20-D EEF proprio; a fail-fast check against the
     checkpoint's training config. (The env's raw 16-D state is converted client-side.)
   - The model emits the repo-standard **20-D EEF**; the client bridges it to the env's
     **12-D OSC**. Set `osc_pos_scale` / `osc_rot_scale` to the eval env's OSC_POSE
     `output_max` (position metres / rotation radians mapped to action 1.0) — a 20-D
     action with these unset raises rather than emitting wrong-magnitude motions. A
     12-D server action (legacy 12-D checkpoint) is passed through unchanged.
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
  `action_sliced` into env keys, and a `checks` block (`state_dim_is_20`,
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
# every fixed-base task (from fixed_base_tasks.json)
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/multi_eval.sh all
# from a task-list file (one task per line, `#` comments)
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/multi_eval.sh my_tasks.txt
```
Tasks run sequentially; per-task `Success rate` lines are aggregated into
`results_robocasa365/summary_<split>.csv`.

## Per-task step limits

`single_eval.py` resolves the rollout horizon from `step_limits.yml` (per-task override),
falling back to `max_steps` in the policy config (default 500) for unlisted tasks; the env's
own `done`/`truncated` still ends an episode early. Values follow robotwin's
`ceil(avg_episode_len / 32) * 32` — seed a new task by computing its mean episode length over
`data/chunk-*/episode_*.parquet`.

## Known limitations

- **Only `OpenDrawer` is validated end-to-end** (train → deploy → real-sim). The other
  fixed-base tasks share the obs/action contract but are unverified.
- **No parallel / distributed eval** (cf. robotwin's `parallel_eval.sh` / DLC path) — tasks
  run sequentially.
- **OSC scales + gripper threshold** in `policy_config.yml` were measured on `OpenDrawer` /
  `default_pandaomron.json`; re-verify for other tasks/controllers.
- `step_limits.yml` is seeded only for tasks whose data is local; add the rest as needed.

## Contract notes

- **Two views (agentview + wrist)**: the client sends `agentview_left` → `head_camera`
  and the `eye_in_hand` wrist → `left_wrist_camera`. The single arm has no 2nd wrist,
  so `right_wrist_camera` stays `null` and the server black-fills that slot when it
  composes the multi-view layout. Mapping lives in `policy_config.yml`; Phase-2
  training must use the same 2-view layout.
- **No client resize / flip**: frames go full resolution; `RoboCasaGymEnv` already
  flips them upright (`image_transform: none`).
- **State / action are raw physical units**: the server (de)normalizes. The client converts
  the env's raw 16-D state to the model's 20-D EEF proprio, which must match the checkpoint's
  training config. The action is 20-D EEF
  out of the model, bridged to the env's 12-D OSC client-side (see the action-spaces
  note above); the 12-D layout/order must match `RoboCasaGymEnv`.
