# RoboDojo

Train and evaluate OpenWAM on the official RoboDojo `arx_x5` HDF5 release.

The reader only accepts the formal layout:

```text
<dataset_dir>/<task>/arx_x5/data/episode_*.hdf5
```

Each episode is converted from RoboDojo's env-origin xyz + world wxyz into
per-arm robot-base **EEF20**, normalized in that 20-D space, then scattered
into OpenWAM's unified 80-D slots `0-9` (left) and `34-43` (right). Evaluation
inverts the same transform and returns native `left/right_ee_pose` + gripper
so RoboDojo's IK runs. Joint-14 and LeRobot dumps are not supported.

## Download HDF5

The official HDF5 dump is about **523GB**. It typically contains **34**
task folders. Hold out `dlc` from training. The eight **open** tasks are
usually absent from this dump and can only be evaluated zero-shot.

Requires `git` and `git-lfs`. Hugging Face and ModelScope are the only
sources; LeRobot, depth, and real-robot packs are ignored.

```bash
# From the OpenWAM repo root. Default target: <openwam>/data/RoboDojo
bash scripts/download_robodojo.sh huggingface
# or
bash scripts/download_robodojo.sh modelscope
```

Override the parent directory with `ROBO_DOJO_DATA_ROOT`. The script writes
`<data_root>/RoboDojo` and that path is `dataloader.dataset_dir`. Do not point
the dataloader at the parent of `RoboDojo`.

Leave `configs/dataloader/robodojo.yaml` on the `/path/to/...` placeholders
and pass real paths as Hydra overrides.

## Compute stats

The stats CLI reads the YAML file only. It does not accept Hydra overrides.
Set `dataset_dir` and `holdout_tasks: [dlc]` in
`configs/dataloader/robodojo.yaml` (or a local copy), then:

```bash
python -m openwam.dataloader.utils.stats_computation.robodojo_stats_computation \
  --config configs/dataloader/robodojo.yaml \
  --output /path/to/robodojo_arx_x5_eef20_stats.npy
```

The pool includes every achieved state and each real next-state target. The
saved calibration fingerprint must match the built-in dual-X5 constants.

## Train

Finetune from `OpenWAM/Pretrained_OpenWAM_Mutual_Final`. A 1-GPU debug run
OOMs on that checkpoint; use 8 GPUs and ZeRO-2 (`training.zero_stage=2` is
the default).

Pass the 20-step debug gate first:

```bash
torchrun --nproc_per_node=8 scripts/train.py dataloader=robodojo \
  dataloader.dataset_dir=/path/to/RoboDojo \
  dataloader.normalization_stats_path=/path/to/robodojo_arx_x5_eef20_stats.npy \
  dataloader.holdout_tasks=[dlc] \
  training.finetune_ckpt_path=/path/to/Pretrained_OpenWAM_Mutual_Final \
  training.output_path=/path/to/openwam_robodojo_sft \
  training.debug=true
```

Then full SFT:

```bash
torchrun --nproc_per_node=8 scripts/train.py dataloader=robodojo \
  dataloader.dataset_dir=/path/to/RoboDojo \
  dataloader.normalization_stats_path=/path/to/robodojo_arx_x5_eef20_stats.npy \
  dataloader.holdout_tasks=[dlc] \
  training.finetune_ckpt_path=/path/to/Pretrained_OpenWAM_Mutual_Final \
  training.learning_rate=2e-5 \
  training.num_epochs=1 \
  training.output_path=/path/to/openwam_robodojo_sft
```

For one task, add `dataloader.task_name=stack_blocks`. The canonical unified
map is `["0-9", "34-43"]`.

## Evaluate

Start the OpenWAM JSON server from the OpenWAM environment:

```bash
python -m openwam.deploy.server \
  --ckpt-dir /path/to/openwam_robodojo_sft/<run> \
  --host 0.0.0.0 \
  --port 8848
```

Do not point RoboDojo's MsgPack client at this port. Then run smokes in order
and a full single-task eval. Isaac evaluation needs a separate RoboDojo
checkout and the `RoboDojo` conda env; see [Smoke progression](#smoke-progression)
and [Full single-task evaluation](#full-single-task-evaluation).

```bash
bash benchmarks/robodojo/run_smoke.sh contract \
  /path/to/RoboDojo stack_blocks
bash benchmarks/robodojo/run_smoke.sh ping
bash benchmarks/robodojo/run_smoke.sh debug stack_blocks
bash benchmarks/robodojo/run_smoke.sh isaac stack_blocks

conda run -n RoboDojo python -m benchmarks.robodojo.single_eval \
  --config benchmarks/robodojo/policy_config.yml \
  --robodojo-root /path/to/RoboDojo-checkout \
  --host 127.0.0.1 \
  --port 8848 \
  --task stack_blocks \
  --device-id 0 \
  --seed 0 \
  --eval-count 25
```

The rest of this file is the native-eval contract: how the adapter talks to
RoboDojo, frame conversion, PhysX resume, and error mapping.

## Native evaluation adapter

This directory connects RoboDojo's native single-environment `arx_x5`
evaluator to an already-running OpenWAM JSON WebSocket policy server. RoboDojo
and XPolicyLab are not modified. Their bundled `WsModelClient` speaks a
different MsgPack protocol, so the runner replaces it with a no-network
placeholder only while `create_eval_env` executes, restores the original symbol
in `finally`, then injects `OpenWAMRoboDojoModelClient`.

## Data and EEF20 contract

Only the formal task-qualified layout is supported:

```text
<dataset_root>/<task>/arx_x5/data/episode_*.hdf5
```

Each episode contains scalar `instruction`, JPEG data under
`vision/{cam_head,cam_left_wrist,cam_right_wrist}/colors`, and achieved state
under:

```text
state/left_ee_poses             (T, 7), xyz+wxyz
state/left_ee_joint_states      (T, 1), gripper in [0, 1]
state/right_ee_poses            (T, 7), xyz+wxyz
state/right_ee_joint_states     (T, 1), gripper in [0, 1]
```

Closed-gripper float noise around `-3e-17` is accepted and clipped to `[0, 1]`.
True out-of-range values are still rejected.

RoboDojo positions are relative to the Isaac environment origin while their
orientations remain world-oriented. A measured transform for each X5
`base_link` converts both into that arm's base frame. Training proprio and
targets, pooled statistics, server state, and server actions all use raw EEF20:

```text
[left.xyz3, left.rot6d6, left.gripper1,
 right.xyz3, right.rot6d6, right.gripper1]
```

Raw EEF20 is normalized before optional scatter to OpenWAM's unified 80-D
training representation. Deployment gathers and denormalizes server-side, so
the client always receives physical raw 20-D output—not joint-14 or unified-80.

## Configure

Edit `benchmarks/robodojo/policy_config.yml`, especially:

- `robodojo_root`: external RoboDojo checkout.
- `task`: a valid RoboDojo task such as `stack_blocks`.
- `host`, `port`, and `timeout`: OpenWAM JSON server.
- `device_id`: physical Isaac GPU id. Before importing Isaac, the runner sets
  `CUDA_VISIBLE_DEVICES=<device_id>`, launches on the resulting visible
  `cuda:0`, and enables the native `isaacsim.replicator.behavior` and
  `isaacsim.sensors.camera` Kit extensions.
- `num_envs: 1`, `eval_batch: false`, and `env_config: arx_x5`: fixed limits.
- `max_steps: null`: preserves RoboDojo's native task limit. Set a positive
  integer or pass `single_eval.py --max-steps N` for an explicit rollout cap;
  the override is applied after RoboDojo finishes processing its config.

The runner puts the OpenWAM checkout first on `sys.path`, followed by
`<robodojo_root>`, `<robodojo_root>/XPolicyLab`, all six vendored
`third_party/IsaacLab/source/<package>` roots, and `third_party/curobo`. Before
any live import it verifies both `benchmarks`/`openwam` and RoboDojo's `env`,
`task`, `utils`, `XPolicyLab`, `client_server`, `src`, `isaaclab`,
`isaaclab_assets`, `isaaclab_tasks`, and `curobo`. Mixed namespace origins or
an already-loaded stale editable checkout cause an explicit startup error.

## Dual-X5 base transforms

Training and evaluation convert RoboDojo env-relative `xyz+wxyz` poses with
the same constants as `env_cfg/robot/dual_x5.yml`, stored in
`openwam/robodojo/contract.py`:

```python
DUAL_X5_LEFT_BASE_POS = (-0.3, -0.45, 0.765)
DUAL_X5_LEFT_BASE_QUAT_WXYZ = (0.707, 0.0, 0.0, 0.707)
DUAL_X5_RIGHT_BASE_POS = (0.3, -0.45, 0.765)
DUAL_X5_RIGHT_BASE_QUAT_WXYZ = (0.707, 0.0, 0.0, 0.707)
```

Do not pass a calibration JSON into the dataloader. Optional
`--calibration-output` on `single_eval` is only a dump of live Isaac poses.

## Smoke progression

Run the checks in order:

```bash
# 1. Built-in dual_x5 conversion; add dataset root + task for formal HDF5.
bash benchmarks/robodojo/run_smoke.sh contract \
  /path/to/RoboDojo stack_blocks

# 2. OpenWAM JSON server ping (no RoboDojo/Isaac import).
bash benchmarks/robodojo/run_smoke.sh ping

# 3. No-Isaac RoboDojo debug-env protocol rollout.
bash benchmarks/robodojo/run_smoke.sh debug stack_blocks

# 4. One native Isaac episode capped at three steps by default.
bash benchmarks/robodojo/run_smoke.sh isaac stack_blocks
```

`debug` exports `EVAL_ENV_TYPE=debug`, constructs
`XPolicyLab/debug_env_client.py` while its MsgPack client factory is replaced by
a no-network placeholder, and then uses static transforms from the calibration.
It exercises reset, observation update, OpenWAM action, and RoboDojo's native
action-dictionary validation without importing Isaac.

Set overrides with `ROBODOJO_ROOT`, `OPENWAM_HOST`, `OPENWAM_PORT`,
`OPENWAM_TIMEOUT`, `ROBODOJO_PYTHON`, and `ROBODOJO_ISAAC_STEPS`. The Isaac
smoke defaults to three steps; `ROBODOJO_ISAAC_STEPS` must be a positive
integer. By default the script uses `conda run -n RoboDojo python`. Host, port,
and timeout CLI overrides are added only when their corresponding environment
variable is explicitly set; otherwise edited values in `policy_config.yml` are
preserved.

The shell launcher always changes to its computed OpenWAM worktree root before
executing `python -m`, so invoking the script from another OpenWAM checkout
cannot win module resolution through the caller's working directory. Relative
config, calibration, dataset-root, and `ROBODOJO_ROOT` paths are resolved
against the caller's original working directory before that change.
Non-Isaac modes execute once and fail fast. `isaac` preserves an existing
`ROBODOJO_RUN_ID` or generates one in the parent shell, then retries exit codes
`99`, `134`, and `139` up to `ROBODOJO_MAX_BASH_RETRIES` times after the
initial launch (default `10`; set `0` to disable retries), with
`ROBODOJO_RETRY_DELAY_SECONDS` controlling the delay (default `5` seconds).

## Full single-task evaluation

```bash
conda run -n RoboDojo python -m benchmarks.robodojo.single_eval \
  --config benchmarks/robodojo/policy_config.yml \
  --robodojo-root /path/to/RoboDojo-checkout \
  --host 127.0.0.1 \
  --port 8848 \
  --task stack_blocks \
  --device-id 0 \
  --seed 0 \
  --eval-count 25
```

Omitting `--max-steps` (and leaving the YAML value null) retains the task's
native `max_steps`. Use `--max-steps N` only for an intentional positive
integer cap.

The runner reuses RoboDojo's `env.reset`, `env.run_eval`, seed manager,
`demo_policy.deploy.eval_one_episode`, result JSON, and video writers. The
runtime policy alias is `openwam`, so native result paths remain recognizable.
The live body runs with its working directory set to `robodojo_root`, so native
relative outputs such as `eval_result/...` land under
`<robodojo_root>/eval_result`; the caller's working directory is restored on
every exit path.

## PhysX recovery and resume

Before `AppLauncher`, the runner resolves the task YAML and enables the current
RoboDojo PhysX warning monitor only when its `Articulation` section is nonempty
(configuration read failures enable it fail-safe). The same flag is passed into
`EvalEnv`. The monitor resets before every single-env attempt.

- `UnStableError` advances the seed manager and continues.
- `PhysXBrokenError` marks the current layout abandoned, closes the simulator,
  and tries the next seed.
- If reset or observation generation raises a generic exception before the
  typed checks run, a fatal monitor state takes the fatal path and valid
  monitor-reported env indices take the broken-seed path. With a clean or
  disabled monitor, the original policy, protocol, or unrelated exception is
  re-raised unchanged.
- `PhysXFatalError` first persists the native resume manifest, then unwinds and
  closes the policy client, env, SimulationApp, monitor, and working-directory
  context. The direct Python entry point uses bounded in-process re-exec;
  shell-required failures or a reached cap return `99`.

Because the current monitor redirects process fd 1 and 2 but does not restore
them, the runner duplicates stdout/stderr before monitor startup. After
SimulationApp and monitor cleanup it restores both original descriptors and
closes the saved copies, including fatal unwind paths, so direct re-exec
inherits working output streams.

Before `create_eval_env`, the runner loads the matching native manifest:

```text
<robodojo_root>/eval_result/RoboDojo/<task>/openwam/<config_name>/
  <seed>_<additional_info>/_resume_<ROBODOJO_RUN_ID>.json
```

The manifest restores completed and abandoned layouts into the current
`create_eval_env(..., resume_state=...)` API. It is deleted best-effort only
after the requested success/fail total completes; failures retain it.

## Why there is no EEF20-to-joint14 conversion

The model predicts absolute per-arm base-frame EEF poses. The adapter converts
each `rot6d` to `wxyz`, inverts the live base transform, and returns RoboDojo's
native action dictionary:

```text
left_ee_pose (7) + left_ee_joint_state (1)
right_ee_pose (7) + right_ee_joint_state (1)
```

RoboDojo sees the `*_ee_pose` keys, invokes its existing `solve_ik`, and drives
the six joints per arm. Converting EEF20 to a fabricated joint-14 vector in the
client would bypass this native IK path and duplicate robot-specific logic.
Only the two gripper scalars are clipped to `[0, 1]`; poses are never clipped.

## Error mapping

- **Data/schema:** missing formal paths, camera datasets, scalar instruction,
  `(T, 7)` poses, `(T, 1)` grippers, finite values, or unit `wxyz` quaternions
  fail in contract/dataloader validation. Fix the source HDF5; do not pad it.
- **Calibration/frame:** wrong schema, non-X5 targets, invalid `base_link`
  transforms, or calibration fingerprint mismatches fail before action
  execution. Regenerate calibration from the same live `arx_x5` setup.
- **Server/protocol:** connection errors, non-`pong`, non-`reset_ack`,
  non-`action`, width 14/80, NaN, or degenerate `rot6d` fail in the adapter.
  Verify that the port serves OpenWAM JSON and the checkpoint's raw action is
  EEF20. Action requests are sent at most once. A transport, response, or
  action-conversion failure poisons the adapter; only a subsequent reset that
  receives `reset_ack` permits another observation.
- **PhysX:** recoverable broken-layout warnings abandon only the current
  single-env seed. Fatal GPU/kernel failures persist resume state and request
  process restart (`99` when a fresh shell process is required). These paths do
  not reinterpret policy/protocol failures as simulator failures.
- **IK/control:** if a valid native action reaches RoboDojo but `solve_ik`
  reports failure, inspect frame calibration, pose reachability, and the
  simulator state. The adapter deliberately does not invent fallback joints.

## Limits and cleanup

This integration is single-env only. Batch calls and any `num_envs != 1` fail
before launch. Cameras are sent as source-resolution JPEGs; existing JPEG bytes
are passed through, and arrays are encoded without client-side resizing.

Policy client, evaluator, and SimulationApp close in nested `finally` blocks.
The temporary MsgPack-client patch is also restored in `finally`, including
construction failures. Every reset attempt clears the cached observation and
remains fail-closed unless `reset_ack` arrives, so an uncertain action is never
silently resent.

The adapter also starts fail-closed immediately after its constructor ping:
the first observation is rejected until the evaluator's initial reset receives
`reset_ack`.
