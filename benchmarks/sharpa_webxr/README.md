# Closed-loop WebXR sim eval for GWP-0.5 Sharpa checkpoints

This harness runs the GWP-0.5 Sharpa EEF policy (`run_giga_sharpa_train_job.py --action-repr eef`) closed loop in the
WebXR-Teleop MuJoCo sim. It uses the training scenes from the live WebXR store and seeded resets, so every checkpoint
sees the same initial states. Each rollout gets automatic stage signals, 16 keyframes for a VLM judge, and a video.

## Layout: env, policy server, runner

- **`env.py` (model-agnostic sim):** `SharpaWebXREnv(scene, obs=ObsConfig(...), action_space=..., eef_frame=...)`.
  - `reset(seed) -> obs` and `step(action, observe=True|False|keys) -> (obs, info)`. One action is one 50 ms control
    step (20 Hz).
  - Action spaces: `joint` (56 joint targets) or `eef` (wrist pose per arm plus hand joints). For `eef`, the env runs
    IK, each solve seeded from the previous one, and returns the residuals in `info`.
  - Observations are configurable: cameras by name and resolution, plus state keys (`joints`, `eef`, `object_poses`,
    `time`). Images render only when a step asks to observe.
  - Physics: TacoSim with the teleop app defaults (2400 Hz physics, 60 Hz tick, each target held for 3 ticks).
  - Reset: `teleop_app.prepare_scene` with a seeded RNG (arm jitter 0.08, object xy ±5 cm, yaw ±180°, 1 s settle).
- **`policies/<model>_server.py` (one per model, in the training `.venv` on a GPU)** follows the
  `policies/common.py` contract. The server owns everything model-specific: image layout, state encoding,
  normalization, sampler and action decode. It returns absolute wrist poses plus hand joints in the ego-camera frame.
  The full request/response spec is in [API.md](API.md). `policies/hold_server.py` is a no-model reference server
  and template, and `check_server.py --port N` checks any server against the contract.
- **`watch_ckpts.sh` (generic eval watcher):** evaluates every checkpoint of any model's run as it lands, given the
  model's server script and a checkpoint-listing command (API.md, "Eval watcher"). `watch_run.sh` is the original
  GWP-specific watcher (EMA + raw pairs, best_val).
  Its `meta` endpoint publishes how the model is deployed upstream:
  - `execute_steps`: actions to run before replanning;
  - `obs_offsets`: past rows the model needs, e.g. LDA-1B's t−5;
  - optionally `cameras`.
  To add a model, write a `BasePolicy` subclass with `act()` and `serve()` it.
- **`rollout.py` (one runner for every model):** reads the server's `meta`, observes only at the rows the next request
  needs, executes the first `execute_steps` of each chunk, then re-observes; there is no ensembling.
  `--execute-steps` overrides the server's value. Episode length is `limits.limit(scene)` = min(`ceil(1.5 × p90)`,
  20 s); signals, judge frames and videos use the same span. The policy only reacts to what it has seen so far, so
  longer recordings are truncated offline: their first N steps are exactly what a run capped at N would have produced.

`policies/gwp_server.py` (GWP-0.5):

- **Sampling:** 10 flow-matching steps, shift 5, no CFG. `ego_view` and `chest_view` are composed into the training
  T-layout at 320x384. The state and actions use the giga codec in the ego-camera frame, and the noise is seeded
  per (scene, seed, chunk).
- **Execution:** `execute_steps` = 20 of 32, the same 62.5% / 1.0 s as upstream's 30 of 48 at 30 Hz
  (`run_inference_openloop.sh` REPLAN_STEPS).
- **Parity:** the runner reproduces both earlier paths bit for bit. qpos and IK results were identical against the
  old `policy_server.py` client-decode path at `--execute-steps 32`, and against `rollout_v2.py` at 20.

Stage signals, computed in `postprocess.py`:

- **reach:** a hand comes within 3 cm of an object.
- **grasp:** at least 0.5 s of hand contact while the object is lifted at least 2 cm or carried at least 3 cm.
- **move:** the object is displaced at least 5 cm, or a joint changes by at least 20° or 3 cm.
- **fallen:** an object fell off the table.
- **Rule verifier** (`benchmarks/sharpa/verifiers.py`): runs for the laptop and box-lid scenes.

Task completion and placement are scored by `judge.py` (Claude, from the keyframes).

Validated before use:

- Rendering matches the recorded dataset frames: PSNR 37.7 dB on the ego view and 42.1 dB on the chest view.
- Replaying recorded setpoints reproduces the recorded outcome to within 1.2 mm.
- Policy inputs are bit-identical to the training dataset (`check_policy_parity.py`).

## Setup

```bash
# headless rendering (no GPU GL needed)
apt-get install -y libosmesa6 libegl1 libglvnd0
git submodule update --init third_party/WebXR-Teleop third_party/giga-world-policy

# sim venv (python 3.12), separate from the training .venv
uv venv .venv_webxr --python 3.12
uv pip install --python .venv_webxr/bin/python -r third_party/WebXR-Teleop/requirements.txt \
    torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv_webxr/bin/python pyzmq imageio imageio-ffmpeg av anthropic openai

# scenes from the live store (needs gsutil access) -> data/webxr_scenes/<scene_id>/
.venv_webxr/bin/python benchmarks/sharpa_webxr/scenes.py $(.venv_webxr/bin/python -c \
    "import json;print(' '.join(json.load(open('benchmarks/sharpa_webxr/tasks.json'))))")
```

The policy server runs in the training `.venv`. It needs the Wan2.2 VAE at `assets/video_backbone_ckpt/Wan2.2-VAE-Diffusers/vae`.

## Automatic eval of a training run

```bash
benchmarks/sharpa_webxr/watch_run.sh <run> <gpu_ema> <gpu_bf16> [seeds_intermediate=0-4] [seeds_final=5-9] \
  > .sharpa_sim_eval/logs/watch_<run>.log 2>&1 &
```

`<run>` is the folder under `.gwp_runs/`. The watcher works like this:

- It copies each `checkpoint_stepN/transformer_bf16.pt` before training's rotation deletes it.
- It evaluates the EMA (`ema_stepN/`) and the raw weights of every checkpoint, oldest first. Each is served by
  `$SERVER` (default `policies/gwp_server.py`) on its own GPU, using about 15 GB next to training.
- It renders videos from one shared low-priority queue, seed 0 first.
- It rates every finished EMA tag with the VLM judge (`score.py`, rubric r2) and writes
  `<tag>/scores_<model>_r2.json` and `.tsv`.
- After training exits, it adds the final seeds for the last step and evaluates the final `best_val`.

The watcher is resumable: rerunning the same command skips finished work. It prints `WATCH_ALL_DONE` when done.

| env | default | meaning |
|---|---|---|
| `VIDEO` | `1` | `0` skips `video.mp4` rendering. Signals and keyframes are always written. |
| `VIDEO_SEEDS` | `all` | Comma list of seeds to render, e.g. `0`. |
| `VIDEO_P` | `32` | Parallel renderers. Each holds about 2.3 GB RAM and renders on the CPU with OSMesa at nice 19. |
| `PORT_BASE` | `11600` | EMA server port; raw uses `PORT_BASE+1`. Change it to run two watchers at once. |
| `SERVER` | `policies/gwp_server.py` | Policy server script; any `policies/common.py` server works. |
| `EXECUTE_STEPS` | server meta | Override actions per replan, e.g. `32` for the full chunk. |
| `JUDGE` | `1` | `0` skips the judge. It is also skipped, with a log line, when no OpenAI key is set. |
| `JUDGE_WEIGHTS` | `ema` | Tags to rate: `ema`, `bf16` or `all`. |
| `JUDGE_MODEL` | `gpt-6-luna` | Judge model; `JUDGE_WORKERS` (default 8) sets parallel requests. |
| `PREP_DIR` | `prep_eef_full_v1` | Training prep dir with norm stats and prompts. Must match the run's `--prep-dir`. |

Assumptions:

- The run uses the EEF action representation (62-D), 2 views in a T-layout, and horizon 32. Check a new run with
  `check_policy_parity.py` first: expect 24/24 identical inputs and a wrist error in the tens of mm or less.
- Training is detected by its `--out-dir .gwp_runs/<run>` argument.
- Rendering is slower than evaluation (about 3 videos/min on a loaded host), so videos lag behind the numbers.

## Manual pieces

```bash
# policy server (training venv)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u benchmarks/sharpa_webxr/policies/gwp_server.py --ckpt <transformer_*.pt> --port 11500
# one checkpoint: 58 scenes x seeds 0-9, 10 workers
RUN=<run> benchmarks/sharpa_webxr/run_suite.sh <tag> 11500 0 9 10
# one scene (server's execute_steps; --execute-steps 32 executes the full chunk)
.venv_webxr/bin/python benchmarks/sharpa_webxr/rollout.py --port 11500 --scene <scene_id> --seeds 0-2 --out <dir>
.venv_webxr/bin/python benchmarks/sharpa_webxr/postprocess.py <dir>/<scene_id>/seed_0
# checkpoint parity on val windows
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/sharpa_webxr/check_policy_parity.py --ckpt <ckpt> --n 24 --out g3.json
# score a folder with a VLM judge -> <folder>/scores_<model>_r2.json + .tsv (per task, per seed)
# rubric r2 needs full-res judge frames: 24 times x (head, chest) at 480x640
.venv_webxr/bin/python benchmarks/sharpa_webxr/postprocess.py --frames-only .sharpa_sim_eval/<run>/<tag>/*/seed_*
.venv_webxr/bin/python benchmarks/sharpa_webxr/score.py .sharpa_sim_eval/<run>/<tag> --model gpt-6-luna   # OPENAI_API_KEY or ~/.config/openai/key
.venv_webxr/bin/python benchmarks/sharpa_webxr/judge.py .sharpa_sim_eval/<run>/<tag>/*/seed_*   # Claude judge, ANTHROPIC_API_KEY
# check a policy server against the API before evaluating it (reference: policies/hold_server.py)
.venv_webxr/bin/python benchmarks/sharpa_webxr/check_server.py --port 11600
# per-category / per-task success (k/n) across checkpoints or models
.venv_webxr/bin/python benchmarks/sharpa_webxr/task_breakdown.py xwam=<tag>/scores_gpt-6-luna_r2.json motus=... --tasks
```

## Results

- Table: `.venv/bin/python benchmarks/sharpa_webxr/aggregate.py <run>`. It prints stage rates, the all-three rate with a
  95% Wilson CI, and the gpt_fleet_depth vs simteleop-only split.
- Notebook: `notebooks/_build_gwp_sim_eval_nb.py`, which builds `notebooks/gwp_sim_eval.ipynb`. It covers stage rates,
  rate vs training step, outcome breakdown, a per-scene heatmap, IK/fallen/episode length, keyframes and embedded videos.
- Video page: `build_review_page.py <run> --tags <tag...> --out <dir>`. It writes one section per scene with
  checkpoints side by side.
- Per rollout: `.sharpa_sim_eval/<run>/<tag>/<scene>/seed_k/`, containing `rollout.json`, `trajectory.npz`,
  `signals.json`, `keyframes/` and `video.mp4`.
