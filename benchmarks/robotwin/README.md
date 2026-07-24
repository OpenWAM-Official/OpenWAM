# RoboTwin Benchmark Evaluation

These scripts assume the OpenWAM policy server is **already running**. They only cover the RoboTwin side of the evaluation loop.

## README TODOs

- [x] Update per-task `limit_steps` based on observed episode lengths.
- [x] Confirm that `state` is correctly forwarded to the server: [`policy_config.yml`](policy_config.yml)'s `send_state` flag is active, and the default is `true` for proprio-conditioned checkpoints.
- [x] Add state-dimension fail-fast checks so RoboTwin eval errors immediately when `action_type` / `state_dim` do not match the checkpoint.
- [ ] Flesh out the debug-mode docs: spell out what gets saved, where, and under what names, so the user experience stays friendly.
- [ ] Investigate the timestamp-folder mismatch in RoboTwin's built-in `eval_results/` directory and see whether it can be fixed.
- [ ] Document `multi_eval.sh`'s `-n <name>` flag (what output path it produces); if `-n` is not strictly required, consider removing it.
- [ ] Clean up [`policy_config.yml`](policy_config.yml) — drop parameters that no longer have an effect.
- [x] Update [`README.md`](README.md): remove the per-task step-analysis section (and the scripts it references). Keep the resolved `limit_steps` numbers inline so users don't have to rerun the analysis.

## Files

| File | Description |
|---|---|
| `openwam2robotwin_interface.py` | RoboTwin client — talks to the WebSocket server. |
| `policy_config.yml` | Config template; `host` / `port` are injected at runtime. |
| `single_eval.sh` | Run evaluation on a single task (whole-task, RoboTwin's native loop). |
| `multi_eval.sh` | Run evaluation on multiple tasks sequentially. |
| `dispatcher.py` | Central **episode-level** scheduler (TCP): per-`(task,mode)` seed allocator + dynamic env-affinity/duplication + result aggregation. Has a `--self-test` fleet simulation and an optional live status HTTP endpoint. |
| `episode_worker.py` | One worker process = one `(task,mode)` assignment: boots the env once, then streams that task's episodes from the dispatcher. `--dry-run` simulates episodes with no RoboTwin. |
| `episode_eval.sh` | Env-setup shim (EGL/PYTHONPATH/CUDA) that execs one `episode_worker.py`. |
| `parallel_eval.sh` | Single-machine multi-GPU eval against already-running servers, using episode-level dynamic scheduling (dispatcher + one supervisor loop per GPU). |
| `dlc_parallel_eval.sh` | DLC multi-node entrypoint; rank 0 runs the dispatcher, every node starts local servers + slot supervisors that all claim episodes from it. |
| `dlc_web_console.py` | Compatibility wrapper for the unified benchmark web control dashboard. |
| `export_results_csv.py` | Export to CSV. Prefers the dispatcher's `results.jsonl` (per-episode); falls back to `summary.tsv` + per-task `Success rate` grepping for legacy runs. |
| `step_limits.yml` | Per-task `step_lim` overrides (see below). |

## Episode-level dynamic scheduling

`parallel_eval.sh` and `dlc_parallel_eval.sh` schedule a **single episode** as
the unit of work, not a whole task. This eliminates the tail-idle waste of the
old whole-task queue: when there are more free GPUs than unstarted tasks, idle
GPUs join an in-progress task (spawn a duplicate env) to drain its remaining
episodes in parallel, so no GPU sits idle while any episode remains.

How it works:

- A central **dispatcher** (`dispatcher.py`; TCP, rank 0 in DLC) owns, per
  `(task, mode)`, a monotonic **seed allocator** and the episode counters. Every
  raw seed is handed out at most once globally, so no scene is ever evaluated
  twice (RoboTwin scenes are fully determined by their integer seed).
- Each GPU **slot** runs a supervisor loop that keeps launching
  `episode_worker.py`. A worker claims one `(task,mode)`, boots its RoboTwin env
  **once**, and streams that task's episodes: `request_seed` → expert-check →
  (valid) `request_commit` → policy rollout → `report_result`. The commit
  handshake makes each job land on **exactly `--test-num` episodes** (no
  overshoot). When the dispatcher drains the job, the worker exits and the
  supervisor launches a fresh one for the next assignment.
- **Duplication policy**: an idle slot first starts any unstarted task; when
  none remain it joins the in-progress task with the longest ETA, but **only if**
  that task still has at least `--min-remaining-for-dup` (θ) episodes left and is
  under its env cap (`ceil(remaining/θ)`). This avoids booting an env that the
  existing env(s) would finish before the new one is even ready.

New flags (both scripts):

| Flag | Default | Description |
|---|---:|---|
| `--test-num` | `100` | Episodes per `(task,mode)`. |
| `--seed` | `0` | Base seed; `st_seed = 100000*(1+seed)`, matching RoboTwin. |
| `--min-remaining-for-dup` | `8` | θ: don't spawn a new env for a task with fewer remaining episodes. |
| `--no-dup` | off | Strict mode: exactly one env per task, never duplicate (most reproducible; equivalent to the old whole-task granularity per job). |
| `--dispatch-port` | `8790` | Dispatcher TCP port. |
| `--http-port` | `0` | Serve a live status page (`/` HTML, `/api/state` JSON); `0` = off. |

Liveness / watchdog (the dispatcher never hangs silently):

- **Stall abort** — if no worker sends any request for `--stall-timeout` seconds
  (default 1800), or zero workers are connected for `--idle-grace` seconds
  (default 120) while jobs remain, the dispatcher writes an `incomplete`
  `summary.tsv` and exits non-zero (2) instead of self-spinning forever (covers
  every slot retiring, or all workers going silent).
- **Hung-worker reclaim** — a worker whose socket is still open but silent for
  `--worker-timeout` seconds (default 1200; must exceed one rollout) has its
  in-flight seed returned and env slot freed so others finish the job. A late
  report from a revived worker is ignored (no double count).
- **Give-up cap** — a task whose expert-check almost never passes stops after
  `target × --max-attempt-factor` seed attempts (default 50; `0` = unlimited),
  is marked `exhausted` in `summary.tsv`, and the run still terminates (exit 3).
- On exit (complete / exhausted / stall) the dispatcher touches a shared
  `.done` file; each node's launcher reaps its local (even wedged) workers on
  that signal, so no node's `wait` hangs on a stuck sim process.

Determinism note: the first `test_num` valid episodes of each task use the same
scenes as an upstream single-process run (seeds are handed out in order and a
seed's validity is policy-independent); only the assignment of episodes to GPUs
is non-deterministic. Use `--no-dup` for the most reproducible, single-stream
behavior.

Outputs land in the log directory: `results.jsonl` (one line per completed
episode — the authoritative record), `summary.tsv` (per-`(task,mode)` success
rate), `state.json` (live snapshot, rewritten atomically as the run progresses),
and `run.env` (parameters). Turn them into a CSV with `export_results_csv.py`.

Live monitoring:

- **`web_control.py`** (recommended, especially for DLC) reads `state.json` /
  `results.jsonl` straight from the shared log directory — no network path to the
  compute nodes needed:
  `python benchmarks/web_control.py <log_dir> --benchmark robotwin --port 8765`.
- **`--http-port`** on the dispatcher serves the same live data directly, but is
  **off by default** and only useful when you can reach the dispatcher host
  (single-machine `parallel_eval.sh`); on DLC the rank-0 port is usually not
  routable, so prefer `web_control.py` on the shared FS.

### Dry-run (no GPUs / no RoboTwin)

Both the scheduling logic and the whole orchestration can be exercised without a
simulator:

```bash
# Fleet simulation: models env-boot cost + episode time, prints dup-vs-no-dup
# makespan/utilization and asserts exact counts + zero duplicate seeds.
python benchmarks/robotwin/dispatcher.py --self-test

# End-to-end orchestration smoke (dispatcher + supervisors + fake workers):
NNODES=1 NODE_RANK=0 ROBOTWIN_RUN_ID=smoke ROBOTWIN_LOG_ROOT=/tmp/rt \
DISPATCHER_ADVERTISE_HOST=127.0.0.1 \
bash benchmarks/robotwin/dlc_parallel_eval.sh --dry-run \
    -m demo_clean -n smoke -d /unused -w 4 --test-num 8 \
    adjust_bottle open_laptop lift_pot
```

## Per-task step_lim overrides

RoboTwin ships upstream per-task step limits in `task_config/_eval_step_limit.yml`. To tweak them without patching the RoboTwin source tree, edit [`step_limits.yml`](step_limits.yml) in this directory:

```yaml
# step_limits.yml (values here match what is checked in)
adjust_bottle: 160
open_laptop: 288
put_bottles_dustbin: 640
```

Semantics:

- Any task listed here overrides RoboTwin's upstream value for that task.
- Tasks not listed keep RoboTwin's original value (which itself falls back to `1000` when the upstream file also lacks the task).
- The file is loaded once at adapter import and applied at the first step of each episode via `TASK_ENV.step_lim = <override>`. Edits to the YAML only take effect in a fresh eval process — restart the evaluation after tweaking values.
- Leaving the file empty (comments only) reproduces stock RoboTwin behavior.

## Environment Setup

### 1. Install the RoboTwin environment

Follow the [official RoboTwin installation guide](https://github.com/RoboTwin-Platform/RoboTwin) to clone the repo, create the Conda environment, install dependencies, and download assets. When you're done you should have a working RoboTwin Conda environment (default name `robotwin`) and a local checkout of the RoboTwin repository.

### 2. Set environment variables

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin      # RoboTwin repo root (required)

export ROBOTWIN_ENV=robotwin                # RoboTwin Conda env name (default: robotwin)
```

### 3. Match the checkpoint action/state mode

`policy_config.yml` must match the checkpoint's saved `config.yaml`:

- `dataloader.action_mode: eef` / `architecture.state_dim: 20` → keep `action_type: ee`, `state_dim: 20`.
- `dataloader.action_mode: joint` / `architecture.state_dim: 14` → set `action_type: qpos`, `state_dim: 14`.
- Keep `send_state: true` for any checkpoint with `architecture.use_proprioception: true`; the adapter will fail fast if the extracted RoboTwin state dimension is wrong.

### 4. Start the OpenWAM server

Start the server separately before running any evaluation (it can live on a remote machine — just make sure the host/IP and the port are reachable):

```bash
bash scripts/deploy.sh --ckpt-dir /path/to/ckpt_dir --port XXXX
```

## Usage

### Single-task evaluation

**Invocation:**

```bash
bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [port] [host]
```

| Argument | Description |
|---|---|
| `task_name` | RoboTwin task name (e.g. `adjust_bottle`). |
| `task_config` | `demo_clean` or `demo_randomized`. |
| `ckpt_setting` | Label written into result filenames (e.g. `openwam`). |
| `gpu_id` | CUDA device for the RoboTwin simulator. |
| `port` | OpenWAM server port (default: `8848`). |
| `host` | OpenWAM server address (default: `127.0.0.1`). |

**Example:**

```bash
bash single_eval.sh adjust_bottle demo_clean openwam 0 8848 127.0.0.1
```


### Multi-task evaluation

```bash
bash multi_eval.sh -m <mode> -n <name> -d <ckpt_dir> [options] <tasks...>
```

**Required flags:**

| Flag | Description |
|---|---|
| `-m`, `--mode` | `demo_clean` or `demo_randomized`. |
| `-n`, `--name` | Label used for the log directory. |
| `-d`, `--ckpt-dir` | OpenWAM checkpoint directory used for log placement and run labeling; the evaluator still talks to an already-running server and does not load weights. |

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | OpenWAM server address. |
| `--port` | `8848` | OpenWAM server port. |
| `-g`, `--gpu` | `0` | CUDA device for the RoboTwin simulator. |

**Examples:**

```bash
# Run two named tasks
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpt_dir \
    adjust_bottle open_laptop

# Run all 50 RoboTwin 2.0 tasks
bash multi_eval.sh -m demo_randomized -n run1 -d /path/to/ckpt_dir all

# Point at a remote server
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpt_dir \
    --host 192.0.2.1 --port 8768 all

# Read the task list from a file (one task per line, `#` comments supported)
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpt_dir tasks.txt
```

### DLC multi-node parallel evaluation

> **Scheduling model updated.** `dlc_parallel_eval.sh` now uses the
> **episode-level dispatcher** described in "Episode-level dynamic scheduling"
> above: rank 0 runs a TCP dispatcher and publishes its address to
> `<log_dir>/.dispatcher_addr`; every node starts local servers + slot
> supervisors that claim *episodes* (not whole tasks) from it. The old
> `queue/pending` + atomic-`mv` per-`task|mode` claim protocol and node-done
> sentinels are **gone** — scheduling, seed dedup, and aggregation all live in
> the dispatcher. Some paragraphs below still describe the old file-queue layout
> for historical reference; the authoritative behavior and flags are in the
> section above. Results are in `results.jsonl` / `summary.tsv` / `state.json`.

`dlc_parallel_eval.sh` is the cluster entrypoint for large runs. Launch the same command on every DLC worker. Each node starts one OpenWAM server per local worker, waits for the server to accept connections, then starts RoboTwin slot supervisors that claim episodes from the rank-0 dispatcher.

This script does **not** require pre-starting OpenWAM servers with `scripts/deploy_multi.sh`; it starts and cleans up its own local servers on every node. It still requires a RoboTwin Python environment for the simulator/client process.

By default, each local policy server is launched as:

```bash
python <repo>/scripts/deploy.py ...
```

Override `SERVER_PYTHON` / `--server-python` or `SERVER_SCRIPT` / `--server-script` if the server must run under a specific Python executable or a custom deploy script. `SERVER_PYTHON` must be an OpenWAM-capable environment with packages such as `torch`, `omegaconf`, `safetensors`, and `websockets`; it is separate from `ROBOTWIN_PYTHON`, which runs the simulator/client side.

**Required environment:**

| Variable | Description |
|---|---|
| `ROBOTWIN_PATH` | RoboTwin repository root. Must be visible on every node. |
| `ROBOTWIN_PYTHON` | Python executable inside the RoboTwin environment. If unset, the script searches for `ROBOTWIN_ENV` as a conda env. |
| `ROBOTWIN_ENV` | Conda env name used only when `ROBOTWIN_PYTHON` is unset. Default: `robotwin`. |
| `SERVER_PYTHON` | Python executable for OpenWAM policy servers. Use the OpenWAM env unless the RoboTwin env also has the OpenWAM server dependencies. |

**DLC / multi-node environment:**

The script prefers DLC variables when present:

| Variable | Description |
|---|---|
| `MLP_WORKER_NUM` | Total node count. |
| `MLP_ROLE_INDEX` | Current node rank. |

It falls back to `NNODES` / `NODE_RANK`, then `WORLD_SIZE` / `RANK`, so local smoke tests can be run manually.

**Invocation:**

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/envs/RoboTwin/bin/python \
SERVER_PYTHON=/path/to/openwam/.venv/bin/python \
ROBOTWIN_RUN_ID=test \
bash benchmarks/robotwin/dlc_parallel_eval.sh \
    -m all -n openwam -d /path/to/openwam_checkpoints/robotwin_dual_system_joint_self_attention \
    --denoise-steps 10 \
    all
```

**Required flags:**

| Flag | Description |
|---|---|
| `-m`, `--mode` | `demo_clean`, `demo_randomized`, or `all`. `all` expands to both modes. |
| `-n`, `--name` | Run label used in log directory names and RoboTwin result naming. |
| `-d`, `--ckpt-dir` | OpenWAM checkpoint directory used by each local policy server. |

Tasks are positional after flags. They can be task names, comma-separated task names, `all`, or a task-list file with one task per line.

**Useful overrides:**

| Variable / flag | Default | Description |
|---|---|---|
| `ROBOTWIN_RUN_ID` | `latest` | Shared run id used in the log path; change it for reruns. |
| `ROBOTWIN_LOG_ROOT` | `<ckpt_dir>/robotwin_eval_logs` | Shared filesystem root for queue, sentinels, and logs. |
| `-w`, `--num-workers` | GPU count | Number of local OpenWAM servers and RoboTwin clients per node. |
| `--gpu-start` | `0` | First local GPU index. |
| `SIM_GPU_STRIDE` | `1` | Stride between worker GPUs. |
| `PORT_BASE`, `--port` | `8848` | Per-node local port base; worker `i` uses base `+ i`. |
| `SERVER_PYTHON`, `--server-python` | `python` | Python executable used to launch each local policy server; must have OpenWAM server dependencies installed. |
| `SERVER_SCRIPT`, `--server-script` | `<repo>/scripts/deploy.py` | Python script used to launch each local policy server. |
| `--bind-host` | `127.0.0.1` | Host passed to `scripts/deploy.py --host`. |
| `--client-host` | `127.0.0.1` | Host passed to RoboTwin clients. Keep this local unless clients must reach a non-local server. |
| `--ckpt-name` | latest checkpoint | Specific checkpoint filename passed to `scripts/deploy.py`. |
| `--denoise-steps` | config default | Denoising step count passed to `scripts/deploy.py`. |
| `--schedule-type` | config default | Schedule type passed to `scripts/deploy.py`. |
| `--shift` | config default | Flow-matching shift passed to `scripts/deploy.py`. |
| `--dry-run` | off | Skip OpenWAM/RoboTwin startup and only exercise DLC/shared-filesystem task assignment. |
| `--fresh` | off | Remove stale queue/sentinel metadata for the same run id before rank 0 initializes the queue. |

**Shared log directory:**

By default:

```text
<ckpt_dir>/robotwin_eval_logs/<name>_<mode>_dlc_<run_id>
```

If `ROBOTWIN_LOG_ROOT` is set:

```text
<ROBOTWIN_LOG_ROOT>/<name>_<mode>_dlc_<run_id>
```

This directory must be on a shared filesystem visible to every node because it stores the queue, sentinels, summary, and logs. The task queue is represented as per-job files under `queue/pending`; workers claim jobs with an atomic same-filesystem `mv` into `queue/claimed`, avoiding concurrent edits to a shared `.queue.txt`. Summary locking still uses an atomic `mkdir` lock directory (`summary.lock.d/`) instead of `flock`, which is safer on many DLC/NFS-style shared filesystems.

**Dry-run task assignment test:**

Use `--dry-run` to validate that a DLC launch can coordinate all nodes and automatically distribute jobs before spending GPU time on policy servers or RoboTwin simulators. Dry-run mode still creates the shared queue, waits for all expected node ranks to rendezvous, starts the requested number of local worker loops per node, atomically claims jobs, writes `summary.tsv`, and runs rank-0 completeness checks; it does not require `ROBOTWIN_PATH`, `ROBOTWIN_PYTHON`, `SERVER_PYTHON`, or a real checkpoint directory.

```bash
ROBOTWIN_LOG_ROOT=/shared/path/robotwin_eval_logs \
ROBOTWIN_RUN_ID=dryrun_$(date +%Y%m%d_%H%M%S) \
bash benchmarks/robotwin/dlc_parallel_eval.sh \
    --dry-run \
    -m all -n dryrun -d /unused/ckpt_dir \
    -w 8 \
    adjust_bottle open_laptop
```

Useful dry-run knobs:

| Variable | Default | Description |
|---|---:|---|
| `DRY_RUN_SLEEP_SEC` | `1` | Simulated duration for each claimed job, useful for observing load balancing. |
| `DRY_RUN_BARRIER_TIMEOUT_SEC` | `QUEUE_READY_TIMEOUT_SEC` | Max time rank 0 waits for every DLC node before workers start claiming jobs. |

Important files:

```text
<log_dir>/
  run.env
  summary.tsv
  .queue.txt       # manifest/debug copy
  queue/
    pending/
    claimed/
  summary.lock.d/
  .queue_ready
  node0/
    servers/
      server_worker0_gpu0.log
      server_worker1_gpu1.log
    worker0/
      worker.log
      adjust_bottle_demo_clean.log
  node1/
    ...
```

Useful commands:

```bash
# Server log for node0 worker0
tail -f <log_dir>/node0/servers/server_worker0_gpu0.log

# Client scheduling log for node0 worker0
tail -f <log_dir>/node0/worker0/worker.log

# Per-task RoboTwin eval log
tail -f <log_dir>/node0/worker0/adjust_bottle_demo_clean.log

# Task-level status table
column -t -s $'\t' < <log_dir>/summary.tsv
```

**Real-time web control console:**

Instead of tailing several files by hand, start the dependency-free benchmark
dashboard against the shared log directory:

```bash
python benchmarks/web_control.py <log_dir> \
    --benchmark robotwin \
    --host 0.0.0.0 \
    --port 8765
```

The legacy RoboTwin command, `python benchmarks/robotwin/dlc_web_console.py
<log_dir>`, remains available and forwards to the same implementation.

Open `http://<node-ip>:8765/` to watch queue progress, task status,
success-rate parsing, consistency checks, failed-task snippets,
node/worker/server logs, and a live tail pane. The dashboard also exposes
`/api/state`, `/api/tail?file=<relative-log-path>`, raw log links, and a CSV
download at `/api/results.csv`. Use `--host 127.0.0.1` for local-only access.

Useful console knobs:

| Option | Default | Description |
|---|---:|---|
| `--tail-bytes` | `200000` | Initial bytes returned by the tail pane. |
| `--state-tail-bytes` | `256000` | Bytes scanned per task log for success-rate parsing. |
| `--max-logs` | `2000` | Maximum log-like files shown in the log browser. |
| `--max-task-log-bytes` | `4000000` | Bytes scanned per failed task for error snippets. |

**Single-node smoke example:**

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/conda/envs/robotwin/bin/python \
NNODES=1 NODE_RANK=0 ROBOTWIN_RUN_ID=smoke \
bash benchmarks/robotwin/dlc_parallel_eval.sh \
    -m demo_clean -n smoke -d /path/to/ckpt_dir \
    -w 1 \
    adjust_bottle
```

**Custom server Python/script example:**

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/conda/envs/robotwin/bin/python \
SERVER_PYTHON=/path/to/server/env/bin/python \
SERVER_SCRIPT=/path/to/custom_deploy.py \
NNODES=1 NODE_RANK=0 ROBOTWIN_RUN_ID=custom_server \
bash benchmarks/robotwin/dlc_parallel_eval.sh \
    -m demo_clean -n custom_server -d /path/to/ckpt_dir \
    -w 1 \
    adjust_bottle
```

### Export evaluation results to CSV

Use `export_results_csv.py` after a DLC run to combine `<log_dir>/summary.tsv` and each per-task log's last `Success rate` line into a single CSV.

**Invocation:**

```bash
python benchmarks/robotwin/export_results_csv.py \
    /path/to/log_dir \
    -o /path/to/log_dir/results.csv
```

If `-o` is omitted, the default output is:

```text
<log_dir>/results.csv
```

The CSV columns are:

| Column | Description |
|---|---|
| `run_id` | `ROBOTWIN_RUN_ID` recorded by `dlc_parallel_eval.sh`. |
| `policy_name` | Value passed with `-n`, `--name`. |
| `requested_mode` | Original `-m`, `--mode` value. |
| `task` | RoboTwin task name. |
| `mode` | Concrete task config: `demo_clean` or `demo_randomized`. |
| `node` | Node rank that ran the job. |
| `worker` | Local worker index that ran the job. |
| `status` | `ok` or `failed` from `summary.tsv`. |
| `exit_code` | Task process exit code. |
| `success_rate` | Parsed numeric success rate from the task log; blank if not found. |
| `episodes` | Number of `Success!` / `Fail!` verdicts parsed from the task log; `0` if the log is readable but has no verdicts yet; blank if the log is missing/unreadable. |
| `step_limit_hits` | `Fail!` episodes truncated at `step_lim` (last `step: N / M` had `N >= M`). These ran out of steps rather than the model reaching a terminal state, so they are **not necessarily model errors** — a high count means `step_lim` may be too tight for this policy (see `step_limits.yml`), not that the model is worse. Same blank-vs-`0` convention as `episodes`. |
| `log_path` | Full path to the per-task log used for parsing. |

The exporter also validates run completeness when `run.env` is available: duplicate `task/mode` rows, missing rows, unexpected rows, or a row-count mismatch are reported. It searches nested `node*/worker*/*.log` files when `summary.tsv` is absent and strips ANSI escape sequences before parsing `Success rate`.

Strict parsing mode returns a non-zero exit code if any task log is missing, does not contain a parseable success rate, or fails the completeness checks above:

```bash
python benchmarks/robotwin/export_results_csv.py /path/to/log_dir --strict
```

## FAQ

### Render Error (headless servers)

On a headless Linux box the SAPIEN renderer fails to find an X display and raises `Render Error`. Start a virtual framebuffer (Xvfb):

```bash
sudo apt-get install -y xvfb   # if not installed yet
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
bash single_eval.sh adjust_bottle demo_clean openwam 0 8848 127.0.0.1
```

Or, in one step, use `xvfb-run`:

```bash
xvfb-run -a bash single_eval.sh adjust_bottle demo_clean openwam 0 8848 127.0.0.1
```

---

### `policy_config.yml` options

```yaml
# Observation settings
# The client always forwards RoboTwin's head / left / right cameras to the server.
# The server inspects the checkpoint's saved config.yaml to decide:
#   - multiview=false → single-view preprocessing using head_camera only
#   - multiview=true  → composed into the L-shape multi-view layout used at training time
# The client no longer needs to configure camera selection or resolution.
send_state: true          # Include the proprio state vector in the obs message.
state_dim: 20             # Fail-fast expected dim. 20 for eef/ee, 14 for joint/qpos.
request_timeout: 300      # WebSocket timeout in seconds.

# Action settings (must match the `action_mode` used at training time).
action_type: ee           # ee   — EEF mode (action_mode: eef at train time, default).
                          #        Server returns 20D (xyz + rot6d + grip) × 2,
                          #        auto-converted to 16D (xyz + quat + grip) × 2 before dispatch.
                          # qpos — Joint-angle mode (action_mode: joint at train time).
                          #        Server returns 14D, passed straight to take_action.

# action_indices: null    # Optional index reordering for the returned action vector (null = no reorder).
```
