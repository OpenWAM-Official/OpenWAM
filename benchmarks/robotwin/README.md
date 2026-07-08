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
| `single_eval.sh` | Run evaluation on a single task. |
| `multi_eval.sh` | Run evaluation on multiple tasks sequentially. |
| `parallel_eval.sh` | Run a shared local queue against already-running local/remote OpenWAM servers. |
| `dlc_parallel_eval.sh` | DLC multi-node entrypoint; starts local OpenWAM servers and RoboTwin clients on every node, then uses a shared queue for cross-node parallel evaluation. |
| `dlc_web_console.py` | Compatibility wrapper for the unified benchmark web control dashboard. |
| `export_results_csv.py` | Export `summary.tsv` plus per-task `Success rate` lines into a CSV file. |
| `step_limits.yml` | Per-task `step_lim` overrides (see below). |

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

`dlc_parallel_eval.sh` is the cluster entrypoint for large runs. Launch the same command on every DLC worker. Rank 0 creates one pending job file per `task|mode` in the shared log directory; each node starts one OpenWAM server per local worker, waits for the server to accept connections, then starts RoboTwin client workers that atomically claim pending job files with `mv`.

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
| `episodes` | Number of `Success!` / `Fail!` verdicts parsed from the task log; blank if none. |
| `step_limit_hits` | `Fail!` episodes truncated at `step_lim` (last `step: N / M` had `N >= M`). These ran out of steps rather than the model reaching a terminal state, so they are **not necessarily model errors** — a high count means `step_lim` may be too tight for this policy (see `step_limits.yml`), not that the model is worse. |
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
