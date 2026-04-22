# RoboTwin Benchmark Evaluation

These scripts assume the OpenWAM policy server is **already running**. They only cover the RoboTwin side of the evaluation loop.

## README TODOs

- [x] Update per-task `limit_steps` based on observed episode lengths.
- [ ] Confirm that `state` is correctly forwarded to the server: check whether [`policy_config.yml`](policy_config.yml)'s `send_state` flag still works, and decide whether to keep the switch or always send `state`.
- [ ] Flesh out the debug-mode docs: spell out what gets saved, where, and under what names, so the user experience stays friendly.
- [ ] Investigate the timestamp-folder mismatch in RoboTwin's built-in `eval_results/` directory and see whether it can be fixed.
- [ ] Document `multi_eval.sh`'s `-n <name>` flag (what output path it produces); if `-n` is not strictly required, consider removing it.
- [ ] Clean up [`policy_config.yml`](policy_config.yml) — drop parameters that no longer have an effect.
- [x] Update [`README.md`](README.md): remove the per-task step-analysis section (and the scripts it references). Keep the resolved `limit_steps` numbers inline so users don't have to rerun the analysis.

## Files

| File | Description |
|---|---|
| `openwam2robotwin_interface.py` | RoboTwin client — calls the HTTP server's `/predict` and `/reset` endpoints. |
| `policy_config.yml` | Config template; `host` / `http_port` are injected at runtime. |
| `single_eval.sh` | Run evaluation on a single task. |
| `multi_eval.sh` | Run evaluation on multiple tasks sequentially. |
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

### 3. Start the OpenWAM server

Start the server separately before running any evaluation (it can live on a remote machine — just make sure the host/IP and the HTTP port are reachable):

```bash
bash scripts/deploy.sh --ckpt-dir /path/to/ckpt_dir --http-port XXXX
```

## Usage

### Single-task evaluation

**Invocation:**

```bash
bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [http_port] [host]
```

| Argument | Description |
|---|---|
| `task_name` | RoboTwin task name (e.g. `adjust_bottle`). |
| `task_config` | `demo_clean` or `demo_randomized`. |
| `ckpt_setting` | Label written into result filenames (e.g. `openwam`). |
| `gpu_id` | CUDA device for the RoboTwin simulator. |
| `http_port` | OpenWAM HTTP port (default: `8848`). |
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
| `-d`, `--ckpt-dir` | OpenWAM checkpoint directory. |

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | OpenWAM server address. |
| `--http-port` | `8848` | OpenWAM HTTP port. |
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
    --host 192.0.2.1 --http-port 8768 all

# Read the task list from a file (one task per line, `#` comments supported)
bash multi_eval.sh -m demo_clean -n run1 -d /path/to/ckpt_dir tasks.txt
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
send_state: true          # Include the joint-state vector in the /predict request.
request_timeout: 300      # HTTP timeout in seconds.

# Action settings (must match the `action_mode` used at training time).
action_type: ee           # ee   — EEF mode (action_mode: eef at train time, default).
                          #        Server returns 20D (xyz + rot6d + grip) × 2,
                          #        auto-converted to 16D (xyz + quat + grip) × 2 before dispatch.
                          # qpos — Joint-angle mode (action_mode: joint at train time).
                          #        Server returns 14D, passed straight to take_action.

# action_indices: null    # Optional index reordering for the returned action vector (null = no reorder).
```
