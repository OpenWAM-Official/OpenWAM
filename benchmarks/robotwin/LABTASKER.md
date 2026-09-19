# RoboTwin with Labtasker

This lightweight integration gives you more built-in features with less custom
orchestration. Labtasker handles task scheduling, retries and progress tracking,
keeping the evaluation adapter small and the workflow simple: submit Tasks,
start Workers and inspect results. On a single machine with a local-filesystem
checkout, you can explicitly authorize its managed local server on first use.

Compared with the traditional launcher, you get:

- **Better load balancing through finer-grained parallelism:** Workers claim
  small episode batches, reducing idle time when some benchmark tasks take longer.
- **Smaller retry units:** a failed batch is retried independently, limiting
  repeated evaluation to that batch rather than the full benchmark task.
- **Real-time progress monitoring:** view running, completed and failed batches,
  along with evaluation progress, in one place through the Labtasker CLI or API.
- **Agent-assisted control:** an agent can inspect Task errors, cancel Tasks
  or requeue failed work via the Labtasker skill and CLI.

After environment setup, choose the path that matches your situation:

- **First evaluation, no manifest cache:** [build the manifest cache](#first-run), then run evaluation.
- **Subsequent evaluations, cache already built:** [reuse the cache](#run-evaluation)
  and submit evaluation directly, including when testing another checkpoint.

## Setup

Activate your **OpenWAM Python environment** and run commands from the repository
root. Every `python` command in this guide uses that environment: dependency
installation, submission, Workers, Labtasker queries and result summaries. The
policy server also uses this Python by default.

`ROBOTWIN_PYTHON` points to the **separate RoboTwin environment's Python**.
The Worker invokes it automatically to build manifests and run simulator episodes;
do not switch to that environment to start the Worker. Install the benchmark
environment first as described in [README.md](README.md).

```bash
# RoboTwin simulator Python, called automatically by the Worker
export ROBOTWIN_PYTHON=/path/to/robotwin/bin/python
export ROBOTWIN_PATH=/path/to/RoboTwin
# OpenWAM Python (active environment)
python -m pip install -r benchmarks/robotwin/requirements-labtasker.txt
```

**On one machine with the repository on a local disk, you can continue to
the first-run commands below.** Their `--auto-start-local-server` flag starts
Labtasker for you; later commands reconnect automatically. No separate server
command, port or database configuration is needed. The project GPU Docker image
already includes the full Labtasker package needed for this workflow.

For multiple machines, or a checkout on NFS, WekaFS or Lustre, use the setup
below first. Use it also if automatic startup reports an unknown filesystem.

<details>
<summary>Setup for multiple machines or a shared-filesystem checkout</summary>

If your team already runs a Labtasker server, ask for its URL and token and
skip to the client settings below. Otherwise, run this once on one machine:

```bash
# Supply your own secret token.
export LABTASKER_SERVER_TOKEN=...  # do not commit the token
labtasker-server serve --connection http --host 0.0.0.0
```

By default, the server saves its records in `.labtasker/` under the directory
where you start it. Its automatically created `server.db` stores submitted
task inputs, task status and reported results. Keep this directory to retain
your evaluation history.

To choose another location, add just `--labtasker-root /path/to/openwam-eval`;
`server.db` follows that directory automatically. `--database` only overrides
the database file's location separately and is unnecessary for this setup.
Local disks and NFS/shared storage are supported: the server detects the
filesystem automatically. On shared storage, run only one server for that
directory/database, even when several machines can access it. The local-disk
restriction applies to `--auto-start-local-server`, not this manual command.

Workers keep their execution logs and run records under their own
`.labtasker/runs/`; setting the server's `--labtasker-root` does not move remote
Worker logs. Benchmark logs, videos and detailed evaluation results still go
to the output directories described below.

The server listens on port **8000** by default. Keep this command running in
its terminal or under your process supervisor.

In each shell that submits tasks, runs Workers or reads results, set:

```bash
export LABTASKER_URL=http://SERVER_HOST:8000
export LABTASKER_TOKEN=...  # same token as the server
```

Replace `SERVER_HOST` with the server machine's reachable hostname or IP.
Then follow the evaluation steps below, omitting `--auto-start-local-server`.
All Workers must be able to access the submitted checkpoint and manifest paths.

</details>

For background, see the optional
[Labtasker guide and demo video](https://luocfprime.github.io/labtasker/latest/).

> [!TIP]
> To let your coding agent help with Labtasker setup, task submission, progress
> monitoring, troubleshooting and result summaries, install the **Labtasker skill**.
> See the [Agent Skill guide](https://github.com/luocfprime/labtasker/blob/main/docs/guides/agent-skill.md)
> for installation instructions. Describe the evaluation you want to run; the
> agent can help translate it into the required Labtasker commands and settings.

<a id="first-run"></a>

## First run: Build the manifest cache

> [!TIP]
> **Build once per evaluation setup; reuse for every checkpoint.**
> The manifest keeps evaluation inputs fixed across parallel batches and retries. [Learn more](#manifest-details).

### 1. Submit manifest-building Tasks

```bash
python benchmarks/robotwin/labtasker_submit.py --auto-start-local-server \
  --operation build_manifest --mode demo_clean
```

Save the submission ID printed by this command.

### 2. Start Workers on four GPUs

Run these four lines in the same terminal, using OpenWAM's `python`. The trailing
`&` starts each Worker in the background so all four run concurrently. Each Worker
uses the configured `ROBOTWIN_PYTHON` for the simulator. For a single GPU,
run only the first line.

```bash
python benchmarks/robotwin/labtasker_worker.py --operation build_manifest --gpu 0 &
python benchmarks/robotwin/labtasker_worker.py --operation build_manifest --gpu 1 &
python benchmarks/robotwin/labtasker_worker.py --operation build_manifest --gpu 2 &
python benchmarks/robotwin/labtasker_worker.py --operation build_manifest --gpu 3 &
```

### 3. Check completion before evaluating

Check the submission with:

```bash
python -m labtasker task count --filter 'metadata.submission_id == "SUBMISSION_ID" and status != "succeeded"'
```

Replace `SUBMISSION_ID` with the printed ID. Continue when the count reaches zero.

<a id="run-evaluation"></a>

## Evaluate

### 1. Submit evaluation Tasks

Once the cache is ready:

```bash
python benchmarks/robotwin/labtasker_submit.py --auto-start-local-server --mode demo_clean \
  -- --ckpt-dir /path/to/checkpoint
```

Save this new submission ID for progress checks and result summaries.

This uses all tasks in the selected mode, 100 episodes per task, unseen instructions and batches of 5.
The latest checkpoint is selected at submission time. Everything after `--` uses
OpenWAM's native deployment arguments; add overrides there only when needed.

### 2. Start evaluation Workers on four GPUs

Use OpenWAM's `python` again. `run_eval` is the default operation:
Ports default to `8848 + GPU ID`, so these four Workers use ports 8848–8851.

```bash
python benchmarks/robotwin/labtasker_worker.py --gpu 0 &
python benchmarks/robotwin/labtasker_worker.py --gpu 1 &
python benchmarks/robotwin/labtasker_worker.py --gpu 2 &
python benchmarks/robotwin/labtasker_worker.py --gpu 3 &
```

Workers load models automatically and reuse them across batches. For a single
GPU, run only the first line. Start these after manifest construction is complete.

**For another checkpoint, submit evaluation again. No cache rebuild is needed.**
If evaluation Workers are still running, they will claim the new Tasks; otherwise,
start them again with the commands above.

## Results

> [!TIP]
> **For routine operations, use the Labtasker Web UI or ask an agent with the Labtasker skill.**
> Monitor progress, inspect errors and manage Tasks without writing commands yourself.

Use the ID printed by the **eval submission**:

```bash
python -m labtasker task list --filter 'metadata.submission_id == "SUBMISSION_ID"'
python benchmarks/robotwin/labtasker_summarize.py \
  SUBMISSION_ID --output-dir outputs/robotwin/summary
```

The summary reads Labtasker Task records only and writes `task_summary.csv` and
`benchmark_summary.csv`. Check the task counts before using a success rate as a
final score: incomplete runs report partial results. Each successful eval Task
also returns an `artifact_dir` for its detailed files.

## Optional settings

The defaults cover the normal workflow. Use `--help` for the full CLI.

<details>
<summary>Cache, evaluation settings and model options</summary>

The cache defaults to `outputs/manifest-cache/robotwin`. Override it with
`--manifest-dir PATH` on both build and eval submissions only if needed. Workers
receive the selected paths from their Tasks. To rebuild, submit `build_manifest`
with `--rebuild`; otherwise a matching cache with enough entries is reused.
A 100-entry cache can serve a shorter `--episodes` prefix. A missing cache is
built normally by `build_manifest`; an existing invalid or too-short cache
requires `--rebuild`.

Use `--mode demo_randomized` for randomized evaluation; build its cache first.
Use `--instruction-type seen` before `--` to evaluate seen instructions.
Use `--tasks all|TASK1,TASK2` to select tasks, `--episodes N` to request an
ordered prefix, and `--note TEXT` to append a display-name suffix. The official
default remains all 50 tasks and 100 episodes per task; smaller values are for
debug or small-scale evaluation.

Changing a checkpoint or minibatch size does not require rebuilding the cache.
To select a checkpoint explicitly, put `--ckpt-name FILE` after `--`; native
`--config` and dotlist overrides also go there. Submission options go before `--`.

Keep code, environments and checkpoint files fixed for reproducibility. Use
versioned checkpoint paths; restart Workers if you overwrite model files in place.

</details>

<details>
<summary>Multiple machines and Worker resources</summary>

Configure the same `LABTASKER_URL` (and `LABTASKER_TOKEN` if required) on every
machine, with a shared Labtasker server you operate. Manifest and checkpoint paths
must be accessible at the submitted locations on every Worker. Use
`--manifest-dir` when the default repository-local cache is not shared.

The Worker uses `--port` exactly when supplied; it does not probe or silently
switch ports. If a port is occupied, server startup fails. For multiple Workers
on the same GPU, or another service using the default port, choose distinct
ports explicitly with `--port`.

Routes match automatically between submit and Worker for each operation. A
custom `--route NAME` changes Worker compatibility; `--queue NAME` selects a
Labtasker Queue. Pass matching values to the submitter and Worker. Each Worker
owns its policy endpoint; model arguments belong on submission.

Workers exit after 300 seconds idle by default; start them again for later work.
Logs, videos and results default to `outputs/robotwin/<route>`; override with
Worker `--output-dir`. Use `--server-python` only if the policy server needs a
different interpreter from the active OpenWAM environment.

</details>

<details>
<summary>Retries and interrupted submissions</summary>

Failed Tasks retry up to 3 attempts by default. An eval retry repeats only its
whole minibatch; a build retry repeats its benchmark task. After fixing an
exhausted failure, use `python -m labtasker task requeue TASK_ID` and start a Worker.

If submission is interrupted, repeat the same command with
`--submission-id PRINTED_ID` before `--`. Existing Tasks are reused and missing
ones submitted. Reusing that ID with changed parameters is unsupported. For a
fresh evaluation, omit this option to generate a new ID. Native `task list` is paginated; follow
`next_cursor` with `--cursor` when inspecting large submissions.

</details>

<a id="manifest-details"></a>

<details>
<summary>Why build an eval manifest first?</summary>

An `eval_manifest` fixes the ordered evaluation instances and their reproduction
information: accepted seeds and instructions for RoboTwin, or initial-state
indices and reset RNG states for LIBERO. It describes what to evaluate, not
motion planning or Worker scheduling.

`build_manifest` prepares a complete benchmark task without policy inference.
`run_eval` distributes its small batches across Workers, reducing idle time at
the end of evaluation while preserving instance identity across retries and
checkpoint comparisons. The stages are started separately by the user.

`manifest_hash` identifies the concrete instances. Rebuilding the cache does not
modify manifests already referenced by submitted Tasks.

</details>
