# EBench (GenManip) Evaluation

Two processes on the OpenWAM side: the **policy server** (this repo's env) and a thin **bridge client**. The bridge polls the GenManip eval server over HTTP (north) and queries the policy server over WebSocket (south, [wire protocol](../README.md)); the GenManip simulation itself belongs to the [EBench](https://github.com/InternRobotics/EBench) repo and may run on a different machine.

Commands below assume the bridge env's python at `/path/to/bridge-env/bin/python` — substitute your actual paths.

## 1. Environment Setup

**Sim server** (per the EBench repo, possibly another machine): follow https://github.com/InternRobotics/EBench — Isaac Sim 4.1.0 + cuRobo, pre-Blackwell GPU, ~11.4 GB EBench-Assets.

**Bridge env** (what OpenWAM's launch scripts run — no torch, no OpenWAM install):

```bash
git clone https://github.com/InternRobotics/EBench && cd EBench
pip install -e genmanip-client              # the only EBench piece the bridge imports
pip install numpy Pillow "websockets>=15"
```

Point `EBENCH_PYTHON` at this env's python. For a no-Isaac sanity check see the mock flow in section 3.

## 2. Start the Policy Server

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-EBench
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench` (or use a checkpoint you trained yourself). Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench
```

WebSocket port 8848 by default. For N parallel workers: `NUM_GPUS=N bash scripts/deploy.sh <ckpt_dir>` → one server per GPU on ports 8848…8848+N-1.

## 3. Run the Evaluation

On the sim machine (EBench tooling): `python ray_eval_server.py --host 0.0.0.0 --port 8087 --no_save_process`, then `gmp submit ebench/generalist/val_train --run_id <run_id>`. Single worker from this repo:

```bash
EBENCH_PYTHON=/path/to/bridge-env/bin/python \
bash benchmarks/ebench/single_eval.sh \
    --url http://<sim-host>:8087 --run-id <run_id> \
    --ckpt-config assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench/config.yaml
```

Parallel run — one policy server per worker:

```bash
NUM_GPUS=4 bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench
NUM_WORKERS=4 EBENCH_PYTHON=/path/to/bridge-env/bin/python \
bash benchmarks/ebench/multi_eval.sh \
    --url http://<sim-host>:8087 --run-id <run_id> \
    --ckpt-config assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-EBench/config.yaml
```

Offline sanity check without Isaac Sim — the mock replays real EBench-Dataset episodes over the exact wire format (verifies bridge + conversion; produces no scores):

```bash
python scripts/download_assets/download_benchmark_data.py    # menu: EBench
python benchmarks/ebench/mock_genmanip_server.py \
    --dataset-dir /path/to/EBench-Dataset --bucket simple_pnp/task1 \
    --episodes 2 --steps-per-episode 8 --port 8087
# in another shell: single_eval.sh as above with --url http://127.0.0.1:8087
```

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Always pass `--ckpt-config` when the checkpoint dir is reachable: it hard-verifies the contract (`dataloader.type=ebench`, `action_mode=eef`, `unify_action=true`); without it you only get an UNVERIFIED warning.
- One policy server per worker (the server-side executor is stateful); `multi_eval.sh` maps worker `i` → south port `SOUTH_PORT_BASE+i` (default 8848+i), matching `deploy.sh`'s `PORT_BASE+i`.
- The bridge waits up to 300 s for the policy server's first ping (compile warmup) — not a hang. `--no-send-state` only for non-proprio checkpoints.
- Arms are absolute EE poses; GenManip solves IK server-side and silently holds joints on IK failure — validate in local sim before online submissions.
- Official online eval: `gmp online submit --base_url https://internrobotics.shlab.org.cn/eval --token $TOK --benchmark_set ebench_generalist` → endpoint + `task_id`; then `single_eval.sh --url "$ENDPOINT" --token "$TOK" --run-id "$TASK_ID"`. ≤16 workers, 10-min inactivity disconnect (warm up first); failed runs resume with the same `task_id`.
- Budget: the generalist split is 794 instances (~1.8 M sim steps) — plan hours and use parallel workers. Scores are stored by the GenManip server under your `run_id`.

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA.

| Method | Type | TableTop SR | TableTop Score | PnP SR | PnP Score | LongHorizon SR | LongHorizon Score | Overall SR | Overall Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| StarVLA-OFT | VLA | - | - | - | - | - | - | 0.0 | 0.2 |
| π₀ | VLA | 15.7 | 30.0 | 35.0 | 39.0 | 17.0 | 41.0 | 23.6 | 37.0 |
| X-VLA | VLA | 8.6 | 24.0 | 50.0 | 54.0 | 6.2 | 25.0 | 23.7 | 36.0 |
| InternVLA-A1 | VLA | 4.3 | 11.0 | 43.0 | 47.0 | 17.9 | 46.0 | 23.9 | 36.0 |
| π₀.₅ | VLA | 12.9 | 32.0 | 45.0 | 50.0 | 18.1 | 39.0 | 27.1 | 41.0 |
| GigaBrain-0.7 | VLA | - | - | - | - | - | - | 33.3 | 46.0 |
| Qwen-RobotManip | VLA | **50.0** | **70.0** | <u>56.5</u> | <u>60.0</u> | <u>29.9</u> | <u>55.0</u> | <u>45.6</u> | <u>60.0</u> |
| Fast-WAM | WAM | - | - | - | - | - | - | 4.7 | 7.6 |
| **OpenWAM-α** | WAM | <u>30.0</u> | <u>44.2</u> | **67.5** | **72.0** | **44.3** | **72.6** | **49.4** | **64.7** |
