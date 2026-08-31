# LIBERO evaluation

This directory is the canonical LIBERO native-action evaluation entry point for
the canonical training snapshot at `/path/to/benchmark_data/libero`.

The OpenWAM server is expected to return raw 10-D model actions:

```text
[native_delta_xyz3, rot6d(Exp(native_delta_axis_angle3)), gripper_open_command]
```

The canonical reader uses separate action/state normalization blocks and keeps
the six rot6d dimensions as an identity mapping during normalization; only
position and gripper dimensions use the configured normalization statistics.

The client sends achieved EEF10 proprioception and converts the response to
LIBERO's runtime 7-D OSC command:

```text
[native_delta_xyz3, native_delta_axis_angle3, native_close_command]
```

Only the rot6d decoding and the gripper sign conversion happen at this
boundary.  There is no absolute-goal composition and no `0.05 m` / `0.5 rad`
controller scaling.  The trained gripper convention is `-1 = closed,
+1 = open`; LIBERO's runtime command is the negation (`+1 = close,
-1 = open`).

## Training

Use the standard dataloader config:

```bash
bash scripts/train.sh dataloader=libero
```

The exact training entry point depends on the local OpenWAM launcher; the
important invariants are `type: libero` and
`action_mode: libero`.

## Evaluation

Start the OpenWAM server from a checkpoint trained with the canonical config,
then run:

```bash
bash benchmarks/libero/single_eval.sh libero_spatial 0 8848 127.0.0.1
```

Or invoke the client directly with a copied YAML config:

```bash
python benchmarks/libero/single_eval.py \
  --config benchmarks/libero/policy_config.yml \
  --suite libero_object --task-id 0
```

Results are written to `result_dir/results.json` when configured.

For a multi-GPU run with dynamic task scheduling, use the standard launcher:

```bash
python benchmarks/libero/run_all_suites.py \
  --task-sample-ratio 0.2 --task-sample-seed 42 \
  --gpus 0,1,2,3,4,5,6,7 --replicas-per-gpu 2 \
  --ckpt-dir /path/to/openwam_checkpoints/new-openwam-libero-sft-10epoch-final \
  --ckpt-name checkpoint_step_10690.safetensors \
  --compile-enabled false
```

The launcher creates one policy server and one queue worker per replica. A worker pulls another suite/task as soon as its previous task exits; failed requests are requeued and the output directory can be resumed safely.

When expanding a completed sampled run to the full task list, pass
`--resume-superset` with a copied output directory.  This preserves valid
results already present in the subset while scheduling only the remaining
tasks; checkpoint, config, seed, trial range, and MuJoCo protocol checks remain
strict.
