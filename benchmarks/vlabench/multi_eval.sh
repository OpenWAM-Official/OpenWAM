#!/usr/bin/env bash
# Fan a VLABench evaluation across tasks / tracks using a worker pool.
#
# ONE SERVER PER WORKER — this is a hard constraint, not a tuning choice.
# The OpenWAM policy server holds a SINGLE receding-horizon policy on the
# server object (`ServerImpl._policy`, see openwam/deploy/server.py), so its
# action-chunk buffer and executor step state are shared by every connected
# client. Point two simulators at one server and each `predict()`
# pops actions generated for the OTHER simulator's observation, while either
# one's `reset()` wipes the buffer for both. Success rates collapse to noise
# with no error surfaced anywhere.
#
# So parallelism == the number of ports you pass. Start one server per port
# BEFORE running this, each on its own GPU, e.g.:
#
#   for i in 0 1 2 3 4 5 6 7; do
#     CUDA_VISIBLE_DEVICES=$i nohup bash scripts/deploy.sh <ckpt_dir> \
#       --ckpt-name checkpoint_step_6000.safetensors \
#       --device cuda:0 --port $((8880+i)) --compile-enabled true &
#   done
#
# VLABench also ships sh/evaluation/example_multi_gpu_eval.sh, but that script
# parses --track/--task into TRACK_OPT/TASK_OPT and then loops over
# ${TRACKS[@]} / ${TASKS[@]}, which are never defined (nor are CKPT and
# job_idx), so its first loop body never runs. This is the working equivalent.
#
# Usage:
#   VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=/path/to/env/bin/python \
#     bash benchmarks/vlabench/multi_eval.sh [tracks] [tasks] [n_episodes] [ports]
#
# Examples:
#   bash benchmarks/vlabench/multi_eval.sh track_1_in_distribution all 50 8880,8881
#   bash benchmarks/vlabench/multi_eval.sh all all 50 8880,8881,8882,8883
#
# `all` for tasks takes each track's OWN task list from its config JSON — the
# tracks do not all cover the same 10 tasks.
#
# Env knobs:
#   GPUS               comma list of GPU ids for the simulators, one per port
#                      (default: 0..N-1 matching the port count)
#   VLABENCH_SAVE_DIR  output root (default benchmarks/vlabench/_eval_out)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_TRACKS=(track_1_in_distribution track_2_cross_category track_3_common_sense
            track_4_semantic_instruction track_5_cross_task track_6_unseen_texture)

# VLABench's README lists six dimensions but ships only five track configs.
# track_5_cross_task deliberately has none — the train/eval task split is the
# user's to choose ("kept open in this setting"), so there is nothing upstream
# could freeze. It therefore runs on seeded episodes (seed=42+i) over a
# held-out task list, which must be supplied here.
#
# The default below is the held-out split for a policy finetuned on VLABench's
# standard 10-task primitive dataset (the tasks of track_1):
#   * every trained family's spatial-grounding sibling — same objects, same
#     motor skill, an instruction family never seen in training;
#   * two entirely unseen object categories.
# select_drink_spatial, select_painting_by_style, insert_bloom_flower,
# replace_wilted_flower and select_billiards_semantic are all excluded: they
# fail to instantiate in VLABench 'main' (PhysicsError / KeyError: 'task' /
# ConfigManager signature mismatch), not by choice.
TRACK5_DEFAULT_TASKS=(
    add_condiment_spatial insert_flower_spatial select_book_spatial
    select_chemistry_tube_spatial select_fruit_spatial select_mahjong_spatial
    select_poker_spatial select_toy_spatial
    select_billiards select_ingredient
)

tracks_arg="${1:-track_1_in_distribution}"
tasks_arg="${2:-all}"
n_episodes="${3:-50}"
ports_arg="${4:-${VLABENCH_PORT:-8880}}"

: "${VLABENCH_PATH:?VLABENCH_PATH must point to the VLABench repo}"
TRACK_DIR="${VLABENCH_ROOT:-${VLABENCH_PATH}/VLABench}/configs/evaluation/tracks"

if [[ "${tracks_arg}" == "all" ]]; then
    tracks=("${ALL_TRACKS[@]}")
else
    IFS=',' read -r -a tracks <<< "${tracks_arg}"
fi
IFS=',' read -r -a ports <<< "${ports_arg}"

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -r -a gpus <<< "${GPUS}"
else
    gpus=(); for i in "${!ports[@]}"; do gpus+=("$i"); done
fi
if (( ${#gpus[@]} != ${#ports[@]} )); then
    echo "[ERROR] GPUS (${#gpus[@]}) must have one entry per port (${#ports[@]})" >&2
    exit 1
fi

save_root="${VLABENCH_SAVE_DIR:-${SCRIPT_DIR}/_eval_out}"
mkdir -p "${save_root}/logs"
queue="${save_root}/.jobqueue"
: > "${queue}"

# Build the job list, taking each track's own task set from its config.
for track in "${tracks[@]}"; do
    cfg="${TRACK_DIR}/${track}.json"
    if [[ "${tasks_arg}" != "all" ]]; then
        IFS=',' read -r -a tt <<< "${tasks_arg}"
    elif [[ -f "${cfg}" ]]; then
        mapfile -t tt < <(python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1])).keys()))" "${cfg}")
    elif [[ "${track}" == "track_5_cross_task" ]]; then
        # No config to enumerate — the open track's split is ours to name.
        IFS=',' read -r -a tt <<< "${VLABENCH_TRACK5_TASKS:-$(IFS=,; echo "${TRACK5_DEFAULT_TASKS[*]}")}"
        echo "[note] ${track} has no upstream episode config; using seeded episodes over" \
             "${#tt[@]} held-out task(s)"
    else
        echo "[ERROR] track config not found: ${cfg}" >&2; exit 1
    fi
    for task in "${tt[@]}"; do
        [[ -n "${task}" ]] && echo "${track} ${task}" >> "${queue}"
    done
done
total=$(wc -l < "${queue}")

echo "tracks    : ${tracks[*]}"
echo "jobs      : ${total}"
echo "episodes  : ${n_episodes}"
echo "workers   : ${#ports[@]}  (one dedicated server each)"
echo "ports     : ${ports[*]}"
echo "gpus      : ${gpus[*]}"
echo "save_root : ${save_root}"
echo

# Worker pool: each worker owns exactly one (gpu, port) pair for its lifetime,
# so a server never serves two simulators at once. Jobs are pulled one at a
# time under a lock.
lock="${save_root}/.queue.lock"
take_job() {
    local line
    exec 9>"${lock}"; flock 9
    line=$(head -1 "${queue}" 2>/dev/null || true)
    [[ -n "${line}" ]] && sed -i '1d' "${queue}"
    flock -u 9; exec 9>&-
    printf '%s' "${line}"
}

worker() {
    local gpu="$1" port="$2" wid="$3"
    while :; do
        local job; job="$(take_job)"
        [[ -z "${job}" ]] && break
        local track task; read -r track task <<< "${job}"
        local log="${save_root}/logs/${track}__${task}.log"
        echo "[w${wid} gpu${gpu} :${port}] ${track}/${task}"
        # EGL enumerates devices independently of CUDA_VISIBLE_DEVICES, so a
        # hardcoded 0 would pin every worker's rendering to physical GPU 0 while
        # compute ran elsewhere. Default to this worker's GPU (cf.
        # benchmarks/robocasa_gr1/single_eval.sh); still overridable.
        CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${gpu}}" \
        VLABENCH_SAVE_DIR="${save_root}/by_task/${task}" \
            bash "${SCRIPT_DIR}/single_eval.sh" "${task}" "${track}" "${n_episodes}" "${port}" \
            > "${log}" 2>&1
    done
}

for i in "${!ports[@]}"; do
    worker "${gpus[$i]}" "${ports[$i]}" "$i" &
done
wait

echo
echo "=== summary ==="
python3 - "${save_root}" "${tracks[*]}" <<'PYEOF'
import json, sys
from pathlib import Path

save_root, tracks = Path(sys.argv[1]), sys.argv[2].split()
NOT_A_METRIC = {"n_episodes"}
combined, failed = {}, []
for result in sorted(save_root.glob("by_task/*/*/openwam/evaluation_result.json")):
    track = result.parent.parent.name
    if track not in tracks:
        continue
    try:
        payload = json.loads(result.read_text())
    except Exception:
        continue
    for task, metrics in payload.items():
        combined.setdefault(track, {})[task] = metrics

for track in tracks:
    per_task = combined.get(track, {})
    for task, metrics in sorted(per_task.items()):
        rendered = "  ".join(
            f"{k}={v}" if k in NOT_A_METRIC else f"{k}={v:.4f}" for k, v in metrics.items()
        )
        print(f"  [ok]   {track}/{task}  {rendered}")
    if not per_task:
        failed.append(track)
        print(f"  [FAIL] {track}: no results (see {save_root}/logs/)")
        continue
    keys = {k for m in per_task.values() for k in m} - NOT_A_METRIC
    means = {
        k: sum(m[k] for m in per_task.values() if k in m) / sum(1 for m in per_task.values() if k in m)
        for k in keys
    }
    episodes = sum(m.get("n_episodes", 0) for m in per_task.values())
    print(f"  ---- {track} mean over {len(per_task)} task(s), {episodes} episodes: "
          + "  ".join(f"{k}={v:.4f}" for k, v in sorted(means.items())))
    combined[track]["__mean__"] = means

out = save_root / "combined_results.json"
out.write_text(json.dumps(combined, indent=2))
print(f"\n  combined -> {out}")
sys.exit(1 if failed else 0)
PYEOF
