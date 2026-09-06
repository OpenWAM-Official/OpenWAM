#!/usr/bin/env bash
# Evaluate the canonical RoboCasa365 native-action policy on a task list.
# Mirrors benchmarks/robotwin/multi_eval.sh. Tasks run sequentially (the client resets server
# state at the start of every episode); per-task "Success rate" lines are aggregated into a CSV.
#
# Start the server first (OpenWAM env):
#   bash scripts/deploy.sh --ckpt-dir /path/to/robocasa365_ckpt --port 8848
#
# Then, inside the robocasa365 env:
#   ROBOCASA365_PYTHON=/path/to/python bash multi_eval.sh [options] <tasks...>
#
# Tasks (positional): task names | "all"/"target" (the 50 eval targets in target_tasks.txt) | a file (one
#                     task per line, '#' comments allowed).
# Options:
#   --split <s>   target | pretrain | all   (default: pretrain)
#   --port  <p>   server WebSocket port      (default: 8848)
#   --host  <h>   server host                (default: 127.0.0.1)
#   --out   <dir> results directory          (default: ./results_robocasa365)
#   -h, --help
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; }

split="pretrain"; port="8848"; host="127.0.0.1"; out="./results_robocasa365"
tasks=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) split="$2"; shift 2 ;;
    --port)  port="$2";  shift 2 ;;
    --host)  host="$2";  shift 2 ;;
    --out)   out="$2";   shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) tasks+=("$1"); shift ;;
  esac
done

python_bin="${ROBOCASA365_PYTHON:-python}"

# Resolve the task list: "all"/"target" -> the official 50 eval targets (target_tasks.txt);
# a file -> its non-comment lines; anything else -> a literal task name.
resolve_tasks() {
  local arg
  for arg in "${tasks[@]:-}"; do
    if [[ "$arg" == "all" || "$arg" == "target" ]]; then
      sed 's/#.*//' "${SCRIPT_DIR}/target_tasks.txt" | awk 'NF'
    elif [[ -f "$arg" ]]; then
      sed 's/#.*//' "$arg" | awk 'NF'
    else
      printf '%s\n' "$arg"
    fi
  done
}

mapfile -t TASKS < <(resolve_tasks)
[[ ${#TASKS[@]} -eq 0 ]] && { echo "[ERROR] no tasks — give task names, 'all', or a file" >&2; usage; exit 1; }

mkdir -p "$out"
csv="${out}/summary_${split}.csv"
echo "task,split,successes,trials,rate_pct" > "$csv"
echo "=== evaluating ${#TASKS[@]} task(s), split=${split}, server=ws://${host}:${port} ==="
for task in "${TASKS[@]}"; do
  log="${out}/${task}_${split}.log"
  echo "--- ${task} ---"
  set +e
  ROBOCASA365_PORT="$port" ROBOCASA365_POLICY_HOST="$host" \
    bash "${SCRIPT_DIR}/single_eval.sh" "$task" "$split" "$port" "$host" 2>&1 | tee "$log"
  set -e
  line="$(grep -aoE 'Success rate: [0-9]+/[0-9]+' "$log" | tail -1 || true)"
  if [[ "$line" =~ ([0-9]+)/([0-9]+) ]]; then
    s="${BASH_REMATCH[1]}"; n="${BASH_REMATCH[2]}"
    rate="$("$python_bin" -c "print(f'{$s/$n*100:.1f}')" 2>/dev/null || echo NA)"
    echo "${task},${split},${s},${n},${rate}" >> "$csv"
  else
    echo "${task},${split},NA,NA,NA" >> "$csv"
  fi
done
echo "=== summary -> ${csv} ==="
column -t -s, "$csv" 2>/dev/null || cat "$csv"
