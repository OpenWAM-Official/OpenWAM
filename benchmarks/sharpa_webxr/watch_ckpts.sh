#!/bin/bash
# Generic eval watcher: closed-loop sim eval of every checkpoint of a training run as it lands, for any model that
# has a policy server (API.md). Nothing here is model-specific; the model side provides two things:
#   --server PATH       its policy server script, started per checkpoint as
#                       <server-python> PATH --ckpt <ckpt> <extra args> --port <port>
#   --list-ckpts CMD    a command (run with bash -c) that prints one line per COMPLETE checkpoint:
#                         <global_step> <ckpt_path> [extra server args...]
#                       The command owns "complete" (e.g. a meta file the trainer writes last, or an mtime settle);
#                       lines may come in any order and repeat. See API.md "Eval watcher" for examples.
# usage: watch_ckpts.sh --run <name> --server <script> --list-ckpts "<cmd>" [options]
#   --seeds 0-4            seeds for every checkpoint            --final-seeds 5-9  extra seeds on the last one
#   --done-file PATH       training is finished once PATH exists (then: final top-up, wait for judge/videos, exit)
#   --once                 evaluate the checkpoints listed now, then finish (no waiting for training)
#   --port 11600           --server-python .venv/bin/python      --workers 12 (parallel rollouts)
#   --stride N             intermediate evals only at steps divisible by N (the last checkpoint is always evaluated)
#   --only-steps a,b,...   evaluate only these steps
#   --sync-cmd CMD         run before every checkpoint scan (e.g. reload a network volume); default none
#   --stop-file PATH       if it exists between evaluations: exit 3 (a supervisor restarts us; state is on disk)
#   --poll 300             seconds between scans while training runs
#   --server-timeout 2400  seconds to wait for a server to accept connections
# env: VIDEO=1 VIDEO_SEEDS=0 VIDEO_P=4 (video.mp4 renderer), JUDGE=1 JUDGE_MODEL=gpt-6-luna JUDGE_WORKERS=8
#      (VLM judge, needs OPENAI_API_KEY or ~/.config/openai/key), MAX_ATTEMPTS=2, EXECUTE_STEPS (override meta)
#
# Results: .sharpa_sim_eval/<run>/step<NNNNNN>/<scene>/seed_<k>/, summaries via aggregate.py, judge scores
# <tag>/scores_<model>_r2.{json,tsv}. Markers per tag: DONE_<seeds> (suite exited 0 and every scene x seed has
# signals.json), ATTEMPTS_<seeds>, FAILED_<seeds> (after MAX_ATTEMPTS; delete to retry), JUDGED_<model>_<seeds>.
# Resumable: rerun the same command; finished tags and rollouts are skipped.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."  # repo root
RUN=; SERVER=; LIST=; SI=0-4; SF=5-9; DONE_FILE=; ONCE=; PORT=11600; SPY=.venv/bin/python; WORKERS=12
STRIDE=0; ONLY=; SYNC_CMD=; STOP_FILE=; POLL=300; STIMEOUT=2400
while [ $# -gt 0 ]; do
  case $1 in
    --run) RUN=$2; shift ;;              --server) SERVER=$2; shift ;;        --list-ckpts) LIST=$2; shift ;;
    --seeds) SI=$2; shift ;;             --final-seeds) SF=$2; shift ;;       --done-file) DONE_FILE=$2; shift ;;
    --once) ONCE=1 ;;                    --port) PORT=$2; shift ;;            --server-python) SPY=$2; shift ;;
    --workers) WORKERS=$2; shift ;;      --stride) STRIDE=$2; shift ;;        --only-steps) ONLY=$2; shift ;;
    --sync-cmd) SYNC_CMD=$2; shift ;;    --stop-file) STOP_FILE=$2; shift ;;  --poll) POLL=$2; shift ;;
    --server-timeout) STIMEOUT=$2; shift ;;
    *) echo "unknown option $1 (see the header of $0)"; exit 2 ;;
  esac
  shift
done
[ -n "$RUN" ] && [ -n "$SERVER" ] && [ -n "$LIST" ] || { echo "need --run, --server and --list-ckpts"; exit 2; }
[ -n "$DONE_FILE" ] || [ -n "$ONCE" ] || { echo "need --done-file (training-finished signal) or --once"; exit 2; }
VIDEO=${VIDEO:-1}; VIDEO_SEEDS=${VIDEO_SEEDS:-0}; VIDEO_P=${VIDEO_P:-4}; MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
JUDGE=${JUDGE:-1}; JUDGE_MODEL=${JUDGE_MODEL:-gpt-6-luna}; JUDGE_WORKERS=${JUDGE_WORKERS:-8}
B=benchmarks/sharpa_webxr; E=.sharpa_sim_eval/$RUN; L=.sharpa_sim_eval/logs/$RUN
mkdir -p $E $L
export RUN MUJOCO_GL=osmesa EXECUTE_STEPS=${EXECUTE_STEPS:-}
NSCENES=$(.venv_webxr/bin/python -c "import json;print(len(json.load(open('$B/tasks.json'))))")
ts() { date -u +%FT%TZ; }
alive() { [ -z "$ONCE" ] && [ ! -e "$DONE_FILE" ]; }
sync_hook() { [ -z "$SYNC_CMD" ] || bash -c "$SYNC_CMD" || echo "[watch] sync command failed $(ts)"; }
tag_of() { printf 'step%06d' $1; }

# ckpts: the model's list, numeric step first, one line per step (first listing wins), optional --only-steps filter
ckpts() {
  bash -c "$LIST" 2>>$L/list_ckpts.log | awk 'NF >= 2 && $1 ~ /^[0-9]+$/ && !seen[$1]++' | sort -n -k1,1 |
    { if [ -n "$ONLY" ]; then awk -v keep=",$ONLY," 'index(keep, ","$1",")>0'; else cat; fi; }
}

# serve <log> <ckpt> <args...>: start the policy server, wait until it accepts connections; pid in SPID
serve() {
  local log=$1; shift
  $SPY -u $SERVER --ckpt "$@" --port $PORT > $log 2>&1 &
  SPID=$!
  local t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt $STIMEOUT ]; do
    .venv_webxr/bin/python -c "import socket;socket.create_connection(('localhost',$PORT),2)" 2>/dev/null && return 0
    kill -0 $SPID 2>/dev/null || { echo "[FAIL server] $1 exited, see $log"; SPID=; return 1; }
    sleep 10
  done
  echo "[FAIL server] $1 not up after ${STIMEOUT}s"; kill $SPID; SPID=; return 1
}

count_done() {  # rollouts with signals for <tag> and seed range a-b (expected: NSCENES per seed)
  local tag=$1 a=${2%-*} b=${2#*-} n=0 s
  for s in $(seq $a $b); do n=$((n + $(find $E/$tag -path "*/seed_$s/signals.json" 2>/dev/null | wc -l))); done
  echo $n
}

# evaluate <tag> <seeds> <ckpt> <server args...>: one suite against one server (blocking)
evaluate() {
  local tag=$1 seeds=$2; shift 2
  [ -f $E/$tag/DONE_$seeds ] || [ -f $E/$tag/FAILED_$seeds ] && return 0
  mkdir -p $E/$tag
  local att=$(( $(cat $E/$tag/ATTEMPTS_$seeds 2>/dev/null || echo 0) + 1 ))
  echo $att > $E/$tag/ATTEMPTS_$seeds
  echo "[eval] $tag seeds $seeds ckpt $1 attempt $att $(ts)"
  local rc=0
  if serve $L/server_$tag.log "$@"; then
    local spid=$SPID
    $B/run_suite.sh $tag $PORT ${seeds%-*} ${seeds#*-} $WORKERS 1 > $L/suite_${tag}_$seeds.log 2>&1 || rc=$?
    kill $spid; wait $spid 2>/dev/null
  else
    rc=1
  fi
  local n want=$(( NSCENES * (${seeds#*-} - ${seeds%-*} + 1) ))
  n=$(count_done $tag $seeds)
  if [ $rc -eq 0 ] && [ $n -eq $want ]; then
    .venv_webxr/bin/python $B/aggregate.py $RUN --tags $tag --tsv $E/$tag/summary_$seeds.tsv | sed 's/^/[summary] /'
    touch $E/$tag/DONE_$seeds
    echo "[eval done] $tag seeds $seeds: $n/$want rollouts $(ts)"
  elif [ $att -ge $MAX_ATTEMPTS ]; then
    touch $E/$tag/FAILED_$seeds
    echo "[FAIL eval] $tag seeds $seeds: $n/$want rollouts, suite rc=$rc after $att attempts; marked FAILED $(ts)"
  else
    echo "[FAIL eval] $tag seeds $seeds: $n/$want rollouts, suite rc=$rc (attempt $att/$MAX_ATTEMPTS, will retry) $(ts)"
  fi
  .venv_webxr/bin/python $B/aggregate.py $RUN --tsv $E/summary.tsv > /dev/null
}

# pending: oldest checkpoint without DONE/FAILED for the intermediate seeds -> "<step> <ckpt> <args...>"
pending() {
  local line t
  while read -r line; do
    [ "$STRIDE" -gt 0 ] && [ $(( ${line%% *} % STRIDE )) -ne 0 ] && continue
    t=$(tag_of ${line%% *})
    [ -f $E/$t/DONE_$SI ] || [ -f $E/$t/FAILED_$SI ] || { echo "$line"; return; }
  done < <(ckpts)
}

todo_videos() {  # finished rollouts without video.mp4, seed 0 first, filtered by VIDEO_SEEDS
  find $E -name signals.json -path '*/seed_*' 2>/dev/null | while read f; do
    d=${f%/signals.json}; [ -f $d/video.mp4 ] && continue
    k=${d##*seed_}; [ "$VIDEO_SEEDS" = all ] || [[ ",$VIDEO_SEEDS," == *",$k,"* ]] || continue
    echo "$k $d"
  done | sort -n -s -k1,1 | cut -d' ' -f2
}
render_loop() {
  while true; do
    list=$(todo_videos)
    if [ -z "$list" ]; then [ -f $E/.evals_done ] && break; sleep 120; continue; fi
    echo "$list" | head -n 400 | LP_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      nice -n 19 xargs -n 4 -P $VIDEO_P .venv_webxr/bin/python $B/render_video.py >> $L/videos.log 2>&1
    echo "[videos] $(find $E -name video.mp4 | wc -l) rendered $(ts)"
  done
}

# judge_pass: rate every finished tag (DONE_<seeds>) not yet rated for those seeds. postprocess.py redoes any
# rollout whose judge_frames lack COMPLETE (e.g. a render killed by a restart) and drops its stale verdict.
judge_pass() {
  local d t m k
  for d in $E/step*; do
    [ -d $d ] || continue
    t=$(basename $d)
    for m in $d/DONE_*; do
      [ -f "$m" ] || continue
      k=${m##*/DONE_}
      [ -f $d/JUDGED_${JUDGE_MODEL}_$k ] && continue
      echo "[judge] $t seeds $k: frames $(ts)"
      find $d -name rollout.json -printf '%h\n' | sort | LP_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        nice -n 10 xargs -n 4 -P 8 .venv_webxr/bin/python $B/postprocess.py --frames-only >> $L/judge_frames_$t.log 2>&1
      if .venv_webxr/bin/python $B/score.py $d --model $JUDGE_MODEL --rubric r2 --workers $JUDGE_WORKERS \
           > $L/judge_${t}_$k.log 2>&1 && ! grep -q 'errors=[1-9]' $L/judge_${t}_$k.log; then
        touch $d/JUDGED_${JUDGE_MODEL}_$k
        tail -1 $L/judge_${t}_$k.log | sed "s/^/[judge] $t: /"
      else
        echo "[FAIL judge] $t seeds $k, see $L/judge_${t}_$k.log (retried next pass) $(ts)"
      fi
    done
  done
}
judge_loop() {
  while true; do
    judge_pass
    [ -f $E/.evals_done ] && { judge_pass; break; }
    sleep 300
  done
}

rm -f $E/.evals_done
RENDER=; JUDGER=
[ "$VIDEO" = 1 ] && { render_loop & RENDER=$!; }
if [ "$JUDGE" = 1 ]; then
  if [ -n "${OPENAI_API_KEY:-}" ] || [ -f ~/.config/openai/key ]; then judge_loop & JUDGER=$!
  else echo "[judge] disabled: no OPENAI_API_KEY or ~/.config/openai/key"; fi
fi

echo "[watch $RUN] start $(ts): server $SERVER, port $PORT, seeds $SI (+$SF on the last), video=$VIDEO, judge=${JUDGER:+$JUDGE_MODEL}${JUDGER:-off}"
while true; do
  # between evaluations only (no server running): a supervisor can stop us, also during a backlog
  [ -n "$STOP_FILE" ] && [ -e "$STOP_FILE" ] && { echo "WATCH_STOP $(ts)"; exit 3; }
  sync_hook
  job=$(pending)
  if [ -n "$job" ]; then
    set -- $job
    step=$1; shift
    evaluate $(tag_of $step) $SI "$@"
    continue
  fi
  if ! alive; then
    sync_hook
    [ -n "$(pending)" ] && continue
    break
  fi
  sleep $POLL
done

echo "[watch $RUN] training finished $(ts); final top-up"
last=$(ckpts | tail -1)
if [ -n "$last" ]; then
  set -- $last
  step=$1; shift
  tag=$(tag_of $step)
  evaluate $tag $SI "$@"  # the last checkpoint is always evaluated, also when --stride skipped it
  [ "$SF" = none ] || evaluate $tag $SF "$@"
fi
touch $E/.evals_done
if [ -n "$JUDGER" ]; then wait $JUDGER; echo "[judge] all rated $(ts)"; fi
if [ -n "$RENDER" ]; then wait $RENDER; echo "[videos] all rendered $(ts)"; fi
.venv_webxr/bin/python $B/aggregate.py $RUN --tsv $E/summary.tsv | sed 's/^/[final] /'
echo "WATCH_ALL_DONE $(ts)"
