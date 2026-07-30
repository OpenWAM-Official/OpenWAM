#!/usr/bin/env bash
# Extract the InternData-A1 LeRobot v3.0 release into a tree the reader can use.
#
# The dataset ships as ~286 .tar.gz archives. gzip streams are not seekable, so
# the LeRobot reader (which needs random access into parquet + mp4) cannot read
# them in place — they must be unpacked once. Budget ~2x the archive size on disk.
#
# Each archive already contains a top-level directory named after itself, so
# unpacking <cat>/<emb>/<name>.tar.gz into <DST>/<cat>/<emb>/ yields either
#   <DST>/<cat>/<emb>/<name>/{data,meta,videos}            (flat archives)
#   <DST>/<cat>/<emb>/<name>/<object>/{data,meta,videos}   (nested archives)
# Both depths occur, and both are found by the reader's recursive bucket walk.
#
# Resumable: a per-archive sentinel is written only after tar exits 0, so an
# interrupted run redoes just the archive it died on. Each archive unpacks into a
# .partial_* staging dir that is renamed into place on success, so a killed tar
# can never leave a half-tree that the sentinel logic or the reader mistakes for
# a complete bucket.
set -uo pipefail

if [[ $# -lt 2 ]]; then
    cat >&2 <<'USAGE'
Usage: extract_interndata_a1_v30.sh SRC_DIR DST_DIR [JOBS]

  SRC_DIR   directory holding the downloaded sim_updated_lerobotv30 archives
  DST_DIR   destination for the extracted tree (created if absent)
  JOBS      parallel tar processes (default 24; gzip is single-threaded per
            archive, so parallelism is across archives)

After extraction, generate the per-embodiment normalization stats:
  python -m openwam.dataloader.utils.stats_computation.interndata_a1_stats_computation \
      --dataset_dir DST_DIR
USAGE
    exit 2
fi

SRC="$(realpath "$1")"
DST="$(realpath -m "$2")"
JOBS="${3:-${A1_JOBS:-24}}"
LOG_DIR="${A1_LOG_DIR:-$DST/.extract_logs}"
SENTINELS="$LOG_DIR/sentinels"
LOG="$LOG_DIR/extract.log"

mkdir -p "$DST" "$SENTINELS"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

one() {
    local arc=$1
    local rel=${arc#"$SRC"/}                       # <cat>/<emb>/<name>.tar.gz
    local dir name key sentinel out part
    dir=$(dirname "$rel")
    name=$(basename "$rel" .tar.gz)
    key=${rel//\//_}
    sentinel="$SENTINELS/$key.done"
    [[ -f "$sentinel" ]] && return 0

    out="$DST/$dir"
    part="$out/.partial_$name"
    rm -rf "$part"
    mkdir -p "$part"
    if tar xzf "$arc" -C "$part" 2>>"$LOG"; then
        rm -rf "${out:?}/$name"
        if [[ -d "$part/$name" ]]; then
            mv "$part/$name" "$out/$name"
            rm -rf "$part"
        else
            mv "$part" "$out/$name"
        fi
        date '+%F %T' >"$sentinel"
        echo "[$(date '+%F %T')] OK   $rel" >>"$LOG"
    else
        rm -rf "$part"
        echo "[$(date '+%F %T')] FAIL $rel" >>"$LOG"
    fi
}
export -f one
export SRC DST SENTINELS LOG

TOTAL=$(find "$SRC" -name '*.tar.gz' | wc -l)
if [[ "$TOTAL" -eq 0 ]]; then
    echo "No .tar.gz archives found under $SRC" >&2
    exit 1
fi
say "START pid=$$ jobs=$JOBS archives=$TOTAL src=$SRC dst=$DST"
find "$SRC" -name '*.tar.gz' | sort | xargs -P "$JOBS" -I{} bash -c 'one "$@"' _ {}

DONE=$(find "$SENTINELS" -name '*.done' | wc -l)
FAILED=$(grep -c 'FAIL ' "$LOG" || true)
say "DONE $DONE/$TOTAL extracted, $FAILED failures, size=$(du -sh "$DST" | cut -f1)"
if [[ "$DONE" -ne "$TOTAL" ]]; then
    say "Incomplete — re-run this script to retry only the missing archives."
    exit 1
fi
