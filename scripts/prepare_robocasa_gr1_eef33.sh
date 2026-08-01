#!/usr/bin/env bash
# One-shot NVIDIA joint44 -> FK EEF33 v2.0 -> LeRobot v3 -> dual stats.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 INPUT_JOINT44_V20 OUTPUT_EEF33_V20 OUTPUT_EEF33_V30" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT="$(realpath "$1")"
EEF_V20="$(realpath -m "$2")"
EEF_V30="$(realpath -m "$3")"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-${ROBOCASA_GR1_PYTHON:-python}}"
OPENWAM_PYTHON="${OPENWAM_PYTHON:-python}"

if [[ -z "${ROBOCASA_GR1_PATH:-}" ]]; then
    echo "ROBOCASA_GR1_PATH must point to robocasa-gr1-tabletop-tasks" >&2
    exit 2
fi

MUJOCO_GL="${MUJOCO_GL:-glfw}" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$ROBOCASA_PYTHON" "$ROOT/scripts/enrich_robocasa_gr1_joint44_to_eef33.py" \
    --input "$INPUT" \
    --output "$EEF_V20" \
    --robocasa-path "$ROBOCASA_GR1_PATH"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$OPENWAM_PYTHON" "$ROOT/scripts/convert_robocasa_gr1_v20_to_v30.py" \
    --input "$EEF_V20" \
    --output "$EEF_V30"

TMP_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$TMP_CONFIG"' EXIT
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$OPENWAM_PYTHON" - "$ROOT" "$EEF_V30" "$TMP_CONFIG" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

root, dataset, output = map(Path, sys.argv[1:])
cfg = OmegaConf.load(root / "configs/dataloader/robocasa_gr1.yaml")
cfg.dataset_dir = str(dataset)
cfg.normalize_mode = None
OmegaConf.save(cfg, output)
PY
# No --output: the default is <dataset_dir>/meta/normalization_stats.npy, the
# fixed location the reader loads (and would otherwise auto-build on first use).
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$OPENWAM_PYTHON" -m openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation \
    --config "$TMP_CONFIG"

echo "Prepared EEF33 dataset: $EEF_V30"
echo "Stats: $EEF_V30/meta/normalization_stats.npy"
