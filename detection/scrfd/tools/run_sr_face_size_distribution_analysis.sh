#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: bash tools/run_sr_face_size_distribution_analysis.sh <baseline_config> <improved_config> <improved_state> <out_dir> [ann_file]" >&2
  exit 1
fi

BASELINE_CONFIG="$1"
IMPROVED_CONFIG="$2"
IMPROVED_STATE="$3"
OUT_DIR="$4"
ANN_FILE="${5:-data/retinaface/train/labelv2.txt}"

python tools/analyze_sr_face_size_distribution.py \
  --baseline-config "${BASELINE_CONFIG}" \
  --improved-config "${IMPROVED_CONFIG}" \
  --improved-state "${IMPROVED_STATE}" \
  --out-dir "${OUT_DIR}" \
  --ann-file "${ANN_FILE}"
