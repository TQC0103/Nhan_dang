#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "Usage: bash tools/run_build_hard_gain_explanation_report.sh <baseline_results> <improved_results> <baseline_work_dir> <improved_work_dir> <out_dir> <gt_dir> [ann_file]" >&2
  exit 1
fi

BASELINE_RESULTS="$1"
IMPROVED_RESULTS="$2"
BASELINE_WORK_DIR="$3"
IMPROVED_WORK_DIR="$4"
OUT_DIR="$5"
GT_DIR="$6"
ANN_FILE="${7:-data/retinaface/train/labelv2.txt}"

python tools/build_hard_gain_explanation_report.py \
  --baseline-results "${BASELINE_RESULTS}" \
  --improved-results "${IMPROVED_RESULTS}" \
  --baseline-work-dir "${BASELINE_WORK_DIR}" \
  --improved-work-dir "${IMPROVED_WORK_DIR}" \
  --baseline-config configs/scrfd/scrfd_2.5g_80e_baseline.py \
  --improved-config configs/scrfd/scrfd_2.5g_80e_asr_jsar.py \
  --out-dir "${OUT_DIR}" \
  --gt-dir "${GT_DIR}" \
  --ann-file "${ANN_FILE}"
