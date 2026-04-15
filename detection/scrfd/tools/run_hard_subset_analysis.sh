#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  cat <<EOF
Usage:
  bash tools/run_hard_subset_analysis.sh <baseline_config> <baseline_ckpt> <improved_config> <improved_ckpt> <out_dir>

This script:
  1. runs WIDERFace eval with --save-preds for both models
  2. runs hard-subset comparison analysis
EOF
  exit 1
fi

BASELINE_CONFIG="$1"
BASELINE_CKPT="$2"
IMPROVED_CONFIG="$3"
IMPROVED_CKPT="$4"
OUT_DIR="$5"

MODE="${MODE:-0}"
THR="${THR:-0.02}"
GT_DIR="${GT_DIR:-data/retinaface/val/gt}"
BASELINE_OUT="${OUT_DIR}/baseline"
IMPROVED_OUT="${OUT_DIR}/improved"
ANALYSIS_OUT="${OUT_DIR}/analysis"

mkdir -p "${BASELINE_OUT}" "${IMPROVED_OUT}" "${ANALYSIS_OUT}"

python tools/test_widerface_enhanced.py \
  "${BASELINE_CONFIG}" \
  "${BASELINE_CKPT}" \
  --out "${BASELINE_OUT}" \
  --mode "${MODE}" \
  --thr "${THR}" \
  --save-preds

python tools/test_widerface_enhanced.py \
  "${IMPROVED_CONFIG}" \
  "${IMPROVED_CKPT}" \
  --out "${IMPROVED_OUT}" \
  --mode "${MODE}" \
  --thr "${THR}" \
  --save-preds

python tools/analyze_hard_subset_comparison.py \
  --baseline "${BASELINE_OUT}" \
  --improved "${IMPROVED_OUT}" \
  --gt-dir "${GT_DIR}" \
  --out-dir "${ANALYSIS_OUT}"
