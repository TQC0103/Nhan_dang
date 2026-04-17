#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash tools/run_bundle_baseline_vs_asr_hard_subset.sh <bundle_root> [output_root]

Example:
  bash tools/run_bundle_baseline_vs_asr_hard_subset.sh \
    /root/analysis_artifacts_bundle \
    /root/analysis_artifacts_bundle/analysis_outputs/baseline_vs_asr_jsar

Environment overrides:
  BASELINE_CONFIG   Default: configs/scrfd/scrfd_2.5g_80e_cosine_baseline.py
  IMPROVED_CONFIG   Default: configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar.py
  GT_DIR            Default: data/retinaface/val/gt
  TEST_MODE         Default: 0
  SCORE_THR         Default: 0.02
  BASELINE_GPU      Default: 0
  IMPROVED_GPU      Default: 0
EOF
  exit 1
fi

BUNDLE_ROOT="$1"
OUT_ROOT="${2:-${BUNDLE_ROOT}/analysis_outputs/baseline_vs_asr_jsar}"

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_baseline.py}"
IMPROVED_CONFIG="${IMPROVED_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar.py}"
GT_DIR="${GT_DIR:-data/retinaface/val/gt}"
TEST_MODE="${TEST_MODE:-0}"
SCORE_THR="${SCORE_THR:-0.02}"
BASELINE_GPU="${BASELINE_GPU:-0}"
IMPROVED_GPU="${IMPROVED_GPU:-0}"

BASELINE_CKPT="${BUNDLE_ROOT}/experiments/baseline/work_dir/latest.pth"
IMPROVED_CKPT="${BUNDLE_ROOT}/experiments/asr_jsar/work_dir/latest.pth"

BASELINE_RERUN_DIR="${OUT_ROOT}/rerun_eval/baseline"
IMPROVED_RERUN_DIR="${OUT_ROOT}/rerun_eval/asr_jsar"
HARD_ANALYSIS_DIR="${OUT_ROOT}/hard_subset_rerun"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${BASELINE_RERUN_DIR}" "${IMPROVED_RERUN_DIR}" "${HARD_ANALYSIS_DIR}" "${LOG_DIR}"

cd "${REPO_ROOT}"

if [[ ! -f "${BASELINE_CKPT}" ]]; then
  echo "Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${IMPROVED_CKPT}" ]]; then
  echo "Missing improved checkpoint: ${IMPROVED_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${GT_DIR}" ]]; then
  echo "Missing WIDERFace gt dir: ${GT_DIR}" >&2
  exit 1
fi

echo "[1/3] Rerun baseline eval with --save-preds"
CUDA_VISIBLE_DEVICES="${BASELINE_GPU}" python tools/test_widerface_enhanced.py \
  "${BASELINE_CONFIG}" \
  "${BASELINE_CKPT}" \
  --out "${BASELINE_RERUN_DIR}" \
  --mode "${TEST_MODE}" \
  --thr "${SCORE_THR}" \
  --save-preds \
  > "${LOG_DIR}/baseline_rerun_eval.log" 2>&1

echo "[2/3] Rerun ASR+JSAR eval with --save-preds"
CUDA_VISIBLE_DEVICES="${IMPROVED_GPU}" python tools/test_widerface_enhanced.py \
  "${IMPROVED_CONFIG}" \
  "${IMPROVED_CKPT}" \
  --out "${IMPROVED_RERUN_DIR}" \
  --mode "${TEST_MODE}" \
  --thr "${SCORE_THR}" \
  --save-preds \
  > "${LOG_DIR}/asr_jsar_rerun_eval.log" 2>&1

echo "[3/3] Run hard-subset per-image analysis"
python tools/analyze_hard_subset_comparison.py \
  --baseline "${BASELINE_RERUN_DIR}" \
  --improved "${IMPROVED_RERUN_DIR}" \
  --gt-dir "${GT_DIR}" \
  --out-dir "${HARD_ANALYSIS_DIR}" \
  > "${LOG_DIR}/hard_subset_rerun.log" 2>&1

echo "Rerun outputs:"
echo "  baseline preds: ${BASELINE_RERUN_DIR}"
echo "  improved preds: ${IMPROVED_RERUN_DIR}"
echo "  hard analysis:  ${HARD_ANALYSIS_DIR}"
echo "  logs:           ${LOG_DIR}"
