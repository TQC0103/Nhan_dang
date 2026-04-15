#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_in_env.sh"

if [[ -x "${RUNNER}" ]]; then
  PYRUN=("${RUNNER}" python)
else
  PYRUN=(python)
fi

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/scrfd/scrfd_2.5g_80e_baseline.py}"
IMPROVED_CONFIG="${IMPROVED_CONFIG:-configs/scrfd/scrfd_2.5g_80e_asr_jsar.py}"
WORK_ROOT="${WORK_ROOT:-work_dirs/compare_2.5g}"
RESULT_ROOT="${RESULT_ROOT:-results/compare_2.5g}"
TEST_MODE="${TEST_MODE:-0}"
SCORE_THR="${SCORE_THR:-0.02}"

BASELINE_WORK_DIR="${WORK_ROOT}/baseline"
IMPROVED_WORK_DIR="${WORK_ROOT}/asr_jsar"
BASELINE_RESULT_DIR="${RESULT_ROOT}/baseline"
IMPROVED_RESULT_DIR="${RESULT_ROOT}/asr_jsar"
COMPARE_RESULT_DIR="${RESULT_ROOT}/comparison"

mkdir -p "${BASELINE_WORK_DIR}" "${IMPROVED_WORK_DIR}" "${BASELINE_RESULT_DIR}" "${IMPROVED_RESULT_DIR}" "${COMPARE_RESULT_DIR}"

cd "${REPO_ROOT}"

echo "[1/6] Train baseline 2.5g for 80 epochs"
"${PYRUN[@]}" tools/train.py "${BASELINE_CONFIG}" --work-dir "${BASELINE_WORK_DIR}"

echo "[2/6] Train improved 2.5g ASR+JSAR for 80 epochs"
"${PYRUN[@]}" tools/train.py "${IMPROVED_CONFIG}" --work-dir "${IMPROVED_WORK_DIR}"

BASELINE_CKPT="${BASELINE_WORK_DIR}/latest.pth"
IMPROVED_CKPT="${IMPROVED_WORK_DIR}/latest.pth"

if [[ ! -f "${BASELINE_CKPT}" ]]; then
  echo "Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${IMPROVED_CKPT}" ]]; then
  echo "Missing improved checkpoint: ${IMPROVED_CKPT}" >&2
  exit 1
fi

echo "[3/6] Evaluate baseline on WIDERFace"
"${PYRUN[@]}" tools/test_widerface_enhanced.py \
  "${BASELINE_CONFIG}" \
  "${BASELINE_CKPT}" \
  --out "${BASELINE_RESULT_DIR}" \
  --mode "${TEST_MODE}" \
  --thr "${SCORE_THR}"

echo "[4/6] Evaluate improved model on WIDERFace"
"${PYRUN[@]}" tools/test_widerface_enhanced.py \
  "${IMPROVED_CONFIG}" \
  "${IMPROVED_CKPT}" \
  --out "${IMPROVED_RESULT_DIR}" \
  --mode "${TEST_MODE}" \
  --thr "${SCORE_THR}"

echo "[5/6] Build comparison report"
"${PYRUN[@]}" tools/compare_widerface_results.py \
  --baseline "${BASELINE_RESULT_DIR}" \
  --improved "${IMPROVED_RESULT_DIR}" \
  --baseline-name "SCRFD-2.5G Baseline 80e" \
  --improved-name "SCRFD-2.5G ASR+JSAR 80e" \
  --out-dir "${COMPARE_RESULT_DIR}"

echo "[6/6] Done"
echo "Baseline results:  ${BASELINE_RESULT_DIR}"
echo "Improved results:  ${IMPROVED_RESULT_DIR}"
echo "Comparison report: ${COMPARE_RESULT_DIR}/comparison.md"
