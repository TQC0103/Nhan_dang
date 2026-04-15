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
BASELINE_GPU="${BASELINE_GPU:-0}"
IMPROVED_GPU="${IMPROVED_GPU:-1}"

BASELINE_WORK_DIR="${WORK_ROOT}/baseline"
IMPROVED_WORK_DIR="${WORK_ROOT}/asr_jsar"
BASELINE_RESULT_DIR="${RESULT_ROOT}/baseline"
IMPROVED_RESULT_DIR="${RESULT_ROOT}/asr_jsar"
COMPARE_RESULT_DIR="${RESULT_ROOT}/comparison"
LOG_DIR="${RESULT_ROOT}/logs"

run_python_on_gpu() {
  local gpu_id="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYRUN[@]}" "$@"
}

wait_for_named_job() {
  local job_name="$1"
  local pid="$2"
  if ! wait "${pid}"; then
    echo "${job_name} failed (pid=${pid})" >&2
    exit 1
  fi
}

mkdir -p \
  "${BASELINE_WORK_DIR}" \
  "${IMPROVED_WORK_DIR}" \
  "${BASELINE_RESULT_DIR}" \
  "${IMPROVED_RESULT_DIR}" \
  "${COMPARE_RESULT_DIR}" \
  "${LOG_DIR}"

cd "${REPO_ROOT}"

if [[ "${BASELINE_GPU}" == "${IMPROVED_GPU}" ]]; then
  echo "BASELINE_GPU and IMPROVED_GPU must be different. Got ${BASELINE_GPU}." >&2
  exit 1
fi

echo "[1/6] Train baseline on GPU ${BASELINE_GPU} and improved model on GPU ${IMPROVED_GPU} in parallel"
run_python_on_gpu "${BASELINE_GPU}" \
  tools/train.py "${BASELINE_CONFIG}" --work-dir "${BASELINE_WORK_DIR}" \
  > "${LOG_DIR}/baseline_train.log" 2>&1 &
BASELINE_TRAIN_PID=$!
run_python_on_gpu "${IMPROVED_GPU}" \
  tools/train.py "${IMPROVED_CONFIG}" --work-dir "${IMPROVED_WORK_DIR}" \
  > "${LOG_DIR}/improved_train.log" 2>&1 &
IMPROVED_TRAIN_PID=$!

echo "Baseline train PID: ${BASELINE_TRAIN_PID} (log: ${LOG_DIR}/baseline_train.log)"
echo "Improved train PID: ${IMPROVED_TRAIN_PID} (log: ${LOG_DIR}/improved_train.log)"

wait_for_named_job "Baseline training" "${BASELINE_TRAIN_PID}"
wait_for_named_job "Improved training" "${IMPROVED_TRAIN_PID}"

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

echo "[2/6] Evaluate both models on WIDERFace in parallel"
run_python_on_gpu "${BASELINE_GPU}" \
  tools/test_widerface_enhanced.py \
  "${BASELINE_CONFIG}" \
  "${BASELINE_CKPT}" \
  --out "${BASELINE_RESULT_DIR}" \
  --mode "${TEST_MODE}" \
  --thr "${SCORE_THR}" \
  > "${LOG_DIR}/baseline_eval.log" 2>&1 &
BASELINE_EVAL_PID=$!
run_python_on_gpu "${IMPROVED_GPU}" \
  tools/test_widerface_enhanced.py \
  "${IMPROVED_CONFIG}" \
  "${IMPROVED_CKPT}" \
  --out "${IMPROVED_RESULT_DIR}" \
  --mode "${TEST_MODE}" \
  --thr "${SCORE_THR}" \
  > "${LOG_DIR}/improved_eval.log" 2>&1 &
IMPROVED_EVAL_PID=$!

echo "Baseline eval PID: ${BASELINE_EVAL_PID} (log: ${LOG_DIR}/baseline_eval.log)"
echo "Improved eval PID: ${IMPROVED_EVAL_PID} (log: ${LOG_DIR}/improved_eval.log)"

wait_for_named_job "Baseline evaluation" "${BASELINE_EVAL_PID}"
wait_for_named_job "Improved evaluation" "${IMPROVED_EVAL_PID}"

echo "[3/6] Build comparison report"
"${PYRUN[@]}" tools/compare_widerface_results.py \
  --baseline "${BASELINE_RESULT_DIR}" \
  --improved "${IMPROVED_RESULT_DIR}" \
  --baseline-name "SCRFD-2.5G Baseline 80e" \
  --improved-name "SCRFD-2.5G ASR+JSAR 80e" \
  --out-dir "${COMPARE_RESULT_DIR}"

echo "[4/6] Done"
echo "Baseline results:  ${BASELINE_RESULT_DIR}"
echo "Improved results:  ${IMPROVED_RESULT_DIR}"
echo "Comparison report: ${COMPARE_RESULT_DIR}/comparison.md"
echo "Train/eval logs:   ${LOG_DIR}"
