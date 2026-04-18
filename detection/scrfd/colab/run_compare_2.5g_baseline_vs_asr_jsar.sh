#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_in_env.sh"

if [[ -f "${RUNNER}" ]]; then
  PYRUN=(bash "${RUNNER}" python)
else
  PYRUN=(python)
fi

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/scrfd/scrfd_2.5g_80e_baseline.py}"
IMPROVED_CONFIG="${IMPROVED_CONFIG:-configs/scrfd/scrfd_2.5g_80e_asr_jsar.py}"
WORK_ROOT="${WORK_ROOT:-work_dirs/compare_2.5g}"
RESULT_ROOT="${RESULT_ROOT:-results/compare_2.5g}"
TEST_MODE="${TEST_MODE:-0}"
SCORE_THR="${SCORE_THR:-0.02}"
SAVE_PREDS="${SAVE_PREDS:-0}"
BASELINE_GPU="${BASELINE_GPU:-0}"
IMPROVED_GPU="${IMPROVED_GPU:-1}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-}"
BASELINE_BATCH_SIZE_PER_GPU="${BASELINE_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
IMPROVED_BATCH_SIZE_PER_GPU="${IMPROVED_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
BASELINE_WORKERS_PER_GPU="${BASELINE_WORKERS_PER_GPU:-${WORKERS_PER_GPU}}"
IMPROVED_WORKERS_PER_GPU="${IMPROVED_WORKERS_PER_GPU:-${WORKERS_PER_GPU}}"
BASELINE_LR="${BASELINE_LR:-}"
IMPROVED_LR="${IMPROVED_LR:-}"
AUTO_SCALE_LR="${AUTO_SCALE_LR:-0}"
BASELINE_BASE_BATCH_SIZE="${BASELINE_BASE_BATCH_SIZE:-8}"
IMPROVED_BASE_BATCH_SIZE="${IMPROVED_BASE_BATCH_SIZE:-8}"
BASELINE_BASE_LR="${BASELINE_BASE_LR:-0.01}"
IMPROVED_BASE_LR="${IMPROVED_BASE_LR:-0.01}"
EXTRA_CFG_OPTIONS="${EXTRA_CFG_OPTIONS:-}"
BASELINE_EXTRA_CFG_OPTIONS="${BASELINE_EXTRA_CFG_OPTIONS:-}"
IMPROVED_EXTRA_CFG_OPTIONS="${IMPROVED_EXTRA_CFG_OPTIONS:-}"
BASELINE_NAME="${BASELINE_NAME:-SCRFD-2.5G Baseline 80e}"
IMPROVED_NAME="${IMPROVED_NAME:-SCRFD-2.5G ASR+JSAR 80e}"

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

detect_gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU '
    return 0
  fi
  echo 0
}

append_cfg_option() {
  local -n cfg_ref=$1
  local key="$2"
  local value="$3"
  cfg_ref+=("${key}=${value}")
}

append_cfg_options_from_string() {
  local -n cfg_ref=$1
  local raw_options="$2"
  local token=""
  if [[ -z "${raw_options}" ]]; then
    return 0
  fi
  # shellcheck disable=SC2206
  local parsed_options=( ${raw_options} )
  for token in "${parsed_options[@]}"; do
    cfg_ref+=("${token}")
  done
}

build_cfg_options() {
  local batch_size="$1"
  local workers="$2"
  local manual_lr="$3"
  local auto_scale="$4"
  local base_batch="$5"
  local base_lr="$6"
  local extra_options="$7"
  local -a cfg_options=()
  local scaled_lr=""
  if [[ -n "${batch_size}" ]]; then
    append_cfg_option cfg_options "data.samples_per_gpu" "${batch_size}"
  fi
  if [[ -n "${workers}" ]]; then
    append_cfg_option cfg_options "data.workers_per_gpu" "${workers}"
  fi
  if [[ -n "${manual_lr}" ]]; then
    append_cfg_option cfg_options "optimizer.lr" "${manual_lr}"
  elif [[ "${auto_scale}" == "1" && -n "${batch_size}" ]]; then
    scaled_lr="$(python - <<PY
base_lr = float("${base_lr}")
batch_size = float("${batch_size}")
base_batch = float("${base_batch}")
print("{:.12g}".format(base_lr * batch_size / base_batch))
PY
)"
    append_cfg_option cfg_options "optimizer.lr" "${scaled_lr}"
  fi
  append_cfg_options_from_string cfg_options "${EXTRA_CFG_OPTIONS}"
  append_cfg_options_from_string cfg_options "${extra_options}"
  if (( ${#cfg_options[@]} > 0 )); then
    printf '%s\n' "${cfg_options[@]}"
  fi
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

GPU_COUNT="$(detect_gpu_count)"
SINGLE_GPU_MODE=0

if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && (( GPU_COUNT > 0 )); then
  if (( BASELINE_GPU >= GPU_COUNT )); then
    echo "BASELINE_GPU=${BASELINE_GPU} is out of range. Detected GPU count: ${GPU_COUNT}" >&2
    exit 1
  fi
  if (( IMPROVED_GPU >= GPU_COUNT )); then
    echo "IMPROVED_GPU=${IMPROVED_GPU} is out of range for detected GPU count ${GPU_COUNT}. Falling back to GPU ${BASELINE_GPU}." >&2
    IMPROVED_GPU="${BASELINE_GPU}"
  fi
fi

if [[ "${BASELINE_GPU}" == "${IMPROVED_GPU}" ]]; then
  SINGLE_GPU_MODE=1
fi

if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && (( GPU_COUNT == 1 )); then
  SINGLE_GPU_MODE=1
  IMPROVED_GPU="${BASELINE_GPU}"
fi

mapfile -t BASELINE_CFG_OPTIONS < <(
  build_cfg_options \
    "${BASELINE_BATCH_SIZE_PER_GPU}" \
    "${BASELINE_WORKERS_PER_GPU}" \
    "${BASELINE_LR}" \
    "${AUTO_SCALE_LR}" \
    "${BASELINE_BASE_BATCH_SIZE}" \
    "${BASELINE_BASE_LR}" \
    "${BASELINE_EXTRA_CFG_OPTIONS}"
)
mapfile -t IMPROVED_CFG_OPTIONS < <(
  build_cfg_options \
    "${IMPROVED_BATCH_SIZE_PER_GPU}" \
    "${IMPROVED_WORKERS_PER_GPU}" \
    "${IMPROVED_LR}" \
    "${AUTO_SCALE_LR}" \
    "${IMPROVED_BASE_BATCH_SIZE}" \
    "${IMPROVED_BASE_LR}" \
    "${IMPROVED_EXTRA_CFG_OPTIONS}"
)

if (( ${#BASELINE_CFG_OPTIONS[@]} > 0 )); then
  echo "Baseline cfg overrides: ${BASELINE_CFG_OPTIONS[*]}"
else
  echo "Baseline cfg overrides: (none)"
fi
if (( ${#IMPROVED_CFG_OPTIONS[@]} > 0 )); then
  echo "Improved cfg overrides: ${IMPROVED_CFG_OPTIONS[*]}"
else
  echo "Improved cfg overrides: (none)"
fi

BASELINE_TRAIN_ARGS=(tools/train.py "${BASELINE_CONFIG}" --work-dir "${BASELINE_WORK_DIR}")
if (( ${#BASELINE_CFG_OPTIONS[@]} > 0 )); then
  BASELINE_TRAIN_ARGS+=(--cfg-options "${BASELINE_CFG_OPTIONS[@]}")
fi

IMPROVED_TRAIN_ARGS=(tools/train.py "${IMPROVED_CONFIG}" --work-dir "${IMPROVED_WORK_DIR}")
if (( ${#IMPROVED_CFG_OPTIONS[@]} > 0 )); then
  IMPROVED_TRAIN_ARGS+=(--cfg-options "${IMPROVED_CFG_OPTIONS[@]}")
fi

if [[ "${SINGLE_GPU_MODE}" == "1" ]]; then
  echo "[1/4] Single-GPU mode: train baseline then improved on GPU ${BASELINE_GPU}"
  run_python_on_gpu "${BASELINE_GPU}" \
    "${BASELINE_TRAIN_ARGS[@]}" \
    > "${LOG_DIR}/baseline_train.log" 2>&1
  run_python_on_gpu "${IMPROVED_GPU}" \
    "${IMPROVED_TRAIN_ARGS[@]}" \
    > "${LOG_DIR}/improved_train.log" 2>&1
else
  echo "[1/4] Train baseline on GPU ${BASELINE_GPU} and improved model on GPU ${IMPROVED_GPU} in parallel"
  run_python_on_gpu "${BASELINE_GPU}" \
    "${BASELINE_TRAIN_ARGS[@]}" \
    > "${LOG_DIR}/baseline_train.log" 2>&1 &
  BASELINE_TRAIN_PID=$!
  run_python_on_gpu "${IMPROVED_GPU}" \
    "${IMPROVED_TRAIN_ARGS[@]}" \
    > "${LOG_DIR}/improved_train.log" 2>&1 &
  IMPROVED_TRAIN_PID=$!

  echo "Baseline train PID: ${BASELINE_TRAIN_PID} (log: ${LOG_DIR}/baseline_train.log)"
  echo "Improved train PID: ${IMPROVED_TRAIN_PID} (log: ${LOG_DIR}/improved_train.log)"

  wait_for_named_job "Baseline training" "${BASELINE_TRAIN_PID}"
  wait_for_named_job "Improved training" "${IMPROVED_TRAIN_PID}"
fi

BASELINE_CKPT="${BASELINE_WORK_DIR}/latest.pth"
IMPROVED_CKPT="${IMPROVED_WORK_DIR}/latest.pth"
BASELINE_EVAL_ARGS=(
  tools/test_widerface_enhanced.py
  "${BASELINE_CONFIG}"
  "${BASELINE_CKPT}"
  --out "${BASELINE_RESULT_DIR}"
  --mode "${TEST_MODE}"
  --thr "${SCORE_THR}"
)
IMPROVED_EVAL_ARGS=(
  tools/test_widerface_enhanced.py
  "${IMPROVED_CONFIG}"
  "${IMPROVED_CKPT}"
  --out "${IMPROVED_RESULT_DIR}"
  --mode "${TEST_MODE}"
  --thr "${SCORE_THR}"
)

if [[ "${SAVE_PREDS}" == "1" ]]; then
  BASELINE_EVAL_ARGS+=(--save-preds)
  IMPROVED_EVAL_ARGS+=(--save-preds)
fi

if [[ ! -f "${BASELINE_CKPT}" ]]; then
  echo "Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${IMPROVED_CKPT}" ]]; then
  echo "Missing improved checkpoint: ${IMPROVED_CKPT}" >&2
  exit 1
fi

if [[ "${SINGLE_GPU_MODE}" == "1" ]]; then
  echo "[2/4] Single-GPU mode: evaluate baseline then improved on GPU ${BASELINE_GPU}"
  run_python_on_gpu "${BASELINE_GPU}" \
    "${BASELINE_EVAL_ARGS[@]}" \
    > "${LOG_DIR}/baseline_eval.log" 2>&1
  run_python_on_gpu "${IMPROVED_GPU}" \
    "${IMPROVED_EVAL_ARGS[@]}" \
    > "${LOG_DIR}/improved_eval.log" 2>&1
else
  echo "[2/4] Evaluate both models on WIDERFace in parallel"
  run_python_on_gpu "${BASELINE_GPU}" \
    "${BASELINE_EVAL_ARGS[@]}" \
    > "${LOG_DIR}/baseline_eval.log" 2>&1 &
  BASELINE_EVAL_PID=$!
  run_python_on_gpu "${IMPROVED_GPU}" \
    "${IMPROVED_EVAL_ARGS[@]}" \
    > "${LOG_DIR}/improved_eval.log" 2>&1 &
  IMPROVED_EVAL_PID=$!

  echo "Baseline eval PID: ${BASELINE_EVAL_PID} (log: ${LOG_DIR}/baseline_eval.log)"
  echo "Improved eval PID: ${IMPROVED_EVAL_PID} (log: ${LOG_DIR}/improved_eval.log)"

  wait_for_named_job "Baseline evaluation" "${BASELINE_EVAL_PID}"
  wait_for_named_job "Improved evaluation" "${IMPROVED_EVAL_PID}"
fi

echo "[3/4] Build comparison report"
"${PYRUN[@]}" tools/compare_widerface_results.py \
  --baseline "${BASELINE_RESULT_DIR}" \
  --improved "${IMPROVED_RESULT_DIR}" \
  --baseline-name "${BASELINE_NAME}" \
  --improved-name "${IMPROVED_NAME}" \
  --out-dir "${COMPARE_RESULT_DIR}"

echo "[4/4] Done"
echo "Baseline results:  ${BASELINE_RESULT_DIR}"
echo "Improved results:  ${IMPROVED_RESULT_DIR}"
echo "Comparison report: ${COMPARE_RESULT_DIR}/comparison.md"
echo "Train/eval logs:   ${LOG_DIR}"
