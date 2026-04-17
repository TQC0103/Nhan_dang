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

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_baseline.py}"
OSH_CONFIG="${OSH_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_online_scheduler_handoff.py}"
ASR_JSAR_CONFIG="${ASR_JSAR_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar.py}"
COMBO_CONFIG="${COMBO_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_online_scheduler_handoff_asr_jsar.py}"

WORK_ROOT="${WORK_ROOT:-work_dirs/ablation_2.5g_cosine}"
RESULT_ROOT="${RESULT_ROOT:-results/ablation_2.5g_cosine}"
LOG_DIR="${RESULT_ROOT}/logs"
COMPARE_RESULT_DIR="${RESULT_ROOT}/comparison"

TEST_MODE="${TEST_MODE:-0}"
SCORE_THR="${SCORE_THR:-0.02}"
SAVE_PREDS="${SAVE_PREDS:-0}"

LATENCY_WARMUP="${LATENCY_WARMUP:-30}"
LATENCY_REPEAT="${LATENCY_REPEAT:-200}"
LATENCY_BATCH_SIZE="${LATENCY_BATCH_SIZE:-1}"
LATENCY_HEIGHT="${LATENCY_HEIGHT:-640}"
LATENCY_WIDTH="${LATENCY_WIDTH:-640}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_LATENCY="${SKIP_LATENCY:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-}"

BASELINE_BATCH_SIZE_PER_GPU="${BASELINE_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
BASELINE_WORKERS_PER_GPU="${BASELINE_WORKERS_PER_GPU:-${WORKERS_PER_GPU}}"
BASELINE_LR="${BASELINE_LR:-}"
BASELINE_GPU="${BASELINE_GPU:-0}"
BASELINE_EXTRA_CFG_OPTIONS="${BASELINE_EXTRA_CFG_OPTIONS:-}"

OSH_BATCH_SIZE_PER_GPU="${OSH_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
OSH_WORKERS_PER_GPU="${OSH_WORKERS_PER_GPU:-${WORKERS_PER_GPU}}"
OSH_LR="${OSH_LR:-}"
OSH_GPU="${OSH_GPU:-0}"
OSH_EXTRA_CFG_OPTIONS="${OSH_EXTRA_CFG_OPTIONS:-}"

ASR_JSAR_BATCH_SIZE_PER_GPU="${ASR_JSAR_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
ASR_JSAR_WORKERS_PER_GPU="${ASR_JSAR_WORKERS_PER_GPU:-${WORKERS_PER_GPU}}"
ASR_JSAR_LR="${ASR_JSAR_LR:-}"
ASR_JSAR_GPU="${ASR_JSAR_GPU:-0}"
ASR_JSAR_EXTRA_CFG_OPTIONS="${ASR_JSAR_EXTRA_CFG_OPTIONS:-}"

COMBO_BATCH_SIZE_PER_GPU="${COMBO_BATCH_SIZE_PER_GPU:-${BATCH_SIZE_PER_GPU}}"
COMBO_WORKERS_PER_GPU="${COMBO_WORKERS_PER_GPU:-${WORKERS_PER_GPU}}"
COMBO_LR="${COMBO_LR:-}"
COMBO_GPU="${COMBO_GPU:-0}"
COMBO_EXTRA_CFG_OPTIONS="${COMBO_EXTRA_CFG_OPTIONS:-}"

EXTRA_CFG_OPTIONS="${EXTRA_CFG_OPTIONS:-}"

run_python_on_gpu() {
  local gpu_id="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYRUN[@]}" "$@"
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
  if [[ -z "${raw_options}" ]]; then
    return 0
  fi
  # shellcheck disable=SC2206
  local parsed_options=( ${raw_options} )
  local token=""
  for token in "${parsed_options[@]}"; do
    cfg_ref+=("${token}")
  done
}

build_cfg_options() {
  local batch_size="$1"
  local workers="$2"
  local manual_lr="$3"
  local extra_options="$4"
  local -a cfg_options=()

  if [[ -n "${batch_size}" ]]; then
    append_cfg_option cfg_options "data.samples_per_gpu" "${batch_size}"
  fi
  if [[ -n "${workers}" ]]; then
    append_cfg_option cfg_options "data.workers_per_gpu" "${workers}"
  fi
  if [[ -n "${manual_lr}" ]]; then
    append_cfg_option cfg_options "optimizer.lr" "${manual_lr}"
  fi

  append_cfg_options_from_string cfg_options "${EXTRA_CFG_OPTIONS}"
  append_cfg_options_from_string cfg_options "${extra_options}"

  if (( ${#cfg_options[@]} > 0 )); then
    printf '%s\n' "${cfg_options[@]}"
  fi
}

require_config() {
  local label="$1"
  local path="$2"
  if [[ -z "${path}" ]]; then
    echo "${label} is not set." >&2
    exit 1
  fi
  if [[ ! -f "${REPO_ROOT}/${path}" && ! -f "${path}" ]]; then
    echo "Missing config for ${label}: ${path}" >&2
    exit 1
  fi
}

run_one_experiment() {
  local name="$1"
  local config="$2"
  local gpu="$3"
  local batch_size="$4"
  local workers="$5"
  local manual_lr="$6"
  local extra_options="$7"

  local work_dir="${WORK_ROOT}/${name}"
  local result_dir="${RESULT_ROOT}/${name}"
  local train_log="${LOG_DIR}/${name}_train.log"
  local eval_log="${LOG_DIR}/${name}_eval.log"
  local latency_log="${LOG_DIR}/${name}_latency.log"
  local ckpt_path="${work_dir}/latest.pth"
  local -a cfg_options=()

  mkdir -p "${work_dir}" "${result_dir}" "${LOG_DIR}"
  mapfile -t cfg_options < <(build_cfg_options "${batch_size}" "${workers}" "${manual_lr}" "${extra_options}")

  echo "============================================================"
  echo "Experiment: ${name}"
  echo "Config:     ${config}"
  echo "GPU:        ${gpu}"
  if (( ${#cfg_options[@]} > 0 )); then
    echo "Overrides:  ${cfg_options[*]}"
  else
    echo "Overrides:  (none)"
  fi

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    local -a train_args=(tools/train.py "${config}" --work-dir "${work_dir}")
    if (( ${#cfg_options[@]} > 0 )); then
      train_args+=(--cfg-options "${cfg_options[@]}")
    fi
    echo "[train] ${name} -> ${train_log}"
    run_python_on_gpu "${gpu}" "${train_args[@]}" > "${train_log}" 2>&1
  fi

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "Missing checkpoint for ${name}: ${ckpt_path}" >&2
    exit 1
  fi

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    local -a eval_args=(
      tools/test_widerface_enhanced.py
      "${config}"
      "${ckpt_path}"
      --out "${result_dir}"
      --mode "${TEST_MODE}"
      --thr "${SCORE_THR}"
    )
    if [[ "${SAVE_PREDS}" == "1" ]]; then
      eval_args+=(--save-preds)
    fi
    echo "[eval] ${name} -> ${eval_log}"
    run_python_on_gpu "${gpu}" "${eval_args[@]}" > "${eval_log}" 2>&1
  fi

  if [[ "${SKIP_LATENCY}" != "1" ]]; then
    echo "[latency] ${name} -> ${latency_log}"
    run_python_on_gpu "${gpu}" \
      tools/profile_detector_runtime.py \
      "${config}" \
      "${ckpt_path}" \
      --out "${result_dir}/latency_summary.json" \
      --device cuda:0 \
      --shape "${LATENCY_HEIGHT}" "${LATENCY_WIDTH}" \
      --batch-size "${LATENCY_BATCH_SIZE}" \
      --warmup "${LATENCY_WARMUP}" \
      --repeat "${LATENCY_REPEAT}" \
      > "${latency_log}" 2>&1
  fi
}

mkdir -p "${WORK_ROOT}" "${RESULT_ROOT}" "${LOG_DIR}" "${COMPARE_RESULT_DIR}"
cd "${REPO_ROOT}"

require_config "BASELINE_CONFIG" "${BASELINE_CONFIG}"
require_config "ASR_JSAR_CONFIG" "${ASR_JSAR_CONFIG}"
require_config "OSH_CONFIG" "${OSH_CONFIG}"
require_config "COMBO_CONFIG" "${COMBO_CONFIG}"

run_one_experiment \
  "baseline" \
  "${BASELINE_CONFIG}" \
  "${BASELINE_GPU}" \
  "${BASELINE_BATCH_SIZE_PER_GPU}" \
  "${BASELINE_WORKERS_PER_GPU}" \
  "${BASELINE_LR}" \
  "${BASELINE_EXTRA_CFG_OPTIONS}"

run_one_experiment \
  "online_scheduler_handoff" \
  "${OSH_CONFIG}" \
  "${OSH_GPU}" \
  "${OSH_BATCH_SIZE_PER_GPU}" \
  "${OSH_WORKERS_PER_GPU}" \
  "${OSH_LR}" \
  "${OSH_EXTRA_CFG_OPTIONS}"

run_one_experiment \
  "asr_jsar" \
  "${ASR_JSAR_CONFIG}" \
  "${ASR_JSAR_GPU}" \
  "${ASR_JSAR_BATCH_SIZE_PER_GPU}" \
  "${ASR_JSAR_WORKERS_PER_GPU}" \
  "${ASR_JSAR_LR}" \
  "${ASR_JSAR_EXTRA_CFG_OPTIONS}"

run_one_experiment \
  "online_scheduler_handoff_asr_jsar" \
  "${COMBO_CONFIG}" \
  "${COMBO_GPU}" \
  "${COMBO_BATCH_SIZE_PER_GPU}" \
  "${COMBO_WORKERS_PER_GPU}" \
  "${COMBO_LR}" \
  "${COMBO_EXTRA_CFG_OPTIONS}"

if [[ "${SKIP_COMPARE}" != "1" ]]; then
  echo "[compare] building ablation report"
  "${PYRUN[@]}" tools/compare_ablation_suite.py \
    --baseline baseline \
    --title "SCRFD 2.5G / 80e / Cosine Ablation Study" \
    --experiment "baseline=${RESULT_ROOT}/baseline" \
    --experiment "online_scheduler_handoff=${RESULT_ROOT}/online_scheduler_handoff" \
    --experiment "asr_jsar=${RESULT_ROOT}/asr_jsar" \
    --experiment "online_scheduler_handoff_asr_jsar=${RESULT_ROOT}/online_scheduler_handoff_asr_jsar" \
    --out-dir "${COMPARE_RESULT_DIR}"
fi

echo "Ablation suite artifacts:"
echo "  work root:    ${WORK_ROOT}"
echo "  result root:  ${RESULT_ROOT}"
echo "  compare file: ${COMPARE_RESULT_DIR}/ablation_suite.md"
echo "  logs:         ${LOG_DIR}"
