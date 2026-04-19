#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
RUNNER="${SCRIPT_DIR}/colab/run_in_env.sh"

SCRFD_ENV_NAME="${SCRFD_ENV_NAME:-scrfd-vps}"
SCRFD_COLAB_ENV="${SCRFD_COLAB_ENV:-${SCRFD_ENV_NAME}}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/.local/bin/micromamba}"

SEARCH_VARIANT="${SEARCH_VARIANT:-backbone}"
NUM_CONFIGS="${NUM_CONFIGS:-16}"
GFLOPS="${GFLOPS:-0.5}"
GENERATE_WORKERS="${GENERATE_WORKERS:-8}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-2.0}"
GENERATE_KEEP_WORKDIR="${GENERATE_KEEP_WORKDIR:-0}"

case "${SEARCH_VARIANT}" in
  backbone)
    GROUP_NAME="${GROUP_NAME:-scrfdgen500m_search16}"
    TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/scrfdgen500m/scrfdgen500m_0.py}"
    SEARCH_MODE="${SEARCH_MODE:-1}"
    KERNEL_SEARCH="${KERNEL_SEARCH:-1}"
    KERNEL_ONLY="${KERNEL_ONLY:-0}"
    ;;
  kernel_only)
    GROUP_NAME="${GROUP_NAME:-scrfd500m_kernel_only_search16}"
    TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/scrfdgen500m/scrfd500m_kernel_seed.py}"
    SEARCH_MODE="${SEARCH_MODE:-1}"
    KERNEL_SEARCH="${KERNEL_SEARCH:-1}"
    KERNEL_ONLY="${KERNEL_ONLY:-1}"
    ;;
  *)
    echo "Unsupported SEARCH_VARIANT=${SEARCH_VARIANT}. Use backbone or kernel_only." >&2
    exit 1
    ;;
esac

GROUP_DIR="configs/${GROUP_NAME}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-80}"
LR_STEPS="${LR_STEPS:-[55,68]}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-80}"
GPUS="${GPUS:-2}"
TASKS_PER_GPU="${TASKS_PER_GPU:-$(( (NUM_CONFIGS + GPUS - 1) / GPUS ))}"
OFFSET="${OFFSET:-0}"
CANDIDATES_PER_GPU="${CANDIDATES_PER_GPU:-1}"
USE_DIST="${USE_DIST:-1}"
PORT_BASE="${PORT_BASE:-29100}"
TEST_THR="${TEST_THR:-0.02}"
TOPK="${TOPK:-10}"
SCORE_KEY="${SCORE_KEY:-hard}"
CLEAN_RUN="${CLEAN_RUN:-1}"
COPY_CHECKPOINTS="${COPY_CHECKPOINTS:-latest}"

LIMIT_HOST_THREADS="${LIMIT_HOST_THREADS:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RESULT_ROOT="${RESULT_ROOT:-wouts}"
RUN_ROOT="${RUN_ROOT:-results/search_500m_16configs}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
VIZ_DIR="${VIZ_DIR:-${RUN_ROOT}/visualization}"
EXPORT_ROOT="${EXPORT_ROOT:-${RUN_ROOT}/analysis_artifacts_bundle}"
EXPORT_ZIP_NAME="${EXPORT_ZIP_NAME:-analysis_artifacts_bundle.zip}"
RUN_METADATA="${RUN_ROOT}/run_metadata.json"

run_in_env() {
  bash "${RUNNER}" "$@"
}

require_data() {
  local missing=0
  for path in \
    "data/retinaface/train/labelv2.txt" \
    "data/retinaface/val/labelv2.txt" \
    "data/retinaface/val/gt"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required dataset path: ${REPO_ROOT}/${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    echo "Prepare the RetinaFace/WIDERFace layout first." >&2
    exit 1
  fi
}

cleanup_logs_in_repo_root() {
  rm -f gpu*_slot*.log test_gpu*.log
}

collect_launcher_logs() {
  local pattern="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  shopt -s nullglob
  local files=(${pattern})
  if (( ${#files[@]} > 0 )); then
    mv "${files[@]}" "${dst_dir}/"
  fi
  shopt -u nullglob
}

candidate_name() {
  local idx="$1"
  printf '%s_%d' "${GROUP_NAME}" "${idx}"
}

verify_generated_configs() {
  local count
  count="$(find "${GROUP_DIR}" -maxdepth 1 -type f -name "${GROUP_NAME}_*.py" | wc -l | tr -d ' ')"
  if [[ "${count}" != "${NUM_CONFIGS}" ]]; then
    echo "Expected ${NUM_CONFIGS} generated configs under ${GROUP_DIR}, found ${count}" >&2
    exit 1
  fi
}

verify_checkpoints() {
  local idx
  for ((idx=0; idx<NUM_CONFIGS; idx++)); do
    local candidate
    candidate="$(candidate_name "${idx}")"
    if [[ ! -f "work_dirs/${candidate}/latest.pth" ]]; then
      echo "Missing checkpoint: ${REPO_ROOT}/work_dirs/${candidate}/latest.pth" >&2
      exit 1
    fi
  done
}

verify_aps() {
  local idx
  for ((idx=0; idx<NUM_CONFIGS; idx++)); do
    local candidate
    candidate="$(candidate_name "${idx}")"
    if [[ ! -f "${RESULT_ROOT}/${GROUP_NAME}/${candidate}/aps" ]]; then
      echo "Missing aps file: ${REPO_ROOT}/${RESULT_ROOT}/${GROUP_NAME}/${candidate}/aps" >&2
      exit 1
    fi
  done
}

write_run_metadata() {
  mkdir -p "${RUN_ROOT}"
  run_in_env python - <<PY
import json
from pathlib import Path

payload = {
    "search_variant": ${SEARCH_VARIANT@Q},
    "group_name": ${GROUP_NAME@Q},
    "group_dir": ${GROUP_DIR@Q},
    "template_config": ${TEMPLATE_CONFIG@Q},
    "num_configs": int(${NUM_CONFIGS}),
    "gflops": float(${GFLOPS}),
    "generate_workers": int(${GENERATE_WORKERS}),
    "oversample_factor": float(${OVERSAMPLE_FACTOR}),
    "search_mode": int(${SEARCH_MODE}),
    "kernel_search": bool(int(${KERNEL_SEARCH})),
    "kernel_only": bool(int(${KERNEL_ONLY})),
    "search_epochs": int(${SEARCH_EPOCHS}),
    "lr_steps": ${LR_STEPS@Q},
    "checkpoint_interval": int(${CHECKPOINT_INTERVAL}),
    "gpus": int(${GPUS}),
    "tasks_per_gpu": int(${TASKS_PER_GPU}),
    "offset": int(${OFFSET}),
    "candidates_per_gpu": int(${CANDIDATES_PER_GPU}),
    "use_dist": bool(int(${USE_DIST})),
    "port_base": int(${PORT_BASE}),
    "test_thr": float(${TEST_THR}),
    "topk": int(${TOPK}),
    "score_key": ${SCORE_KEY@Q},
    "result_root": ${RESULT_ROOT@Q},
    "run_root": ${RUN_ROOT@Q},
    "log_dir": ${LOG_DIR@Q},
    "viz_dir": ${VIZ_DIR@Q},
}
path = Path(${RUN_METADATA@Q})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(path)
PY
}

clean_previous_run() {
  if [[ "${CLEAN_RUN}" != "1" ]]; then
    return
  fi
  rm -rf "${GROUP_DIR}" "${RUN_ROOT}" "${VIZ_DIR}" "${EXPORT_ROOT}"
  rm -rf "${RESULT_ROOT}/${GROUP_NAME}"
  shopt -s nullglob
  local candidate_dirs=(work_dirs/"${GROUP_NAME}"_*)
  if (( ${#candidate_dirs[@]} > 0 )); then
    rm -rf "${candidate_dirs[@]}"
  fi
  shopt -u nullglob
  cleanup_logs_in_repo_root
}

print_step() {
  echo
  echo "[$1] $2"
}

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  echo "micromamba not found: ${MICROMAMBA_BIN}" >&2
  echo "Run setup_vps_env.sh first or override MICROMAMBA_BIN/MAMBA_ROOT_PREFIX." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export SCRFD_COLAB_ENV
export SCRFD_ENV_NAME
export MAMBA_ROOT_PREFIX
export MICROMAMBA_BIN

print_step "info" "Using environment ${SCRFD_COLAB_ENV}"
echo "Repo root: ${REPO_ROOT}"
echo "Group: ${GROUP_NAME}"
echo "Run root: ${RUN_ROOT}"
echo "Bundle root: ${EXPORT_ROOT}"

print_step "prep" "Cleaning previous run directories"
clean_previous_run
mkdir -p "${LOG_DIR}"
require_data

print_step "1/5" "Generating ${NUM_CONFIGS} candidate configs"
GEN_CMD=(python search_tools/parallel_generate.py
  --group "${GROUP_DIR}"
  --template-config "${TEMPLATE_CONFIG}"
  --mode "${SEARCH_MODE}"
  --gflops "${GFLOPS}"
  --num-configs "${NUM_CONFIGS}"
  --workers "${GENERATE_WORKERS}"
  --oversample-factor "${OVERSAMPLE_FACTOR}")
if [[ "${KERNEL_SEARCH}" == "1" ]]; then
  GEN_CMD+=(--kernel-search)
fi
if [[ "${KERNEL_ONLY}" == "1" ]]; then
  GEN_CMD+=(--kernel-only)
fi
if [[ "${GENERATE_KEEP_WORKDIR}" == "1" ]]; then
  GEN_CMD+=(--keep-workdir)
fi
run_in_env "${GEN_CMD[@]}" > "${LOG_DIR}/generate.log" 2>&1
verify_generated_configs

print_step "2/5" "Training ${NUM_CONFIGS} candidates on ${GPUS} GPUs"
cleanup_logs_in_repo_root
SCRFD_TRAIN_EXTRA_ARGS="--cfg-options total_epochs=${SEARCH_EPOCHS} checkpoint_config.interval=${CHECKPOINT_INTERVAL} evaluation.interval=${CHECKPOINT_INTERVAL} lr_config.step=${LR_STEPS}" \
run_in_env env \
  SCRFD_LIMIT_HOST_THREADS="${LIMIT_HOST_THREADS}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
  bash search_tools/search_train.sh "${GROUP_NAME}" "${GPUS}" "${TASKS_PER_GPU}" "${OFFSET}" "${CANDIDATES_PER_GPU}" "${USE_DIST}" "${PORT_BASE}" \
  > "${LOG_DIR}/train_launcher.log" 2>&1
collect_launcher_logs "gpu*_slot*.log" "${LOG_DIR}/train"
verify_checkpoints

print_step "3/5" "Evaluating all candidates on WIDERFace"
cleanup_logs_in_repo_root
run_in_env env \
  SCRFD_LIMIT_HOST_THREADS="${LIMIT_HOST_THREADS}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
  bash search_tools/search_test_parallel.sh "${GROUP_NAME}" "${GPUS}" "${TASKS_PER_GPU}" "${OFFSET}" "${RESULT_ROOT}" "${TEST_THR}" "${GROUP_NAME}" \
  > "${LOG_DIR}/test_launcher.log" 2>&1
collect_launcher_logs "test_gpu*.log" "${LOG_DIR}/test"
verify_aps

print_step "4/5" "Building search visualizations"
mkdir -p "${VIZ_DIR}"
run_in_env env \
  SCRFD_LIMIT_HOST_THREADS="${LIMIT_HOST_THREADS}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
  python search_tools/visualize_search.py \
    --group "${GROUP_DIR}" \
    --result-dir "${RESULT_ROOT}" \
    --prefix "${GROUP_NAME}" \
    --idx-from 0 \
    --idx-to "${NUM_CONFIGS}" \
    --score-key "${SCORE_KEY}" \
    --topk "${TOPK}" \
    --output-dir "${VIZ_DIR}" \
    > "${LOG_DIR}/visualize.log" 2>&1
write_run_metadata

print_step "5/5" "Packaging bundle zip"
rm -rf "${EXPORT_ROOT}"
run_in_env python tools/package_search_artifacts.py \
  --group "${GROUP_DIR}" \
  --result-root "${RESULT_ROOT}" \
  --work-root "work_dirs" \
  --viz-dir "${VIZ_DIR}" \
  --log-dir "${LOG_DIR}" \
  --run-metadata "${RUN_METADATA}" \
  --copy-checkpoints "${COPY_CHECKPOINTS}" \
  --out-dir "${EXPORT_ROOT}" \
  --zip-name "${EXPORT_ZIP_NAME}" \
  > "${LOG_DIR}/package.log" 2>&1

BEST_NAME="$(run_in_env python - <<PY
import json
from pathlib import Path
stats = Path(${VIZ_DIR@Q}) / 'search_stats.jsonl'
if not stats.exists():
    raise SystemExit('')
line = stats.read_text(encoding='utf-8').splitlines()[0]
record = json.loads(line)
print(record['name'])
print('{:.4f}'.format(record['aps'][2]))
PY
)"

echo
echo "Done."
echo "Best candidate:"
printf '%s\n' "${BEST_NAME}" | sed -n '1p'
echo "Best hard AP:"
printf '%s\n' "${BEST_NAME}" | sed -n '2p'
echo "Visualization dir: ${VIZ_DIR}"
echo "Bundle dir: ${EXPORT_ROOT}"
echo "Bundle zip: ${EXPORT_ROOT}/${EXPORT_ZIP_NAME}"
echo "Logs dir: ${LOG_DIR}"
