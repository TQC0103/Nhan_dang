#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRFD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRFD_DIR}/../.." && pwd)"

RUN_IN_ENV="${SCRIPT_DIR}/run_in_env.sh"
ACTION="${1:-all}"

SEARCH_KIND="${SCRFD_COLAB_SEARCH_KIND:-backbone}"
NUM_CONFIGS="${SCRFD_COLAB_NUM_CONFIGS:-4}"
SEARCH_EPOCHS="${SCRFD_COLAB_SEARCH_EPOCHS:-2}"
GEN_MODE="${SCRFD_COLAB_GEN_MODE:-}"
GEN_SEED="${SCRFD_COLAB_GEN_SEED:-3407}"
OUTPUT_DIR="${SCRFD_COLAB_OUTPUT_DIR:-wouts}"
TOPK="${SCRFD_COLAB_TOPK:-4}"
TEST_THR="${SCRFD_COLAB_TEST_THR:-0.02}"
SAMPLES_PER_GPU="${SCRFD_COLAB_SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${SCRFD_COLAB_WORKERS_PER_GPU:-2}"
OPTIMIZER_LR="${SCRFD_COLAB_LR:-0.005}"
CLEAN_RUN="${SCRFD_COLAB_CLEAN:-1}"

case "${SEARCH_KIND}" in
  backbone)
    DEFAULT_GROUP_NAME="scrfdgen2.5g_colab_quick_demo"
    DEFAULT_TEMPLATE="configs/scrfdgen2.5g/scrfdgen2.5g_0.py"
    DEFAULT_GEN_MODE="1"
    ;;
  all)
    DEFAULT_GROUP_NAME="scrfdgen2.5g_colab_quick_all_demo"
    DEFAULT_TEMPLATE="configs/scrfdgen2.5g/scrfdgen2.5g_0.py"
    DEFAULT_GEN_MODE="2"
    ;;
  kernel_only)
    DEFAULT_GROUP_NAME="scrfdgen2.5g_colab_quick_demo"
    DEFAULT_TEMPLATE="configs/scrfdgen2.5g/scrfdgen2.5g_0.py"
    DEFAULT_GEN_MODE="1"
    echo "SCRFD_COLAB_SEARCH_KIND=kernel_only is mapped to original SCRFD quick generation mode 1." >&2
    ;;
  *)
    echo "Unsupported SCRFD_COLAB_SEARCH_KIND: ${SEARCH_KIND}" >&2
    echo "Use 'backbone' or 'all'." >&2
    exit 1
    ;;
esac

GROUP_NAME="${SCRFD_COLAB_SEARCH_GROUP:-${DEFAULT_GROUP_NAME}}"
GROUP_DIR="configs/${GROUP_NAME}"
TEMPLATE_CONFIG="${SCRFD_COLAB_TEMPLATE:-${DEFAULT_TEMPLATE}}"
GEN_MODE="${GEN_MODE:-${DEFAULT_GEN_MODE}}"
STEP_POINT=$(( SEARCH_EPOCHS > 1 ? SEARCH_EPOCHS - 1 : 1 ))

TRAIN_EXTRA_ARGS="total_epochs=${SEARCH_EPOCHS} checkpoint_config.interval=1 evaluation.interval=1 data.samples_per_gpu=${SAMPLES_PER_GPU} data.workers_per_gpu=${WORKERS_PER_GPU} optimizer.lr=${OPTIMIZER_LR} lr_config.warmup_iters=10 lr_config.step=[${STEP_POINT}] log_config.interval=20"

print_step() {
  echo
  echo "[$1] $2"
}

run_in_env() {
  if [[ -f "${RUN_IN_ENV}" ]]; then
    bash "${RUN_IN_ENV}" "$@"
    return
  fi

  local env_name="${SCRFD_COLAB_ENV:-scrfd-colab}"
  local mamba_root_prefix="${MAMBA_ROOT_PREFIX:-/content/micromamba}"
  local micromamba_bin="${MICROMAMBA_BIN:-/content/bin/micromamba}"

  if [[ ! -x "${micromamba_bin}" ]]; then
    echo "Missing helper runner and micromamba binary: ${RUN_IN_ENV} / ${micromamba_bin}" >&2
    exit 1
  fi

  "${micromamba_bin}" run -r "${mamba_root_prefix}" -n "${env_name}" "$@"
}

require_data() {
  local missing=0
  for path in \
    "data/retinaface/train/labelv2.txt" \
    "data/retinaface/val/labelv2.txt" \
    "data/retinaface/val/gt"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required dataset path: ${SCRFD_DIR}/${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" != "0" ]]; then
    echo "Run detection/scrfd/colab/setup_colab_env.sh first." >&2
    exit 1
  fi
}

clean_run() {
  if [[ "${CLEAN_RUN}" != "1" ]]; then
    return
  fi

  print_step "clean" "Removing previous demo outputs for ${GROUP_NAME}"
  rm -rf "${GROUP_DIR}"
  rm -rf "${OUTPUT_DIR}/${GROUP_NAME}"
  rm -rf "${OUTPUT_DIR}/${GROUP_NAME}_viz"
  shopt -s nullglob
  local work_dirs=(work_dirs/"${GROUP_NAME}"_*)
  if (( ${#work_dirs[@]} > 0 )); then
    rm -rf "${work_dirs[@]}"
  fi
  shopt -u nullglob
}

generate_candidates() {
  print_step "generate" "Generating ${NUM_CONFIGS} quick configs for ${GROUP_NAME}"
  mkdir -p "${GROUP_DIR}"
  run_in_env python search_tools/generate_configs_quick.py \
    --group "${GROUP_DIR}" \
    --template-config "${TEMPLATE_CONFIG}" \
    --mode "${GEN_MODE}" \
    --num-configs "${NUM_CONFIGS}" \
    --seed "${GEN_SEED}"
}

train_candidates() {
  print_step "train" "Training ${NUM_CONFIGS} candidates on 1 GPU"
  require_data
  run_in_env env \
    SCRFD_LIMIT_HOST_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    SCRFD_TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS}" \
    bash search_tools/search_train.sh "${GROUP_NAME}" 1 "${NUM_CONFIGS}" 0 1 0
}

test_candidates() {
  print_step "test" "Evaluating ${NUM_CONFIGS} candidates on WIDERFace"
  require_data
  run_in_env env \
    SCRFD_LIMIT_HOST_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    bash search_tools/search_test_parallel.sh "${GROUP_NAME}" 1 "${NUM_CONFIGS}" 0 "${OUTPUT_DIR}" "${TEST_THR}" "${GROUP_NAME}"
}

visualize_search() {
  print_step "viz" "Summarizing search results"
  run_in_env python search_tools/visualize_search.py \
    --group "${GROUP_DIR}" \
    --result-dir "${OUTPUT_DIR}" \
    --prefix "${GROUP_NAME}" \
    --idx-from 0 \
    --idx-to "${NUM_CONFIGS}" \
    --score-key hard \
    --topk "${TOPK}"
}

print_best_candidate() {
  local best_output
  best_output="$(
    run_in_env python - <<PY
import json
from pathlib import Path

stats_file = Path("${OUTPUT_DIR}/${GROUP_NAME}_viz/search_stats.jsonl")
if not stats_file.exists():
    raise SystemExit("Missing stats file: %s" % stats_file)
first_line = stats_file.read_text(encoding="utf-8").splitlines()[0]
record = json.loads(first_line)
print(record["name"])
print("{:.4f}".format(record["aps"][2]))
PY
  )"
  local best_name
  local best_hard
  best_name="$(printf '%s\n' "${best_output}" | sed -n '1p')"
  best_hard="$(printf '%s\n' "${best_output}" | sed -n '2p')"

  cat <<EOF

Quick search finished.

Best candidate:
  ${best_name}

Hard AP:
  ${best_hard}

Useful outputs:
  ${SCRFD_DIR}/${GROUP_DIR}
  ${SCRFD_DIR}/${OUTPUT_DIR}/${GROUP_NAME}
  ${SCRFD_DIR}/${OUTPUT_DIR}/${GROUP_NAME}_viz/topk_summary.md

To rerun only one step:
  bash detection/scrfd/colab/run_quick_search.sh generate
  bash detection/scrfd/colab/run_quick_search.sh train
  bash detection/scrfd/colab/run_quick_search.sh test
  bash detection/scrfd/colab/run_quick_search.sh viz
EOF
}

main() {
  cd "${SCRFD_DIR}"

  case "${ACTION}" in
    all)
      clean_run
      generate_candidates
      train_candidates
      test_candidates
      visualize_search
      print_best_candidate
      ;;
    generate)
      clean_run
      generate_candidates
      ;;
    train)
      train_candidates
      ;;
    test)
      test_candidates
      ;;
    viz)
      visualize_search
      print_best_candidate
      ;;
    *)
      echo "Usage: $0 [all|generate|train|test|viz]" >&2
      exit 1
      ;;
  esac
}

main "$@"
