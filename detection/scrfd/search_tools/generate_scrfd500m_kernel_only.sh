#!/usr/bin/env bash

set -euo pipefail

NUM_CONFIGS="${1:-16}"
GROUP="${2:-configs/scrfd500m_kernel_only}"
WORKERS="${3:-8}"
OVERSAMPLE="${4:-2.0}"
TEMPLATE="${5:-configs/scrfdgen500m/scrfd500m_kernel_seed.py}"
GFLOPS="${6:-0.5}"

if [[ ! "${NUM_CONFIGS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_CONFIGS must be an integer, got: ${NUM_CONFIGS}" >&2
  exit 1
fi

mkdir -p "${GROUP}"

CURRENT_ENV_NAME="${SCRFD_ENV_NAME:-${ENV:-${CONDA_DEFAULT_ENV:-}}}"
if [[ -z "${SCRFD_MMCV_MAX_VERSION:-}" && "${CURRENT_ENV_NAME}" == "scrfd-rtx50" ]]; then
  export SCRFD_MMCV_MAX_VERSION=1.7.2
fi
if [[ -z "${SCRFD_TORCH_SHARING_STRATEGY:-}" && "${CURRENT_ENV_NAME}" == "scrfd-rtx50" ]]; then
  export SCRFD_TORCH_SHARING_STRATEGY=file_system
fi

PYTHON_CMD=()
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=("python")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=("python3")
elif [[ -n "${MM:-}" && -n "${ROOT:-}" && -n "${ENV:-}" ]]; then
  PYTHON_CMD=("${MM}" "run" "-r" "${ROOT}" "-n" "${ENV}" "python")
elif command -v micromamba >/dev/null 2>&1 && [[ -n "${MAMBA_ROOT_PREFIX:-}" && -n "${SCRFD_ENV_NAME:-}" ]]; then
  PYTHON_CMD=("micromamba" "run" "-r" "${MAMBA_ROOT_PREFIX}" "-n" "${SCRFD_ENV_NAME}" "python")
else
  echo "Could not find a usable Python interpreter." >&2
  echo "Set PYTHON_BIN=python3 or export MM/ROOT/ENV for micromamba-based runs." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" search_tools/parallel_generate.py \
  --group "${GROUP}" \
  --template-config "${TEMPLATE}" \
  --mode 1 \
  --kernel-search \
  --kernel-only \
  --gflops "${GFLOPS}" \
  --num-configs "${NUM_CONFIGS}" \
  --workers "${WORKERS}" \
  --oversample-factor "${OVERSAMPLE}" \
  --keep-workdir
