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

python search_tools/parallel_generate.py \
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
