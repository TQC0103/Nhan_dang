#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash tools/run_plot_scale_prob_history.sh <source> <out_dir> [config]" >&2
  exit 1
fi

SOURCE="$1"
OUT_DIR="$2"
CONFIG="${3:-}"

ARGS=(
  --source "${SOURCE}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${CONFIG}" ]]; then
  ARGS+=(--config "${CONFIG}")
fi

python tools/plot_scale_prob_history.py "${ARGS[@]}"
