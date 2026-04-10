#!/usr/bin/env bash

set -euo pipefail

GPU="${1:-0}"
GROUP="${2:-scrfdgen2.5g}"
IDX_FROM="${3:-1}"
IDX_TO="${4:-320}"
OUTPUT_DIR="${5:-wouts}"
THR="${6:-0.02}"
PREFIX="${7:-$GROUP}"

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

for ((i=IDX_FROM; i<=IDX_TO; i++))
do
    TASK="$PREFIX"_"$i"
    echo $TASK
    CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON_CMD[@]}" -u tools/test_widerface.py ./configs/"$GROUP"/"$TASK".py ./work_dirs/"$TASK"/latest.pth --mode 0 --thr "$THR" --out "$OUTPUT_DIR"/"$GROUP"/"$TASK"
done

