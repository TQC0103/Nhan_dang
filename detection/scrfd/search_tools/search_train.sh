#!/usr/bin/env bash

set -euo pipefail

GROUP="${1:-scrfdgen2.5g}"
GPUS="${2:-8}"
TASKS_PER_GPU="${3:-8}"
OFFSET="${4:-1}"
CANDIDATES_PER_GPU="${5:-1}"
USE_DIST="${6:-1}"
PORT_BASE="${7:-29100}"

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

for ((i=0; i<GPUS; i++))
do
    a=$((TASKS_PER_GPU*i+OFFSET))
    b=$((TASKS_PER_GPU*(i+1)+OFFSET))
    echo "$i,$a,$b,$GROUP,candidates_per_gpu=${CANDIDATES_PER_GPU}"
    for ((slot=0; slot<CANDIDATES_PER_GPU; slot++))
    do
        start=$((a+slot))
        if (( start >= b )); then
            continue
        fi
        "${PYTHON_CMD[@]}" -u search_tools/search_train.py \
            "$i" "$start" "$b" "$GROUP" "$USE_DIST" "$PORT_BASE" "$CANDIDATES_PER_GPU" > "gpu${i}_slot${slot}.log" 2>&1 &
    done
done

wait

