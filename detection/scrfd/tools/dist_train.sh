#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
PYTHON_BIN="${PYTHON_BIN:-}"
CURRENT_ENV_NAME="${SCRFD_ENV_NAME:-${ENV:-${CONDA_DEFAULT_ENV:-}}}"

if [[ -z "${SCRFD_MMCV_MAX_VERSION:-}" && "${CURRENT_ENV_NAME}" == "scrfd-rtx50" ]]; then
  export SCRFD_MMCV_MAX_VERSION=1.7.2
fi
if [[ -z "${SCRFD_TORCH_SHARING_STRATEGY:-}" && "${CURRENT_ENV_NAME}" == "scrfd-rtx50" ]]; then
  export SCRFD_TORCH_SHARING_STRATEGY=file_system
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find a usable Python interpreter." >&2
    exit 1
  fi
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
"${PYTHON_BIN}" -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
