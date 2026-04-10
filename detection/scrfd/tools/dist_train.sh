#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
PYTHON_BIN="${PYTHON_BIN:-}"

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
