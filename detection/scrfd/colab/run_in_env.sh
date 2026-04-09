#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${SCRFD_COLAB_ENV:-scrfd-colab}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/content/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/content/bin/micromamba}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

exec "${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" "$@"
