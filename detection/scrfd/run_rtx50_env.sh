#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${SCRFD_ENV_NAME:-scrfd-rtx50}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/.local/bin/micromamba}"
MMCV_MAX_VERSION="${SCRFD_MMCV_MAX_VERSION:-1.7.2}"
TORCH_SHARING_STRATEGY="${SCRFD_TORCH_SHARING_STRATEGY:-file_system}"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash ${SCRIPT_DIR}/run_rtx50_env.sh <command> [args...]" >&2
  exit 1
fi

exec "${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
  env \
  SCRFD_MMCV_MAX_VERSION="${MMCV_MAX_VERSION}" \
  SCRFD_TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY}" \
  "$@"
