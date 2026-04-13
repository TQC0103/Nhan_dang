#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRFD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_IN_ENV="${SCRIPT_DIR}/run_in_env.sh"

MODE="${SCRFD_QUICK_GEN_MODE:-1}"
NUM_CONFIGS="${SCRFD_QUICK_GEN_NUM_CONFIGS:-4}"
SEED="${SCRFD_QUICK_GEN_SEED:-3407}"
CLEAN_GROUP="${SCRFD_QUICK_GEN_CLEAN:-1}"
ALLOW_DUPLICATES="${SCRFD_QUICK_GEN_ALLOW_DUPLICATES:-0}"
GROUP="${SCRFD_QUICK_GEN_GROUP:-configs/scrfdgen2.5g_quick}"
TEMPLATE="${SCRFD_QUICK_GEN_TEMPLATE:-configs/scrfdgen2.5g/scrfdgen2.5g_0.py}"

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

cd "${SCRFD_DIR}"

if [[ "${CLEAN_GROUP}" == "1" ]]; then
  rm -rf "${GROUP}"
fi

mkdir -p "${GROUP}"

ARGS=(
  --group "${GROUP}"
  --template-config "${TEMPLATE}"
  --mode "${MODE}"
  --num-configs "${NUM_CONFIGS}"
  --seed "${SEED}"
)

if [[ "${ALLOW_DUPLICATES}" == "1" ]]; then
  ARGS+=(--allow-duplicates)
fi

run_in_env python search_tools/generate_configs_quick.py "${ARGS[@]}"

cat <<EOF

Quick config generation finished.

Group:
  ${SCRFD_DIR}/${GROUP}

Template:
  ${SCRFD_DIR}/${TEMPLATE}

Example:
  bash colab/run_in_env.sh python tools/print_config.py ${GROUP}/$(basename "${GROUP}")_0.py
EOF
