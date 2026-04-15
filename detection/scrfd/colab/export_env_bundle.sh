#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRFD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_IN_ENV="${SCRIPT_DIR}/run_in_env.sh"

ENV_NAME="${SCRFD_COLAB_ENV:-scrfd-colab}"
WORKSPACE_ROOT="${SCRFD_WORKSPACE_ROOT:-/content}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${WORKSPACE_ROOT}/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${WORKSPACE_ROOT}/bin/micromamba}"
ENV_PREFIX="${SCRFD_EXPORT_ENV_PREFIX:-${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}}"

OUTPUT_DIR="${SCRFD_EXPORT_OUTPUT_DIR:-${WORKSPACE_ROOT}/scrfd_env_bundle}"
BUNDLE_NAME="${SCRFD_EXPORT_BUNDLE_NAME:-${ENV_NAME}}"
ARCHIVE_FORMAT="${SCRFD_EXPORT_ARCHIVE_FORMAT:-zip}"

case "${ARCHIVE_FORMAT}" in
  zip)
    ENV_ARCHIVE="${OUTPUT_DIR}/${BUNDLE_NAME}.zip"
    ;;
  tar.gz|tgz)
    ENV_ARCHIVE="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
    ARCHIVE_FORMAT="tar.gz"
    ;;
  *)
    echo "Unsupported SCRFD_EXPORT_ARCHIVE_FORMAT: ${ARCHIVE_FORMAT}" >&2
    echo "Use 'zip' or 'tar.gz'." >&2
    exit 1
    ;;
esac

FORCE_NONEDITABLE="${SCRFD_EXPORT_FORCE_NONEDITABLE:-1}"
INSTALL_CONDA_PACK="${SCRFD_EXPORT_INSTALL_CONDA_PACK:-1}"

print_step() {
  echo
  echo "[$1] $2"
}

run_in_env() {
  if [[ -f "${RUN_IN_ENV}" ]]; then
    bash "${RUN_IN_ENV}" "$@"
    return
  fi

  if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
    echo "Missing helper runner and micromamba binary: ${RUN_IN_ENV} / ${MICROMAMBA_BIN}" >&2
    exit 1
  fi

  "${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" "$@"
}

if [[ ! -d "${ENV_PREFIX}" ]]; then
  echo "Environment prefix not found: ${ENV_PREFIX}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

print_step "1/3" "Preparing environment for packing"
if [[ "${FORCE_NONEDITABLE}" == "1" ]]; then
  run_in_env python -m pip uninstall -y mmdet || true
  run_in_env python -m pip install --no-deps "${SCRFD_DIR}"
fi

print_step "2/3" "Ensuring conda-pack is available"
if ! run_in_env python -c "import conda_pack" >/dev/null 2>&1; then
  if [[ "${INSTALL_CONDA_PACK}" != "1" ]]; then
    echo "conda-pack is not installed in the environment." >&2
    echo "Set SCRFD_EXPORT_INSTALL_CONDA_PACK=1 or install it manually." >&2
    exit 1
  fi
  run_in_env python -m pip install conda-pack
fi

print_step "3/3" "Packing environment"
rm -f "${ENV_ARCHIVE}"
run_in_env conda-pack -p "${ENV_PREFIX}" -o "${ENV_ARCHIVE}" --format "${ARCHIVE_FORMAT}" --ignore-missing-files

cat <<EOF

Export complete.

Main artifact:
  ${ENV_ARCHIVE}

Use this file as your offline Kaggle environment bundle.

Notes:
  - This script only exports the packed environment archive.
  - Default format is .zip because it is usually more portable for Kaggle uploads than tarballs with Unix symlinks.
  - Your repo can be uploaded separately as a Kaggle Dataset.
  - The export reinstalls local SCRFD/MMDet as a non-editable package before packing,
    so the bundle does not depend on the original Colab path.
EOF
