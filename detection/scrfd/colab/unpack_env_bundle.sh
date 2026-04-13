#!/usr/bin/env bash

set -euo pipefail

ENV_ARCHIVE="${1:-}"
TARGET_PREFIX="${2:-/kaggle/working/envs/scrfd-colab}"
REPO_ARCHIVE="${3:-}"
REPO_TARGET="${SCRFD_UNPACK_REPO_TARGET:-/kaggle/working}"
DATASET_ARCHIVE="${4:-}"
DATASET_TARGET="${SCRFD_UNPACK_DATASET_TARGET:-/kaggle/working}"

usage() {
  cat <<EOF
Usage:
  bash $0 <env_archive.tar.gz> [target_prefix] [repo_archive.tar.gz] [dataset_archive.tar.gz]

Examples:
  bash $0 /kaggle/input/mybundle/scrfd-colab.tar.gz
  bash $0 /kaggle/input/mybundle/scrfd-colab.tar.gz /kaggle/working/envs/scrfd-colab /kaggle/input/mybundle/scrfd-colab-repo.tar.gz
EOF
}

if [[ -z "${ENV_ARCHIVE}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${ENV_ARCHIVE}" ]]; then
  echo "Environment archive not found: ${ENV_ARCHIVE}" >&2
  exit 1
fi

mkdir -p "${TARGET_PREFIX}"
tar -xzf "${ENV_ARCHIVE}" -C "${TARGET_PREFIX}"

if [[ -x "${TARGET_PREFIX}/bin/conda-unpack" ]]; then
  "${TARGET_PREFIX}/bin/conda-unpack"
fi

if [[ -n "${REPO_ARCHIVE}" ]]; then
  if [[ ! -f "${REPO_ARCHIVE}" ]]; then
    echo "Repo archive not found: ${REPO_ARCHIVE}" >&2
    exit 1
  fi
  mkdir -p "${REPO_TARGET}"
  tar -xzf "${REPO_ARCHIVE}" -C "${REPO_TARGET}"
fi

if [[ -n "${DATASET_ARCHIVE}" ]]; then
  if [[ ! -f "${DATASET_ARCHIVE}" ]]; then
    echo "Dataset archive not found: ${DATASET_ARCHIVE}" >&2
    exit 1
  fi
  mkdir -p "${DATASET_TARGET}"
  tar -xzf "${DATASET_ARCHIVE}" -C "${DATASET_TARGET}"
fi

cat <<EOF

Environment unpacked.

Python:
  ${TARGET_PREFIX}/bin/python

If you unpacked the repo too, example commands are:
  ${TARGET_PREFIX}/bin/python /kaggle/working/detection/scrfd/tools/print_config.py /kaggle/working/detection/scrfd/configs/scrfd/scrfd_500m.py

Note:
  This does not change the running Kaggle notebook kernel automatically.
  Use the packed env's python binary to run scripts.
EOF
