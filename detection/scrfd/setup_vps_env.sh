#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRFD_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRFD_DIR}/../.." && pwd)"

ENV_NAME="${SCRFD_ENV_NAME:-scrfd-vps}"
PYTHON_VERSION="${SCRFD_PYTHON_VERSION:-3.8}"
PYTORCH_VERSION="${SCRFD_TORCH_VERSION:-1.10.0}"
TORCHVISION_VERSION="${SCRFD_TORCHVISION_VERSION:-0.11.0}"
TORCHAUDIO_VERSION="${SCRFD_TORCHAUDIO_VERSION:-0.10.0}"
CUDA_TOOLKIT_VERSION="${SCRFD_CUDATOOLKIT_VERSION:-11.3}"
MMCV_VERSION="${SCRFD_MMCV_VERSION:-1.4.0}"

INSTALL_SYSTEM_DEPS="${SCRFD_INSTALL_SYSTEM_DEPS:-1}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/.local/bin/micromamba}"
MICROMAMBA_PARENT="$(dirname "${MICROMAMBA_BIN}")"

print_step() {
  echo
  echo "[$1] $2"
}

run_in_env() {
  "${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" "$@"
}

env_exists() {
  "${MICROMAMBA_BIN}" env list -r "${MAMBA_ROOT_PREFIX}" | awk '{print $1}' | grep -Fxq "${ENV_NAME}"
}

if [[ ! -d "${SCRFD_DIR}" ]]; then
  echo "SCRFD directory not found: ${SCRFD_DIR}" >&2
  exit 1
fi

mkdir -p "${MICROMAMBA_PARENT}" "${MAMBA_ROOT_PREFIX}"

print_step "1/8" "Optional system package setup"
if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  if [[ "${EUID}" -eq 0 ]]; then
    apt-get update
    apt-get install -y \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      pkg-config \
      unzip \
      wget
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      pkg-config \
      unzip \
      wget
  else
    echo "Skipping apt packages: no root/sudo. Set SCRFD_INSTALL_SYSTEM_DEPS=0 to silence this."
  fi
else
  echo "Skipping system package setup."
fi

print_step "2/8" "Installing micromamba"
if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "${MICROMAMBA_PARENT}" bin/micromamba
  if [[ ! -x "${MICROMAMBA_PARENT}/bin/micromamba" && -x "${MICROMAMBA_PARENT}/micromamba" ]]; then
    mv "${MICROMAMBA_PARENT}/micromamba" "${MICROMAMBA_BIN}"
  elif [[ -x "${MICROMAMBA_PARENT}/bin/micromamba" ]]; then
    mv "${MICROMAMBA_PARENT}/bin/micromamba" "${MICROMAMBA_BIN}"
    rmdir "${MICROMAMBA_PARENT}/bin" 2>/dev/null || true
  fi
else
  echo "micromamba already exists: ${MICROMAMBA_BIN}"
fi

print_step "3/8" "Creating environment ${ENV_NAME}"
if env_exists; then
  echo "Environment already exists, reusing it: ${ENV_NAME}"
else
  "${MICROMAMBA_BIN}" create -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
    -c conda-forge -c pytorch \
    "python=${PYTHON_VERSION}" pip
fi

print_step "4/8" "Installing PyTorch stack"
"${MICROMAMBA_BIN}" install -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
  -c pytorch -c conda-forge \
  "pytorch=${PYTORCH_VERSION}" \
  "torchvision=${TORCHVISION_VERSION}" \
  "torchaudio=${TORCHAUDIO_VERSION}" \
  "cudatoolkit=${CUDA_TOOLKIT_VERSION}"
"${MICROMAMBA_BIN}" install -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
  -c conda-forge \
  "mkl=2024.0.0"
run_in_env python - <<'PY'
import torch
print("Torch import OK:", torch.__version__)
PY

print_step "5/8" "Installing Python dependencies"
run_in_env python -m pip install -U "pip<24.1" "setuptools<58" "wheel<0.38"
run_in_env python -m pip install \
  "numpy<2" \
  "cython<3" \
  matplotlib scipy Pillow tqdm terminaltables \
  "yapf<0.40.2" \
  tensorboard opencv-python==4.8.1.78 \
  onnxruntime-gpu==1.14.0 "pycocotools>=2.0.6"

print_step "6/8" "Installing mmcv-full ${MMCV_VERSION}"
run_in_env python -m pip uninstall -y mmcv mmcv-full mmdet || true
run_in_env python -m pip install \
  "mmcv-full==${MMCV_VERSION}" \
  -f "https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"

print_step "7/8" "Installing local SCRFD package"
run_in_env python -m pip install -r "${SCRFD_DIR}/requirements/build.txt"
run_in_env python -m pip install -v -e "${SCRFD_DIR}"

print_step "8/8" "Running sanity checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

run_in_env python - <<'PY'
import os
import sys
import torch
import mmcv
import mmdet

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("MMCV:", mmcv.__version__)
print("MMDet:", mmdet.__version__)

from mmdet.models.detectors.scrfd import SCRFD
from mmdet.models.detectors.scrfd_kd import SCRFDKD
from mmdet.models.backbones.mobilenet_v1_ks import MobileNetV1KS

print("SCRFD import OK")
print("SCRFDKD import OK")
print("MobileNetV1KS import OK")
PY

cat <<EOF

VPS setup completed.

Repo root:
  ${REPO_ROOT}

SCRFD dir:
  ${SCRFD_DIR}

Environment:
  ${ENV_NAME}

Run commands inside the environment with:
  ${MICROMAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} <command>

Examples:
  ${MICROMAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} python ${SCRFD_DIR}/tools/print_config.py ${SCRFD_DIR}/configs/scrfd/scrfd_500m.py
  ${MICROMAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} bash ${SCRFD_DIR}/tools/dist_train.sh ${SCRFD_DIR}/configs/scrfd/scrfd_500m.py 8

If your VPS already has required system packages, you can skip apt by running:
  SCRFD_INSTALL_SYSTEM_DEPS=0 bash ${SCRFD_DIR}/setup_vps_env.sh
EOF
