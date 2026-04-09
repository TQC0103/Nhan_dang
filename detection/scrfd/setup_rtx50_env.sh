#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRFD_DIR="${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRFD_DIR}/../.." && pwd)"

ENV_NAME="${SCRFD_ENV_NAME:-scrfd-rtx50}"
PYTHON_VERSION="${SCRFD_PYTHON_VERSION:-3.10}"
PYTORCH_VERSION="${SCRFD_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${SCRFD_TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${SCRFD_TORCHAUDIO_VERSION:-2.7.1}"
TORCH_INDEX_URL="${SCRFD_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MMCV_VERSION="${SCRFD_MMCV_VERSION:-1.7.2}"
MMCV_GIT_TAG="${SCRFD_MMCV_GIT_TAG:-v1.7.2}"
MMCV_REPO_URL="${SCRFD_MMCV_REPO_URL:-https://github.com/open-mmlab/mmcv.git}"
TORCH_CUDA_ARCH_LIST="${SCRFD_TORCH_CUDA_ARCH_LIST:-12.0}"
MAX_JOBS="${SCRFD_MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
BUILD_MMCV_OPS="${SCRFD_BUILD_MMCV_OPS:-auto}"

INSTALL_SYSTEM_DEPS="${SCRFD_INSTALL_SYSTEM_DEPS:-1}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/.local/bin/micromamba}"
MICROMAMBA_PARENT="$(dirname "${MICROMAMBA_BIN}")"
EXTERNALS_DIR="${SCRFD_EXTERNALS_DIR:-${REPO_ROOT}/.externals}"
MMCV_SRC_DIR="${SCRFD_MMCV_SRC_DIR:-${EXTERNALS_DIR}/mmcv-${MMCV_VERSION}}"

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

mkdir -p "${MICROMAMBA_PARENT}" "${MAMBA_ROOT_PREFIX}" "${EXTERNALS_DIR}"

print_step "1/9" "Optional system package setup"
if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  INSTALL_CMD=()
  if [[ "${EUID}" -eq 0 ]]; then
    INSTALL_CMD=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    INSTALL_CMD=(sudo apt-get)
  fi
  if [[ ${#INSTALL_CMD[@]} -gt 0 ]]; then
    "${INSTALL_CMD[@]}" update
    "${INSTALL_CMD[@]}" install -y \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      ninja-build \
      pkg-config \
      unzip \
      wget
  else
    echo "Skipping apt packages: no root/sudo. Set SCRFD_INSTALL_SYSTEM_DEPS=0 to silence this."
  fi
else
  echo "Skipping system package setup."
fi

print_step "2/9" "Checking CUDA toolkit"
CUDA_HOME=""
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"
  export CUDA_HOME
  nvcc --version
  if [[ "${BUILD_MMCV_OPS}" == "auto" ]]; then
    BUILD_MMCV_OPS="1"
  fi
else
  if [[ "${BUILD_MMCV_OPS}" == "1" ]]; then
    echo "nvcc not found but SCRFD_BUILD_MMCV_OPS=1 was requested." >&2
    exit 1
  fi
  BUILD_MMCV_OPS="0"
  echo "nvcc not found. Falling back to mmcv without compiled ops."
fi

print_step "3/9" "Installing micromamba"
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

print_step "4/9" "Creating environment ${ENV_NAME}"
if env_exists; then
  echo "Environment already exists, reusing it: ${ENV_NAME}"
else
  "${MICROMAMBA_BIN}" create -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
    -c conda-forge \
    "python=${PYTHON_VERSION}" git pip
fi

print_step "5/9" "Installing PyTorch ${PYTORCH_VERSION} (CUDA 12.8 wheels)"
run_in_env python -m pip install -U pip "setuptools<81" wheel
run_in_env python -m pip install \
  "torch==${PYTORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
run_in_env python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

print_step "6/9" "Installing Python dependencies"
run_in_env python -m pip install \
  "numpy<2" \
  "cython<3" \
  packaging psutil ninja \
  matplotlib scipy Pillow tqdm terminaltables \
  "yapf<0.40.2" \
  tensorboard opencv-python-headless==4.10.0.84 \
  "pycocotools>=2.0.6"

print_step "7/9" "Installing MMCV ${MMCV_VERSION}"
run_in_env python -m pip uninstall -y mmcv mmcv-full mmdet || true
if [[ "${BUILD_MMCV_OPS}" == "1" ]]; then
  if [[ ! -d "${MMCV_SRC_DIR}/.git" ]]; then
    git clone "${MMCV_REPO_URL}" "${MMCV_SRC_DIR}"
  fi
  git -C "${MMCV_SRC_DIR}" fetch --tags --force origin
  git -C "${MMCV_SRC_DIR}" checkout "${MMCV_GIT_TAG}"
  run_in_env bash -lc \
    "cd '${MMCV_SRC_DIR}' && \
     export CUDA_HOME='${CUDA_HOME}' && \
     export MMCV_WITH_OPS=1 && \
     export FORCE_CUDA=1 && \
     export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
     export MAX_JOBS='${MAX_JOBS}' && \
     python -m pip install -v -e ."
else
  run_in_env python -m pip install --no-build-isolation "mmcv==${MMCV_VERSION}"
fi

print_step "8/9" "Installing local SCRFD package"
run_in_env python -m pip install -r "${SCRFD_DIR}/requirements/build.txt"
run_in_env env SCRFD_MMCV_MAX_VERSION=1.7.2 python -m pip install -v -e "${SCRFD_DIR}"

print_step "9/9" "Running sanity checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi
run_in_env env SCRFD_MMCV_MAX_VERSION=1.7.2 python - <<'PY'
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

RTX 50 experimental setup completed.

Repo root:
  ${REPO_ROOT}

SCRFD dir:
  ${SCRFD_DIR}

Environment:
  ${ENV_NAME}

MMCV source:
  ${MMCV_SRC_DIR}

Important:
  This setup is experimental. It keeps SCRFD on the MMDetection 2.x path,
  but runs it with PyTorch ${PYTORCH_VERSION} and MMCV ${MMCV_VERSION} on
  a Blackwell GPU path.
  MMCV ops build enabled: ${BUILD_MMCV_OPS}

Use commands through the helper wrapper so SCRFD sees the experimental MMCV
compatibility override:
  bash ${SCRFD_DIR}/run_rtx50_env.sh python ${SCRFD_DIR}/tools/print_config.py ${SCRFD_DIR}/configs/scrfd/scrfd_500m.py
  bash ${SCRFD_DIR}/run_rtx50_env.sh bash ${SCRFD_DIR}/search_tools/search_train.sh scrfdgen500m_kernel 1 1 0 1 0

If you prefer manual commands, export:
  export SCRFD_MMCV_MAX_VERSION=1.7.2
  ${MICROMAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} <command>
EOF
