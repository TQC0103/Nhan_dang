#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

REPO_ROOT="${1:-${SCRFD_REPO_ROOT:-${DEFAULT_REPO_ROOT}}}"
SCRFD_DIR="${REPO_ROOT}/detection/scrfd"

ENV_NAME="${SCRFD_COLAB_ENV:-scrfd-colab}"
WORKSPACE_ROOT="${SCRFD_WORKSPACE_ROOT:-/content}"
PYTHON_VERSION="${SCRFD_PYTHON_VERSION:-3.8}"
PYTORCH_VERSION="${SCRFD_TORCH_VERSION:-1.10.0}"
TORCHVISION_VERSION="${SCRFD_TORCHVISION_VERSION:-0.11.0}"
TORCHAUDIO_VERSION="${SCRFD_TORCHAUDIO_VERSION:-0.10.0}"
CUDA_TOOLKIT_VERSION="${SCRFD_CUDATOOLKIT_VERSION:-11.3}"
MMCV_VERSION="${SCRFD_MMCV_VERSION:-1.4.0}"
MMCV_WHEEL_URL="${SCRFD_MMCV_WHEEL_URL:-https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html}"

INSTALL_SYSTEM_DEPS="${SCRFD_INSTALL_SYSTEM_DEPS:-1}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${WORKSPACE_ROOT}/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${WORKSPACE_ROOT}/bin/micromamba}"
MICROMAMBA_PARENT="$(dirname "${MICROMAMBA_BIN}")"

PREPARE_RETINAFACE="${SCRFD_PREPARE_RETINAFACE:-1}"
DOWNLOAD_RETINAFACE="${SCRFD_DOWNLOAD_RETINAFACE:-1}"
DATA_ROOT="${SCRFD_DATA_ROOT:-${SCRFD_DIR}/data/retinaface}"
DATA_MODE="${SCRFD_DATA_MODE:-copy}"
FORCE_DATA_PREPARE="${SCRFD_FORCE_DATA_PREPARE:-1}"
SOURCE_ROOT="${SCRFD_SOURCE_ROOT:-}"
ANN_ROOT="${SCRFD_ANN_ROOT:-}"
TRAIN_LABEL="${SCRFD_TRAIN_LABEL:-}"
VAL_LABEL="${SCRFD_VAL_LABEL:-}"
GT_DIR="${SCRFD_GT_DIR:-}"

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

if [[ "${DOWNLOAD_RETINAFACE}" == "1" ]]; then
  PREPARE_RETINAFACE="1"
fi

if [[ "${DATA_MODE}" != "copy" && "${DATA_MODE}" != "symlink" ]]; then
  echo "SCRFD_DATA_MODE must be 'copy' or 'symlink', got: ${DATA_MODE}" >&2
  exit 1
fi

mkdir -p "${MICROMAMBA_PARENT}" "${MAMBA_ROOT_PREFIX}" "${SCRFD_DIR}/work_dirs"

print_step "1/9" "Optional system package setup"
if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]] && command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    pkg-config \
    unzip \
    wget
else
  echo "Skipping apt packages."
fi

print_step "2/9" "Installing micromamba"
if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  TMP_DIR="$(mktemp -d)"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "${TMP_DIR}" bin/micromamba
  mv "${TMP_DIR}/bin/micromamba" "${MICROMAMBA_BIN}"
  chmod +x "${MICROMAMBA_BIN}"
  rm -rf "${TMP_DIR}"
else
  echo "micromamba already exists: ${MICROMAMBA_BIN}"
fi

print_step "3/9" "Creating environment ${ENV_NAME}"
if env_exists; then
  echo "Environment already exists, reusing it: ${ENV_NAME}"
else
  "${MICROMAMBA_BIN}" create -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
    -c conda-forge -c pytorch \
    "python=${PYTHON_VERSION}" pip
fi

print_step "4/9" "Installing PyTorch stack"
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
print("CUDA available:", torch.cuda.is_available())
PY

print_step "5/9" "Installing Python dependencies"
run_in_env python -m pip install -U "pip<24.1" "setuptools<58" "wheel<0.38"
run_in_env python -m pip install \
  "numpy<1.24" \
  "cython<3" \
  matplotlib scipy Pillow tqdm terminaltables \
  "yapf<0.40.2" \
  tensorboard gdown \
  opencv-python-headless==4.8.1.78 \
  onnx onnxsim onnxruntime-gpu==1.14.0 \
  "pycocotools>=2.0.6"

print_step "6/9" "Installing mmcv-full ${MMCV_VERSION}"
run_in_env python -m pip uninstall -y mmcv mmcv-full mmdet || true
run_in_env python -m pip install \
  "mmcv-full==${MMCV_VERSION}" \
  -f "${MMCV_WHEEL_URL}"

print_step "7/9" "Installing local SCRFD package"
run_in_env python -m pip install -r "${SCRFD_DIR}/requirements/build.txt"
run_in_env python -m pip install -v -e "${SCRFD_DIR}"

print_step "8/9" "Optional retinaface dataset preparation"
if [[ "${PREPARE_RETINAFACE}" == "1" ]]; then
  PREPARE_ARGS=(--dest-root "${DATA_ROOT}" --mode "${DATA_MODE}")
  if [[ "${FORCE_DATA_PREPARE}" == "1" ]]; then
    PREPARE_ARGS+=(--force)
  fi

  if [[ "${DOWNLOAD_RETINAFACE}" == "1" ]]; then
    PREPARE_ARGS+=(--download-all)
  else
    if [[ -z "${SOURCE_ROOT}" ]]; then
      echo "SCRFD_SOURCE_ROOT is required when SCRFD_PREPARE_RETINAFACE=1 and SCRFD_DOWNLOAD_RETINAFACE=0." >&2
      exit 1
    fi
    PREPARE_ARGS+=(--source-root "${SOURCE_ROOT}")
    if [[ -n "${ANN_ROOT}" ]]; then
      PREPARE_ARGS+=(--ann-root "${ANN_ROOT}")
    fi
    if [[ -n "${TRAIN_LABEL}" ]]; then
      PREPARE_ARGS+=(--train-label "${TRAIN_LABEL}")
    fi
    if [[ -n "${VAL_LABEL}" ]]; then
      PREPARE_ARGS+=(--val-label "${VAL_LABEL}")
    fi
    if [[ -n "${GT_DIR}" ]]; then
      PREPARE_ARGS+=(--gt-dir "${GT_DIR}")
    fi
  fi

  bash "${SCRFD_DIR}/prepare_retinaface_data.sh" "${PREPARE_ARGS[@]}"
else
  echo "Skipping dataset preparation."
fi

print_step "9/9" "Running sanity checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

run_in_env python - <<'PY'
import sys
import torch
import mmcv
import mmdet
import onnx
import onnxruntime as ort

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("MMCV:", mmcv.__version__)
print("MMDet:", mmdet.__version__)
print("ONNX:", onnx.__version__)
print("ONNXRuntime providers:", ort.get_available_providers())

from mmdet.models.detectors.scrfd import SCRFD
from mmdet.models.detectors.scrfd_kd import SCRFDKD
from mmdet.models.backbones.mobilenet_v1_ks import MobileNetV1KS

print("SCRFD import OK")
print("SCRFDKD import OK")
print("MobileNetV1KS import OK")
PY

cat <<EOF

Colab environment is ready.

Repo root:
  ${REPO_ROOT}

SCRFD dir:
  ${SCRFD_DIR}

Workspace root:
  ${WORKSPACE_ROOT}

Environment:
  ${ENV_NAME}

Run commands inside the environment with:
  bash ${SCRFD_DIR}/colab/run_in_env.sh python ${SCRFD_DIR}/tools/print_config.py ${SCRFD_DIR}/configs/scrfd/scrfd_500m.py
  bash ${SCRFD_DIR}/colab/run_in_env.sh bash ${SCRFD_DIR}/tools/dist_train.sh ${SCRFD_DIR}/configs/scrfd/scrfd_500m.py 1

Dataset setup:
  By default this script also downloads and prepares WIDERFace/SCRFD retinaface data into:
    ${DATA_ROOT}

Override examples:
  SCRFD_PREPARE_RETINAFACE=0 SCRFD_DOWNLOAD_RETINAFACE=0 bash ${SCRFD_DIR}/colab/setup_colab_env.sh ${REPO_ROOT}
  SCRFD_DOWNLOAD_RETINAFACE=0 SCRFD_SOURCE_ROOT=/content/WIDERFace SCRFD_ANN_ROOT=/content/retinaface_ann bash ${SCRFD_DIR}/colab/setup_colab_env.sh ${REPO_ROOT}
EOF
