#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${1:-/content/insightface}"
SCRFD_DIR="${REPO_ROOT}/detection/scrfd"

ENV_NAME="${SCRFD_COLAB_ENV:-scrfd-colab}"
PYTHON_VERSION="${SCRFD_PYTHON_VERSION:-3.8}"
PYTORCH_VERSION="${SCRFD_TORCH_VERSION:-1.10.0}"
TORCHVISION_VERSION="${SCRFD_TORCHVISION_VERSION:-0.11.0}"
TORCHAUDIO_VERSION="${SCRFD_TORCHAUDIO_VERSION:-0.10.0}"
CUDA_TOOLKIT_VERSION="${SCRFD_CUDATOOLKIT_VERSION:-11.3}"
MMCV_VERSION="${SCRFD_MMCV_VERSION:-1.4.0}"

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/content/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/content/bin/micromamba}"
MICROMAMBA_PARENT="$(dirname "${MICROMAMBA_BIN}")"

if [[ ! -d "${SCRFD_DIR}" ]]; then
  echo "SCRFD directory not found: ${SCRFD_DIR}" >&2
  exit 1
fi

mkdir -p "${MICROMAMBA_PARENT}" "${MAMBA_ROOT_PREFIX}"

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  echo "[1/7] Installing micromamba into ${MICROMAMBA_PARENT}"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "${MICROMAMBA_PARENT}" bin/micromamba
  if [[ ! -x "${MICROMAMBA_PARENT}/bin/micromamba" && -x "${MICROMAMBA_PARENT}/micromamba" ]]; then
    mv "${MICROMAMBA_PARENT}/micromamba" "${MICROMAMBA_BIN}"
  fi
else
  echo "[1/7] micromamba already available at ${MICROMAMBA_BIN}"
fi

echo "[2/7] Creating/refreshing environment ${ENV_NAME}"
"${MICROMAMBA_BIN}" create -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
  -c conda-forge -c pytorch \
  "python=${PYTHON_VERSION}" pip

echo "[3/7] Installing PyTorch ${PYTORCH_VERSION} with CUDA toolkit ${CUDA_TOOLKIT_VERSION}"
"${MICROMAMBA_BIN}" install -y -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" \
  -c pytorch -c conda-forge \
  "pytorch=${PYTORCH_VERSION}" \
  "torchvision=${TORCHVISION_VERSION}" \
  "torchaudio=${TORCHAUDIO_VERSION}" \
  "cudatoolkit=${CUDA_TOOLKIT_VERSION}"

echo "[4/7] Installing Python build/runtime dependencies"
"${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" python -m pip install -U "pip<25" setuptools wheel
"${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" python -m pip install \
  "numpy<2" \
  "cython<3" \
  matplotlib scipy Pillow tqdm terminaltables \
  autotorch tensorboard gdown opencv-python==4.8.1.78 \
  onnxruntime-gpu==1.14.0 mmpycocotools

echo "[5/7] Installing mmcv-full ${MMCV_VERSION}"
"${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" python -m pip uninstall -y mmcv mmcv-full mmdet || true
"${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" python -m pip install \
  "mmcv-full==${MMCV_VERSION}" \
  -f "https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"

echo "[6/7] Installing local SCRFD package in editable mode"
"${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" python -m pip install -r "${SCRFD_DIR}/requirements/build.txt"
"${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" python -m pip install -v -e "${SCRFD_DIR}"

echo "[7/7] Running sanity checks"
"${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${ENV_NAME}" python - <<'PY'
import sys
import torch
import mmcv
import mmdet

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
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

Colab environment is ready.

Use commands like:
  ${MICROMAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} python detection/scrfd/tools/print_config.py detection/scrfd/configs/scrfd/scrfd_500m.py
  ${MICROMAMBA_BIN} run -r ${MAMBA_ROOT_PREFIX} -n ${ENV_NAME} bash detection/scrfd/tools/dist_train.sh detection/scrfd/configs/scrfd/scrfd_500m.py 1

Repository root:
  ${REPO_ROOT}
EOF
