#!/bin/bash
# Setup script for SCRFD experiments on remote server

set -euo pipefail

echo "=== SCRFD Environment Setup ==="

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Use setup_vps_env.sh instead, or install conda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment
conda create -n scrfd python=3.8 -y
conda activate scrfd

# Install PyTorch with CUDA
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch -c nvidia

# Install mmcv and mmdet
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/torch1.10.0/cu113/index.html
pip install mmdet==2.17.0

# Install other dependencies
pip install -U "pip<24.1" "setuptools<58" "wheel<0.38"
conda install -y -c conda-forge configspace=0.4.11
pip install onnxruntime-gpu==1.14.0
pip install opencv-python==4.5.5.64
pip install Pillow scipy

# Install autotorch for architecture search
pip install --no-build-isolation autotorch

echo "=== Setup Complete ==="
echo "Activate environment: conda activate scrfd"
