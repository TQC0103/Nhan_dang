#!/bin/bash
# Download WIDERFace dataset and SCRFD pretrained models

set -e

echo "=== Downloading SCRFD Data ==="

# Create data directory
mkdir -p data/retinaface
cd data/retinaface

# Download WIDERFace dataset
# 1. Download from official source
WIDER_URL="https://wu-kan.cn/WIDERFACE/widerface.zip"
echo "Downloading WIDERFace dataset from $WIDER_URL ..."
wget -c "$WIDER_URL" -O widerface.zip

# 2. Extract
echo "Extracting WIDERFace..."
unzip -q widerface.zip
rm widerface.zip

# 3. Download annotation file (labelv2.txt)
# From Google Drive link mentioned in README
echo "Downloading annotation file..."
# Note: You need to manually download from:
# https://drive.google.com/file/d/1UW3KoApOhusyqSHX96yEDRYiNkd3Iv3Z/view?usp=sharing
# And place as data/retinaface/train/labelv2.txt

# Download pretrained SCRFD models
cd ../..
mkdir -p weights
cd weights

echo "Downloading SCRFD pretrained models..."
# SCRFD 500M
wget -c "https://1drv.ms/u/s!AswpsDO2toNKqyYWxScdiTITY4TQ?e=DjXof9" -O SCRFD_500M.onnx

# SCRFD 2.5G
wget -c "https://1drv.ms/u/s!AswpsDO2toNKqyTIXnzB1ujPq4th?e=5t1VNv" -O SCRFD_2.5G.onnx

# SCRFD 10G
wget -c "https://1drv.ms/u/s!AswpsDO2toNKqyUKwTiwXv2kaa8o?e=umfepO" -O SCRFD_10G.onnx

echo "=== Download Complete ==="
echo "IMPORTANT: Manual steps required:"
echo "1. Download labelv2.txt from Google Drive and place in data/retinaface/train/"
echo "2. Prepare the final SCRFD data layout with prepare_retinaface_data.sh if needed"
echo "3. Convert ONNX models to PyTorch checkpoints using tools/scrfd2onnx.py"
