#!/usr/bin/env bash

set -euo pipefail

GPU="${1:-0}"
GROUP="${2:-scrfdgen2.5g}"
IDX_FROM="${3:-1}"
IDX_TO="${4:-320}"
OUTPUT_DIR="${5:-wouts}"
THR="${6:-0.02}"
PREFIX="${7:-$GROUP}"

for ((i=IDX_FROM; i<=IDX_TO; i++))
do
    TASK="$PREFIX"_"$i"
    echo $TASK
    CUDA_VISIBLE_DEVICES="$GPU" python -u tools/test_widerface.py ./configs/"$GROUP"/"$TASK".py ./work_dirs/"$TASK"/latest.pth --mode 0 --thr "$THR" --out "$OUTPUT_DIR"/"$GROUP"/"$TASK"
done

