#!/usr/bin/env bash

set -euo pipefail

GROUP="${1:-scrfdgen2.5g}"
GPUS="${2:-8}"
TASKS_PER_GPU="${3:-8}"
OFFSET="${4:-1}"
CANDIDATES_PER_GPU="${5:-1}"
USE_DIST="${6:-1}"
PORT_BASE="${7:-29100}"

for ((i=0; i<GPUS; i++))
do
    a=$((TASKS_PER_GPU*i+OFFSET))
    b=$((TASKS_PER_GPU*(i+1)+OFFSET))
    echo "$i,$a,$b,$GROUP,candidates_per_gpu=${CANDIDATES_PER_GPU}"
    for ((slot=0; slot<CANDIDATES_PER_GPU; slot++))
    do
        start=$((a+slot))
        if (( start >= b )); then
            continue
        fi
        python -u search_tools/search_train.py \
            "$i" "$start" "$b" "$GROUP" "$USE_DIST" "$PORT_BASE" "$CANDIDATES_PER_GPU" > "gpu${i}_slot${slot}.log" 2>&1 &
    done
done

