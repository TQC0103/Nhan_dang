#!/usr/bin/env bash

set -euo pipefail

GROUP="${1:-scrfdgen2.5g}"
GPUS="${2:-8}"
TASKS_PER_GPU="${3:-8}"
OFFSET="${4:-1}"

for ((i=0; i<GPUS; i++))
do
    a=$((TASKS_PER_GPU*i+OFFSET))
    b=$((TASKS_PER_GPU*(i+1)+OFFSET))
    echo "$i,$a,$b,$GROUP"
    python -u search_tools/search_train.py "$i" "$a" "$b" "$GROUP" > "gpu${i}.log" 2>&1 &
done

