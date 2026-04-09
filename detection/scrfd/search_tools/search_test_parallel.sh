#!/usr/bin/env bash

set -euo pipefail

GROUP="${1:-scrfdgen2.5g}"
GPUS="${2:-8}"
TASKS_PER_GPU="${3:-8}"
OFFSET="${4:-1}"
OUTPUT_DIR="${5:-wouts}"
THR="${6:-0.02}"
PREFIX="${7:-$GROUP}"

for ((i=0; i<GPUS; i++))
do
    a=$((TASKS_PER_GPU*i+OFFSET))
    b=$((TASKS_PER_GPU*(i+1)+OFFSET-1))
    echo "$i,$a,$b,$GROUP,$OUTPUT_DIR"
    bash search_tools/search_test.sh "$i" "$GROUP" "$a" "$b" "$OUTPUT_DIR" "$THR" "$PREFIX" > "test_gpu${i}.log" 2>&1 &
done

wait
