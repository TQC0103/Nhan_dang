#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASELINE_CONFIG="${BASELINE_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_baseline_paper_sr12.py}"
export IMPROVED_CONFIG="${IMPROVED_CONFIG:-configs/scrfd/scrfd_2.5g_80e_cosine_asr_jsar_paper_sr12.py}"
export BASELINE_NAME="${BASELINE_NAME:-SCRFD-2.5G PaperSR12 Cosine Baseline 80e}"
export IMPROVED_NAME="${IMPROVED_NAME:-SCRFD-2.5G PaperSR12 Cosine ASR+JSAR 80e}"
export WORK_ROOT="${WORK_ROOT:-work_dirs/compare_2.5g_paper_sr12_cosine}"
export RESULT_ROOT="${RESULT_ROOT:-results/compare_2.5g_paper_sr12_cosine}"

exec bash "${SCRIPT_DIR}/run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh"
