#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
RUNNER="${SCRIPT_DIR}/colab/run_in_env.sh"

SCRFD_ENV_NAME="${SCRFD_ENV_NAME:-scrfd-vps}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/.local/micromamba}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-${HOME}/.local/bin/micromamba}"
SCRFD_COLAB_ENV="${SCRFD_COLAB_ENV:-${SCRFD_ENV_NAME}}"

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/scrfd/scrfd_2.5g_80e_baseline_paper_sr12.py}"
IMPROVED_CONFIG="${IMPROVED_CONFIG:-configs/scrfd/scrfd_2.5g_80e_asr_jsar_paper_sr12.py}"
BASELINE_NAME="${BASELINE_NAME:-SCRFD-2.5G PaperSR12 Baseline 80e}"
IMPROVED_NAME="${IMPROVED_NAME:-SCRFD-2.5G PaperSR12 ASR+JSAR 80e}"

WORK_ROOT="${WORK_ROOT:-work_dirs/compare_2.5g_paper_sr12}"
RESULT_ROOT="${RESULT_ROOT:-results/compare_2.5g_paper_sr12}"
EXPORT_ROOT="${EXPORT_ROOT:-${RESULT_ROOT}/analysis_artifacts_bundle}"
EXPORT_ZIP_NAME="${EXPORT_ZIP_NAME:-analysis_artifacts_bundle.zip}"
COPY_CHECKPOINTS="${COPY_CHECKPOINTS:-latest}"
SAVE_PREDS="${SAVE_PREDS:-1}"

copy_tree_if_exists() {
  local src_dir="$1"
  local dst_dir="$2"
  if [[ -d "${src_dir}" ]]; then
    mkdir -p "${dst_dir}"
    cp -a "${src_dir}/." "${dst_dir}/"
  fi
}

run_bundle_zip_refresh() {
  local bundle_root="$1"
  local zip_path="$2"
  "${MICROMAMBA_BIN}" run -r "${MAMBA_ROOT_PREFIX}" -n "${SCRFD_COLAB_ENV}" python - <<PY
import os
import os.path as osp
import zipfile

bundle_root = osp.abspath(${bundle_root@Q})
zip_path = osp.abspath(${zip_path@Q})

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for root, _, files in os.walk(bundle_root):
        for file_name in files:
            src_path = osp.join(root, file_name)
            if osp.abspath(src_path) == zip_path:
                continue
            archive.write(src_path, osp.relpath(src_path, bundle_root))
PY
}

if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  echo "micromamba not found: ${MICROMAMBA_BIN}" >&2
  echo "Run setup_vps_env.sh first or override MICROMAMBA_BIN/MAMBA_ROOT_PREFIX." >&2
  exit 1
fi

cd "${REPO_ROOT}"

export SCRFD_COLAB_ENV
export MAMBA_ROOT_PREFIX
export MICROMAMBA_BIN
export BASELINE_CONFIG
export IMPROVED_CONFIG
export BASELINE_NAME
export IMPROVED_NAME
export WORK_ROOT
export RESULT_ROOT
export SAVE_PREDS

echo "Using environment: ${SCRFD_COLAB_ENV}"
echo "Repo root: ${REPO_ROOT}"
echo "Work root: ${WORK_ROOT}"
echo "Result root: ${RESULT_ROOT}"
echo "Export root: ${EXPORT_ROOT}"

echo
echo "[1/3] Train + eval + compare"
bash "${SCRIPT_DIR}/colab/run_compare_2.5g_paper_sr12_baseline_vs_asr_jsar.sh"

echo
echo "[2/3] Package analysis artifacts"
rm -rf "${EXPORT_ROOT}"
bash "${RUNNER}" python tools/package_analysis_artifacts.py \
  --experiment baseline "${WORK_ROOT}/baseline" "${RESULT_ROOT}/baseline" "${BASELINE_CONFIG}" \
  --experiment asr_jsar "${WORK_ROOT}/asr_jsar" "${RESULT_ROOT}/asr_jsar" "${IMPROVED_CONFIG}" \
  --copy-checkpoints "${COPY_CHECKPOINTS}" \
  --out-dir "${EXPORT_ROOT}" \
  --zip-name "${EXPORT_ZIP_NAME}"

echo
echo "[3/3] Add compare outputs to bundle"
ANALYSIS_ROOT="${EXPORT_ROOT}/analysis_outputs/baseline_vs_asr_jsar"
copy_tree_if_exists "${RESULT_ROOT}/comparison" "${ANALYSIS_ROOT}/comparison"
copy_tree_if_exists "${RESULT_ROOT}/logs" "${ANALYSIS_ROOT}/logs"
run_bundle_zip_refresh "${EXPORT_ROOT}" "${EXPORT_ROOT}/${EXPORT_ZIP_NAME}"

echo
echo "Done."
echo "Bundle dir: ${EXPORT_ROOT}"
echo "Bundle zip: ${EXPORT_ROOT}/${EXPORT_ZIP_NAME}"
echo "Comparison: ${ANALYSIS_ROOT}/comparison/comparison.md"
echo "Logs:       ${ANALYSIS_ROOT}/logs"
