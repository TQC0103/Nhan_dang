#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DEST="${SCRIPT_DIR}/data/retinaface"

SOURCE_ROOT=""
ANN_ROOT=""
TRAIN_LABEL=""
VAL_LABEL=""
GT_DIR=""
DEST_ROOT="${DEFAULT_DEST}"
MODE="symlink"
FORCE=0
DOWNLOAD_ALL=0
DOWNLOAD_WIDERFACE=0
DOWNLOAD_ANNOTATIONS=0
DOWNLOAD_ROOT=""
WIDER_TRAIN_URL="${WIDER_TRAIN_URL:-https://data.brainchip.com/dataset-mirror/widerface/WIDER_train.zip}"
WIDER_VAL_URL="${WIDER_VAL_URL:-https://data.brainchip.com/dataset-mirror/widerface/WIDER_val.zip}"
WIDER_SPLIT_URL="${WIDER_SPLIT_URL:-https://data.brainchip.com/dataset-mirror/widerface/wider_face_split.zip}"
ANNOTATION_GDRIVE_ID="${ANNOTATION_GDRIVE_ID:-1UW3KoApOhusyqSHX96yEDRYiNkd3Iv3Z}"

usage() {
  cat <<EOF
Usage:
  bash prepare_retinaface_data.sh --source-root <path> [options]
  bash prepare_retinaface_data.sh --download-all [options]

Prepare the dataset layout expected by SCRFD:
  data/retinaface/
    train/
      images/
      labelv2.txt
    val/
      images/
      labelv2.txt
      gt/

Options:
  --source-root <path>   Source dataset root. Supports either:
                         1. Prepared layout with train/images and val/images
                         2. WIDERFace raw layout with WIDER_train/images and WIDER_val/images
  --ann-root <path>      Annotation root. If omitted, defaults to --source-root.
  --train-label <path>   Explicit path to train/labelv2.txt
  --val-label <path>     Explicit path to val/labelv2.txt
  --gt-dir <path>        Explicit path to validation gt directory containing *.mat
  --dest-root <path>     Destination root. Default: ${DEFAULT_DEST}
  --mode <symlink|copy>  Create symlinks or copy files/directories. Default: symlink
  --download-all         Download WIDERFace raw files and SCRFD annotations automatically.
  --download-widerface   Download WIDER_train/WIDER_val/wider_face_split.
  --download-annotations Download SCRFD annotation bundle from Google Drive.
  --download-root <path> Download/extract cache root. Default: <dest_parent>/.downloads/retinaface
  --wider-train-url <u>  Override WIDER_train.zip URL.
  --wider-val-url <u>    Override WIDER_val.zip URL.
  --wider-split-url <u>  Override wider_face_split.zip URL.
  --annotation-gdrive-id <id>
                         Override the Google Drive file id for SCRFD annotation bundle.
  --force                Replace existing destination entries
  -h, --help             Show this help

Examples:
  bash prepare_retinaface_data.sh \\
    --source-root /datasets/retinaface_prepared

  bash prepare_retinaface_data.sh \\
    --source-root /datasets/WIDERFace \\
    --ann-root /datasets/retinaface_annotations

  bash prepare_retinaface_data.sh \\
    --source-root /datasets/WIDERFace \\
    --train-label /datasets/ann/train/labelv2.txt \\
    --val-label /datasets/ann/val/labelv2.txt \\
    --gt-dir /datasets/ann/val/gt

  bash prepare_retinaface_data.sh \\
    --download-all \\
    --force
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"
  if command -v wget >/dev/null 2>&1; then
    wget -c "${url}" -O "${output}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output}"
  else
    die "Need wget or curl to download files"
  fi
}

download_gdrive_file() {
  local file_id="$1"
  local output="$2"
  local cookie_file=""
  local confirm=""
  mkdir -p "$(dirname "${output}")"
  cookie_file="$(mktemp)"
  if command -v wget >/dev/null 2>&1; then
    confirm="$(
      wget --quiet \
        --save-cookies "${cookie_file}" \
        --keep-session-cookies \
        "https://drive.google.com/uc?export=download&id=${file_id}" \
        -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' | head -n 1
    )"
    if [[ -n "${confirm}" ]]; then
      wget --load-cookies "${cookie_file}" \
        "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" \
        -O "${output}"
    else
      wget "https://drive.google.com/uc?export=download&id=${file_id}" -O "${output}"
    fi
  elif command -v curl >/dev/null 2>&1; then
    curl -c "${cookie_file}" -L \
      "https://drive.google.com/uc?export=download&id=${file_id}" \
      -o /tmp/gdrive_probe.html
    confirm="$(sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' /tmp/gdrive_probe.html | head -n 1)"
    if [[ -n "${confirm}" ]]; then
      curl -L -b "${cookie_file}" \
        "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" \
        -o "${output}"
    else
      curl -L "https://drive.google.com/uc?export=download&id=${file_id}" -o "${output}"
    fi
    rm -f /tmp/gdrive_probe.html
  else
    rm -f "${cookie_file}"
    die "Need wget or curl to download Google Drive files"
  fi
  rm -f "${cookie_file}"
}

extract_zip() {
  local zip_path="$1"
  local dest_dir="$2"
  require_command unzip
  mkdir -p "${dest_dir}"
  unzip -q -o "${zip_path}" -d "${dest_dir}"
}

find_first_existing() {
  local base="$1"
  shift
  local candidate=""
  for candidate in "$@"; do
    if [[ -e "${base}/${candidate}" ]]; then
      printf '%s\n' "${base}/${candidate}"
      return 0
    fi
  done
  return 1
}

find_first_recursive() {
  local base="$1"
  local type_flag="$2"
  local name="$3"
  find "${base}" -maxdepth 5 "${type_flag}" -name "${name}" 2>/dev/null | head -n 1
}

find_train_images() {
  local root="$1"
  local found=""
  found="$(find_first_existing "${root}" \
    train/images \
    WIDER_train/images \
    WIDERFACE/WIDER_train/images \
    widerface/WIDER_train/images || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  find "${root}" -maxdepth 5 -type d \( -path '*/train/images' -o -path '*/WIDER_train/images' \) 2>/dev/null | head -n 1
}

find_val_images() {
  local root="$1"
  local found=""
  found="$(find_first_existing "${root}" \
    val/images \
    WIDER_val/images \
    WIDERFACE/WIDER_val/images \
    widerface/WIDER_val/images || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  find "${root}" -maxdepth 5 -type d \( -path '*/val/images' -o -path '*/WIDER_val/images' \) 2>/dev/null | head -n 1
}

find_train_label() {
  local root="$1"
  local found=""
  found="$(find_first_existing "${root}" \
    train/labelv2.txt \
    labelv2.txt \
    retinaface/train/labelv2.txt || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  find "${root}" -maxdepth 5 -type f -path '*/train/labelv2.txt' 2>/dev/null | head -n 1
}

find_val_label() {
  local root="$1"
  local found=""
  found="$(find_first_existing "${root}" \
    val/labelv2.txt \
    retinaface/val/labelv2.txt || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  find "${root}" -maxdepth 5 -type f -path '*/val/labelv2.txt' 2>/dev/null | head -n 1
}

find_gt_dir() {
  local root="$1"
  local found=""
  found="$(find_first_existing "${root}" \
    val/gt \
    gt \
    eval_tools/ground_truth \
    wider_face_split \
    retinaface/val/gt || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi
  find "${root}" -maxdepth 5 -type d \( -path '*/val/gt' -o -name gt -o -name ground_truth \) 2>/dev/null | head -n 1
}

replace_target() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    if [[ "${FORCE}" != "1" ]]; then
      die "Destination already exists: ${dst} (use --force to replace)"
    fi
    rm -rf "${dst}"
  fi

  if [[ "${MODE}" == "symlink" ]]; then
    ln -s "${src}" "${dst}"
  else
    cp -a "${src}" "${dst}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --ann-root)
      ANN_ROOT="$2"
      shift 2
      ;;
    --train-label)
      TRAIN_LABEL="$2"
      shift 2
      ;;
    --val-label)
      VAL_LABEL="$2"
      shift 2
      ;;
    --gt-dir)
      GT_DIR="$2"
      shift 2
      ;;
    --dest-root)
      DEST_ROOT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --download-all)
      DOWNLOAD_ALL=1
      shift
      ;;
    --download-widerface)
      DOWNLOAD_WIDERFACE=1
      shift
      ;;
    --download-annotations)
      DOWNLOAD_ANNOTATIONS=1
      shift
      ;;
    --download-root)
      DOWNLOAD_ROOT="$2"
      shift 2
      ;;
    --wider-train-url)
      WIDER_TRAIN_URL="$2"
      shift 2
      ;;
    --wider-val-url)
      WIDER_VAL_URL="$2"
      shift 2
      ;;
    --wider-split-url)
      WIDER_SPLIT_URL="$2"
      shift 2
      ;;
    --annotation-gdrive-id)
      ANNOTATION_GDRIVE_ID="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ "${MODE}" == "symlink" || "${MODE}" == "copy" ]] || die "--mode must be symlink or copy"
DEST_ROOT="$(mkdir -p "$(dirname "${DEST_ROOT}")" && cd "$(dirname "${DEST_ROOT}")" && pwd)/$(basename "${DEST_ROOT}")"

if [[ "${DOWNLOAD_ALL}" == "1" ]]; then
  DOWNLOAD_WIDERFACE=1
  DOWNLOAD_ANNOTATIONS=1
fi

if [[ -z "${SOURCE_ROOT}" && "${DOWNLOAD_WIDERFACE}" != "1" ]]; then
  die "--source-root is required unless --download-widerface/--download-all is used"
fi

if [[ -z "${DOWNLOAD_ROOT}" ]]; then
  DOWNLOAD_ROOT="$(dirname "${DEST_ROOT}")/.downloads/retinaface"
else
  DOWNLOAD_ROOT="$(mkdir -p "$(dirname "${DOWNLOAD_ROOT}")" && cd "$(dirname "${DOWNLOAD_ROOT}")" && pwd)/$(basename "${DOWNLOAD_ROOT}")"
fi

if [[ "${DOWNLOAD_WIDERFACE}" == "1" || "${DOWNLOAD_ANNOTATIONS}" == "1" ]]; then
  mkdir -p "${DOWNLOAD_ROOT}"
fi

if [[ "${DOWNLOAD_WIDERFACE}" == "1" ]]; then
  RAW_ROOT="${DOWNLOAD_ROOT}/raw"
  EXTRACT_ROOT="${DOWNLOAD_ROOT}/extracted"
  mkdir -p "${RAW_ROOT}" "${EXTRACT_ROOT}"
  echo "Downloading WIDERFace raw files into ${RAW_ROOT}"
  download_file "${WIDER_TRAIN_URL}" "${RAW_ROOT}/WIDER_train.zip"
  download_file "${WIDER_VAL_URL}" "${RAW_ROOT}/WIDER_val.zip"
  download_file "${WIDER_SPLIT_URL}" "${RAW_ROOT}/wider_face_split.zip"
  extract_zip "${RAW_ROOT}/WIDER_train.zip" "${EXTRACT_ROOT}"
  extract_zip "${RAW_ROOT}/WIDER_val.zip" "${EXTRACT_ROOT}"
  extract_zip "${RAW_ROOT}/wider_face_split.zip" "${EXTRACT_ROOT}"
  SOURCE_ROOT="${EXTRACT_ROOT}"
fi

if [[ "${DOWNLOAD_ANNOTATIONS}" == "1" ]]; then
  ANN_DL_ROOT="${DOWNLOAD_ROOT}/annotations"
  mkdir -p "${ANN_DL_ROOT}"
  ANN_ARCHIVE="${ANN_DL_ROOT}/scrfd_annotations_download"
  echo "Downloading SCRFD annotation bundle into ${ANN_DL_ROOT}"
  download_gdrive_file "${ANNOTATION_GDRIVE_ID}" "${ANN_ARCHIVE}"
  if unzip -tqq "${ANN_ARCHIVE}" >/dev/null 2>&1; then
    extract_zip "${ANN_ARCHIVE}" "${ANN_DL_ROOT}/extracted"
    ANN_ROOT="${ANN_DL_ROOT}/extracted"
  elif command -v file >/dev/null 2>&1 && file "${ANN_ARCHIVE}" 2>/dev/null | grep -qi 'Zip archive'; then
    extract_zip "${ANN_ARCHIVE}" "${ANN_DL_ROOT}/extracted"
    ANN_ROOT="${ANN_DL_ROOT}/extracted"
  else
    mkdir -p "${ANN_DL_ROOT}/extracted/train"
    cp -f "${ANN_ARCHIVE}" "${ANN_DL_ROOT}/extracted/train/labelv2.txt"
    ANN_ROOT="${ANN_DL_ROOT}/extracted"
  fi
fi

[[ -n "${SOURCE_ROOT}" ]] || die "Could not resolve source root"
SOURCE_ROOT="$(cd "${SOURCE_ROOT}" && pwd)"
if [[ -z "${ANN_ROOT}" ]]; then
  ANN_ROOT="${SOURCE_ROOT}"
else
  ANN_ROOT="$(cd "${ANN_ROOT}" && pwd)"
fi

[[ -d "${SOURCE_ROOT}" ]] || die "Source root not found: ${SOURCE_ROOT}"
[[ -d "${ANN_ROOT}" ]] || die "Annotation root not found: ${ANN_ROOT}"

TRAIN_IMAGES="$(find_train_images "${SOURCE_ROOT}")"
VAL_IMAGES="$(find_val_images "${SOURCE_ROOT}")"
[[ -n "${TRAIN_IMAGES}" ]] || die "Could not find training images under ${SOURCE_ROOT}"
[[ -n "${VAL_IMAGES}" ]] || die "Could not find validation images under ${SOURCE_ROOT}"

if [[ -z "${TRAIN_LABEL}" ]]; then
  TRAIN_LABEL="$(find_train_label "${ANN_ROOT}")"
fi
if [[ -z "${VAL_LABEL}" ]]; then
  VAL_LABEL="$(find_val_label "${ANN_ROOT}")"
fi
if [[ -z "${GT_DIR}" ]]; then
  GT_DIR="$(find_gt_dir "${ANN_ROOT}")"
fi

[[ -n "${TRAIN_LABEL}" ]] || die "Could not find train/labelv2.txt under ${ANN_ROOT}. Pass --train-label explicitly."
[[ -n "${VAL_LABEL}" ]] || die "Could not find val/labelv2.txt under ${ANN_ROOT}. Pass --val-label explicitly."
[[ -n "${GT_DIR}" ]] || die "Could not find validation gt directory under ${ANN_ROOT}. Pass --gt-dir explicitly."

TRAIN_LABEL="$(cd "$(dirname "${TRAIN_LABEL}")" && pwd)/$(basename "${TRAIN_LABEL}")"
VAL_LABEL="$(cd "$(dirname "${VAL_LABEL}")" && pwd)/$(basename "${VAL_LABEL}")"
GT_DIR="$(cd "${GT_DIR}" && pwd)"

[[ -d "${TRAIN_IMAGES}" ]] || die "Training images directory not found: ${TRAIN_IMAGES}"
[[ -d "${VAL_IMAGES}" ]] || die "Validation images directory not found: ${VAL_IMAGES}"
[[ -f "${TRAIN_LABEL}" ]] || die "Train label file not found: ${TRAIN_LABEL}"
[[ -f "${VAL_LABEL}" ]] || die "Val label file not found: ${VAL_LABEL}"
[[ -d "${GT_DIR}" ]] || die "Validation gt directory not found: ${GT_DIR}"

mkdir -p "${DEST_ROOT}/train" "${DEST_ROOT}/val"

replace_target "${TRAIN_IMAGES}" "${DEST_ROOT}/train/images"
replace_target "${TRAIN_LABEL}" "${DEST_ROOT}/train/labelv2.txt"
replace_target "${VAL_IMAGES}" "${DEST_ROOT}/val/images"
replace_target "${VAL_LABEL}" "${DEST_ROOT}/val/labelv2.txt"
replace_target "${GT_DIR}" "${DEST_ROOT}/val/gt"

TRAIN_IMAGE_COUNT="$(find "${DEST_ROOT}/train/images" -type f | wc -l | tr -d ' ')"
VAL_IMAGE_COUNT="$(find "${DEST_ROOT}/val/images" -type f | wc -l | tr -d ' ')"
GT_COUNT="$(find "${DEST_ROOT}/val/gt" -type f | wc -l | tr -d ' ')"

echo
echo "Prepared SCRFD retinaface dataset layout."
echo "Destination: ${DEST_ROOT}"
echo "Mode: ${MODE}"
echo
echo "Resolved sources:"
echo "  train/images   -> ${TRAIN_IMAGES}"
echo "  train/labelv2  -> ${TRAIN_LABEL}"
echo "  val/images     -> ${VAL_IMAGES}"
echo "  val/labelv2    -> ${VAL_LABEL}"
echo "  val/gt         -> ${GT_DIR}"
echo
echo "Counts:"
echo "  train images: ${TRAIN_IMAGE_COUNT}"
echo "  val images:   ${VAL_IMAGE_COUNT}"
echo "  val gt files: ${GT_COUNT}"
echo
echo "Quick checks:"
echo "  ls ${DEST_ROOT}/train/labelv2.txt"
echo "  ls ${DEST_ROOT}/val/labelv2.txt"
echo "  ls ${DEST_ROOT}/val/gt | head"
