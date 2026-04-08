#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

CRITERIA_ZIP_BASE_URL="https://zenodo.org/records/15516900/files"
CHUNK_PREFIX="criteria_part"
CHUNK_COUNT=6

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d "processed_criteria" ]; then
  echo "[INFO] Downloading and extracting processed_criteria chunks..."
  mkdir -p processed_criteria

  for i in $(seq 0 $((CHUNK_COUNT - 1))); do
    chunk_zip="${CHUNK_PREFIX}_${i}.zip"
    chunk_url="${CRITERIA_ZIP_BASE_URL}/${chunk_zip}?download=1"

    if [ ! -f "$chunk_zip" ]; then
      echo "[INFO] Downloading $chunk_zip..."
      wget --quiet "$chunk_url" -O "$chunk_zip"
    else
      echo "[INFO] $chunk_zip already exists. Skipping download."
    fi

    echo "[INFO] Extracting $chunk_zip into processed_criteria..."
    unzip -q "$chunk_zip" -d processed_criteria
  done
else
  echo "[INFO] processed_criteria already exists. Skipping extraction."
fi
