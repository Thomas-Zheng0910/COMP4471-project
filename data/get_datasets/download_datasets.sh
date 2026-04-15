#!/bin/bash
# download_datasets.sh — Download training/evaluation datasets from HuggingFace.
#
# All datasets are stored as tar archives on HuggingFace. This script downloads
# the tar files and extracts them into the datasets/ directory.
#
# Prerequisites:
#   pip install -U huggingface_hub
#   hf auth login   # required for private repos
#
# Usage:
#   bash data/get_datasets/download_datasets.sh                  # download all
#   bash data/get_datasets/download_datasets.sh unidepth_data    # download one
#   bash data/get_datasets/download_datasets.sh sunrgbd vkitti2  # download specific ones

set -e

REPO_ID="Mianbul/COMP4471-project"
REPO_TYPE="dataset"
LOCAL_DIR="datasets"

# Dataset name → tar filename on HuggingFace, and extracted directory name
declare -A DATASET_TAR=(
    [sunrgbd]="SUNRGBD.tar"
    [vkitti2]="virtual_kitti_2.tar"
    [diode]="diode_indoor.tar"
    [unidepth_data]="unidepth_data.tar"
)

declare -A DATASET_DIR=(
    [sunrgbd]="SUNRGBD"
    [vkitti2]="virtual_kitti_2"
    [diode]="diode_indoor"
    [unidepth_data]="unidepth_data"
)

ALL_DATASETS=(sunrgbd vkitti2 diode unidepth_data)

# ─── Helper: download + extract one dataset ──────────────────────────────────
download_and_extract() {
    local ds="$1"
    local tar_file="${DATASET_TAR[$ds]}"
    local dir_name="${DATASET_DIR[$ds]}"

    if [ -z "$tar_file" ]; then
        echo "WARNING: Unknown dataset '$ds'. Available: ${ALL_DATASETS[*]}"
        return 1
    fi

    # Skip if already extracted
    if [ -d "$LOCAL_DIR/$dir_name" ]; then
        echo "  Skipping $ds — $LOCAL_DIR/$dir_name already exists."
        return 0
    fi

    echo "  Downloading $tar_file from $REPO_ID ..."
    hf download "$REPO_ID" "datasets/$tar_file" \
        --repo-type "$REPO_TYPE" --local-dir "$LOCAL_DIR"

    local tar_path="$LOCAL_DIR/datasets/$tar_file"
    if [ ! -f "$tar_path" ]; then
        echo "ERROR: Expected $tar_path not found after download."
        return 1
    fi

    echo "  Extracting $tar_file → $LOCAL_DIR/ ..."
    tar -xf "$tar_path" -C "$LOCAL_DIR"

    echo "  Cleaning up $tar_path ..."
    rm -f "$tar_path"

    echo "  Done: $ds → $LOCAL_DIR/$dir_name"
}

# ─── Parse arguments ─────────────────────────────────────────────────────────
DATASETS_TO_DOWNLOAD=("$@")

# If no args, download everything
if [ ${#DATASETS_TO_DOWNLOAD[@]} -eq 0 ]; then
    DATASETS_TO_DOWNLOAD=("${ALL_DATASETS[@]}")
fi

echo "=== Downloading datasets from $REPO_ID ==="
echo ""

for ds in "${DATASETS_TO_DOWNLOAD[@]}"; do
    echo "[$ds]"
    download_and_extract "$ds"
    echo ""
done

# Clean up the nested datasets/datasets/ dir if empty
rmdir "$LOCAL_DIR/datasets" 2>/dev/null || true

echo "All requested downloads complete."
