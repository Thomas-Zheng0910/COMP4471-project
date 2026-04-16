#!/bin/bash

# This script downloads the sunrgbd, vkitti2, sintel, diode from huggingface
# repo_id: Mianbul/COMP4471-project
# The dataset would be downloaded under ./datasets/
# - "sunrgbd": "datasets/SUNRGBD",
# - "vkitti2": "datasets/virtual_kitti_2",
# - "sintel": "datasets/unidepth_data",
# - "diode": "datasets/diode_indoor"

# Set Hugging face cache directory
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"
export HF_HUB_ENABLE_HF_TRANSFER="0"  # Disable hf_transfer to avoid potential issues with large files
export HF_HUB_DOWNLOAD_TIMEOUT="60"  # Increased timeout for large files

# Create the datasets directory if it doesn't exist
mkdir -p datasets

# Download the dataset using Hugging Face's snapshot_download
python data/get_datasets/hf_downloader.py --repo_id Mianbul/COMP4471-project --target_path datasets/extras

# Verbose
echo -e "\033[0;32mDatasets downloaded successfully to datasets/extras/\033[0m"

# Prepare to extract the downloaded files
# Target tars:
# [sunrgbd]="SUNRGBD.tar"
# [vkitti2]="virtual_kitti_2.tar"
# [diode]="diode_indoor.tar"
# [unidepth_data]="unidepth_data.tar"

# Downloaded root: datasets/extras/Mianbul/COMP4471-project/
DOWNLOAD_ROOT="datasets/extras/Mianbul/COMP4471-project"

# We extract under root of ./datasets
for dataset in "SUNRGBD" "virtual_kitti_2" "diode_indoor" "unidepth_data"; do
    tar_path="${DOWNLOAD_ROOT}/${dataset}.tar"
    if [ -f "$tar_path" ]; then
        echo -e "\033[0;34mExtracting $dataset...\033[0m"
        tar -xf "$tar_path" -C datasets/
        echo -e "\033[0;32m$dataset extracted successfully.\033[0m"
    else
        echo -e "\033[0;31mError: $tar_path not found!\033[0m"
    fi
done

# Delete the ./datasets/extras/ directory after extraction
rm -rf datasets/extras

# Verbose
echo -e "\033[0;32mAll datasets extracted successfully to ./datasets/\033[0m"
