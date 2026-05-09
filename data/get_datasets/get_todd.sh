#!/bin/bash
# Download TODD (Toronto Transparent Object Depth Dataset)
# https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/ZJJAJ3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$(dirname "$SCRIPT_DIR")"
TODD_DIR="${DATASETS_DIR}/todd"

# Borealis Dataverse file IDs for TODD dataset
# https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/ZJJAJ3
TRAIN_FILE_ID=200799
TEST_FILE_ID=200798
VAL_FILE_ID=200797

# Create directory
mkdir -p "${TODD_DIR}"
cd "${TODD_DIR}"

echo "========================================="
echo "Downloading TODD Dataset"
echo "========================================="
echo "Target directory: ${TODD_DIR}"
echo ""

# Function to download a file from Borealis Dataverse
download_file() {
    local file_id=$1
    local output_name=$2
    local url="https://borealisdata.ca/api/access/datafile/${file_id}"
    
    echo "Downloading ${output_name}..."
    echo "URL: ${url}"
    
    if command -v wget &> /dev/null; then
        wget --progress=bar:force -O "${output_name}" "${url}" || {
            echo "Warning: wget failed for ${output_name}, trying with curl..."
            rm -f "${output_name}"
            curl -L -o "${output_name}" "${url}"
        }
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "${output_name}" "${url}"
    else
        echo "Error: Neither wget nor curl is installed. Please install one of them."
        exit 1
    fi
    
    if [ -f "${output_name}" ] && [ -s "${output_name}" ]; then
        echo "✓ Successfully downloaded ${output_name}"
        ls -lh "${output_name}"
    else
        echo "✗ Failed to download ${output_name}"
        return 1
    fi
}

# Download all splits
echo "Downloading train set..."
download_file ${TRAIN_FILE_ID} "train.7z"
echo ""

echo "Downloading test set..."
download_file ${TEST_FILE_ID} "test.7z"
echo ""

echo "Downloading val set..."
download_file ${VAL_FILE_ID} "val.7z"
echo ""

# Extract archives if 7z is available
echo "========================================="
echo "Extracting archives..."
echo "========================================="

if command -v 7z &> /dev/null; then
    for archive in train.7z test.7z val.7z; do
        if [ -f "${archive}" ]; then
            echo "Extracting ${archive}..."
            7z x -y "${archive}" || echo "Warning: Failed to extract ${archive}"
        fi
    done
elif command -v 7za &> /dev/null; then
    for archive in train.7z test.7z val.7z; do
        if [ -f "${archive}" ]; then
            echo "Extracting ${archive}..."
            7za x -y "${archive}" || echo "Warning: Failed to extract ${archive}"
        fi
    done
else
    echo "Warning: 7z not found. Archives downloaded but not extracted."
    echo "To extract manually, install p7zip and run:"
    echo "  7z x train.7z"
    echo "  7z x test.7z"
    echo "  7z x val.7z"
fi

echo ""
echo "========================================="
echo "TODD Dataset Download Complete!"
echo "========================================="
echo "Location: ${TODD_DIR}"
echo ""
echo "Contents:"
ls -lh "${TODD_DIR}"
