#!/bin/bash

# This script downloads the KITTI Eigen Split dataset from Kaggle (~5 GB)
# Kaggle Dataset: https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset
# The file would be downloaded and extracted under ./datasets/kitti_eigen/

# Set cache directories
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"
export HF_HUB_ENABLE_HF_TRANSFER="0"  # Disable hf_transfer to avoid potential issues with large files
export HF_HUB_DOWNLOAD_TIMEOUT="60"  # Increased timeout for large files

# Create the datasets directory if it doesn't exist
mkdir -p datasets

# Download the dataset using kagglehub (recommended) or fallback to wget
# Note: For Kaggle download, you need to have kagglehub installed:
#   pip install kagglehub
# Or configure Kaggle API credentials:
#   ~/.kaggle/kaggle.json with {"username":"YOUR_USERNAME","key":"YOUR_KEY"}

if command -v python3 &> /dev/null; then
    echo "Attempting to download using kagglehub..."
    python3 << 'EOF'
import kagglehub
import os
import shutil

# Download the dataset
dataset_path = kagglehub.dataset_download("awsaf49/kitti-eigen-split-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# Move to datasets/kitti_eigen
target_dir = "datasets/kitti_eigen"
os.makedirs(target_dir, exist_ok=True)

# Move contents
for item in os.listdir(dataset_path):
    src = os.path.join(dataset_path, item)
    dst = os.path.join(target_dir, item)
    if os.path.exists(dst):
        shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
    shutil.move(src, dst)

print(f"Dataset organized at {target_dir}/")
EOF
    if [ $? -eq 0 ]; then
        echo "KITTI Eigen Split dataset downloaded successfully!"
        exit 0
    else
        echo "kagglehub download failed, falling back to manual download instructions..."
    fi
fi

# Fallback: Manual download instructions
echo ""
echo "============================================"
echo "  Manual Download Required"
echo "============================================"
echo ""
echo "The KITTI Eigen Split dataset requires Kaggle authentication."
echo "Please download manually from:"
echo "  https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset"
echo ""
echo "Option 1 - Using Kaggle API:"
echo "  1. Install Kaggle API: pip install kaggle"
echo "  2. Get API credentials from https://www.kaggle.com/settings"
echo "  3. Place kaggle.json in ~/.kaggle/"
echo "  4. Run: kaggle datasets download -d awsaf49/kitti-eigen-split-dataset"
echo ""
echo "Option 2 - Manual download:"
echo "  1. Visit: https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset"
echo "  2. Click 'Download' button"
echo "  3. Extract the zip to: datasets/kitti_eigen/"
echo ""
echo "Expected directory structure after extraction:"
echo "  ./datasets/kitti_eigen/"
echo "      test/          - Test split files"
echo "      train/         - Training split files"
echo "      val/           - Validation split files"
echo "      test_files.txt     - List of test files"
echo "      train_files.txt    - List of train files"
echo "      val_files.txt      - List of validation files"
echo ""
