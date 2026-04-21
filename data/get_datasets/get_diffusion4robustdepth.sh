#!/bin/bash

# This script downloads the Diffusion4RobustDepth dataset from Hugging Face
# (testing - 23 GB)
# repo_id: fabiotosi92/Diffusion4RobustDepth
# The dataset would be downloaded under ./datasets/Diffusion4RobustDepth/

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
python data/get_datasets/hf_downloader.py --repo_id fabiotosi92/Diffusion4RobustDepth --target_path datasets/Diffusion4RobustDepth
# Verbose
echo -e "\033[0;32mDiffusion4RobustDepth dataset downloaded successfully to datasets/Diffusion4RobustDepth/\033[0m"

# We'll have these tars under datasets/Diffusion4RobustDepth/:
# apolloscapes.tar     mapillary_part3.tar  ToM_part2.tar       weights_Table4.tar
# cityscapes.tar       mapillary_part4.tar  ToM_part3.tar       weights_Table5.tar
# kitti.tar            nuscenes.tar         weights_Table1.tar
# mapillary_part1.tar  robotcar.tar         weights_Table2.tar
# mapillary_part2.tar  ToM_part1.tar        weights_Table3.tar

# Move into the dataset directory to perform extraction
cd datasets/Diffusion4RobustDepth/ || exit

echo -e "\nStarting extraction and merging..."

# Handle split files (mapillary, ToM, weights)
# We extract them into the same directory; 
# tar will merge the folder contents automatically.
for prefix in mapillary ToM weights_Table; do
    # Create a directory
    mkdir -p "${prefix}"
    if ls ${prefix}*.tar >/dev/null 2>&1; then
        echo -e "\nMerging and extracting ${prefix} parts..."
        for f in ${prefix}*.tar; do
            echo "Extracting $f..."
            tar -xf "$f" -C "${prefix}" && rm "$f"
        done
        echo -e "\033[0;32mMerged and extracted ${prefix} parts successfully.\033[0m"
    fi
done

# Handle standalone tar files
for dataset in apolloscapes cityscapes kitti nuscenes robotcar; do
    if [ -f "${dataset}.tar" ]; then
        echo -e "\nExtracting ${dataset}..."
        # Create a directory for the dataset if the tar doesn't contain a root folder
        mkdir -p "$dataset"
        tar -xf "${dataset}.tar" -C "$dataset" && rm "${dataset}.tar"
        echo -e "\033[0;32mExtracted ${dataset}.tar into $dataset/\033[0m"
    fi
done

echo -e "\n\033[0;32mAll datasets extracted and organized successfully.\033[0m"

# NOTE: (file structure):
# datasets/
#     Diffusion4RobustDepth/
#         apolloscapes/driving/apolloscapes/challenging/
#             stereo_train_001/camera_5 -> *.jpg (no depth)
#             stereo_train_002/camera_5 -> *.jpg (no depth)
#             stereo_train_003/camera_5 -> *.jpg (no depth)
#         cityscapes/challenging/left/
#             test/*/ -> *.jpg (no depth) [under ./test/ we have category folders like "aachen", "bochum", etc.]
#             train/*/ -> *.jpg (no depth)
#             val/*/ -> *.jpg (no depth)
#         kitti/challenging/
#             2012/
#                 testing/colored_0/ -> *.jpg (no depth)
#                 training/colored_0/ -> *.jpg (no depth)
#             2015/
#                 testing/
#                     image_2/ -> *.jpg (no depth)
#                     image_3/ -> *.jpg (no depth)
#                 training/
#                     image_2/ -> *.jpg (no depth)
#             test_completion/ -> *.jpg (no depth)
#             test_prediction/ -> *.jpg (no depth)
#             val/ -> *.jpg (no depth)
#         mapillary/mapillary/challenging/
#             training/images -> *.jpg (no depth)
#             validation/images -> *.jpg (no depth)
#         nuscenes/driving/nuscenes/challenging/
#             night/samples/CAM_FRONT/ -> *.jpg (no depth)
#             rain/samples/CAM_FRONT/ -> *.jpg (no depth)
#         robotcar/driving/robotcar/challenging -> *.jpg (no depth)
#         ToM/
#             */
#                 challenging/ -> *.jpg (no depth)
#                 easy/ -> *.jpg (no depth)
#         weights_Table/weights/
#             Table1/
#                 depth_anything/weights.pth
#                 dpt_large/weights.pth
#                 midas/weights.pth
#                 zoe/weights.pth
#             Table2/
#                 md4all-DD_ft_DA/weights.ckpt
#                 md4all-DD_ft_Ours/weights.ckpt
#             Table3/
#                 md4all-DD_ft_DA/weights.ckpt
#                 md4all-DD_ft_Ours/weights.ckpt
#             Table4/
#                 depth_anything/weights.pth
#                 dpt_large/weights.pth
#             Table5/
#                 depth_anything/weights.pth
#                 dpt_large/weights.pth