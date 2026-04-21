#!/bin/bash

# This script generates depth maps for the RobustDepth dataset using the
# DepthAnything model.

# Cache directory
# torch.hub download location
export TORCH_HOME="./cache"
# Set hugging face and transformers cache location
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# Root directory
DATASET_ROOT="./datasets/Diffusion4RobustDepth"
# Sub dirs to process (relative to root)
SUBDIRS=(...)
# (
#     "apolloscapes"
#     "cityscapes"
#     "kitti"
#     "mapillary"
#     "nuscenes"
#     "robotcar"
#     "ToM"
# )
# Output suffix for depth maps
OUTPUT_SUFFIX="_depth_anything"
# Model ID for DepthAnything (HuggingFace)
# NOTE: Please use the "indoor" or "outdoor" model based on the dataset type
#       if use the "small" model, the generated depth maps are relative depths
#       NOT metric depths (in metres).
MODEL_ID="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
# Device to use (e.g. "0" for cuda:0, "cpu" for CPU)
CUDA_DEVICE="0"
# Whether to overwrite existing depth maps
OVERWRITE=false
# Whether to use automatic mixed precision (AMP) for faster inference on supported GPUs
AMP=false
# Optional: limit number of images to process for quick testing (set to empty for no limit)
MAX_IMAGES=""
# Optional: save visualization
SAVE_VIS=false

# Loop through each subdir and generate depth maps
for SUBDIR in "${SUBDIRS[@]}"; do
    echo -e "\n\033[1mProcessing subdir: $SUBDIR\033[0m"
    CMD="python -m data.preprocess.generate_GT_diffusion4robustdepth_depthanything \
        --dataset_root \"$DATASET_ROOT/$SUBDIR\" \
        --output_suffix \"$OUTPUT_SUFFIX\" \
        --model_id \"$MODEL_ID\" \
        --cuda \"$CUDA_DEVICE\""

    if [ "$OVERWRITE" = true ]; then
        CMD+=" --overwrite"
    fi
    if [ "$AMP" = true ]; then
        CMD+=" --amp"
    fi
    if [ -n "$MAX_IMAGES" ]; then
        CMD+=" --max_images \"$MAX_IMAGES\""
    fi
    if [ "$SAVE_VIS" = true ]; then
        CMD+=" --save_vis"
    fi

    eval $CMD
done