#!/bin/bash

# run_infer.sh — Inference and evaluation script for UniDepthV1.
#
# Pairs with infer/infer_depth.py.
# Configure the variables in each section below, then run:
#   bash run_script/run_infer.sh

# Cache directory
# torch.hub download location
export TORCH_HOME="./cache"
# Set hugging face and transformers cache location
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# Checkpoint & output
# Path to the .pth checkpoint saved by train_depth.py
CHECKPOINT="..."

# Where to write the metrics JSON
OUTPUT_DIR="..."

# Device
CUDA=...

# Dataset
# Path to the dataset file
# NOTE: please make sure to import the correct dataset class in infer/infer_depth.py
DATA_ROOT="datasets/nyu_depth_v2_labeled.mat"
# Which split to evaluate
SPLIT="test"

# Optional: infers on given folder of RGB images
# IMAGE_FOLDER
# -> images
# -> depths (optional, for evaluation)
# -> intrinsics.json (optional, for better inference)
IMAGE_FOLDER=""

# Scale factor to convert raw depth values to metres.
# NOTE: global to all datasets
DEPTH_SCALE=1.0

# Maximum depth (metres) used as an upper cap during evaluation.
# NOTE: global to all datasets
MAX_DEPTH=10.0

# Model architecture — must exactly match the checkpoint
ENCODER_NAME="convnext_large_pt"

# Feature-map indices for the encoder.
# Leave OUTPUT_IDX empty to use the encoder's default indices.
# Examples:
#   convnextv2_large : OUTPUT_IDX=""          (uses encoder default)
#   dinov3_vits16    : OUTPUT_IDX="3 6 9 12"
OUTPUT_IDX=""

USE_CHECKPOINT="false"

# Decoder settings
HIDDEN_DIM=512
DROPOUT=0.0
DEPTHS="3 2 1"
NUM_HEADS=8
EXPANSION=4

# Network input resolution (H W) — must match training
IMAGE_SHAPE="480 640"

# DataLoader
BATCH_SIZE=4
NUM_WORKERS=4

# Build command
CMD="python -m infer.infer_depth \
    --checkpoint $CHECKPOINT \
    --split $SPLIT \
    --output_dir $OUTPUT_DIR \
    --cuda $CUDA \
    --encoder_name $ENCODER_NAME \
    --use_checkpoint $USE_CHECKPOINT \
    --hidden_dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --depths $DEPTHS \
    --num_heads $NUM_HEADS \
    --expansion $EXPANSION \
    --image_shape $IMAGE_SHAPE \
    --depth_scale $DEPTH_SCALE \
    --max_depth $MAX_DEPTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS"

# Append data_root only when it is non-empty
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data_root $DATA_ROOT"
fi

# Append image_folder only when it is non-empty
if [ -n "$IMAGE_FOLDER" ]; then
    CMD="$CMD --image_folder $IMAGE_FOLDER"
fi

# Append output_idx only when it is non-empty
if [ -n "$OUTPUT_IDX" ]; then
    CMD="$CMD --output_idx $OUTPUT_IDX"
fi

echo "Running command:"
echo "$CMD"
eval "$CMD"