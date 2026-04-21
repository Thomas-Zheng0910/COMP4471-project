#!/bin/bash

# run_infer.sh — Inference and evaluation script for UniDepthV1.
#
# Evaluates a trained checkpoint on the NYUv2 test set and/or a custom
# image folder. Copy and fill in the "..." placeholders before running:
#   bash run_script/run_infer.sh

# Cache directory
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# ─── Checkpoint & output ─────────────────────────────────────────────────────
# Path to the .pth checkpoint saved by train_depth.py
CHECKPOINT="..."

# Where to write metrics JSON and visualisations
OUTPUT_DIR="runs/infer"

# ─── Device ──────────────────────────────────────────────────────────────────
CUDA=0

# ─── Dataset evaluation ─────────────────────────────────────────────────────
# Path to dataset file (set to "" to skip dataset evaluation)
DATA_ROOT="datasets/nyu_depth_v2_labeled.mat"
# Which split to evaluate
SPLIT="test"

# ─── Folder inference (optional) ─────────────────────────────────────────────
# Set to a folder path to run inference on arbitrary images
# Expected layout:
#   IMAGE_FOLDER/
#     images/        — RGB images (png/jpg)
#     depths/        — (optional) GT depth PNGs for evaluation
#     intrinsics.json — (optional) per-image intrinsics {stem: [fx, fy, cx, cy]}
IMAGE_FOLDER=""

# ─── Depth settings ──────────────────────────────────────────────────────────
# Scale factor to convert raw depth values to metres (dataset eval uses its own)
DEPTH_SCALE=0.001

# Maximum depth (metres) used as upper cap during evaluation
MAX_DEPTH=10.0

# ─── Model architecture (must exactly match the checkpoint) ──────────────────
ENCODER_NAME="dinov3_vitl16"

# Feature-map indices for the encoder
# dinov3_vits16: "3 6 9 12",  dinov3_vitl16: "5 12 18 24"
OUTPUT_IDX="5 12 18 24"

USE_CHECKPOINT="false"

# Decoder settings
HIDDEN_DIM=512
DROPOUT=0.0
DEPTHS="3 2 1"
NUM_HEADS=8
EXPANSION=4

# LiDAR fusion (set to "true" only for Phase 3+ checkpoints)
USE_LIDAR_FUSION="false"
# Fusion type: "late" or "token" (must match checkpoint)
LIDAR_FUSION_TYPE="late"

# Network input resolution (H W) — must match training
IMAGE_SHAPE="480 640"

# DataLoader
BATCH_SIZE=4
NUM_WORKERS=4

# ─── Build command ───────────────────────────────────────────────────────────
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
    --num_workers $NUM_WORKERS \
    --use_lidar_fusion $USE_LIDAR_FUSION \
    --lidar_fusion_type $LIDAR_FUSION_TYPE"

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