#!/bin/bash

# run_infer_template.sh — Template for inference & evaluation with UniDepthV1.
#
# Copy this file and fill in the "..." placeholders before running:
#   cp run_script/run_infer_template.sh run_script/run_infer_myexp.sh
#   bash run_script/run_infer_myexp.sh

# Cache directory
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# ─── Checkpoint & output ─────────────────────────────────────────────────────
# Path to the .pth checkpoint saved by train_depth.py
CHECKPOINT="..."                          # e.g. "runs/train_depth_.../best.pth"

# Where to write metrics JSON and visualisations
OUTPUT_DIR="runs/infer"

# ─── Device ──────────────────────────────────────────────────────────────────
CUDA=...                                    # GPU index

# ─── Dataset evaluation ─────────────────────────────────────────────────────
# Path to NYUv2 .mat file (set to "" to skip dataset evaluation)
DATA_ROOT="datasets/nyu_depth_v2_labeled.mat"
# Which split to evaluate: "train" (795 samples) or "test" (654 samples)
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
# Scale factor to convert raw depth values to metres (for folder-mode GT PNGs)
DEPTH_SCALE=0.001

# Maximum depth (metres) used as upper cap during evaluation
MAX_DEPTH=10.0

# ─── Model architecture (must exactly match the checkpoint) ──────────────────
# Options: "dinov3_vits16" (small), "dinov3_vitl16" (large), "convnextv2_large"
ENCODER_NAME="dinov3_vitl16"

# Encoder feature-map output indices
# dinov3_vits16: "3 6 9 12",  dinov3_vitl16: "5 12 18 24"
OUTPUT_IDX="5 12 18 24"

# Gradient checkpointing (not needed at inference, but must match if loading state)
USE_CHECKPOINT="false"

# Decoder settings (must match training)
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

# Append optional arguments only when non-empty
if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data_root $DATA_ROOT"
fi

if [ -n "$IMAGE_FOLDER" ]; then
    CMD="$CMD --image_folder $IMAGE_FOLDER"
fi

if [ -n "$OUTPUT_IDX" ]; then
    CMD="$CMD --output_idx $OUTPUT_IDX"
fi

echo "Running command:"
echo "$CMD"
eval "$CMD"
