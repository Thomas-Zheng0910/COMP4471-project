#!/bin/bash

# ──────────────────────────────────────────────────────────────────────────────
# UniDepthV1 Inference Script
# ──────────────────────────────────────────────────────────────────────────────
# Runs depth inference on a folder of images and (optionally) evaluates
# against ground-truth depth maps.
#
# Expected data layout (same as training):
#   DATA_ROOT/
#       images/       *.png RGB images
#       depths/       *.png 16-bit depth maps in mm  (optional, for eval)
#       intrinsics.json                              (optional)
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT="runs/train_depth_*/checkpoints/epoch_50.pth"   # adjust to your checkpoint
DATA_ROOT="data/demo"
OUTPUT_DIR="runs/infer"

# ── Device ────────────────────────────────────────────────────────────────────
CUDA=0

# ── Model Architecture (must match the checkpoint) ───────────────────────────
ENCODER_NAME="convnextv2_large"
OUTPUT_IDX="3 6 33 36"
USE_CHECKPOINT="false"
HIDDEN_DIM=512
DROPOUT=0.0
DEPTHS="1 2 3"
NUM_HEADS=8
EXPANSION=4

# ── Data ──────────────────────────────────────────────────────────────────────
IMAGE_SHAPE="384 384"
DEPTH_SCALE=0.001

# ── Evaluation ────────────────────────────────────────────────────────────────
# MAX_DEPTH=""   # uncomment and set to limit eval range, e.g. MAX_DEPTH=80.0

# ── Build & Run ──────────────────────────────────────────────────────────────
CMD="python -m infer.infer_depth \
    --checkpoint $CHECKPOINT \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --cuda $CUDA \
    --encoder_name $ENCODER_NAME \
    --output_idx $OUTPUT_IDX \
    --use_checkpoint $USE_CHECKPOINT \
    --hidden_dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --depths $DEPTHS \
    --num_heads $NUM_HEADS \
    --expansion $EXPANSION \
    --image_shape $IMAGE_SHAPE \
    --depth_scale $DEPTH_SCALE"

# Add optional max_depth
if [ -n "${MAX_DEPTH:-}" ]; then
    CMD="$CMD --max_depth $MAX_DEPTH"
fi

echo "Running command:"
echo $CMD
eval $CMD
