#!/bin/bash

# run_eval_baselines.sh — Evaluate baseline depth models on NYUv2.
#
# Usage:
#   bash run_script/run_eval_baselines.sh
#
# Runs all three baselines sequentially and saves metrics to runs/baselines/.

# Cache directory
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# ─── Common settings ─────────────────────────────────────────────────────────
DATA_ROOT="datasets/nyu_depth_v2_labeled.mat"
SPLIT="test"
OUTPUT_DIR="runs/baselines"
CUDA=0
BATCH_SIZE=4
NUM_WORKERS=4
MAX_DEPTH=10.0
IMAGE_SHAPE="480 640"

# ─── UniDepth V2 ─────────────────────────────────────────────────────────────
echo "============================================"
echo "  Evaluating: UniDepth V2"
echo "============================================"
python -m infer.eval_baselines \
    --baseline unidepthv2 \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --cuda $CUDA \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_depth $MAX_DEPTH \
    --image_shape $IMAGE_SHAPE

# ─── Depth Anything V2 ───────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Evaluating: Depth Anything V2"
echo "============================================"
python -m infer.eval_baselines \
    --baseline depth_anything_v2 \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --cuda $CUDA \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_depth $MAX_DEPTH \
    --image_shape $IMAGE_SHAPE

# ─── Marigold (LCM, 4 steps) ─────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Evaluating: Marigold"
echo "============================================"
python -m infer.eval_baselines \
    --baseline marigold \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --cuda $CUDA \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_depth $MAX_DEPTH \
    --image_shape $IMAGE_SHAPE \
    --num_inference_steps 4 \
    --ensemble_size 1

echo ""
echo "Done. Metrics saved to $OUTPUT_DIR/"
