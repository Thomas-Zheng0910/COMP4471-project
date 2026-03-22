#!/bin/bash

# Phase 5 Ablation Study Runner
# Trains all four variants for comprehensive evaluation

set -e

# Configuration
MAT_PATH="${1:-datasets/nyu_depth_v2_labeled.mat}"
LIDAR_H5_KEY="${2:-lidar_depths}"
LIDAR_ROOT="${3:-}"
EPOCHS="${4:-20}"  # Reduced for demo; use 100 for production
BATCH_SIZE="${5:-4}"
DEVICE="${6:-0}"

echo "=========================================="
echo "Phase 5 Ablation Study"
echo "=========================================="
echo "MAT_PATH: $MAT_PATH"
echo "LIDAR_H5_KEY: $LIDAR_H5_KEY"
echo "LIDAR_ROOT: $LIDAR_ROOT"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "DEVICE: $DEVICE"
echo "=========================================="
echo ""

# Common arguments
CMD_BASE="/localdata/yhip/COMP4471-project/.venv/bin/python -m train.train_depth"
COMMON_ARGS="
  --train_root $MAT_PATH \
  --val_root $MAT_PATH \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --cuda $DEVICE \
  --image_shape 384 384 \
  --encoder_name convnextv2_large \
  --hidden_dim 512 \
  --num_heads 8 \
  --expansion 4 \
  --depth_loss_weight 10.0 \
  --camera_loss_weight 0.5 \
  --invariance_loss_weight 0.1 \
  --lidar_loss_weight 0.5 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --save_every 1 \
  --log_every 50 \
  --num_workers 4 \
  --phase4_eval_fallback true
"

# Add LiDAR configuration if HDF5 key provided
if [ -n "$LIDAR_H5_KEY" ]; then
  COMMON_ARGS="$COMMON_ARGS --use_lidar true --lidar_h5_key $LIDAR_H5_KEY"
fi

# Add external LiDAR root if provided
if [ -n "$LIDAR_ROOT" ]; then
  COMMON_ARGS="$COMMON_ARGS --lidar_root $LIDAR_ROOT"
fi

# Timestamp for grouping
TIMESTAMP=$(date +%s)
echo "Experiment timestamp: $TIMESTAMP"
echo ""

# Variant 1: RGB-only Baseline
echo "════════════════════════════════════════════════════════════════"
echo "[1/4] RGB-Only Baseline (no LiDAR)"
echo "════════════════════════════════════════════════════════════════"
eval "$CMD_BASE $COMMON_ARGS --phase5_ablation rgb_only"
echo ""

# Variant 2: LiDAR Supervision Only (no fusion)
echo "════════════════════════════════════════════════════════════════"
echo "[2/4] LiDAR Supervision Only (supervision loss, no fusion)"
echo "════════════════════════════════════════════════════════════════"
eval "$CMD_BASE $COMMON_ARGS --phase5_ablation supervision_only"
echo ""

# Variant 3: Late Fusion (1/16 scale)
echo "════════════════════════════════════════════════════════════════"
echo "[3/4] LiDAR Late Fusion (1/16 scale)"
echo "════════════════════════════════════════════════════════════════"
eval "$CMD_BASE $COMMON_ARGS \
  --phase5_ablation late_fusion \
  --lidar_fusion_type late \
  --lidar_dropout_prob 0.2"
echo ""

# Variant 4: Token Fusion (multi-scale)
echo "════════════════════════════════════════════════════════════════"
echo "[4/4] LiDAR Token Fusion (multi-scale 16/8/4)"
echo "════════════════════════════════════════════════════════════════"
eval "$CMD_BASE $COMMON_ARGS \
  --phase5_ablation token_fusion \
  --lidar_fusion_type token \
  --lidar_dropout_prob 0.2"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ Phase 5 Ablation Complete"
echo "════════════════════════════════════════════════════════════════"
echo "Check ./runs/ for experiment outputs"
echo ""
