#!/bin/bash
# run_eval_all.sh — Evaluate all custom models + baselines on NYUv2 test set.
# Results saved as JSON files in runs/eval_all/

set -e

# Cache
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Common settings
DATA_ROOT="datasets/nyu_depth_v2_labeled.mat"
SPLIT="test"
OUTPUT_BASE="runs/eval_all"
CUDA=4
BATCH_SIZE=4
NUM_WORKERS=4
MAX_DEPTH=10.0
IMAGE_SHAPE="480 640"

mkdir -p "$OUTPUT_BASE"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ViT Vanilla (dinov3_vitl16, no LiDAR, 30 epochs)
# ═══════════════════════════════════════════════════════════════════════════════
echo "============================================"
echo "  [1/7] ViT Vanilla (30 epoch)"
echo "============================================"
python -m infer.infer_depth \
    --checkpoint runs/experiments/vit-without-guide-30-epoches/checkpoints/epoch_30.pth \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_BASE/vit_vanilla" \
    --cuda $CUDA \
    --encoder_name dinov3_vitl16 \
    --output_idx 5 12 18 24 \
    --use_checkpoint true \
    --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
    --num_heads 8 --expansion 4 \
    --use_lidar_fusion true --lidar_fusion_type token \
    --image_shape $IMAGE_SHAPE \
    --max_depth $MAX_DEPTH \
    --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ConvNeXt Vanilla (convnextv2_large, no LiDAR, 50 epochs)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  [2/7] ConvNeXt Vanilla (50 epoch)"
echo "============================================"
python -m infer.infer_depth \
    --checkpoint runs/experiments/convnext-50-epoch/checkpoints/epoch_50.pth \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_BASE/convnext_vanilla" \
    --cuda $CUDA \
    --encoder_name convnextv2_large \
    --use_checkpoint true \
    --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
    --num_heads 8 --expansion 4 \
    --use_lidar_fusion true --lidar_fusion_type token \
    --image_shape $IMAGE_SHAPE \
    --max_depth $MAX_DEPTH \
    --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Teacher / LiDAR-guided (convnextv2_large, trained WITH LiDAR, 50 epochs)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  [3/7] Teacher (LiDAR-guided, 50 epoch)"
echo "============================================"
python -m infer.infer_depth \
    --checkpoint datasets/teacher/train_depth_1776839443668_3116335/checkpoints/epoch_50.pth \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_BASE/teacher_lidar" \
    --cuda $CUDA \
    --encoder_name convnextv2_large \
    --use_checkpoint true \
    --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
    --num_heads 8 --expansion 4 \
    --use_lidar_fusion true --lidar_fusion_type token \
    --image_shape $IMAGE_SHAPE \
    --max_depth $MAX_DEPTH \
    --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Self-Distillation (convnextv2_large, distilled from teacher, 60 epochs)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  [4/7] Self-Distillation (60 epoch)"
echo "============================================"
python -m infer.infer_depth \
    --checkpoint runs/experiments/self-distillation-60-epoch-best/checkpoints/epoch_60.pth \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_BASE/self_distill" \
    --cuda $CUDA \
    --encoder_name convnextv2_large \
    --use_checkpoint true \
    --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
    --num_heads 8 --expansion 4 \
    --use_lidar_fusion true --lidar_fusion_type token \
    --image_shape $IMAGE_SHAPE \
    --max_depth $MAX_DEPTH \
    --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Baseline: Depth Anything V2
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  [5/7] Baseline: Depth Anything V2"
echo "============================================"
python -m infer.eval_baselines \
    --baseline depth_anything_v2 \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_BASE/baselines" \
    --cuda $CUDA \
    --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
    --max_depth $MAX_DEPTH \
    --image_shape $IMAGE_SHAPE \
    --eval_datasets nyuv2

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Baseline: UniDepth V2
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  [6/7] Baseline: UniDepth V2"
echo "============================================"
python -m infer.eval_baselines \
    --baseline unidepthv2 \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_BASE/baselines" \
    --cuda $CUDA \
    --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
    --max_depth $MAX_DEPTH \
    --image_shape $IMAGE_SHAPE \
    --eval_datasets nyuv2

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Baseline: Marigold
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "  [7/7] Baseline: Marigold"
echo "============================================"
python -m infer.eval_baselines \
    --baseline marigold \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_BASE/baselines" \
    --cuda $CUDA \
    --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS \
    --max_depth $MAX_DEPTH \
    --image_shape $IMAGE_SHAPE \
    --eval_datasets nyuv2 \
    --num_inference_steps 4 --ensemble_size 1

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Results in: $OUTPUT_BASE/"
echo "============================================"
