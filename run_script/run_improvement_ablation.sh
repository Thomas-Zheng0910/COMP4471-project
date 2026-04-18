#!/bin/bash
# ======================================================================
# Architecture Improvement Ablation Experiments
# ======================================================================
# 6 runs to compare improvements against baseline:
#   1) baseline       - Current config (SILog + Regression + SelfDistill)
#   2) grad_matching  - + Gradient Matching Loss (from DAV2/MiDaS)
#   3) edge_ssi       - + Edge-Guided Local SSI Loss (from UniDepthV2)
#   4) color_jitter   - + Color Jitter Augmentation
#   5) deep_sup       - + Multi-Scale Deep Supervision
#   6) combined       - All improvements together
# ======================================================================

# Cache directory
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Shared configuration ─────────────────────────────────────────────
SEED=648
CUDA=4
EPOCHS=30          # Enough to see clear differences
BATCH_SIZE=1
ACCUM_STEPS=16     # effective BS = 16
LR=1e-4
ENCODER_LR=1e-5
LAYER_DECAY=0.9
LR_MIN=1e-6
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
FREEZE_ENCODER_EPOCHS=3
LOG_EVERY=50
SAVE_EVERY=10
AMP=true

# Encoder
ENCODER_NAME="dinov3_vitl16"
PRETRAINED="./model/backbones/metadinov3/dinov3-weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
OUTPUT_IDX="5 12 18 24"
USE_CHECKPOINT="true"

# Decoder
HIDDEN_DIM=512
DROPOUT=0.0
DEPTHS="3 2 1"
NUM_HEADS=8
EXPANSION=4

# Data
VAL_ROOT="datasets/nyu_depth_v2_labeled.mat"
IMAGE_SHAPE="480 640"
DEPTH_SCALE=1.0
NUM_WORKERS=8
MAX_TRAIN_SAMPLES=3000  # Smaller for faster ablation comparison
DATASETS="nyuv2,sunrgbd,vkitti2,sintel"

# Base loss weights
DEPTH_LOSS_WEIGHT=10.0
CAMERA_LOSS_WEIGHT=0.1
INVARIANCE_LOSS_WEIGHT=0.01

# ── Helper function ──────────────────────────────────────────────────
run_experiment() {
    local RUN_NAME=$1
    local EXTRA_ARGS=$2

    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: $RUN_NAME"
    echo "============================================================"
    echo "  Extra args: $EXTRA_ARGS"
    echo ""

    CMD="python -m train.train_depth \
        --seed $SEED \
        --cuda $CUDA \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --encoder_lr $ENCODER_LR \
        --layer_decay $LAYER_DECAY \
        --lr_min $LR_MIN \
        --weight_decay $WEIGHT_DECAY \
        --log_every $LOG_EVERY \
        --save_every $SAVE_EVERY \
        --accum_steps $ACCUM_STEPS \
        --warmup_steps $WARMUP_STEPS \
        --freeze_encoder_epochs $FREEZE_ENCODER_EPOCHS \
        --encoder_name $ENCODER_NAME \
        --use_checkpoint $USE_CHECKPOINT \
        --hidden_dim $HIDDEN_DIM \
        --dropout $DROPOUT \
        --depths $DEPTHS \
        --num_heads $NUM_HEADS \
        --expansion $EXPANSION \
        --depth_loss_name SILog \
        --depth_loss_weight $DEPTH_LOSS_WEIGHT \
        --camera_loss_name Regression \
        --camera_loss_weight $CAMERA_LOSS_WEIGHT \
        --invariance_loss_name SelfDistill \
        --invariance_loss_weight $INVARIANCE_LOSS_WEIGHT \
        --image_shape $IMAGE_SHAPE \
        --depth_scale $DEPTH_SCALE \
        --num_workers $NUM_WORKERS \
        --datasets $DATASETS \
        --max_train_samples $MAX_TRAIN_SAMPLES \
        --val_root $VAL_ROOT \
        --run_name $RUN_NAME \
        --script_path $0 \
        $EXTRA_ARGS"

    # Add conditional arguments
    if [ -n "$OUTPUT_IDX" ]; then
        CMD="$CMD --output_idx $OUTPUT_IDX"
    fi
    if [ -n "$PRETRAINED" ]; then
        CMD="$CMD --pretrained $PRETRAINED"
    fi
    if [ "$AMP" = "true" ]; then
        CMD="$CMD --amp"
    fi

    echo "Running: $CMD"
    echo ""
    eval $CMD

    echo ""
    echo "  EXPERIMENT $RUN_NAME COMPLETE"
    echo "============================================================"
}

# ── Select which experiment to run ───────────────────────────────────
# Usage: bash run_improvement_ablation.sh <experiment_name>
# Where experiment_name is one of:
#   baseline, grad_matching, edge_ssi, color_jitter, deep_sup, combined, all

EXPERIMENT=${1:-"all"}

case "$EXPERIMENT" in
    "baseline")
        run_experiment "ablation_baseline" ""
        ;;
    "grad_matching")
        run_experiment "ablation_grad_matching" "--grad_matching_weight 1.0 --grad_matching_scales 4"
        ;;
    "edge_ssi")
        run_experiment "ablation_edge_ssi" "--edge_guided_ssi_weight 0.5"
        ;;
    "color_jitter")
        run_experiment "ablation_color_jitter" "--color_jitter 0.3"
        ;;
    "deep_sup")
        run_experiment "ablation_deep_sup" "--deep_supervision true"
        ;;
    "combined")
        run_experiment "ablation_combined" "--grad_matching_weight 1.0 --grad_matching_scales 4 --edge_guided_ssi_weight 0.5 --color_jitter 0.3 --deep_supervision true"
        ;;
    "all")
        run_experiment "ablation_baseline" ""
        run_experiment "ablation_grad_matching" "--grad_matching_weight 1.0 --grad_matching_scales 4"
        run_experiment "ablation_edge_ssi" "--edge_guided_ssi_weight 0.5"
        run_experiment "ablation_color_jitter" "--color_jitter 0.3"
        run_experiment "ablation_deep_sup" "--deep_supervision true"
        run_experiment "ablation_combined" "--grad_matching_weight 1.0 --grad_matching_scales 4 --edge_guided_ssi_weight 0.5 --color_jitter 0.3 --deep_supervision true"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: bash run_improvement_ablation.sh {baseline|grad_matching|edge_ssi|color_jitter|deep_sup|combined|all}"
        exit 1
        ;;
esac

echo ""
echo "All requested experiments complete!"
