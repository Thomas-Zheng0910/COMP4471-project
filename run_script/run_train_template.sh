#!/bin/bash

# run_train_template.sh — Template for training UniDepthV1.
#
# Copy this file and fill in the "..." placeholders before running:
#   cp run_script/run_train_template.sh run_script/run_train_myexp.sh
#   bash run_script/run_train_myexp.sh

# Cache directory
# torch.hub download location
export TORCH_HOME="./cache"
# Set hugging face and transformers cache location
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# Experiment Configuration
SEED=648
CUDA=...                 # GPU index (e.g. 0)
EPOCHS=200
BATCH_SIZE=4
LR=1e-4                 # Decoder learning rate
ENCODER_LR=1e-5          # Encoder learning rate (10x lower avoids catastrophic forgetting)
LAYER_DECAY=0.9           # Layer-wise LR decay for encoder
LR_MIN=1e-6
WEIGHT_DECAY=0.01
CLIP_VALUE=1.0
LOG_EVERY=50
SAVE_EVERY=10
ACCUM_STEPS=4             # Gradient accumulation (effective batch = BATCH_SIZE * ACCUM_STEPS * 2)
WARMUP_STEPS=500          # Linear LR warmup steps (0 to disable)
FREEZE_ENCODER_EPOCHS=5   # Freeze encoder for first N epochs (0 to disable)

# Model Architecture — Pixel Encoder
# Options: "dinov3_vits16" (small, fast), "dinov3_vitl16" (large, better)
ENCODER_NAME="dinov3_vitl16"
# Pretrained encoder weights (leave empty to use ImageNet default)
PRETRAINED="./model/backbones/metadinov3/dinov3-weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# Encoder feature-map output indices
# dinov3_vits16: "3 6 9 12",  dinov3_vitl16: "5 12 18 24"
OUTPUT_IDX="5 12 18 24"
# Gradient checkpointing (recommended for ViT-L on 24GB GPU)
USE_CHECKPOINT="true"

# Model Architecture — Pixel Decoder
HIDDEN_DIM=512
DROPOUT=0.0
DEPTHS="3 2 1"
NUM_HEADS=8
EXPANSION=4

# Loss Configuration
DEPTH_LOSS_NAME="SILog"
DEPTH_LOSS_WEIGHT=10.0
CAMERA_LOSS_NAME="Regression"
CAMERA_LOSS_WEIGHT=0.1     # Reduced from 0.5 — camera loss can be counterproductive on single-camera datasets
INVARIANCE_LOSS_NAME="SelfDistill"
INVARIANCE_LOSS_WEIGHT=0.01 # Reduced from 0.1 — keeps regularization mild

# Data Configuration
# NYUv2 root (used for validation; training roots are set per-dataset in DATASET_DEFAULT_ROOTS)
VAL_ROOT="datasets/nyu_depth_v2_labeled.mat"
IMAGE_SHAPE="480 640"
DEPTH_SCALE=1.0
NUM_WORKERS=4

# Multi-dataset training (comma-separated)
# Available: nyuv2, sunrgbd, vkitti2, sintel
# Each dataset uses its default root unless overridden with DATASET_ROOTS
DATASETS="nyuv2,sunrgbd,vkitti2,sintel"
# Optional: comma-separated roots matching DATASETS (leave empty for defaults)
DATASET_ROOTS=""

# Output run name (leave empty for auto-generated name)
RUN_NAME=""

# Checkpoint Resume (leave empty for fresh start)
RESUME=""

# Build Command
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
    --clip_value $CLIP_VALUE \
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
    --depth_loss_name $DEPTH_LOSS_NAME \
    --depth_loss_weight $DEPTH_LOSS_WEIGHT \
    --camera_loss_name $CAMERA_LOSS_NAME \
    --camera_loss_weight $CAMERA_LOSS_WEIGHT \
    --invariance_loss_name $INVARIANCE_LOSS_NAME \
    --invariance_loss_weight $INVARIANCE_LOSS_WEIGHT \
    --image_shape $IMAGE_SHAPE \
    --depth_scale $DEPTH_SCALE \
    --num_workers $NUM_WORKERS \
    --datasets $DATASETS \
    --script_path $0"

# Add conditional arguments
if [ -n "$VAL_ROOT" ]; then
    CMD="$CMD --val_root $VAL_ROOT"
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

if [ -n "$OUTPUT_IDX" ]; then
    CMD="$CMD --output_idx $OUTPUT_IDX"
fi

if [ -n "$PRETRAINED" ]; then
    CMD="$CMD --pretrained $PRETRAINED"
fi

if [ -n "$DATASET_ROOTS" ]; then
    CMD="$CMD --dataset_roots $DATASET_ROOTS"
fi

if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run_name $RUN_NAME"
fi

# Execute Command
echo "Running command:"
echo $CMD
eval $CMD
