#!/bin/bash

# Cache directory
# torch.hub download location
export TORCH_HOME="./cache"
# Set hugging face and transformers cache location
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Experiment Configuration
SEED=648
CUDA=2
# EPOCHS=200
EPOCHS=100
BATCH_SIZE=4
LR=1e-4 # try 1e-2
ENCODER_LR=1e-5
LAYER_DECAY=0.9
LR_MIN=1e-6
WEIGHT_DECAY=0.01
CLIP_VALUE=1.0
LOG_EVERY=50
SAVE_EVERY=10
ACCUM_STEPS=4
WARMUP_STEPS=500
FREEZE_ENCODER_EPOCHS=5
AMP=false  # set true when GPU memory is limited (shared GPUs, etc.)

# Model Architecture — Pixel Encoder
ENCODER_NAME="convnextv2_large"
OUTPUT_IDX=""
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
CAMERA_LOSS_WEIGHT=0.1
INVARIANCE_LOSS_NAME="SelfDistill"
INVARIANCE_LOSS_WEIGHT=0.01

# Data Configuration
VAL_ROOT="datasets/nyu_depth_v2_labeled.mat"
IMAGE_SHAPE="480 640"
DEPTH_SCALE=1.0
NUM_WORKERS=8
MAX_TRAIN_SAMPLES=5000  # cap samples per epoch (0=use all; full dataset is ~49k)
DATASETS="nyuv2,ToM"

# LiDAR Configuration (set USE_LIDAR=true to enable)
USE_LIDAR=false
LIDAR_ROOT="datasets/nyuv2_lidar_projected,datasets/tom_lidar_projected"  # global fallback; used when DATASET_LIDAR_ROOTS entry is empty
# Per-dataset LiDAR roots, comma-separated and parallel to DATASETS.
# Leave an entry empty to fall back to LIDAR_ROOT, or omit entirely to use LIDAR_ROOT for all.
# Example (nyuv2 + ToM, others have no LiDAR):
#   DATASETS="nyuv2,sunrgbd,vkitti2,sintel,ToM"
#   DATASET_LIDAR_ROOTS="datasets/nyuv2_lidar_projected,datasets/tom_lidar_projected"
DATASET_LIDAR_ROOTS=""       # leave empty to use LIDAR_ROOT for all LiDAR-capable datasets
LIDAR_DEPTH_SCALE=1.0
LIDAR_LOSS_WEIGHT=0.5
LIDAR_DROPOUT_PROB=0.0
USE_LIDAR_FUSION=true        # enable LiDAR fusion in the decoder
LIDAR_FUSION_TYPE="token"    # "late" or "token"

# ──────────────────────────────────────────────────────────────────────────────
# Data Augmentation (RGB-only, applied before ToTensor/Normalize)
# ──────────────────────────────────────────────────────────────────────────────
# Set AUGMENT=true to enable the full augmentation pipeline.
# These augmentations match the original UniDepth V2 training config.
# Only affects RGB images during training — depth/LiDAR maps are NOT augmented.
# Set individual values to 0 to disable that specific augmentation.
AUGMENT=true

# ColorJitter: randomly adjusts brightness, contrast, saturation by up to this amount.
# Range [0, 1]. UniDepth V2 default: 0.4
AUG_JITTER=0.4

# ColorJitter hue: max hue shift. Range [0, 0.5]. UniDepth V2 default: 0.1
AUG_JITTER_HUE=0.1

# Probability of applying ColorJitter per sample. UniDepth V2 default: 0.8
AUG_JITTER_P=0.8

# GaussianBlur max sigma. Kernel size is fixed at 5. UniDepth V2 default: 2.0
AUG_BLUR_SIGMA=2.0

# Probability of applying GaussianBlur per sample. UniDepth V2 default: 0.2
AUG_BLUR_P=0.2

# RandomGamma range: gamma sampled from [1-range, 1+range]. UniDepth V2 default: 0.2
AUG_GAMMA=0.2

# Probability of applying RandomGamma per sample. UniDepth V2 default: 0.8
AUG_GAMMA_P=0.8

# Probability of converting image to grayscale (3-ch output). UniDepth V2 default: 0.2
AUG_GRAYSCALE_P=0.2

# Checkpoint Resume (leave empty for fresh start)
RESUME="runs/train_depth_1777002371825_1285234/checkpoints/epoch_50.pth"

# ──────────────────────────────────────────────────────────────────────────────
# Scheduled Self-Distillation
#
# MODE A — Train teacher  (current defaults):
#   DISTILL_WEIGHT=0, USE_LIDAR=true, USE_LIDAR_FUSION=true
#   LiDAR hints are fused directly into the model. Save the checkpoint.
#
# MODE B — Train student via distillation:
#   DISTILL_WEIGHT>0, TEACHER_CHECKPOINT=<path to teacher epoch_N.pth>
#   USE_LIDAR=true  (dataset must still load LiDAR so the teacher can use it)
#   USE_LIDAR_FUSION=false  (student model itself never sees hints)
#   LIDAR_ROOT must still be set — teacher forward pass needs lidar_depth/mask.
# ──────────────────────────────────────────────────────────────────────────────
DISTILL_WEIGHT=0.0          # 0 = disabled (teacher-training mode)
TEACHER_CHECKPOINT="datasets/teacher/train_depth_1776839443668_3116335/checkpoints/epoch_50.pth"       # path to teacher epoch_N.pth; leave empty when training teacher
DISTILL_WARMUP_STEPS=3000   # steps before distill loss turns on
DISTILL_TOTAL_STEPS=0       # 0 = auto (num_epochs * steps_per_epoch)
DISTILL_PEAK_STEPS=30000    # step at which cosine bell-curve peaks (then decays)
DISTILL_TEMPERATURE=4.0     # temperature for soft-KL
DISTILL_ENTROPY_THRESHOLD=0.5  # teacher confidence mask threshold
DISTILL_LAMBDA_LOGIT=1.0    # weight for soft-KL component
DISTILL_LAMBDA_FEAT=0.1     # weight for feature-MSE component
DISTILL_EMA_ALPHA=0.999     # EMA decay for teacher update (only used in lidar_only mode)
TEACHER_EMA_MODE="gated"    # frozen: static oracle; lidar_only: EMA on lidar layers; gated: full EMA on val improvement

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
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --use_lidar $USE_LIDAR \
    --lidar_depth_scale $LIDAR_DEPTH_SCALE \
    --lidar_loss_weight $LIDAR_LOSS_WEIGHT \
    --lidar_dropout_prob $LIDAR_DROPOUT_PROB \
    --use_lidar_fusion $USE_LIDAR_FUSION \
    --lidar_fusion_type $LIDAR_FUSION_TYPE \
    --aug_jitter $AUG_JITTER \
    --aug_jitter_hue $AUG_JITTER_HUE \
    --aug_jitter_p $AUG_JITTER_P \
    --aug_blur_sigma $AUG_BLUR_SIGMA \
    --aug_blur_p $AUG_BLUR_P \
    --aug_gamma $AUG_GAMMA \
    --aug_gamma_p $AUG_GAMMA_P \
    --aug_grayscale_p $AUG_GRAYSCALE_P \
    --distill_weight $DISTILL_WEIGHT \
    --distill_warmup_steps $DISTILL_WARMUP_STEPS \
    --distill_total_steps $DISTILL_TOTAL_STEPS \
    --distill_peak_steps $DISTILL_PEAK_STEPS \
    --distill_temperature $DISTILL_TEMPERATURE \
    --distill_entropy_threshold $DISTILL_ENTROPY_THRESHOLD \
    --distill_lambda_logit $DISTILL_LAMBDA_LOGIT \
    --distill_lambda_feat $DISTILL_LAMBDA_FEAT \
    --distill_ema_alpha $DISTILL_EMA_ALPHA \
    --teacher_ema_mode $TEACHER_EMA_MODE \
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

if [ "$AMP" = "true" ]; then
    CMD="$CMD --amp"
fi

if [ -n "$LIDAR_ROOT" ]; then
    CMD="$CMD --lidar_root $LIDAR_ROOT"
fi

if [ -n "$DATASET_LIDAR_ROOTS" ]; then
    CMD="$CMD --dataset_lidar_roots $DATASET_LIDAR_ROOTS"
fi

if [ "$AUGMENT" = "true" ]; then
    CMD="$CMD --augment"
fi

if [ -n "$TEACHER_CHECKPOINT" ]; then
    CMD="$CMD --teacher_checkpoint $TEACHER_CHECKPOINT"
fi

# Execute Command
echo "Running command:"
echo $CMD
eval $CMD
