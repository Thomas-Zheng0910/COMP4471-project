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
CUDA=4
# EPOCHS=200
EPOCHS=100
BATCH_SIZE=1
LR=1e-4 # try 1e-2
ENCODER_LR=1e-5
LAYER_DECAY=0.9
LR_MIN=1e-6
WEIGHT_DECAY=0.01
CLIP_VALUE=1.0
LOG_EVERY=50
SAVE_EVERY=10
ACCUM_STEPS=16
WARMUP_STEPS=500
FREEZE_ENCODER_EPOCHS=5
AMP=true  # set true when GPU memory is limited (shared GPUs, etc.)

# Model Architecture — Pixel Encoder
ENCODER_NAME="dinov3_vitl16"
PRETRAINED="./model/backbones/metadinov3/dinov3-weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
OUTPUT_IDX="5 12 18 24"
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
MAX_TRAIN_SAMPLES=10000  # cap samples per epoch (0=use all; full dataset is ~49k)
DATASETS="nyuv2,sunrgbd,vkitti2,sintel"

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
    --max_train_samples $MAX_TRAIN_SAMPLES \
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

# Execute Command
echo "Running command:"
echo $CMD
eval $CMD
