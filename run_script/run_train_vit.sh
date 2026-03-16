#!/bin/bash

# Cache directory
# torch.hub download location
export TORCH_HOME="./cache"
# Set hugging face and transformers cache location
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# Experiment Configuration
SEED=648
CUDA=1
EPOCHS=50
BATCH_SIZE=2
LR=1e-4
LR_MIN=1e-6
WEIGHT_DECAY=0.01
CLIP_VALUE=1.0
LOG_EVERY=50
SAVE_EVERY=1

# Model Architecture — Pixel Encoder
ENCODER_NAME="dinov3_vits16"
PRETRAINED="./model/backbones/metadinov3/dinov3-weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
OUTPUT_IDX="3 6 9 12"
USE_CHECKPOINT="false"

# Model Architecture — Pixel Decoder
HIDDEN_DIM=512
DROPOUT=0.0
DEPTHS="2 2 2"
NUM_HEADS=8
EXPANSION=4

# Loss Configuration
DEPTH_LOSS_NAME="SILog"
DEPTH_LOSS_WEIGHT=10.0
CAMERA_LOSS_NAME="Regression"
CAMERA_LOSS_WEIGHT=0.5
INVARIANCE_LOSS_NAME="SelfDistill"
INVARIANCE_LOSS_WEIGHT=0.1

# Data Configuration
TRAIN_ROOT="datasets/nyu_depth_v2_labeled.mat"
VAL_ROOT="datasets/nyu_depth_v2_labeled.mat"
IMAGE_SHAPE="480 640"
DEPTH_SCALE=1.0
NUM_WORKERS=4

# Checkpoint Resume (leave empty for fresh start)
RESUME=""

# Build Command
CMD="python -m train.train_depth \
    --seed $SEED \
    --cuda $CUDA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_min $LR_MIN \
    --weight_decay $WEIGHT_DECAY \
    --clip_value $CLIP_VALUE \
    --log_every $LOG_EVERY \
    --save_every $SAVE_EVERY \
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
    --train_root $TRAIN_ROOT \
    --image_shape $IMAGE_SHAPE \
    --depth_scale $DEPTH_SCALE \
    --num_workers $NUM_WORKERS \
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

# Execute Command
echo "Running command:"
echo $CMD
eval $CMD
