#!/bin/bash
# Inference Script

# Cache directory
# torch.hub download location
export TORCH_HOME="./cache"
# Set hugging face and transformers cache location
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"

# Paths
CHECKPOINT="..."
DATA_ROOT="datasets/nyu_depth_v2_labeled.mat"
OUTPUT_DIR="runs/infer"

# Device
CUDA=...

# Model Architecture (must match the checkpoint)
ENCODER_NAME="convnextv2_large"
OUTPUT_IDX="3 6 33 36"
USE_CHECKPOINT="false"
HIDDEN_DIM=512
DROPOUT=0.0
DEPTHS="1 2 3"
NUM_HEADS=8
EXPANSION=4
IMAGE_SHAPE="384 384"
DEPTH_SCALE=0.001

# Evaluation (optional)
# MAX_DEPTH=""   # uncomment and set to limit eval range, e.g. MAX_DEPTH=80.0

# Build CMD
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
