#!/bin/bash
# ======================================================================
# Launch all 6 ablation experiments in parallel on GPUs 1-5
# GPU 1: baseline + combined (sequential)
# GPU 2: grad_matching
# GPU 3: edge_ssi
# GPU 4: color_jitter
# GPU 5: deep_sup
# ======================================================================

cd /homes/chzheng/userhome/COMP4471-project

export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/localdata/chzheng/miniconda3/envs/DepthSense/lib:$LD_LIBRARY_PATH

PYTHON="/localdata/chzheng/miniconda3/envs/DepthSense/bin/python"

# ── Shared config ─────────────────────────────────────────────
COMMON_ARGS="--seed 648 \
    --epochs 15 \
    --batch_size 1 \
    --lr 1e-4 \
    --encoder_lr 1e-5 \
    --layer_decay 0.9 \
    --lr_min 1e-6 \
    --weight_decay 0.01 \
    --log_every 50 \
    --save_every 5 \
    --accum_steps 16 \
    --warmup_steps 500 \
    --freeze_encoder_epochs 3 \
    --encoder_name dinov3_vitl16 \
    --use_checkpoint true \
    --hidden_dim 512 \
    --dropout 0.0 \
    --depths 3 2 1 \
    --num_heads 8 \
    --expansion 4 \
    --depth_loss_name SILog \
    --depth_loss_weight 10.0 \
    --camera_loss_name Regression \
    --camera_loss_weight 0.1 \
    --invariance_loss_name SelfDistill \
    --invariance_loss_weight 0.01 \
    --image_shape 480 640 \
    --depth_scale 1.0 \
    --num_workers 8 \
    --datasets nyuv2,sunrgbd,vkitti2,sintel \
    --max_train_samples 3000 \
    --val_root datasets/nyu_depth_v2_labeled.mat \
    --output_idx 5 12 18 24 \
    --pretrained ./model/backbones/metadinov3/dinov3-weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --amp \
    --script_path run_script/launch_parallel_ablation.sh"

LOG_DIR="runs/ablation_logs"
mkdir -p $LOG_DIR

echo "Starting 6 ablation experiments in parallel..."
echo "Logs will be in $LOG_DIR/"
echo ""

# 1) baseline on GPU 2
echo "[GPU 2] Launching baseline..."
$PYTHON -m train.train_depth $COMMON_ARGS \
    --cuda 2 \
    --run_name ablation_baseline \
    > $LOG_DIR/baseline.log 2>&1 &
PID_BASELINE=$!

# 2) grad_matching on GPU 3
echo "[GPU 3] Launching grad_matching..."
$PYTHON -m train.train_depth $COMMON_ARGS \
    --cuda 3 \
    --run_name ablation_grad_matching \
    --grad_matching_weight 1.0 --grad_matching_scales 4 \
    > $LOG_DIR/grad_matching.log 2>&1 &
PID_GRAD=$!

# 3) edge_ssi on GPU 4
echo "[GPU 4] Launching edge_ssi..."
$PYTHON -m train.train_depth $COMMON_ARGS \
    --cuda 4 \
    --run_name ablation_edge_ssi \
    --edge_guided_ssi_weight 0.5 \
    > $LOG_DIR/edge_ssi.log 2>&1 &
PID_EDGE=$!

# 4) color_jitter on GPU 5
echo "[GPU 5] Launching color_jitter..."
$PYTHON -m train.train_depth $COMMON_ARGS \
    --cuda 5 \
    --run_name ablation_color_jitter \
    --color_jitter 0.3 \
    > $LOG_DIR/color_jitter.log 2>&1 &
PID_JITTER=$!

# 5) deep_sup on GPU 1
echo "[GPU 1] Launching deep_sup..."
$PYTHON -m train.train_depth $COMMON_ARGS \
    --cuda 1 \
    --run_name ablation_deep_sup \
    --deep_supervision true \
    > $LOG_DIR/deep_sup.log 2>&1 &
PID_DEEP=$!

echo ""
echo "5 experiments launched. Waiting for GPU 1 (deep_sup) to finish before launching combined..."
echo "PIDs: baseline=$PID_BASELINE grad=$PID_GRAD edge=$PID_EDGE jitter=$PID_JITTER deep=$PID_DEEP"
echo ""

# Wait for deep_sup to finish, then launch combined on GPU 1
wait $PID_DEEP
echo "[GPU 1] deep_sup finished. Launching combined..."
$PYTHON -m train.train_depth $COMMON_ARGS \
    --cuda 1 \
    --run_name ablation_combined \
    --grad_matching_weight 1.0 --grad_matching_scales 4 \
    --edge_guided_ssi_weight 0.5 \
    --color_jitter 0.3 \
    --deep_supervision true \
    > $LOG_DIR/combined.log 2>&1 &
PID_COMBINED=$!

echo "PIDs: baseline=$PID_BASELINE grad=$PID_GRAD edge=$PID_EDGE jitter=$PID_JITTER combined=$PID_COMBINED"

# Wait for all
wait $PID_BASELINE $PID_GRAD $PID_EDGE $PID_JITTER $PID_COMBINED

echo ""
echo "========================================"
echo "  ALL 6 EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Logs:"
for f in $LOG_DIR/*.log; do
    echo "  $f"
done
