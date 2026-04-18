#!/bin/bash
# =============================================================================
# Transparent Surface Depth Estimation — Ablation Launch Script
# =============================================================================
# Dual-channel architecture ablation study:
#   E1: Baseline (si=True, NYUv2+ToM, color_jitter, deep_supervision)
#   E2: +Edge head (low-level channel)
#   E3: +Seg auxiliary (high-level channel)
#   E4: Full dual-channel (edge + seg)
#   E5: Full + boundary loss
#   E6: Full + grad_matching + edge_ssi (additional ablations)
#
# Usage:
#   bash run_script/run_transparent_ablation.sh [GPU_LIST] [EPOCHS]
#   e.g.  bash run_script/run_transparent_ablation.sh "0,1,2,3,4,5" 20
# =============================================================================

set -e

# ── Configuration ──
GPU_LIST_RAW="${1:-0,1,2,3,4,5}"
EPOCHS="${2:-20}"
BATCH_SIZE=2
ACCUM_STEPS=8
SAVE_EVERY=5

# Cache
export TORCH_HOME="./cache"
export TRANSFORMERS_CACHE="./cache"
export HF_HOME="./cache"
export HF_HUB_CACHE="./cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${PYTHON_BIN:-/localdata/chzheng/miniconda3/envs/DepthSense/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

# ── Parse GPU list ──
IFS=',' read -ra GPU_IDS <<< "$GPU_LIST_RAW"
NUM_GPUS=${#GPU_IDS[@]}
echo "==========================================================="
echo " Transparent Surface Ablation (${NUM_GPUS} GPUs available)"
echo "==========================================================="
echo " Epochs: $EPOCHS  |  Batch: $BATCH_SIZE  |  Accum: $ACCUM_STEPS"
echo " GPUs: ${GPU_LIST_RAW}"
echo "==========================================================="

# ── Common flags ──
COMMON_FLAGS="\
  --seed 42 \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr 1e-4 \
  --encoder_lr 1e-5 \
  --layer_decay 0.9 \
  --lr_min 1e-6 \
  --weight_decay 0.01 \
  --clip_value 1.0 \
  --log_every 50 \
  --save_every $SAVE_EVERY \
  --accum_steps $ACCUM_STEPS \
  --warmup_steps 500 \
  --freeze_encoder_epochs 3 \
  --encoder_name dinov3_vitl16 \
  --pretrained ./model/backbones/metadinov3/dinov3-weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output_idx 5 12 18 24 \
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
  --image_shape 384 384 \
  --depth_scale 1.0 \
  --num_workers 4 \
  --val_root datasets/nyu_depth_v2_labeled.mat \
  --datasets nyuv2,tom \
  --color_jitter 0.3 \
  --deep_supervision true \
  --amp \
  --script_path $0"

PIDS=()
GPU_IDX=0

launch_experiment() {
  local name="$1"
  local extra_flags="$2"
  local gpu="${GPU_IDS[$GPU_IDX]}"

  echo ""
  echo ">>> Launching $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON_BIN -m train.train_depth \
    --cuda 0 \
    --run_name "${name}" \
    $COMMON_FLAGS \
    $extra_flags \
    > "runs/${name}.log" 2>&1 &
  PIDS+=($!)
  echo "    PID: ${PIDS[-1]}  |  Log: runs/${name}.log"

  GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
}

mkdir -p runs

# ── E1: Baseline (si=True, NYUv2+ToM, color_jitter, deep_supervision) ──
launch_experiment "E1_baseline" ""

# ── E2: +Edge head (low-level channel) ──
launch_experiment "E2_edge_head" "\
  --use_edge_head true \
  --edge_loss_weight 0.5"

# ── E3: +Seg auxiliary (high-level channel) ──
launch_experiment "E3_seg_aux" "\
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy"

# ── E4: Full dual-channel (edge + seg) ──
launch_experiment "E4_dual_channel" "\
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy"

# ── E5: Full + boundary loss ──
launch_experiment "E5_dual_boundary" "\
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy \
  --boundary_loss_weight 0.5"

# ── E6: Full + grad_matching + edge_ssi (kitchen sink) ──
launch_experiment "E6_full_plus_extras" "\
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy \
  --boundary_loss_weight 0.5 \
  --grad_matching_weight 1.0 \
  --edge_guided_ssi_weight 0.5"

echo ""
echo "==========================================================="
echo " All ${#PIDS[@]} experiments launched!"
echo " PIDs: ${PIDS[*]}"
echo "==========================================================="
echo " Monitor with: tail -f runs/E*_*.log"
echo " Check GPUs:   nvidia-smi"
echo "==========================================================="

# Wait for all
wait "${PIDS[@]}"
echo ""
echo "All experiments completed!"
