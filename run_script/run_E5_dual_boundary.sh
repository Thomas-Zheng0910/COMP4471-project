#!/bin/bash
# E5: Dual-Channel + Boundary Loss — gradient matching at transparent object edges
set -e
GPU="${1:-0}"

export TORCH_HOME="./cache"
export HF_HOME="./cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p runs

CUDA_VISIBLE_DEVICES=$GPU python -m train.train_depth \
  --cuda 0 --run_name E5_dual_boundary \
  --seed 42 --epochs 20 --batch_size 2 --accum_steps 8 \
  --lr 1e-4 --encoder_lr 1e-5 --layer_decay 0.9 --lr_min 1e-6 \
  --weight_decay 0.01 --clip_value 1.0 --warmup_steps 500 \
  --freeze_encoder_epochs 3 --log_every 50 --save_every 5 \
  --encoder_name dinov3_vitl16 \
  --pretrained ./model/backbones/metadinov3/dinov3-weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --output_idx 5 12 18 24 --use_checkpoint true \
  --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
  --num_heads 8 --expansion 4 \
  --depth_loss_name SILog --depth_loss_weight 10.0 \
  --camera_loss_name Regression --camera_loss_weight 0.1 \
  --invariance_loss_name SelfDistill --invariance_loss_weight 0.01 \
  --image_shape 384 384 --depth_scale 1.0 --num_workers 4 \
  --val_root datasets/nyu_depth_v2_labeled.mat \
  --datasets nyuv2,tom --color_jitter 0.3 --deep_supervision true --amp \
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy \
  --boundary_loss_weight 0.5 \
  2>&1 | tee runs/E5_dual_boundary.log
