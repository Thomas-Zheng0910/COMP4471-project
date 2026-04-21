#!/bin/bash
# =============================================================================
# Generate PromptDA-style simulated LiDAR from dense depth maps.
#
# Implements "sparse anchor interpolation" from PromptDA (CVPR 2025, Sec 3.3):
#   1. Downsample GT depth to LiDAR resolution (192x256, iPhone ARKit)
#   2. Sample sparse anchors on a distorted grid (stride 7)
#   3. Interpolate remaining pixels via RGB-similarity KNN
#
# This produces a filled low-res depth map with interpolation artifacts at
# depth boundaries, mimicking real LiDAR noise patterns.
#
# Usage:
#   bash data/preprocess/generate_simulated_lidar.sh [DATASET]
#
# Examples:
#   bash data/preprocess/generate_simulated_lidar.sh nyuv2
#   bash data/preprocess/generate_simulated_lidar.sh tom
#   bash data/preprocess/generate_simulated_lidar.sh all
# =============================================================================

set -e

DATASET="${1:-all}"
LIDAR_H=192
LIDAR_W=256
STRIDE=7
KNN_K=4
RGB_SIGMA=20.0
SEED=42

echo "==========================================================="
echo " Generating simulated LiDAR (PromptDA sparse-anchor interp)"
echo " Dataset:    $DATASET"
echo " LiDAR res:  ${LIDAR_H}x${LIDAR_W}"
echo " Stride:     $STRIDE  |  KNN-K: $KNN_K"
echo " Seed:       $SEED"
echo "==========================================================="

case "$DATASET" in
  nyuv2)
    echo ">>> Generating for NYUv2..."
    python data/preprocess/generate_simulated_lidar.py \
      --input-mat datasets/nyu_depth_v2_labeled.mat \
      --output-dir "datasets/nyuv2_lidar_projected" \
      --lidar-h "$LIDAR_H" --lidar-w "$LIDAR_W" \
      --stride "$STRIDE" --knn-k "$KNN_K" \
      --rgb-sigma "$RGB_SIGMA" \
      --seed "$SEED"
    echo "Done. Output: datasets/nyuv2_lidar_projected/"
    ;;

  tom)
    echo ">>> Generating for ToM (Diffusion4RobustDepth)..."
    python data/preprocess/generate_simulated_lidar.py \
      --input-image-dir "datasets/Diffusion4RobustDepth/ToM" \
      --depth-suffix "_depth_anything" \
      --output-dir "datasets/tom_lidar_projected" \
      --lidar-h "$LIDAR_H" --lidar-w "$LIDAR_W" \
      --stride "$STRIDE" --knn-k "$KNN_K" \
      --rgb-sigma "$RGB_SIGMA" \
      --seed "$SEED"
    echo "Done. Output: datasets/tom_lidar_projected/"
    ;;

  all)
    echo ">>> Generating for all datasets..."
    bash "$0" nyuv2
    bash "$0" tom
    ;;

  *)
    echo "Unknown dataset: $DATASET"
    echo "Available: nyuv2, tom, all"
    exit 1
    ;;
esac

echo ""
echo "==========================================================="
echo " Simulated LiDAR generation complete!"
echo "==========================================================="
