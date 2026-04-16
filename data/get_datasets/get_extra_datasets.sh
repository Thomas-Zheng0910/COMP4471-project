#!/bin/bash
# Download additional training and evaluation datasets.
# All downloads are programmatic — no manual browser steps required.
#
# Datasets downloaded:
#   TRAINING:
#     - SUN-RGBD         (~5 GB)  → datasets/SUNRGBD/
#     - Virtual KITTI 2  (~15 GB) → datasets/virtual_kitti_2/
#     - Sintel (HDF5)    (~1 GB)  → datasets/unidepth_data/ (from UniDepth Google Drive)
#   EVALUATION:
#     - iBims-1 (HDF5)   (~0.5 GB) → datasets/unidepth_data/ (from UniDepth Google Drive)
#     - DIODE Indoor val  (~2.6 GB) → datasets/diode_indoor/
#
# Usage:
#   bash data/get_datasets/get_training_datasets.sh [--skip-sunrgbd] [--skip-vkitti2] [--skip-unidepth-hdf5] [--skip-diode]

set -e

DATASETS_DIR="datasets"
mkdir -p "$DATASETS_DIR"

SKIP_SUNRGBD=false
SKIP_VKITTI2=false
SKIP_UNIDEPTH=false
SKIP_DIODE=false

for arg in "$@"; do
    case $arg in
        --skip-sunrgbd)       SKIP_SUNRGBD=true ;;
        --skip-vkitti2)       SKIP_VKITTI2=true ;;
        --skip-unidepth-hdf5) SKIP_UNIDEPTH=true ;;
        --skip-diode)         SKIP_DIODE=true ;;
    esac
done

# ─── 1. SUN-RGBD (~5 GB) ─────────────────────────────────────────────────────
if [ "$SKIP_SUNRGBD" = false ]; then
    echo "============================================"
    echo "  Downloading SUN-RGBD (~5 GB)"
    echo "============================================"
    if [ -d "$DATASETS_DIR/SUNRGBD" ]; then
        echo "  Already exists, skipping."
    else
        wget -c -O "$DATASETS_DIR/SUNRGBD.zip" \
            "http://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
        echo "  Extracting..."
        unzip -q "$DATASETS_DIR/SUNRGBD.zip" -d "$DATASETS_DIR/"
        rm "$DATASETS_DIR/SUNRGBD.zip"
        echo "  Done → $DATASETS_DIR/SUNRGBD/"
    fi
fi

# ─── 2. Virtual KITTI 2 — RGB + Depth (~15 GB) ──────────────────────────────
if [ "$SKIP_VKITTI2" = false ]; then
    echo ""
    echo "============================================"
    echo "  Downloading Virtual KITTI 2 (~15 GB)"
    echo "============================================"
    VKITTI_DIR="$DATASETS_DIR/virtual_kitti_2"
    mkdir -p "$VKITTI_DIR"
    if [ -d "$VKITTI_DIR/vkitti_2.0.3_rgb" ] && [ -d "$VKITTI_DIR/vkitti_2.0.3_depth" ]; then
        echo "  Already exists, skipping."
    else
        VKITTI_BASE="https://download.europe.naverlabs.com/virtual_kitti_2.0.3"
        # RGB (~7 GB)
        if [ ! -d "$VKITTI_DIR/vkitti_2.0.3_rgb" ]; then
            wget -c -O "$VKITTI_DIR/vkitti_2.0.3_rgb.tar" "$VKITTI_BASE/vkitti_2.0.3_rgb.tar"
            tar -xf "$VKITTI_DIR/vkitti_2.0.3_rgb.tar" -C "$VKITTI_DIR/"
            rm "$VKITTI_DIR/vkitti_2.0.3_rgb.tar"
        fi
        # Depth (~7.5 GB)
        if [ ! -d "$VKITTI_DIR/vkitti_2.0.3_depth" ]; then
            wget -c -O "$VKITTI_DIR/vkitti_2.0.3_depth.tar" "$VKITTI_BASE/vkitti_2.0.3_depth.tar"
            tar -xf "$VKITTI_DIR/vkitti_2.0.3_depth.tar" -C "$VKITTI_DIR/"
            rm "$VKITTI_DIR/vkitti_2.0.3_depth.tar"
        fi
        echo "  Done → $VKITTI_DIR/"
    fi
fi

# ─── 3. iBims-1 + Sintel HDF5 from UniDepth Google Drive (~1.5 GB) ──────────
if [ "$SKIP_UNIDEPTH" = false ]; then
    echo ""
    echo "============================================"
    echo "  Downloading iBims-1 + Sintel (UniDepth HDF5)"
    echo "============================================"
    UNIDEPTH_DIR="$DATASETS_DIR/unidepth_data"
    if [ -d "$UNIDEPTH_DIR" ] && [ "$(ls -A "$UNIDEPTH_DIR" 2>/dev/null)" ]; then
        echo "  Already exists, skipping."
    else
        pip install -q gdown 2>/dev/null || true
        gdown --folder "https://drive.google.com/drive/folders/1FKsa5-b3EX0ukZq7bxord5fC5OfUiy16" \
            -O "$UNIDEPTH_DIR"
        echo "  Done → $UNIDEPTH_DIR/"
    fi
fi

# ─── 4. DIODE Indoor Validation (~2.6 GB) ────────────────────────────────────
if [ "$SKIP_DIODE" = false ]; then
    echo ""
    echo "============================================"
    echo "  Downloading DIODE Indoor Validation (~2.6 GB)"
    echo "============================================"
    DIODE_DIR="$DATASETS_DIR/diode_indoor"
    if [ -d "$DIODE_DIR" ] && [ "$(ls -A "$DIODE_DIR" 2>/dev/null)" ]; then
        echo "  Already exists, skipping."
    else
        mkdir -p "$DIODE_DIR"
        wget -c -O "$DIODE_DIR/val.tar.gz" \
            "http://diode-dataset.s3.amazonaws.com/val.tar.gz"
        echo "  Extracting..."
        tar -xzf "$DIODE_DIR/val.tar.gz" -C "$DIODE_DIR/"
        rm "$DIODE_DIR/val.tar.gz"
        echo "  Done → $DIODE_DIR/"
    fi
fi

echo ""
echo "============================================"
echo "  All downloads complete!"
echo "============================================"
echo "  $DATASETS_DIR/SUNRGBD/         - SUN-RGBD (training)"
echo "  $DATASETS_DIR/virtual_kitti_2/ - Virtual KITTI 2 (training)"
echo "  $DATASETS_DIR/unidepth_data/   - iBims-1 + Sintel HDF5 (eval + training)"
echo "  $DATASETS_DIR/diode_indoor/    - DIODE Indoor val (eval)"
