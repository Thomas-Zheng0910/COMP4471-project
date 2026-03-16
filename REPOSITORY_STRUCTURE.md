# COMP4471 UniDepthV1 - Repository Structure Guide
Last Updated: 2026-03-10

## Project Overview

**UniDepthV1** is a monocular depth estimation model that predicts per-pixel depth from single RGB images. The project includes training and inference pipelines with support for evaluation against ground-truth depth maps.

**Key Features:**
- Single-GPU training (no distributed training)
- ConvNeXtV2-based encoder with pre-trained ImageNet weights
- Multi-scale decoder with attention mechanisms
- Multiple loss functions (regression, distillation, confidence, etc.)
- Comprehensive evaluation metrics (D1, D2, D3, RMSE, SILog, etc.)
- TensorBoard logging during training

---

## Directory Structure & File Purposes

```
COMP4471-project/
├── README.md                           # Project overview and documentation
├── environment.yml                     # Conda environment dependencies
├── libc++FIX.sh                        # Fix libc++ / pandas import issues on Linux
│
├── data/                               # Data loading modules
│   ├── __init__.py
│   ├── nyuv2_dataset.py               # NYU Depth V2 dataset loader (from .mat file)
│   └── get_datasets/
│       └── get_nyu_v2.sh              # Script to download nyu_depth_v2_labeled.mat
│
├── model/                              # Model architecture components
│   ├── __init__.py
│   ├── encoder.py                     # Encoder wrapper + backbone loading functions
│   │
│   ├── backbones/                     # Pre-trained backbone architectures
│   │   ├── __init__.py
│   │   ├── convnext.py                # Original ConvNeXt backbone
│   │   ├── convnext2.py               # ConvNeXtV2 backbone (improved)
│   │   └── dinov2.py                  # DINO-V2 vision transformer
│   │
│   ├── layers/                        # Custom neural network layers
│   │   ├── __init__.py
│   │   ├── activation.py              # SwiGLU, GEGLU activations
│   │   ├── attention.py               # Attention mechanisms (SimpleAttention, etc.)
│   │   ├── convnext.py                # ConvNeXt blocks
│   │   ├── drop_path.py               # DropPath regularization
│   │   ├── layer_scale.py             # Layer scaling
│   │   ├── mlp.py                     # Multi-layer perceptron blocks
│   │   ├── nystrom_attention.py       # Nyström approximation for efficient attention
│   │   ├── positional_encoding.py     # Positional encoding for spatial information
│   │   └── upsample.py                # Upsampling layers (ConvUpsample, shuffle, etc.)
│   │
│   ├── ops/                          # Operations and loss functions
│   │   ├── __init__.py
│   │   ├── scheduler.py              # Learning rate scheduler implementations
│   │   │
│   │   └── losses/                   # Loss function modules
│   │       ├── __init__.py
│   │       ├── regression.py         # Base regression loss with alpha-gamma weighting
│   │       ├── silog.py              # Scale-Invariant Log loss
│   │       ├── arel.py               # Absolute Relative Error loss
│   │       ├── confidence.py         # Confidence-weighted loss
│   │       ├── distill.py            # Teacher-student distillation losses
│   │       ├── local_ssi.py          # Local Scale-Shift invariance loss
│   │       ├── dummy.py              # Placeholder loss for testing
│   │       └── utils.py              # Loss utility functions
│   │
│   └── unidepthv1/                   # Main UniDepthV1 model architecture
│       ├── __init__.py
│       ├── unidepthv1.py             # Main model class (encoder + decoder)
│       └── decoder.py                # Depth decoder with camera head
│
├── train/                              # Training pipeline
│   ├── __init__.py
│   └── train_depth.py                 # Single-GPU training script with TensorBoard logging
│
├── infer/                              # Inference pipeline
│   ├── __init__.py
│   └── infer_depth.py                 # Inference script + optional evaluation
│
├── utils/                              # Utility modules
│   ├── __init__.py
│   ├── camera.py                      # Pinhole camera model (intrinsics handling)
│   ├── constants.py                   # Dataset normalization constants, depth bins
│   ├── coordinate.py                  # Coordinate grid generation
│   ├── distributed.py                 # Distributed training utilities (not used)
│   ├── evaluation_depth.py            # Evaluation metrics (D1, RMSE, SILog, etc.)
│   ├── geometric.py                   # Geometric operations (ray generation, depth conversion)
│   ├── misc.py                        # Miscellaneous utilities (stacking, matching, profiling)
│   ├── sht.py                         # Spherical harmonics transform
│   └── visualization.py               # Depth colorization and visualization
│
└── run_script/                         # Bash scripts for training and inference
    ├── run_train.sh                   # Training script with configurable parameters (NYUv2)
    └── run_infer_demo.sh              # Inference script with configurable parameters
```

---

## Module Purposes

### Core Model Architecture

#### **model/unidepthv1/unidepthv1.py** - Main Model
- Combines encoder (ConvNeXtV2) with decoder
- Handles image preprocessing and padding/resizing
- Outputs depth maps and camera intrinsics
- Uses `utils.geometric.generate_rays()` for 3D ray generation
- Leverages `utils.misc.match_gt()` and `utils.misc.match_intrinsics()` for evaluation
- **Dependencies:** encoder.py, decoder.py, utils (camera, geometric, misc, constants, visualization)

#### **model/unidepthv1/decoder.py** - Decoder Network
- Decodes multi-scale encoder features into depth
- Components:
  - `ListAdapter`: Adapts encoder features to hidden dimension
  - `CameraHead`: Predicts camera intrinsics
  - `DepthHead`: Predicts depth maps
  - Multi-scale fusion and refinement stages
- Uses custom layers: `AttentionBlock`, `NystromBlock`, `ConvUpsample`, `MLP`, `PositionEmbeddingSine`
- **Dependencies:** model/layers, utils (geometric, misc, sht)

#### **model/encoder.py** - Encoder Wrapper
- Loads backbone architectures (ConvNeXtV2, ConvNeXt, DINO-V2)
- Wraps backbone to extract multi-scale features
- Functions:
  - `convnextv2_base()`: Loads ConvNeXtV2 base (128→256→512→1024 dims)
  - `convnextv2_large()`: Loads ConvNeXtV2 large (192→384→768→1536 dims)
  - Downloads pre-trained weights from Facebook AI's hub
- **Dependencies:** backbones

### Backbone Architectures

#### **model/backbones/convnext2.py** - ConvNeXtV2
- Modern CNN backbone with improved design
- Uses ModernBlock with depthwise conv + 1×1 conv
- Efficient multi-scale feature extraction
- Supports gradient checkpointing for memory efficiency

#### **model/backbones/convnext.py** - ConvNeXt
- Original ConvNeXt architecture
- Alternative to ConvNeXtV2 if needed

#### **model/backbones/dinov2.py** - DINO-V2
- Self-supervised vision transformer
- Alternative backbone for feature extraction

### Custom Layers

#### **model/layers/attention.py** - Attention Mechanisms
- `SimpleAttention`: Basic multi-head attention
- `AttentionBlock`: Attention block with residual
- `AttentionLayer`: Full attention layer
- `AttentionDecoderBlock`: Attention for decoder
- Used in decoder's feature refinement

#### **model/layers/nystrom_attention.py** - Efficient Attention
- Nyström approximation for fast attention
- Reduces quadratic complexity for large feature maps

#### **model/layers/mlp.py** - MLP Blocks
- Feed-forward networks in transformer-like architecture
- GLU-style variants

#### **model/layers/activation.py** - Custom Activations
- `SwiGLU`: SiLU-based gated linear unit
- `GEGLU`: GELU-based gated linear unit
- Used in feed-forward blocks

#### **model/layers/upsample.py** - Upsampling Layers
- `ConvUpsample`: Convolution + upsampling
- `ConvUpsampleShuffle`: Pixel shuffle for upsampling
- `ConvUpsampleShuffleResidual`: With residual connection
- `ResUpsampleBil`: Bilinear upsampling with residual

#### **model/layers/positional_encoding.py** - Positional Encoding
- `PositionEmbeddingSine`: Sinusoidal positional encoding
- Injects spatial information into decoder

#### **model/layers/convnext.py** - ConvNeXt Blocks
- `CvnxtBlock`: ResNet-style ConvNeXt blocks for layers

#### **model/layers/layer_scale.py & drop_path.py**
- Layer scaling and DropPath for training stability

### Loss Functions

#### **model/ops/losses/regression.py** - Base Regression Loss
- `Regression`: Alpha-gamma weighted loss
- Supports scaling and different regression functions
- Used as base for depth prediction

#### **model/ops/losses/silog.py** - Scale-Invariant Log Loss
- SILog loss for monocular depth estimation
- Common metric in depth estimation literature

#### **model/ops/losses/arel.py** - Absolute Relative Error
- ARel metric adapted as loss term

#### **model/ops/losses/confidence.py** - Confidence Loss
- Learns per-pixel depth confidence/uncertainty
- Weights loss by confidence

#### **model/ops/losses/distill.py** - Distillation Losses
- `SelfDistill`: Self-ensemble distillation
- `TeacherDistill`: Teacher-student knowledge distillation
- For knowledge transfer and model refinement

#### **model/ops/losses/local_ssi.py** - Local SSI Loss
- `LocalSSI`: Local scale-shift invariance
- `EdgeGuidedLocalSSI`: Edge-guided variant
- Preserves local depth relationships

#### **model/ops/losses/utils.py**
- Helper functions for loss computation
- Masked mean/quantile calculations

### Data Loading

#### **data/nyuv2_dataset.py** - NYU Depth V2 Dataset
- `NYUv2Dataset`: Loads the official NYU Depth V2 labeled split from `nyu_depth_v2_labeled.mat` (MATLAB v7.3 / HDF5 format)
- 1449 densely labelled indoor RGBD pairs; uses the standard **Eigen et al. 654-image test split**
- Args: `root`, `image_shape`, `depth_scale`, `split` (`"train"` / `"test"` / `"all"`), `flip_aug`, `return_intrinsics`
- Built-in `flip_aug`: returns `(original, flipped)` tuples for **SelfDistill** invariance loss
- Provides `NYUv2Dataset.collate_fn` that interleaves flipped pairs and separates tensor data from `img_metas`
- Hard-coded `NYUV2_INTRINSICS` (fx, fy, cx, cy from the NYUv2 toolbox), depth range 0.005 – 10.0 m
- Applies Eigen border-crop eval mask when `split="test"`
- Lazy per-worker HDF5 file handle for DataLoader fork-safety

#### **data/get_datasets/get_nyu_v2.sh** - Dataset Download Script
- Downloads `nyu_depth_v2_labeled.mat` (~2.8 GB) from the MIT Silberman host
- Saves to `./datasets/nyu_depth_v2_labeled.mat`

### Training Pipeline

#### **train/train_depth.py** - Training Script
- **Main Components:**
  - Argument parsing with 40+ configurable parameters
  - Model initialization from UniDepthV1
  - DataLoader setup (DummyDataset or real data)
  - Loss function selection and composition
  - AdamW optimizer + CosineAnnealingLR scheduler
  - TensorBoard logging (losses, LR, sample visualizations)
  - Periodic checkpointing
  - Gradient clipping and mixed precision support

- **Key Functions:**
  - `get_args()`: Parameter definitions
  - `train_epoch()`: Single epoch training loop
  - `validate()`: Evaluation on validation set
  - `main()`: Training orchestration

- **Outputs:**
  ```
  runs/train_depth_<timestamp>/
    checkpoints/     # Model weights
    logs/            # TensorBoard logs
  ```

- **Dependencies:** 
  - model.unidepthv1.UniDepthV1
  - data (DummyDataset or DemoImageDataset)
  - model.ops.losses (various loss functions)
  - utils (camera, visualization, misc)

### Inference Pipeline

#### **infer/infer_depth.py** - Inference Script
- **Main Components:**
  - Argument parsing
  - Checkpoint loading and model reconstruction
  - Batch inference on video frames or image folders
  - Optional GT depth evaluation
  - Visualization and result saving

- **Key Functions:**
  - `get_args()`: Parameter definitions
  - `infer_single()`: Single image inference
  - `infer_batch()`: Batch processing
  - `main()`: Inference orchestration

- **Outputs:**
  ```
  output_dir/
    predictions/     # Depth predictions (.npy)
    visualizations/  # Colorized depth (.png)
    metrics.json     # Evaluation scores (if GT available)
  ```

- **Dependencies:**
  - model.unidepthv1.UniDepthV1
  - utils (evaluation_depth, visualization, camera, misc)

### Utilities

#### **utils/camera.py** - Camera Model
- `Pinhole`: Pinhole camera model
  - Intrinsic matrix (fx, fy, cx, cy)
  - Extrinsic matrix (rotation, translation)
  - Ray generation, depth warping, etc.
- `invert_pinhole()`: Inverts intrinsic matrix
- Used in geometric transformations

#### **utils/geometric.py** - Geometric Operations
- `generate_rays()`: Creates 3D rays from pixel coords using intrinsics
- `spherical_zbuffer_to_euclidean()`: Converts spherical depth to Euclidean
- `flat_interpolate()`: Interpolation in flattened space
- Critical for depth-to-3D conversion

#### **utils/evaluation_depth.py** - Evaluation Metrics
- Metrics computed:
  - **δ¹, δ², δ³**: Threshold metrics (% pixels within accuracy threshold)
  - **RMSE**: Root mean squared error
  - **RMSELog**: RMSE in log-space
  - **SILog**: Scale-Invariant Log error
  - **ARel**: Absolute Relative Error
  - **SRel**: Squared Relative Error
  - **DellAUC**: Area under delta curve
  - **taup, tauτ**: Accuracy thresholds
  - **SSIM**: Structural similarity
- `ssi()`: Scale-shift invariant depth estimation
- Supports masking for invalid depth regions

#### **utils/misc.py** - Miscellaneous Utilities
- `max_stack()`: Stack tensors and take max
- `get_params()`: Count model parameters
- `match_gt()`: Align prediction with GT depth for evaluation
- `match_intrinsics()`: Handle intrinsic matrix alignment
- `profile_method()`: Timing/profiling decorator
- `recursive_to()`: Recursively move tensors to device
- `squeeze_list()`: List squeezing utilities

#### **utils/coordinate.py** - Coordinate Grid
- `coords_grid()`: Generate pixel coordinate grids
- Used for ray generation and geometric operations

#### **utils/visualization.py** - Visualization
- `colorize()`: Convert depth maps to RGB visualization
- `image_grid()`: Create image grids for TensorBoard
- Supports different colormaps and depth ranges

#### **utils/constants.py** - Constants
- `IMAGENET_DATASET_MEAN`, `IMAGENET_DATASET_STD`: Normalization
- `OPENAI_DATASET_MEAN`, `OPENAI_DATASET_STD`: CLIP normalization
- `DEPTH_BINS`: Pre-defined depth quantization bins (0.1m to 260m)

#### **utils/sht.py** - Spherical Harmonics
- Spherical Harmonics Transform operations
- Used for spherical depth representation

#### **utils/distributed.py** - Distributed Training (Not Used)
- Utilities for multi-GPU training
- Not used in current implementation (single-GPU only)

---

## Dependency Graph

```
Training Flow:
train/train_depth.py
  ├─→ model/unidepthv1/unidepthv1.py (model)
  │   ├─→ model/encoder.py
  │   │   └─→ model/backbones/ (ConvNeXtV2)
  │   ├─→ model/unidepthv1/decoder.py
  │   │   ├─→ model/layers/ (attention, mlp, upsample, etc.)
  │   │   └─→ utils/geometric.py
  │   └─→ utils/ (camera, misc, constants, visualization)
  │
  ├─→ data/dummy_dataset.py or demo_dataset.py (data)
  │
  ├─→ model/ops/losses/ (loss functions)
  │   ├─→ regression.py
  │   ├─→ silog.py
  │   ├─→ confidence.py
  │   ├─→ distill.py
  │   ├─→ local_ssi.py
  │   └─→ utils.py
  │
  └─→ utils/
      ├─→ camera.py
      ├─→ evaluation_depth.py
      ├─→ geometric.py
      ├─→ misc.py
      └─→ visualization.py

Inference Flow:
infer/infer_depth.py
  ├─→ model/unidepthv1/unidepthv1.py (model)
  ├─→ utils/evaluation_depth.py (metrics, if GT available)
  ├─→ utils/visualization.py (depth colorization)
  └─→ utils/misc.py (GT matching)
```

---

## Data Flow

### Training Data Flow
```
DummyDataset/DemoImageDataset
    ↓ (batches of images + depth + camera intrinsics)
DataLoader
    ↓
train_depth.py
    ├─→ UniDepthV1 (forward pass)
    │   └─→ generates predictions (depth, camera params)
    ├─→ Loss Functions (compute loss)
    └─→ Backward + optimize
```

### Inference Data Flow
```
Input Images (folder or file)
    ↓
infer_depth.py
    ├─→ Load checkpoint
    ├─→ Reconstruct UniDepthV1
    └─→ Forward pass
        ├─→ generate depth predictions
        ├─→ colorize_visualization
        ├─→ save_predictions
        └─→ (optional) evaluate_vs_gt
```

---

## Configuration via Shell Scripts

### **run_script/run_train.sh**
Configures:
- Data root pointing to `datasets/nyu_depth_v2_labeled.mat`
- GPU/device selection and random seed
- Model architecture (encoder name, depths, heads, hidden dim, etc.)
- Training hyperparameters (batch size, LR, lr_min, weight decay, grad clip, epochs)
- Loss composition (depth loss, camera loss, invariance/SelfDistill loss + weights)
- Data preprocessing (image shape 480×640, depth scale)
- Optional checkpoint resume path
- Optional `output_idx` for selecting encoder feature levels

### **run_script/run_infer_demo.sh**
Configures:
- Checkpoint to load
- Input data directory
- Output directory
- Model architecture (must match checkpoint)
- Inference parameters (image shape, depth scale)
- Optional evaluation parameters (max_depth for gt range)

---

## Key Integration Points

### 1. **Model Architecture**
- Encoder: ConvNeXtV2 backbone → multi-scale features
- Decoder: Multi-scale fusion → depth and camera head
- Output: Depth map + intrinsic parameters

### 2. **Loss Composition**
- Multiple loss terms computed in training
- Combined with learnable weights
- Supports mixed objectives (regression + distillation + confidence)

### 3. **Evaluation**
- Training: TensorBoard metrics during training
- Inference: Full metrics suite (δ, RMSE, SILog) if GT available

### 4. **Visualization**
- Depth maps colorized via Jet colormap
- Sample visualizations logged to TensorBoard
- Error visualizations during inference

---

## Usage Example

```bash
# Training (from run_script/run_train.sh)
python -m train.train_depth \
  --encoder_name convnext_large_pt \
  --batch_size 2 \
  --epochs 50 \
  --lr 1e-4 \
  --train_root datasets/nyu_depth_v2_labeled.mat \
  --val_root datasets/nyu_depth_v2_labeled.mat

# Inference (from run_script/run_infer_demo.sh)
python -m infer.infer_depth \
  --checkpoint runs/train_depth_*/checkpoints/epoch_50.pth \
  --data_root data/demo \
  --encoder_name convnextv2_large \
  --output_idx 3 6 33 36
```

---

## Summary

The **COMP4471 UniDepthV1** project is a well-structured monocular depth estimation framework with:
- **Clear separation of concerns**: model, data, train, infer, utils
- **Modular components**: encoder, decoder, losses, metrics all pluggable
- **Comprehensive utilities**: camera geometry, visualization, evaluation
- **Production-ready**: TensorBoard logging, checkpointing, evaluation metrics
- **Single-GPU focused**: Simplified training without distributed complexities
- **Real dataset integration**: NYU Depth V2 via HDF5 with Eigen split, flip augmentation, and custom collate_fn

---

## Changelog

### 2026-03-10
- **Added** `data/nyuv2_dataset.py`: Full `NYUv2Dataset` replacing `dummy_dataset.py` and `demo_dataset.py`. Loads directly from the official `.mat` file, supports train/test/all splits, flip augmentation for SelfDistill, and a custom `collate_fn`.
- **Added** `data/get_datasets/get_nyu_v2.sh`: Download script for `nyu_depth_v2_labeled.mat`.
- **Renamed** `run_script/run_train_demo.sh` → `run_script/run_train.sh`: Updated to target the NYUv2 `.mat` dataset path and revised default hyperparameters.
- **Added** `libc++FIX.sh`: Shell script to patch `LD_LIBRARY_PATH` inside the `DepthSense` conda environment when pandas/h5py fail to import due to `libc++` conflicts.
- **Removed** `data/dummy_dataset.py` (synthetic random data, no longer needed).
- **Removed** `data/demo_dataset.py` (disk-based demo loader, superseded by `nyuv2_dataset.py`).

