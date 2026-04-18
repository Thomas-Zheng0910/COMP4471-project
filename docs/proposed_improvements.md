# Proposed Improvements: Dual-Channel Architecture for Transparent Surface Depth Estimation

## Motivation

Monocular depth estimation models struggle with transparent and reflective surfaces (glass, mirrors, water) because these materials violate the photometric consistency assumptions that most depth networks rely on. Standard depth decoders use a single feature stream that entangles low-level structural cues (edges, gradients) with high-level semantic understanding (object identity, material type). For transparent surfaces, these two levels of information must cooperate: the model needs to *know* that a region is glass (semantic) and simultaneously preserve the *sharp depth discontinuity* at its boundary (structural).

We propose a **dual-channel decoder architecture** that explicitly separates feature processing into:

1. **High-level channel (SegHead)** — extracts semantic object-class information via auxiliary segmentation, fused into the decoder through cross-attention.
2. **Low-level channel (EdgeHead)** — extracts structural edge features via learned Sobel-style convolutions, fused through gated residual refinement.

Both channels feed back into the main depth decoder at different spatial scales, providing complementary signals that improve depth accuracy at transparent object boundaries.

## Architecture Overview

```
                        DINOv3 ViT-L/16 Encoder
                                 │
                    ┌────────────┴────────────┐
                    │     Depth Decoder        │
                    │                          │
              ┌─────┴─────┐                    │
              │  1/16 scale │◄── SegHead ──── Cross-Attention + Gate
              │  (semantic) │    (high-level)   (auxiliary segmentation)
              └─────┬─────┘
                    │  upsample
              ┌─────┴─────┐
              │  1/8 scale  │
              └─────┬─────┘
                    │  upsample
              ┌─────┴─────┐
              │  1/4 scale  │
              └─────┬─────┘
                    │  upsample
              ┌─────┴─────┐
              │  1/2 scale  │◄── EdgeHead ──── Gated Residual Refinement
              │ (structural)│    (low-level)    (learned edge detection)
              └─────┬─────┘
                    │
                  Depth Output (H × W)
```

## Dual-Channel Components

### High-Level Channel: SegHead

**Purpose:** Inject object-class awareness into the depth decoder so the model can distinguish transparent materials from their surroundings.

**Architecture:**
- Operates on **1/16-scale** latent features (24×24 for 384×384 input)
- Two-layer CNN: `Conv2d(dim → dim/2, 3×3) → GELU → Dropout → Conv2d(dim/2 → C, 1×1)`
- Produces **C-class segmentation logits** (C=81 for COCO classes)
- Supervised by YOLO-generated pseudo-labels during training

**Fusion mechanism — Cross-attention with gating:**
1. Segmentation logits are projected to query features via `seg_to_query` (linear projection)
2. Cross-attention between seg queries and depth features extracts class-conditioned depth cues
3. A learned gate (sigmoid) controls how much semantic information blends into the depth stream:

$$\text{fused} = \text{latents}_{16} + \sigma(g) \cdot (\text{CrossAttn}(\text{seg\_query}, \text{latents}_{16}) - \text{latents}_{16})$$

**Parameter cost:** ~5.2M parameters (2.1% of total model)

### Low-Level Channel: EdgeHead

**Purpose:** Preserve sharp depth transitions at object boundaries, particularly where transparent surfaces create abrupt depth changes invisible in RGB.

**Architecture:**
- Operates on **1/2-scale** latent features (192×192 for 384×384 input)
- Edge detection branch: `Conv2d(dim → dim, 3×3) → GELU → Dropout → Conv2d(dim → 1, 1×1)`
- Edge-to-feature projection: `Conv2d(1 → dim, 1×1) → GELU`
- Gated refinement: `Conv2d(2·dim → dim, 1×1) → GELU → Conv2d(dim → dim, 1×1) → Sigmoid`

**Fusion mechanism — Gated residual refinement:**
1. Edge logits are converted to feature-space representation via `edge_to_feat`
2. Depth features and edge features are concatenated and passed through a gate network
3. The gate controls per-channel blending:

$$\text{refined} = \text{latents}_2 + \sigma(\text{gate}(\text{latents}_2 \| \text{edge\_feat})) \cdot \text{edge\_feat}$$

**Parameter cost:** ~50K parameters (0.02% of total model)

## New Loss Functions

### 1. SegCrossEntropy — Auxiliary Segmentation Loss

Supervises the SegHead with pseudo-labels from a pretrained YOLO segmentation model.

$$\mathcal{L}_{\text{seg}} = \text{CE}(\hat{y}_{\text{seg}},\; y_{\text{seg}}^{\downarrow})$$

where $y_{\text{seg}}^{\downarrow}$ is the ground-truth label map downsampled to prediction resolution via nearest-neighbor interpolation. Pixels with label 255 are ignored.

- **Weight:** 0.5 (default)
- **Why:** Teaches the decoder which regions are transparent/reflective, enabling material-aware depth reasoning.

### 2. EdgeBCE — Scale-Invariant Edge Supervision

Supervises the EdgeHead using automatically generated edge maps from depth ground truth.

**Edge GT generation:**
1. Compute Sobel gradients $(g_x, g_y)$ on the depth map
2. Compute gradient magnitude: $m = \sqrt{g_x^2 + g_y^2}$
3. Normalize by local depth for scale invariance: $\hat{m} = m / (d + \epsilon)$
4. Threshold to binary: $e = \mathbb{1}[\hat{m} > 0.1]$

**Loss:**

$$\mathcal{L}_{\text{edge}} = -\frac{1}{N}\sum\left[w_+ \cdot e \cdot \log\sigma(\hat{e}) + w_- \cdot (1-e) \cdot \log(1 - \sigma(\hat{e}))\right]$$

with balanced weights $w_+ = N / (2 \cdot N_+)$, $w_- = N / (2 \cdot N_-)$ to handle class imbalance (edges are sparse).

- **Weight:** 0.5 (default)
- **Why:** Teaches the decoder to detect depth discontinuities, which are especially important at transparent surface boundaries where RGB edges may be weak or absent.

### 3. TransparencyBoundaryLoss — Gradient Matching at Object Boundaries

Enforces accurate depth gradients specifically in the boundary zones of segmented objects.

**Boundary zone extraction:**
1. Dilate the segmentation mask by $k$ pixels via max-pooling
2. Erode the segmentation mask by $k$ pixels via min-pooling (negated max-pool)
3. Boundary zone = dilated − eroded (a band around each object contour)

**Loss:**

$$\mathcal{L}_{\text{boundary}} = \frac{1}{|\mathcal{Z}|} \sum_{(i,j) \in \mathcal{Z}} \left(|\nabla_x \hat{d} - \nabla_x d| + |\nabla_y \hat{d} - \nabla_y d|\right)$$

where $\mathcal{Z}$ is the boundary zone and gradients are computed with Sobel filters.

- **Weight:** 0.5 (default), **Dilation:** 8 pixels
- **Why:** Transparent surfaces often have correct depth in their interior but incorrect transitions at boundaries. This loss specifically penalizes gradient errors where they matter most.

## Scale-Invariant Training

All experiments use **scale-invariant (SI) depth evaluation**, which removes the global scale ambiguity. This is particularly important for transparent surfaces because:

- Transparent objects are often at intermediate depths where scale errors compound
- The ToM (Transparent Object Matting) dataset has different depth distributions than NYUv2
- SI metrics focus on *relative depth ordering* rather than absolute values

The SILog loss with `si=True` is used:

$$\mathcal{L}_{\text{SILog}} = \sqrt{\frac{1}{N}\sum d_i^2 - \frac{\lambda}{N^2}\left(\sum d_i\right)^2}$$

where $d_i = \log \hat{d}_i - \log d_i$ and $\lambda = 0.85$ (integrated parameter).

## Training Data

| Dataset | Samples | Content | Role |
|---------|---------|---------|------|
| **NYUv2** | 795 train / 654 val | Indoor RGB-D scenes | Primary depth + seg labels |
| **ToM** | ~13,500 | Transparent objects (glass, mirrors) | Domain-specific transparent surfaces |

NYUv2 segmentation labels are generated offline using a YOLO model and stored as a NumPy array (`nyuv2_yolo_seg_labels.npy`, 81 COCO classes).

## Ablation Study Design

Six experiments systematically evaluate each component:

| Exp | Name | Edge | Seg | Boundary | Extras | Purpose |
|-----|------|------|-----|----------|--------|---------|
| E1 | Baseline | — | — | — | — | Reference (SI + ToM + color jitter + deep supervision) |
| E2 | +Edge | ✓ | — | — | — | Isolate low-level channel contribution |
| E3 | +Seg | — | ✓ | — | — | Isolate high-level channel contribution |
| E4 | Dual-channel | ✓ | ✓ | — | — | Test synergy of both channels |
| E5 | +Boundary | ✓ | ✓ | ✓ | — | Add boundary gradient supervision |
| E6 | Full | ✓ | ✓ | ✓ | grad_match + edge_ssi | Kitchen sink with all auxiliary losses |

**Common configuration across all experiments:**
- Encoder: DINOv3 ViT-L/16 (frozen for first 3 epochs)
- Image resolution: 384 × 384
- Optimizer: AdamW (decoder LR=1e-4, encoder LR=1e-5, weight decay=0.01)
- Scheduler: Cosine annealing (warmup 500 steps, min LR=1e-6)
- Batch size: 2 × 8 accumulation steps = effective batch 16
- Epochs: 20
- Augmentation: Color jitter (σ=0.3)
- Deep supervision: Enabled (losses at 1/8, 1/4, 1/2 scales)

## Expected Outcomes

| Hypothesis | Experiment | Expected Result |
|------------|------------|-----------------|
| Edge features improve boundary sharpness | E2 vs E1 | Lower edge-related error metrics |
| Semantic cues help material identification | E3 vs E1 | Better depth on transparent regions |
| Dual-channel is better than either alone | E4 vs E2, E3 | Synergistic improvement |
| Boundary loss sharpens transitions | E5 vs E4 | Improved gradient accuracy at edges |
| Additional losses have diminishing returns | E6 vs E5 | Marginal gains, possible overfitting |

## Running the Experiments

### Prerequisites

```bash
# Activate environment
conda activate DepthSense

# Fix libc++ issue (if on the HKUST cluster)
export LD_LIBRARY_PATH=/localdata/chzheng/miniconda3/envs/DepthSense/lib:$LD_LIBRARY_PATH

# Set cache directories
export TORCH_HOME=./cache
export HF_HOME=./cache

# Python binary (adjust if different)
PYTHON=/localdata/chzheng/miniconda3/envs/DepthSense/bin/python
```

### Launch All Experiments at Once

The ablation script assigns one experiment per GPU and launches them in parallel:

```bash
# Launch all 6 experiments on GPUs 0-5 (one per GPU):
bash run_script/run_transparent_ablation.sh "0,1,2,3,4,5" 20

# If only 2 GPUs are free, experiments run in round-robin (2 at a time):
bash run_script/run_transparent_ablation.sh "2,3" 20

# Monitor progress:
tail -f runs/E*_*.log
```

### Launch Individual Experiments

All experiments share these common flags. Set `GPU` to a free GPU index:

```bash
GPU=0  # <-- change to your free GPU

COMMON="\
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
  --datasets nyuv2,tom --color_jitter 0.3 --deep_supervision true --amp"
```

#### E1 — Baseline

Reference experiment with scale-invariant training on NYUv2+ToM, color jitter, and deep supervision. No auxiliary heads.

```bash
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m train.train_depth \
  --cuda 0 --run_name E1_baseline \
  $COMMON \
  > runs/E1_baseline.log 2>&1 &
```

#### E2 — +Edge Head (Low-Level Channel Only)

Adds the EdgeHead at 1/2 scale with gated residual refinement. Tests whether learned edge detection alone improves boundary depth.

```bash
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m train.train_depth \
  --cuda 0 --run_name E2_edge_head \
  $COMMON \
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  > runs/E2_edge_head.log 2>&1 &
```

#### E3 — +Seg Auxiliary (High-Level Channel Only)

Adds the SegHead at 1/16 scale with cross-attention fusion. Tests whether semantic class awareness alone helps transparent surface depth.

```bash
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m train.train_depth \
  --cuda 0 --run_name E3_seg_aux \
  $COMMON \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy \
  > runs/E3_seg_aux.log 2>&1 &
```

#### E4 — Dual-Channel (Edge + Seg Together)

Combines both channels: EdgeHead for low-level boundary features and SegHead for high-level semantic features. Tests whether the two channels complement each other.

```bash
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m train.train_depth \
  --cuda 0 --run_name E4_dual_channel \
  $COMMON \
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy \
  > runs/E4_dual_channel.log 2>&1 &
```

#### E5 — Dual-Channel + Boundary Loss

Adds the TransparencyBoundaryLoss on top of E4. This loss penalizes depth gradient errors specifically in the dilated contour zone around segmented objects, encouraging sharp transitions at transparent surface edges.

```bash
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m train.train_depth \
  --cuda 0 --run_name E5_dual_boundary \
  $COMMON \
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy \
  --boundary_loss_weight 0.5 \
  > runs/E5_dual_boundary.log 2>&1 &
```

#### E6 — Full (All Losses)

Kitchen sink: adds gradient matching (from DepthAnythingV2/MiDaS) and edge-guided local SSI (from UniDepthV2) on top of E5. Tests whether additional established losses provide further gains.

```bash
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m train.train_depth \
  --cuda 0 --run_name E6_full_plus_extras \
  $COMMON \
  --use_edge_head true \
  --edge_loss_weight 0.5 \
  --use_seg_auxiliary true \
  --seg_loss_weight 0.5 \
  --seg_labels_path datasets/nyuv2_yolo_seg_labels.npy \
  --boundary_loss_weight 0.5 \
  --grad_matching_weight 1.0 \
  --edge_guided_ssi_weight 0.5 \
  > runs/E6_full_plus_extras.log 2>&1 &
```

### Monitoring and Checking Results

```bash
# Watch a specific experiment's loss curve:
tail -f runs/E4_dual_channel.log

# Check GPU memory usage:
nvidia-smi

# Compare results after training:
python compare_ablation.py
```

## File Changes Summary

| File | Change |
|------|--------|
| `model/unidepthv1/decoder.py` | Added SegHead, EdgeHead classes; dual-channel fusion in DepthHead |
| `model/unidepthv1/unidepthv1.py` | Updated output unpacking, added seg/edge/boundary loss computation |
| `model/ops/losses/seg_ce.py` | New: SegCrossEntropy loss |
| `model/ops/losses/edge_bce.py` | New: EdgeBCE loss |
| `model/ops/losses/boundary_loss.py` | New: TransparencyBoundaryLoss |
| `model/ops/losses/__init__.py` | Registered new loss modules |
| `train/train_depth.py` | New CLI args, config wiring, ToM dataset registration, seg label pipeline |
| `data/nyuv2_dataset.py` | Added seg label loading, SI=True |
| `data/ToM_dataset.py` | SI=True |
| `run_script/run_transparent_ablation.sh` | New: 6-experiment ablation launch script |
