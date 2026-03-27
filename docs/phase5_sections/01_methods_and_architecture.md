# 1) Method and Architecture Comparison (4 Phase 5 Methods)

## 1.1 Four methods used in training

From `run_script/run_phase5_ablation.sh`, the four variants are launched as:

1. **RGB-only** (`phase5_ablation=rgb_only`)
2. **Supervision-only** (`phase5_ablation=supervision_only`)
3. **Late fusion** (`phase5_ablation=late_fusion`, `--lidar_fusion_type late`)
4. **Token fusion** (`phase5_ablation=token_fusion`, `--lidar_fusion_type token`)

These are reflected in `runs/phase5_fullscale_latest.json` under `runs[].ablation`.

---

## 1.2 High-level architecture: what is shared vs what changes

### Shared backbone and decoder pipeline (all four methods)

All methods use the same base model path:

- `train/train_depth.py` creates `UniDepthV1(config)`.
- `model/unidepthv1/unidepthv1.py`:
  - `encode_decode(...)` performs RGB feature extraction via `self.pixel_encoder`.
  - Then calls `self.pixel_decoder.forward(inputs, {})`.
- `model/unidepthv1/decoder.py`:
  - `Decoder.forward(...)` computes camera and depth branches.
  - `DepthHead.forward(...)` predicts multi-scale depth outputs (`out8`, `out4`, `out2`) and fusion stats.

So the core network skeleton is shared; the ablation changes are mostly LiDAR-related feature usage and supervision path.

### What changes across methods

- **RGB-only**: no LiDAR input, no LiDAR fusion branch.
- **Supervision-only**: LiDAR is loaded and used in sparse LiDAR loss, but no architectural fusion in decoder.
- **Late fusion**: LiDAR is fused once at 1/16 feature scale through prompt + gate.
- **Token fusion**: LiDAR is fused at multiple scales (1/16, 1/8, 1/4) with scale-specific cross-attention + gates.

---

## 1.3 Method-level differences (algorithmic behavior)

## A) RGB-only

In `train/train_depth.py`, ablation logic sets:

- `config["data"]["use_lidar"] = False`
- `config["model"]["pixel_decoder"]["use_lidar_fusion"] = False`

Effects:

- Dataset does not load LiDAR tensors.
- No LiDAR sparse loss can be computed (`compute_lidar_sparse_loss` path not activated).
- Decoder runs pure RGB depth estimation.

## B) Supervision-only

Ablation logic sets only:

- `use_lidar_fusion = False`

But `use_lidar` remains enabled from common run arguments.

Effects:

- LiDAR tensors (`lidar_depth`, `lidar_mask`, optional confidence) are loaded by `data/nyuv2_dataset.py`.
- Training adds sparse LiDAR supervision (`LiDARSparse`) in `train/train_depth.py` via `compute_lidar_sparse_loss(...)` and `losses["opt"]["LiDARSparse"] = lidar_weight * lidar_raw_loss`.
- Decoder does not inject LiDAR features into latent tokens.

Interpretation: this is a supervision-side ablation without representation fusion.

## C) Late fusion

Ablation logic sets:

- `use_lidar_fusion = True`
- `lidar_fusion_type = "late"`

In `model/unidepthv1/decoder.py` (`DepthHead`):

- LiDAR pack is built as 3 channels: `[lidar_depth, lidar_mask, lidar_confidence]`.
- LiDAR is downsampled to 1/16 scale.
- `lidar_encoder` converts LiDAR pack to latent features.
- `prompt_lidar` cross-attends LiDAR tokens with depth latents.
- `lidar_gate` learns how much LiDAR prompt should update latents.

Formula-like update at 1/16 scale:

$$
\mathbf{z}_{16} \leftarrow \mathbf{z}_{16} + g\big(\mathbf{z}_{16}, \hat{\mathbf{z}}_{lidar}\big) \odot \big(\hat{\mathbf{z}}_{lidar} - \mathbf{z}_{16}\big)
$$

where $g(\cdot)$ is the learned gate in `[0,1]`.

## D) Token fusion

Ablation logic sets:

- `use_lidar_fusion = True`
- `lidar_fusion_type = "token"`

In `DepthHead`, token-fusion-specific modules are enabled:

- Multi-scale LiDAR encoders: `lidar_encoder_16`, `lidar_encoder_8`, `lidar_encoder_4`.
- Multi-scale cross-attention fusion: `lidar_fusion_16`, `lidar_fusion_8`, `lidar_fusion_4`.
- Scale-specific gates: `lidar_gate_16`, `lidar_gate_8`, `lidar_gate_4`.

This creates progressive LiDAR-RGB interactions at coarse-to-fine levels rather than a single late injection.

---

## 1.4 How structure is built in code (file-by-file mapping)

## Entry and experiment orchestration

- `run_script/run_phase5_ablation.sh`
  - Defines and launches the four variants.
  - Writes run mapping JSON (`runs/phase5_fullscale_<timestamp>.json` + latest alias).
  - Auto-runs plotting script after training.

## Training and ablation switching

- `train/train_depth.py`
  - CLI defines `--phase5_ablation` and `--lidar_fusion_type`.
  - Ablation block rewrites config flags before model creation.
  - Computes sparse LiDAR loss through `compute_lidar_sparse_loss`.
  - Logs both optimization losses and validation metrics (`val_rmse`, `val_abs_rel`).

## Model architecture and fusion mechanics

- `model/unidepthv1/unidepthv1.py`
  - `UniDepthV1.build(...)` instantiates encoder and decoder.
  - `encode_decode(...)` routes batch tensors into decoder and returns `fusion_stats`.

- `model/unidepthv1/decoder.py`
  - `Decoder.build(...)` passes `use_lidar_fusion` and `lidar_fusion_type` into `DepthHead`.
  - `DepthHead.forward(...)` contains the key branch:
    - no LiDAR fusion,
    - late fusion at 1/16,
    - token fusion at 1/16 + 1/8 + 1/4.

## Data loading and LiDAR tensor construction

- `data/nyuv2_dataset.py`
  - Loads RGB/depth from `.mat`.
  - If LiDAR is enabled, loads LiDAR depth (and optional confidence) either from HDF5 keys or external files.
  - Produces `lidar_mask` from valid sparse depth pixels.
  - `collate_fn` packs tensor fields and metadata for training.

## Consistency checker for Phase 5 implementation

- `validate_phase5.py`
  - Validates that late/token modules exist.
  - Verifies ablation config behavior.
  - Sanity-checks forward pass under variant settings.

---

## 1.5 Practical interpretation of each method

- **RGB-only**: baseline monocular model capacity without LiDAR information.
- **Supervision-only**: tests whether sparse LiDAR as an extra loss (but not feature fusion) is enough.
- **Late fusion**: tests single-stage feature injection at coarse semantic scale.
- **Token fusion**: tests multi-scale LiDAR interaction as richer geometric conditioning.

This sequence is methodologically clean because it separates the effect of:

1. LiDAR supervision alone, and
2. architectural fusion design.
