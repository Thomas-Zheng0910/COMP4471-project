# COMP4471-project

Monocular metric depth estimation based on UniDepthV1 with DINOv3 encoder, trained and evaluated on NYUv2 (and optionally additional datasets).

## Setup

```bash
conda env create -f environment.yml
conda activate <env_name>
bash libc++FIX.sh   # fix libc++/pandas import issues on Linux if needed
```

## Data Preparation

### NYUv2 (required)

```bash
bash data/get_datasets/get_nyu_v2.sh
```

Places `datasets/nyu_depth_v2_labeled.mat` (~2.8 GB).

### Additional Training Datasets (optional)

Download SUN-RGBD, Virtual KITTI 2, Sintel, iBims-1, and DIODE Indoor:

```bash
bash data/get_datasets/get_training_datasets.sh
```

Use `--skip-sunrgbd`, `--skip-vkitti2`, etc. to skip individual datasets. See script for details.

## Training

### Quick Start (NYUv2 only)

```bash
bash run_script/run_train.sh
```

This runs UniDepthV1 with ViT-L encoder on NYUv2. Key settings in the script:
- `CUDA=3` — GPU index
- `DATASETS="nyuv2"` — training datasets (comma-separated; add `sunrgbd`, `sintel`, `vkitti2` after downloading)
- `EPOCHS=200`, `BATCH_SIZE=2`, `ACCUM_STEPS=4` — effective batch size 16
- `ENCODER_LR=1e-5`, `FREEZE_ENCODER_EPOCHS=5` — prevents encoder catastrophic forgetting
- `WARMUP_STEPS=500` — linear LR warmup before cosine annealing

### Custom Experiment

Copy and edit the template:

```bash
cp run_script/run_train_template.sh run_script/run_train_myexp.sh
# Edit CUDA, DATASETS, EPOCHS, etc.
bash run_script/run_train_myexp.sh
```

### Multi-Dataset Training

After downloading extra datasets, set `DATASETS` in the run script:

```bash
DATASETS="nyuv2,sunrgbd,sintel"
```

Available training datasets: `nyuv2`, `sunrgbd`, `vkitti2`, `sintel`.

### Outputs

Training outputs are saved under `runs/train_depth_<timestamp>/`:
- `checkpoints/` — model checkpoints every `SAVE_EVERY` epochs
- `tensorboard/` — TensorBoard logs (loss, LR, depth visualisations)
- `run_script.sh` — copy of the launch script for reproducibility

Monitor training:
```bash
tensorboard --logdir runs/
```

## Inference & Evaluation (Our Model)

Evaluate a trained checkpoint on NYUv2 test set:

```bash
bash run_script/run_infer.sh
```

Edit the script to set:
- `CHECKPOINT="runs/<run_name>/checkpoints/epoch_200.pth"` — path to your checkpoint
- `ENCODER_NAME` / `OUTPUT_IDX` — must match training config
- `DATA_ROOT` — dataset to evaluate on
- `IMAGE_FOLDER` — (optional) run on arbitrary images with visualisations

Metrics JSON and visualisations are saved to `OUTPUT_DIR` (default `runs/infer/`).

## Baseline Evaluation

Evaluate pretrained baselines (UniDepth V2, Depth Anything V2, Marigold) on NYUv2:

```bash
bash run_script/run_eval_baselines.sh
```

Baseline weights are auto-downloaded from HuggingFace on first run. Results are saved to `runs/baselines/`.

To evaluate on multiple benchmarks (after downloading iBims-1 / DIODE), set in the script:

```bash
EVAL_DATASETS="nyuv2,ibims1,diode_indoor"
```

## Project Structure

See [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) for detailed file descriptions.

```
run_script/
  run_train.sh              # Main training script (ready to run)
  run_train_template.sh     # Training template (copy & edit)
  run_infer.sh              # Inference / evaluation script
  run_eval_baselines.sh     # Baseline model evaluation
data/
  get_datasets/
    get_nyu_v2.sh           # Download NYUv2
    get_training_datasets.sh # Download extra datasets
train/
  train_depth.py            # Training entry point
infer/
  infer_depth.py            # Inference entry point (our model)
  eval_baselines.py         # Baseline evaluation entry point
```