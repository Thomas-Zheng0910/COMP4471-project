#!/usr/bin/env python3
"""
visualize_datasets.py — View RGB images and their GT depth maps for ALL datasets.

Covers:
  Training:  nyuv2, sunrgbd, vkitti2, sintel
  Eval-only: ibims1, diode_indoor
  (KITTI/ToM skipped if _depth_anything.png files not yet generated)

Produces matplotlib figures (RGB | Depth with colorbar) saved under outputs/dataset_preview/.

Usage:
    python visualize_datasets.py [--n 5] [--datasets all]
    python visualize_datasets.py --n 3 --datasets nyuv2,sintel,ibims1
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.train_depth import build_dataset, DATASET_DEFAULT_ROOTS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Additional eval-only datasets not in DATASET_DEFAULT_ROOTS
EVAL_DATASETS = {
    "ibims1": "datasets/unidepth_data",
    "diode_indoor": "datasets/diode_indoor",
    "kitti": "datasets/Diffusion4RobustDepth/kitti/driving/kitti/challenging",
    "tom": "datasets/Diffusion4RobustDepth/ToM",
}

ALL_DATASETS = list(DATASET_DEFAULT_ROOTS.keys()) + list(EVAL_DATASETS.keys())


def build_any_dataset(name: str, split: str, image_shape=(480, 640)):
    """Build any dataset (training or eval-only) by name."""
    data_cfg = {"image_shape": list(image_shape), "depth_scale": 1.0, "use_lidar": False}

    # Training datasets go through the standard factory
    if name in DATASET_DEFAULT_ROOTS:
        return build_dataset(name, split=split, data_cfg=data_cfg, flip_aug=False)

    # Eval-only datasets
    if name == "ibims1":
        from data.ibims_dataset import IBims1Dataset
        return IBims1Dataset(root=EVAL_DATASETS[name], split="test", image_shape=image_shape)
    elif name == "diode_indoor":
        from data.diode_dataset import DIODEIndoorDataset
        return DIODEIndoorDataset(root=EVAL_DATASETS[name], split="val", image_shape=image_shape)
    elif name == "kitti":
        from data.kitti_dataset import KITTIDataset
        return KITTIDataset(root=EVAL_DATASETS[name], split=split, image_shape=image_shape)
    elif name == "tom":
        from data.ToM_dataset import ToMDataset
        return ToMDataset(root=EVAL_DATASETS[name], split=split, image_shape=image_shape)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: {ALL_DATASETS}")


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized (3,H,W) tensor back to (H,W,3) float [0,1]."""
    img = image_tensor.cpu().float().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def save_sample_figure(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray,
                       title: str, path: str):
    """Save a single-sample figure: RGB | Depth (plasma cmap, colorbar in metres)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Depth (mask invalid pixels)
    depth_vis = np.where(mask, depth, np.nan)
    vmin = np.nanmin(depth_vis) if mask.any() else 0
    vmax = np.nanmax(depth_vis) if mask.any() else 1
    im = axes[1].imshow(depth_vis, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"GT Depth  [{vmin:.2f}, {vmax:.2f}] m")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (m)")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_grid_figure(samples: list, ds_name: str, path: str):
    """Save an N-row grid: each row is RGB | Depth for one sample."""
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(f"{ds_name} — {n} random samples", fontsize=15, fontweight="bold")

    for i, (rgb, depth, mask, stats_str, idx) in enumerate(samples):
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"#{idx}  RGB", fontsize=10)
        axes[i, 0].axis("off")

        depth_vis = np.where(mask, depth, np.nan)
        vmin = np.nanmin(depth_vis) if mask.any() else 0
        vmax = np.nanmax(depth_vis) if mask.any() else 1
        im = axes[i, 1].imshow(depth_vis, cmap="plasma", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"GT Depth  {stats_str}", fontsize=10)
        axes[i, 1].axis("off")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04, label="m")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--n", type=int, default=5, help="Number of samples per dataset")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated dataset names, or 'all'")
    parser.add_argument("--image_shape", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--outdir", type=str, default="outputs/dataset_preview")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.datasets.strip().lower() == "all":
        dataset_names = list(ALL_DATASETS)
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]

    rng = np.random.RandomState(args.seed)

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")
        try:
            ds = build_any_dataset(ds_name, split=args.split,
                                   image_shape=tuple(args.image_shape))
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        total = len(ds)
        if total == 0:
            print(f"  SKIP: 0 samples")
            continue
        n = min(args.n, total)
        indices = rng.choice(total, size=n, replace=False)
        indices.sort()

        grid_data = []
        for j, idx in enumerate(indices):
            try:
                sample = ds[idx]
            except Exception as e:
                print(f"  [{j}] idx={idx}: ERROR loading sample: {e}")
                continue
            if isinstance(sample, (list, tuple)):
                sample = sample[0]

            rgb = denormalize_image(sample["image"])
            depth = sample["depth"][0].cpu().numpy()
            mask = sample["depth_mask"][0].cpu().numpy().astype(bool)

            valid_depth = depth[mask]
            if len(valid_depth) > 0:
                dmin, dmax = valid_depth.min(), valid_depth.max()
                valid_pct = 100 * mask.sum() / mask.size
                stats = f"[{dmin:.2f}, {dmax:.2f}] m  ({valid_pct:.0f}% valid)"
            else:
                stats = "NO VALID PIXELS"

            title = f"{ds_name}  sample #{idx}"
            fname = f"{ds_name}_{j:02d}_idx{idx}.png"
            save_sample_figure(rgb, depth, mask, title, os.path.join(args.outdir, fname))
            print(f"  [{j}] idx={idx:5d}  {stats}  -> {fname}")

            grid_data.append((rgb, depth, mask, stats, idx))

        if grid_data:
            grid_fname = f"{ds_name}_grid.png"
            save_grid_figure(grid_data, ds_name, os.path.join(args.outdir, grid_fname))
            print(f"  Grid -> {grid_fname}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("  SUMMARY (checking up to 50 random samples per dataset)")
    print(f"{'='*60}")
    for ds_name in dataset_names:
        try:
            ds = build_any_dataset(ds_name, split=args.split,
                                   image_shape=tuple(args.image_shape))
        except Exception:
            print(f"  {ds_name}: LOAD FAILED")
            continue

        total = len(ds)
        if total == 0:
            print(f"  {ds_name}: 0 samples")
            continue
        check_n = min(50, total)
        check_idx = rng.choice(total, size=check_n, replace=False)

        depth_mins, depth_maxs, valid_pcts, all_zeros = [], [], [], 0
        for idx in check_idx:
            try:
                sample = ds[idx]
            except Exception:
                continue
            if isinstance(sample, (list, tuple)):
                sample = sample[0]
            depth = sample["depth"][0].cpu().numpy()
            mask = sample["depth_mask"][0].cpu().numpy().astype(bool)
            valid = depth[mask]
            if len(valid) > 0:
                depth_mins.append(valid.min())
                depth_maxs.append(valid.max())
                valid_pcts.append(100 * mask.sum() / mask.size)
            else:
                all_zeros += 1
                valid_pcts.append(0)

        print(f"\n  {ds_name} ({total} samples, checked {check_n}):")
        if depth_mins:
            print(f"    Depth range: [{np.min(depth_mins):.3f}, {np.max(depth_maxs):.3f}]m")
            print(f"    Mean depth range: [{np.mean(depth_mins):.3f}, {np.mean(depth_maxs):.3f}]m")
        if valid_pcts:
            print(f"    Valid pixel %: mean={np.mean(valid_pcts):.1f}%, min={np.min(valid_pcts):.1f}%, max={np.max(valid_pcts):.1f}%")
        if all_zeros:
            print(f"    WARNING: {all_zeros}/{check_n} samples have ZERO valid pixels!")

    print(f"\nAll outputs saved to {args.outdir}/")


if __name__ == "__main__":
    main()
