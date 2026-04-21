"""
Generate PromptDA-style simulated LiDAR from dense depth maps.

Implements the "sparse anchor interpolation" method from:
  PromptDA (Lin et al., CVPR 2025) - Section 3.3 "Synthetic data: LiDAR simulation"

Algorithm:
  1. Downsample GT depth to low resolution (default 192x256, iPhone ARKit depth res)
  2. Sample sparse anchor points on the low-res depth using a distorted grid
     with a stride (default 7), adding random jitter to grid positions
  3. Interpolate remaining depth values from anchor points using RGB-similarity
     KNN (K nearest anchors weighted by color similarity)

This produces a filled low-res depth map with interpolation artifacts at depth
boundaries, mimicking real LiDAR noise patterns. Naive downsampling alone leads
to the model learning depth super-resolution rather than noise correction.

Supports three input modes:
  1) NYUv2 .mat file  (--input-mat)
  2) Directory of .npy depth files  (--input-npy-dir)
  3) Image directory tree with depth PNGs  (--input-image-dir + --depth-suffix)

Usage:
    # NYUv2
    python data/preprocess/generate_simulated_lidar.py \
        --input-mat datasets/nyu_depth_v2_labeled.mat \
        --output-dir datasets/nyuv2_lidar_projected \
        --lidar-h 192 --lidar-w 256 --stride 7 --knn-k 4

    # ToM (Depth-Anything PNGs in a tree)
    python data/preprocess/generate_simulated_lidar.py \
        --input-image-dir datasets/Diffusion4RobustDepth/ToM \
        --depth-suffix _depth_anything \
        --output-dir datasets/tom_lidar_projected \
        --lidar-h 192 --lidar-w 256 --stride 7 --knn-k 4
"""
import argparse
from pathlib import Path

import numpy as np
import h5py
from PIL import Image
from scipy.spatial import cKDTree


# ─── Core simulation ────────────────────────────────────────────────────────

def _downsample_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Downsample depth to (target_h, target_w) via PIL bilinear."""
    img = Image.fromarray(depth.astype(np.float32))
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def _downsample_rgb(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Downsample RGB to (target_h, target_w) via PIL bilinear."""
    img = Image.fromarray(rgb.astype(np.uint8))
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def _sample_distorted_grid(h: int, w: int, stride: int, rng: np.random.Generator,
                           jitter: float = 0.5) -> np.ndarray:
    """Sample anchor positions on a regular grid with random jitter.

    Returns (N, 2) array of (row, col) integer coordinates clamped to [0, h) x [0, w).
    jitter: max displacement as a fraction of stride (0.5 = up to half-stride).
    """
    rows = np.arange(stride // 2, h, stride)
    cols = np.arange(stride // 2, w, stride)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    grid_r = grid_r.ravel().astype(np.float64)
    grid_c = grid_c.ravel().astype(np.float64)

    max_offset = jitter * stride
    grid_r += rng.uniform(-max_offset, max_offset, size=grid_r.shape)
    grid_c += rng.uniform(-max_offset, max_offset, size=grid_c.shape)

    grid_r = np.clip(np.round(grid_r), 0, h - 1).astype(np.int32)
    grid_c = np.clip(np.round(grid_c), 0, w - 1).astype(np.int32)

    coords = np.stack([grid_r, grid_c], axis=1)
    # Remove duplicates after rounding
    coords = np.unique(coords, axis=0)
    return coords


def simulate_lidar(depth: np.ndarray,
                   rgb: np.ndarray | None,
                   lidar_h: int = 192,
                   lidar_w: int = 256,
                   stride: int = 7,
                   knn_k: int = 4,
                   rgb_sigma: float = 20.0,
                   jitter: float = 0.5,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate LiDAR via PromptDA sparse-anchor interpolation.

    Args:
        depth: (H, W) dense GT depth.
        rgb: (H, W, 3) corresponding RGB image, or None. If None, KNN uses
             spatial distance only (no RGB weighting).
        lidar_h, lidar_w: target LiDAR resolution.
        stride: grid stride for anchor sampling.
        knn_k: number of nearest anchors for interpolation.
        rgb_sigma: bandwidth for RGB similarity kernel exp(-||c1-c2||^2 / sigma^2).
        jitter: max jitter as fraction of stride.
        rng: numpy random Generator.

    Returns:
        (lidar_h, lidar_w) simulated LiDAR depth map.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Step 1: Downsample depth (and RGB) to LiDAR resolution
    depth_lr = _downsample_depth(depth, lidar_h, lidar_w)
    if rgb is not None:
        rgb_lr = _downsample_rgb(rgb, lidar_h, lidar_w)
    else:
        rgb_lr = None

    # Step 2: Sample sparse anchor points on distorted grid
    anchors = _sample_distorted_grid(lidar_h, lidar_w, stride, rng, jitter)

    # Filter anchors to valid depth locations
    anchor_depths = depth_lr[anchors[:, 0], anchors[:, 1]]
    valid = np.isfinite(anchor_depths) & (anchor_depths > 0)
    anchors = anchors[valid]
    anchor_depths = anchor_depths[valid]

    if len(anchors) == 0:
        return np.zeros((lidar_h, lidar_w), dtype=np.float32)

    # Step 3: Interpolate remaining pixels from anchors using RGB-similarity KNN
    all_rows, all_cols = np.mgrid[0:lidar_h, 0:lidar_w]
    all_coords = np.stack([all_rows.ravel(), all_cols.ravel()], axis=1)  # (N_total, 2)

    # Build KD-tree on anchor spatial coordinates
    anchor_tree = cKDTree(anchors.astype(np.float64))
    k = min(knn_k, len(anchors))
    dists, indices = anchor_tree.query(all_coords.astype(np.float64), k=k)

    if k == 1:
        dists = dists[:, np.newaxis]
        indices = indices[:, np.newaxis]

    # Compute spatial weights (inverse distance, avoid div-by-zero)
    eps = 1e-8
    spatial_w = 1.0 / (dists + eps)

    # Compute RGB similarity weights if RGB is available
    if rgb_lr is not None:
        anchor_colors = rgb_lr[anchors[:, 0], anchors[:, 1]]  # (n_anchors, 3)
        pixel_colors = rgb_lr[all_coords[:, 0], all_coords[:, 1]]  # (N_total, 3)

        neighbor_colors = anchor_colors[indices]  # (N_total, k, 3)
        pixel_colors_exp = pixel_colors[:, np.newaxis, :]  # (N_total, 1, 3)

        color_diff_sq = np.sum((neighbor_colors - pixel_colors_exp) ** 2, axis=2)
        rgb_w = np.exp(-color_diff_sq / (rgb_sigma ** 2))
        weights = spatial_w * rgb_w
    else:
        weights = spatial_w

    # Normalize weights
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum < eps, 1.0, weight_sum)
    weights = weights / weight_sum

    # Weighted average of anchor depths
    neighbor_depths = anchor_depths[indices]  # (N_total, k)
    interp_depths = np.sum(weights * neighbor_depths, axis=1)

    simulated = interp_depths.reshape(lidar_h, lidar_w).astype(np.float32)
    return simulated


# ─── Dataset processing ─────────────────────────────────────────────────────

def process_mat(input_mat, output_dir, lidar_h, lidar_w, stride, knn_k,
                rgb_sigma, jitter, split=None, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with h5py.File(input_mat, "r") as h5:
        depths = h5["depths"]
        # NYUv2 .mat also has "images" (3, H, W) uint8
        has_images = "images" in h5
        total = depths.shape[0]
        indices = list(range(total))
        if split is not None and split != "all":
            split_file = Path(input_mat).parent / f"nyuv2_{split}.txt"
            if split_file.exists():
                indices = [int(x.strip()) for x in split_file.read_text().split()
                           if x.strip().isdigit()]
        for idx in indices:
            depth = np.array(depths[idx]).T  # HDF5 stores transposed
            if has_images:
                rgb = np.array(h5["images"][idx]).transpose(2, 1, 0)  # (H, W, 3)
            else:
                rgb = None
            sim = simulate_lidar(depth, rgb, lidar_h, lidar_w, stride, knn_k,
                                 rgb_sigma, jitter, rng)
            np.save(output_dir / f"{idx:05d}.npy", sim)
            if idx % 100 == 0:
                print(f"[{idx}/{total}] saved {output_dir / f'{idx:05d}.npy'}")


def process_npy_dir(input_dir, output_dir, lidar_h, lidar_w, stride, knn_k,
                    rgb_sigma, jitter, seed=42):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    files = sorted(input_dir.glob("*.npy"))
    for idx, f in enumerate(files):
        depth = np.load(f)
        sim = simulate_lidar(depth, None, lidar_h, lidar_w, stride, knn_k,
                             rgb_sigma, jitter, rng)
        np.save(output_dir / f.name, sim)
        if idx % 100 == 0:
            print(f"[{idx}/{len(files)}] saved {output_dir / f.name}")


def process_image_dir(input_dir, depth_suffix, output_dir, lidar_h, lidar_w,
                      stride, knn_k, rgb_sigma, jitter, seed=42):
    """Walk a directory tree, find depth maps matching *<suffix>.png,
    and corresponding RGB images (same stem without the suffix)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    suffix = depth_suffix if depth_suffix.startswith("_") else f"_{depth_suffix}"
    if not suffix.lower().endswith(".png"):
        suffix = f"{suffix}.png"

    depth_files = sorted(input_dir.rglob(f"*{suffix}"))
    if not depth_files:
        raise RuntimeError(
            f"No depth files matching *{suffix} found under {input_dir}")
    print(f"Found {len(depth_files)} depth maps matching *{suffix}")

    for idx, f in enumerate(depth_files):
        depth_img = Image.open(f)
        depth = np.array(depth_img, dtype=np.float32)

        # Try to find corresponding RGB image
        stem = f.stem.replace(suffix.replace(".png", ""), "")
        rgb = None
        for ext in [".jpg", ".png", ".jpeg"]:
            rgb_path = f.parent / f"{stem}{ext}"
            if rgb_path.exists() and rgb_path != f:
                rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
                break

        sim = simulate_lidar(depth, rgb, lidar_h, lidar_w, stride, knn_k,
                             rgb_sigma, jitter, rng)

        rel = f.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, sim)

        if idx % 200 == 0:
            print(f"[{idx}/{len(depth_files)}] saved {out_path}")

    print(f"Done. Saved {len(depth_files)} simulated LiDAR maps to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PromptDA-style simulated LiDAR via sparse anchor interpolation.")
    parser.add_argument("--input-mat", type=str, default=None,
                        help="Input .mat file with 'depths' (e.g. NYUv2)")
    parser.add_argument("--input-npy-dir", type=str, default=None,
                        help="Input directory of .npy depth maps")
    parser.add_argument("--input-image-dir", type=str, default=None,
                        help="Input directory tree with depth PNG files")
    parser.add_argument("--depth-suffix", type=str, default="_depth_anything",
                        help="Suffix for depth PNG files (for --input-image-dir)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--lidar-h", type=int, default=192,
                        help="Simulated LiDAR height (iPhone ARKit = 192)")
    parser.add_argument("--lidar-w", type=int, default=256,
                        help="Simulated LiDAR width (iPhone ARKit = 256)")
    parser.add_argument("--stride", type=int, default=7,
                        help="Grid stride for anchor sampling (paper uses 7)")
    parser.add_argument("--knn-k", type=int, default=4,
                        help="K nearest anchors for interpolation")
    parser.add_argument("--rgb-sigma", type=float, default=20.0,
                        help="RGB similarity bandwidth for KNN weighting")
    parser.add_argument("--jitter", type=float, default=0.5,
                        help="Max grid jitter as fraction of stride")
    parser.add_argument("--split", type=str, default=None,
                        help="train|test|all (for .mat input)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input_mat:
        process_mat(args.input_mat, args.output_dir, args.lidar_h, args.lidar_w,
                    args.stride, args.knn_k, args.rgb_sigma, args.jitter,
                    args.split, args.seed)
    elif args.input_npy_dir:
        process_npy_dir(args.input_npy_dir, args.output_dir, args.lidar_h,
                        args.lidar_w, args.stride, args.knn_k, args.rgb_sigma,
                        args.jitter, args.seed)
    elif args.input_image_dir:
        process_image_dir(args.input_image_dir, args.depth_suffix, args.output_dir,
                          args.lidar_h, args.lidar_w, args.stride, args.knn_k,
                          args.rgb_sigma, args.jitter, args.seed)
    else:
        raise ValueError(
            "Must provide --input-mat, --input-npy-dir, or --input-image-dir")


if __name__ == "__main__":
    main()
