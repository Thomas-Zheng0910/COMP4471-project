#!/usr/bin/env python3
"""
Phase 0-1 preparation utility for NYUv2 + LiDAR integration.

What this script does:
1) Phase 0 checks
   - Validates that RGB/depth frames exist in the NYUv2 .mat file.
   - Validates LiDAR source accessibility (HDF5 key or external folder).
2) Phase 1 outputs
   - Builds train/test split manifests (Eigen-style split aligned to NYUv2Dataset).
   - Computes LiDAR sparsity/range statistics.
   - Writes a machine-readable summary JSON for experiment tracking.

Usage examples:
    python -m data.get_datasets.phase01_prepare_nyuv2_lidar \
        --mat-path datasets/nyu_depth_v2_labeled.mat \
        --lidar-h5-key lidar_depths \
        --output-dir datasets/nyuv2_lidar_phase01

    python -m data.get_datasets.phase01_prepare_nyuv2_lidar \
        --mat-path datasets/nyu_depth_v2_labeled.mat \
        --lidar-root datasets/nyuv2_lidar_projected \
        --output-dir datasets/nyuv2_lidar_phase01
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

EIGEN_TEST_INDICES = [i - 1 for i in range(1, 655)]
MIN_DEPTH = 0.005
MAX_DEPTH = 10.0


def _load_external_lidar(lidar_root: Path, idx: int) -> np.ndarray:
    candidates = [
        lidar_root / f"{idx:05d}.npy",
        lidar_root / f"{idx:04d}.npy",
        lidar_root / f"{idx}.npy",
        lidar_root / f"{idx:05d}.npz",
        lidar_root / f"{idx:04d}.npz",
        lidar_root / f"{idx}.npz",
    ]
    src = next((path for path in candidates if path.is_file()), None)
    if src is None:
        raise FileNotFoundError(f"Missing LiDAR file for index {idx} under {lidar_root}")

    if src.suffix == ".npy":
        return np.load(src).astype(np.float32)

    obj = np.load(src)
    for key in ("lidar_depth", "depth", "sparse_depth"):
        if key in obj:
            return obj[key].astype(np.float32)
    raise KeyError(f"NPZ file {src} has no lidar depth key.")


def _get_lidar_frame(
    h5: h5py.File,
    idx: int,
    lidar_h5_key: Optional[str],
    lidar_root: Optional[Path],
) -> np.ndarray:
    if lidar_h5_key is not None:
        arr = h5[lidar_h5_key][idx]
        if arr.shape == (640, 480):
            return np.transpose(arr, (1, 0)).astype(np.float32)
        return arr.astype(np.float32)

    if lidar_root is None:
        raise RuntimeError("Neither lidar_h5_key nor lidar_root is provided.")

    return _load_external_lidar(lidar_root, idx)


def _compute_stats(
    h5: h5py.File,
    total: int,
    lidar_h5_key: Optional[str],
    lidar_root: Optional[Path],
    max_samples: int,
) -> Dict:
    sample_count = min(total, max_samples) if max_samples > 0 else total
    valid_ratios = []
    all_valid_depth_values = []

    missing_count = 0
    for idx in range(sample_count):
        try:
            lidar = _get_lidar_frame(h5, idx, lidar_h5_key, lidar_root)
        except (FileNotFoundError, KeyError):
            missing_count += 1
            continue

        mask = np.isfinite(lidar) & (lidar > MIN_DEPTH) & (lidar < MAX_DEPTH)
        valid_ratio = float(mask.mean())
        valid_ratios.append(valid_ratio)

        if mask.any():
            all_valid_depth_values.append(lidar[mask])

    if all_valid_depth_values:
        stacked = np.concatenate(all_valid_depth_values, axis=0)
        depth_stats = {
            "min": float(stacked.min()),
            "max": float(stacked.max()),
            "mean": float(stacked.mean()),
            "median": float(np.median(stacked)),
            "p95": float(np.percentile(stacked, 95)),
        }
    else:
        depth_stats = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }

    sparsity_stats = {
        "mean_valid_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "median_valid_ratio": float(np.median(valid_ratios)) if valid_ratios else 0.0,
        "p05_valid_ratio": float(np.percentile(valid_ratios, 5)) if valid_ratios else 0.0,
        "p95_valid_ratio": float(np.percentile(valid_ratios, 95)) if valid_ratios else 0.0,
    }

    return {
        "sampled_frames": sample_count,
        "missing_lidar_frames": missing_count,
        "sparsity": sparsity_stats,
        "depth_distribution": depth_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NYUv2+LiDAR phase 0-1 artifacts.")
    parser.add_argument("--mat-path", type=str, required=True)
    parser.add_argument("--lidar-h5-key", type=str, default=None)
    parser.add_argument("--lidar-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="datasets/nyuv2_lidar_phase01")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all frames")
    args = parser.parse_args()

    mat_path = Path(args.mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"NYUv2 mat file not found: {mat_path}")

    lidar_root = Path(args.lidar_root) if args.lidar_root is not None else None
    if lidar_root is not None and not lidar_root.exists():
        raise FileNotFoundError(f"LiDAR root does not exist: {lidar_root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(mat_path, "r") as h5:
        if "images" not in h5 or "depths" not in h5:
            raise KeyError("NYUv2 mat file must contain 'images' and 'depths' keys.")

        total = int(h5["images"].shape[0])

        if args.lidar_h5_key is not None and args.lidar_h5_key not in h5:
            raise KeyError(f"LiDAR HDF5 key not found: {args.lidar_h5_key}")

        if args.lidar_h5_key is None and lidar_root is None:
            raise ValueError("Please provide either --lidar-h5-key or --lidar-root.")

        test_set = set(EIGEN_TEST_INDICES)
        all_indices = list(range(total))
        train_indices = [i for i in all_indices if i not in test_set]
        test_indices = [i for i in all_indices if i in test_set]

        stats = _compute_stats(
            h5=h5,
            total=total,
            lidar_h5_key=args.lidar_h5_key,
            lidar_root=lidar_root,
            max_samples=args.max_samples,
        )

    (output_dir / "splits").mkdir(parents=True, exist_ok=True)
    (output_dir / "splits" / "train.txt").write_text("\n".join(map(str, train_indices)) + "\n")
    (output_dir / "splits" / "test.txt").write_text("\n".join(map(str, test_indices)) + "\n")

    summary = {
        "phase": "0-1",
        "dataset": "NYUv2",
        "mat_path": str(mat_path),
        "lidar_source": {
            "h5_key": args.lidar_h5_key,
            "lidar_root": str(lidar_root) if lidar_root is not None else None,
        },
        "num_total": total,
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "metrics": ["AbsRel", "RMSE", "delta1", "delta2", "delta3"],
        "quality_checks": {
            "expected_depth_range_m": [MIN_DEPTH, MAX_DEPTH],
            "missing_lidar_frames": stats["missing_lidar_frames"],
        },
        "lidar_statistics": stats,
    }

    with (output_dir / "phase01_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote summary to {output_dir / 'phase01_summary.json'}")
    print(f"[OK] Wrote splits to {output_dir / 'splits'}")


if __name__ == "__main__":
    main()
