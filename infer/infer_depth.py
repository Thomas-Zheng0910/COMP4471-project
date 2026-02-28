"""
infer_depth.py – Inference (and optional evaluation) script for UniDepthV1.

Usage:
    python -m infer.infer_depth <args>
    (see run_script/run_infer_demo.sh for full configuration)

Features:
    - Single-image or dataset-folder inference
    - Loads a trained checkpoint and reconstructs the model
    - Computes evaluation metrics (d1, d2, d3, arel, rmse, silog, …) when GT is available
    - Saves colorized depth predictions and error visualisations
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.unidepthv1 import UniDepthV1
from utils.evaluation_depth import eval_depth
from utils.visualization import colorize, image_grid


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniDepthV1 inference (and evaluation) on images.",
    )

    # --- Input / Output ---
    parser.add_argument(
        "--data_root", type=str, default="data/demo",
        help="Root folder with images/ (and optionally depths/, intrinsics.json).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/infer",
        help="Directory where predictions and visualisations are saved.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a training checkpoint (.pth) that contains model weights.",
    )

    # --- Device ---
    parser.add_argument("--cuda", type=int, default=0)

    # --- Model architecture (must match the checkpoint) ---
    parser.add_argument("--encoder_name", type=str, default="convnextv2_large")
    parser.add_argument("--output_idx", type=int, nargs="+", default=[3, 6, 33, 36])
    parser.add_argument(
        "--use_checkpoint",
        type=lambda x: x.lower() == "true",
        default=False,
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--expansion", type=int, default=4)

    # --- Data ---
    parser.add_argument(
        "--image_shape", type=int, nargs=2, default=[384, 384],
        help="Network input resolution (H W).",
    )
    parser.add_argument(
        "--depth_scale", type=float, default=0.001,
        help="Scale factor applied to raw 16-bit depth PNGs to get metres.",
    )
    parser.add_argument(
        "--max_depth", type=float, default=None,
        help="Optional cap on GT depth used during metric evaluation.",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Config builder (matches training script)
# ──────────────────────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace) -> dict:
    """Build the nested config dict expected by UniDepthV1 from flat args."""
    return {
        "model": {
            "name": "UniDepthV1",
            "pixel_encoder": {
                "name": args.encoder_name,
                "output_idx": args.output_idx,
                "use_checkpoint": args.use_checkpoint,
            },
            "pixel_decoder": {
                "name": "Decoder",
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "depths": args.depths,
            },
            "num_heads": args.num_heads,
            "expansion": args.expansion,
        },
        "training": {
            "lr": 1e-4,
            "wd": 0.01,
            "losses": {},
        },
        "data": {
            "image_shape": args.image_shape,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_intrinsics(intrinsics_path: str):
    """Load optional intrinsics JSON → dict  {stem: [fx, fy, cx, cy]}."""
    if intrinsics_path and os.path.isfile(intrinsics_path):
        with open(intrinsics_path) as f:
            return json.load(f)
    return None


def build_K(fx, fy, cx, cy) -> torch.Tensor:
    """Return a 3×3 intrinsics matrix."""
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def default_intrinsics(orig_h: int, orig_w: int) -> torch.Tensor:
    """Simple pinhole with ~60° horizontal FoV."""
    fx = orig_w / (2.0 * np.tan(np.radians(30)))
    fy = fx
    cx = orig_w / 2.0
    cy = orig_h / 2.0
    return build_K(fx, fy, cx, cy)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    config = build_config(args)

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ── Build model & load checkpoint ────────────────────────────────────────
    print("Building UniDepthV1 …")
    model = UniDepthV1(config)
    model.load_pretrained(args.checkpoint)
    model.to(device).eval()
    print("Model loaded and set to eval mode.")

    # ── Discover images ──────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    depths_dir = data_root / "depths"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No *.png images found in {images_dir}")

    has_gt = depths_dir.exists()
    intrinsics_map = load_intrinsics(str(data_root / "intrinsics.json"))

    print(f"Found {len(image_paths)} image(s) in {images_dir}")
    if has_gt:
        print(f"Ground-truth depth available in {depths_dir}")

    # ── Output dirs ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "pred_depth"
    vis_dir = out_dir / "visualisations"
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────────────
    all_pred_depths = []
    all_gt_depths = []
    all_masks = []

    for img_path in image_paths:
        stem = img_path.stem
        print(f"  Processing: {stem}")

        # Load RGB (H, W, 3) uint8 → (3, H, W) torch
        rgb_np = np.array(Image.open(img_path).convert("RGB"))
        rgb_torch = torch.from_numpy(rgb_np).permute(2, 0, 1)  # C, H, W
        orig_h, orig_w = rgb_np.shape[:2]

        # Optional intrinsics
        intrinsics = None
        if intrinsics_map is not None and stem in intrinsics_map:
            fx, fy, cx, cy = intrinsics_map[stem]
            intrinsics = build_K(fx, fy, cx, cy)

        # Run inference
        with torch.no_grad():
            predictions = model.infer(rgb_torch, intrinsics=intrinsics)

        depth_pred = predictions["depth"].squeeze().cpu()  # (H, W)

        # Save raw predicted depth as 16-bit PNG.
        # depth_pred is in metres; dividing by depth_scale (e.g. 0.001) converts
        # back to the raw unit used in the dataset (e.g. millimetres).
        depth_mm = (depth_pred.numpy() / args.depth_scale).clip(0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(str(pred_dir / f"{stem}.png"))

        # --- Visualisation --------------------------------------------------
        depth_pred_np = depth_pred.numpy()
        rgb_vis = rgb_np  # (H, W, 3) uint8

        # Colour-map range follows the reference UniDepth demo (0.01–10 m)
        depth_pred_col = colorize(depth_pred_np, vmin=0.01, vmax=10.0, cmap="magma_r")

        vis_panels = [rgb_vis, depth_pred_col]

        # If GT depth exists, add GT and error panels
        gt_depth_path = depths_dir / f"{stem}.png"
        if has_gt and gt_depth_path.exists():
            gt_pil = Image.open(gt_depth_path)
            # GT depth stored as raw int (e.g. mm); multiply by depth_scale to metres
            gt_np = np.array(gt_pil, dtype=np.float32) * args.depth_scale

            # Resize prediction to GT size for metric computation
            gt_h, gt_w = gt_np.shape[:2]
            depth_pred_resized = (
                F.interpolate(
                    depth_pred.unsqueeze(0).unsqueeze(0),
                    size=(gt_h, gt_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )

            mask = gt_np > 0
            if args.max_depth is not None:
                mask = mask & (gt_np <= args.max_depth)

            # ARel error map for visualisation
            arel_map = np.zeros_like(gt_np)
            arel_map[mask] = np.abs(gt_np[mask] - depth_pred_resized[mask]) / gt_np[mask]

            gt_col = colorize(gt_np, vmin=0.01, vmax=10.0, cmap="magma_r")
            err_col = colorize(arel_map, vmin=0.0, vmax=0.2, cmap="coolwarm")

            vis_panels.extend([gt_col, err_col])

            # Accumulate for batch metric computation
            all_pred_depths.append(
                torch.from_numpy(depth_pred_resized).unsqueeze(0)  # (1, H, W)
            )
            all_gt_depths.append(
                torch.from_numpy(gt_np).unsqueeze(0)  # (1, H, W)
            )
            all_masks.append(torch.from_numpy(mask).unsqueeze(0))  # (1, H, W)

        # Save grid: 1×2 (no GT) or 2×2 (with GT)
        n_panels = len(vis_panels)
        if n_panels == 4:
            grid = image_grid(vis_panels, rows=2, cols=2)
        else:
            grid = image_grid(vis_panels, rows=1, cols=2)
        if grid is not None:
            Image.fromarray(grid).save(str(vis_dir / f"{stem}.png"))

    # ── Evaluation metrics ───────────────────────────────────────────────────
    if all_gt_depths:
        print("\n── Evaluation Metrics ──")

        # Pad / stack to common size for eval_depth (samples may differ in size)
        # Process per-sample to handle different resolutions
        agg_metrics = defaultdict(list)
        for pred_d, gt_d, mask_d in zip(all_pred_depths, all_gt_depths, all_masks):
            sample_metrics = eval_depth(
                gts=gt_d.unsqueeze(0),
                preds=pred_d.unsqueeze(0),
                masks=mask_d.unsqueeze(0),
                max_depth=args.max_depth,
            )
            for name, vals in sample_metrics.items():
                agg_metrics[name].append(vals.mean())

        # Print summary
        results = {}
        for name in agg_metrics:
            mean_val = torch.stack(agg_metrics[name]).mean().item()
            results[name] = mean_val

        # Organise nicely: accuracy metrics first, then errors
        accuracy_keys = [k for k in results if k.startswith("d")]
        error_keys = [k for k in results if k not in accuracy_keys]
        for key in sorted(accuracy_keys) + sorted(error_keys):
            print(f"  {key:20s}: {results[key]:.4f}")

        # Save metrics to JSON
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
    else:
        print("\nNo ground-truth depth found — skipping evaluation metrics.")

    print(f"Predictions saved to {pred_dir}")
    print(f"Visualisations saved to {vis_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
