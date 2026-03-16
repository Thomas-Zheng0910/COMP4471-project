"""
infer_depth.py - Inference (and optional evaluation) script for UniDepthV1.

Usage:
    python -m infer.infer_depth <args>
    (see run_script/run_infer.sh for full configuration)

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.nyuv2_dataset import NYUv2Dataset as ROOT_DATASET
from model.unidepthv1.unidepthv1 import UniDepthV1
from utils.evaluation_depth import eval_depth
from utils.visualization import colorize, image_grid

from typing import Optional, Dict, List

# Import IMAGENET_MEAN and IMAGENET_STD from the dataset module
from data.nyuv2_dataset import IMAGENET_MEAN, IMAGENET_STD

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniDepthV1 inference (and evaluation) on images.",
    )

    # --- Input / Output ---
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a training checkpoint (.pth) saved by train_depth.py.",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to data_root's dataset file",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "test"],
        help="Dataset split to evaluate: 'train' (795 samples) or 'test' (654 samples).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/infer",
        help="Directory where the metrics JSON is saved.",
    )

    # --- Device ---
    parser.add_argument("--cuda", type=int, default=0,
                        help="GPU index to use (ignored if CUDA is unavailable).")

    # --- Model architecture (must match the checkpoint) ---
    parser.add_argument("--encoder_name", type=str, default="convnextv2_large",
                        help="Pixel-encoder backbone name.")
    parser.add_argument("--output_idx", type=int, nargs="+", default=None,
                        help="Encoder feature-map indices. Leave unset to use encoder default.")
    parser.add_argument(
        "--use_checkpoint",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable gradient checkpointing in the encoder (true/false).",
    )
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Decoder hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Decoder dropout rate.")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3],
                        help="Number of decoder blocks per stage.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads.")
    parser.add_argument("--expansion", type=int, default=4,
                        help="MLP expansion factor.")

    # --- Data ---
    parser.add_argument(
        "--image_shape", type=int, nargs=2, default=[384, 384],
        help="Network input resolution (H W) — must match training.",
    )
    parser.add_argument(
        "--depth_scale", type=float, default=0.001,
        help="Scale factor applied to raw 16-bit depth PNGs to get metres.",
    )
    parser.add_argument(
        "--max_depth", type=float, default=10.0,
        help="Maximum depth (metres) used as an upper cap during evaluation.",
    )
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes.")
    
    # Additional images to evaluate (folder mode)
    parser.add_argument(
        "--image_folder", type=str, default=None,
        help="Path to a folder of RGB images for inference (PNG/JPEG). "
             "If set, runs in 'folder mode' and ignores the dataset split.",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Config builder (matches training script)
# ──────────────────────────────────────────────────────────────────────────────

def build_config(args: argparse.Namespace) -> dict:

    """
    Construct the nested config dict expected by UniDepthV1.
    """

    return {
        "model": {
            "name": "UniDepthV1",
            "pixel_encoder": {
                "name": args.encoder_name,
                **({"output_idx": args.output_idx} if args.output_idx is not None else {}),
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

def load_checkpoint(model: UniDepthV1, ckpt_path: str, device: torch.device) -> None:
    """
    Load model weights from a training checkpoint produced by train_depth.py.

    Handles three checkpoint layouts:
      1. {"model_state_dict": ...}  — produced by train_depth.py save_checkpoint
      2. {"model": ...}             — alternative / pre-trained format
      3. raw state-dict             — direct weight file
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format in {ckpt_path}")

    # Strip DDP "module." prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    info = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded from {ckpt_path}")
    print(f"  missing keys : {info.missing_keys}")
    print(f"  unexpected keys: {info.unexpected_keys}")

def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Compute RMSE, SILog, REL (abs_rel), and log10 for one sample.

    Args:
        pred:  Predicted depth  [H, W], float32, metres.
        gt:    Ground-truth depth [H, W], float32, metres.
        mask:  Boolean validity mask [H, W].

    Returns:
        Dict with keys "rmse", "silog", "rel", "log10".
    """
    p = pred[mask].float().clamp(min=1e-6)
    g = gt[mask].float().clamp(min=1e-6)

    rmse  = torch.sqrt(((g - p) ** 2).mean()).item()
    silog = (100.0 * torch.std(torch.log(p) - torch.log(g))).item()
    rel   = (torch.abs(g - p) / g).mean().item()
    log10 = torch.abs(torch.log10(p) - torch.log10(g)).mean().item()

    return {"rmse": rmse, "silog": silog, "rel": rel, "log10": log10}

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

    print("\nArguments:")
    for arg in vars(args):
        print(f"  \033[1m{arg}:\033[0m {getattr(args, arg)}")

    # Set Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model & load checkpoint
    print("\n>>> Building UniDepthV1 >>>")
    model = UniDepthV1(config)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device).eval()
    print("\033[92mModel loaded and set to eval mode.\033[0m")

    # Create output directory if it doesn't exist
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    # if data_root is not none
    if args.data_root is not None:
        print(f"\n>>> Loading \033[1;33m{ROOT_DATASET.__name__}\033[0m {args.split}-set from {args.data_root} >>>")
        # Dataset & DataLoader
        # TODO: extend to other datasets
        dataset = ROOT_DATASET(
            root = args.data_root,
            split = args.split,
            image_shape = args.image_shape,
            depth_scale = args.depth_scale,
            flip_aug = False,
            return_intrinsics = True,
        )
        loader = DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = (device.type == "cuda"),
            collate_fn = ROOT_DATASET.collate_fn,
        )
        print(f"\033[92mLoaded {len(dataset)} samples.\033[0m")

        # Inference
        agg: dict = defaultdict(list)

        # NOTE: This is not a good practice to hardcode them here
        # TODO: Use the constants from utils/constants.py instead
        # We now use the defined macro in the dataset class
        imagenet_mean = torch.tensor(
            IMAGENET_MEAN, device = device
        ).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(
            IMAGENET_STD, device = device
        ).view(1, 3, 1, 1)

        # Inference loop
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(loader), 
                                         total = len(loader),
                                         desc = f"Inferencing"):
                
                # Un-normalise images for model.infer() -> [0,255] uint8
                images: torch.Tensor = batch["data"]["image"]             # [B, 3, H, W] ImageNet-normed
                gts: torch.Tensor    = batch["data"]["depth"].to(device)  # [B, 1, H, W]
                Ks: torch.Tensor     = batch["data"].get("K")             # [B, 3, 3] or None
                # get batch size
                B = images.shape[0]
                for i in range(B):
                    # Undo ImageNet normalisation -> [0,255] uint8 for model.infer()
                    rgb_i = (images[i].to(device) * imagenet_std + imagenet_mean)
                    rgb_uint8 = (rgb_i * 255).clamp(0, 255).to(torch.uint8)
                    # Optional intrinsics per sample (if dataset provides them)
                    K_i = Ks[i].to(device) if Ks is not None else None
                    # Run inference for this sample
                    pred_i: torch.Tensor = model.infer(rgb_uint8, intrinsics = K_i)["depth"]
                    # Resize tensor
                    if pred_i.ndim == 4:
                        pred_i = pred_i.squeeze(0)  # (1, H, W)
                    gt_i   = gts[i]                 # (1, H, W)
                    mask_i = (gt_i > 0)             # (1, H, W)
                    # Apply max depth mask if specified
                    if args.max_depth is not None:
                        mask_i = mask_i & (gt_i <= args.max_depth)
                    # Compute metrics for this sample and accumulate
                    sample_m = eval_depth(
                        gts = gt_i.unsqueeze(0),
                        preds = pred_i.unsqueeze(0),
                        masks = mask_i.unsqueeze(0),
                        max_depth = args.max_depth,
                    )
                    for name, vals in sample_m.items():
                        agg[name].append(vals.mean().item())

        # Collect results and print metrics
        print(f"\n{ROOT_DATASET.__name__} {args.split}-Set Metrics")
        results = {name: float(np.mean(v)) for name, v in agg.items()}
        acc_keys = sorted(k for k in results if k.startswith("d"))
        err_keys = sorted(k for k in results if k not in acc_keys)
        for key in acc_keys + err_keys:
            print(f"  \033[1m{key:20s}:\033[0m {results[key]:.4f}")
        m_path = out_dir / f"metrics_{ROOT_DATASET.__name__.lower()}_{args.split}.json"
        with open(m_path, "w") as f:
            json.dump(results, f, indent = 2)
        print(f"\n\033[92mMetrics saved to {m_path}\033[0m")
    else:
        # No dataset evaluation — only folder inference if image_folder is set
        pass

    # -------------------------------------------------------------------------- #

    # Run inference on additional image folder
    if args.image_folder is not None:
        # verbose
        print(f"\n>>> Running inference on images in {args.image_folder} >>>")

        # Collect image paths
        data_root = Path(args.data_root)
        images_dir = data_root / "images"
        depths_dir = data_root / "depths"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Support PNG and JPEG
        # Collect and sort all image paths with supported extensions
        image_paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
            for p in images_dir.glob(ext)
        )
        if not image_paths:
            raise RuntimeError(f"No images (.png/.jpg/.jpeg) found in {images_dir}")

        # Checks if GTs are available
        has_gt = depths_dir.exists()

        # Load optional intrinsics JSON
        # NOTE: if not found, returns NONE and the following code will use a default intrinsics
        intrinsics_map: Optional[Dict[str, List[float]]] = load_intrinsics(str(data_root / "intrinsics.json"))

        print(f"\033[92mFound {len(image_paths)} image(s) in {images_dir}\033[0m")
        if has_gt:
            print(f"\033[92mGround-truth depth available in {depths_dir}\033[0m")

        # Set-up Output dirs
        pred_dir = out_dir / "additional_image" / "pred_depth"
        vis_dir = out_dir / "additional_image" / "visualisations"
        pred_dir.mkdir(parents = True, exist_ok = True)
        vis_dir.mkdir(parents = True, exist_ok = True)

        # Inference 
        all_pred_depths: List[torch.Tensor] = []
        all_gt_depths: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        for img_path in tqdm(image_paths, desc = "Inferencing"):

            # Get stem for matching GT and naming outputs
            stem = img_path.stem

            # Load RGB (H, W, 3) uint8 -> (3, H, W) torch
            rgb_np = np.array(Image.open(img_path).convert("RGB"))
            rgb_torch = torch.from_numpy(rgb_np).permute(2, 0, 1)  # C, H, W
            orig_h, orig_w = rgb_np.shape[:2]

            # Optional intrinsics: JSON map -> fallback to default pinhole
            if intrinsics_map is not None and stem in intrinsics_map:
                fx, fy, cx, cy = intrinsics_map[stem]
                intrinsics = build_K(fx, fy, cx, cy)
            else:
                intrinsics = default_intrinsics(orig_h, orig_w)

            # Run inference
            with torch.no_grad():
                depth_pred: torch.Tensor = \
                    model.infer(rgb_torch, intrinsics = intrinsics)["depth"]

            depth_pred = depth_pred.squeeze().cpu()  # (H, W)

            # Save raw predicted depth as 16-bit PNG.
            # depth_pred is in metres; dividing by depth_scale (e.g. 0.001) converts
            # back to the raw unit used in the dataset (e.g. millimetres).
            depth_mm = (depth_pred.numpy() / args.depth_scale).clip(0, 65535).astype(np.uint16)
            Image.fromarray(depth_mm).save(str(pred_dir / f"{stem}.png"))

            # Visualisation
            depth_pred_np = depth_pred.numpy()
            rgb_vis = rgb_np  # (H, W, 3) uint8

            # Colour-map range follows the reference UniDepth demo (0.01–10 m)
            depth_pred_col = colorize(depth_pred_np, 
                                      vmin = 0.01, vmax = 10.0, cmap = "magma_r")

            vis_panels: List[np.ndarray] = [rgb_vis, depth_pred_col]

            # If GT depth exists, add GT and error panels
            gt_depth_path = depths_dir / f"{stem}.png"

            if has_gt and gt_depth_path.exists():

                # Load GT depth (H, W) float32 in metres
                gt_pil = Image.open(gt_depth_path)

                # GT depth stored as raw int (e.g. mm); multiply by depth_scale to metres
                gt_np = np.array(gt_pil, dtype = np.float32) * args.depth_scale

                # Resize prediction to GT size for metric computation
                gt_h, gt_w = gt_np.shape[:2]
                depth_pred_resized = (
                    F.interpolate(
                        depth_pred.unsqueeze(0).unsqueeze(0),
                        size = (gt_h, gt_w),
                        mode = "bilinear",
                        align_corners = False,
                    )
                    .squeeze()
                    .numpy()
                )

                # Validity mask: GT > 0 and optionally GT <= max_depth
                mask = gt_np > 0
                if args.max_depth is not None:
                    mask = mask & (gt_np <= args.max_depth)

                # ARel error map for visualisation
                arel_map = np.zeros_like(gt_np)
                arel_map[mask] = np.abs(gt_np[mask] - depth_pred_resized[mask]) / gt_np[mask]

                # Colour-maps for GT and error
                gt_col = colorize(gt_np, vmin = 0.01, vmax = 10.0, cmap = "magma_r")
                err_col = colorize(arel_map, vmin = 0.0, vmax = 0.2, cmap = "coolwarm")

                # Add GT and error panels to visualisation
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
                grid = image_grid(vis_panels, rows = 2, cols = 2)
            else:
                grid = image_grid(vis_panels, rows = 1, cols = 2)
            if grid is not None:
                Image.fromarray(grid).save(str(vis_dir / f"{stem}.png"))

        # Evaluation metrics
        # NOTE: Only available if GT depths were found
        if all_gt_depths:
            print("\n>>> Evaluation Metrics >>>")

            # Pad / stack to common size for eval_depth (samples may differ in size)
            # Process per-sample to handle different resolutions
            agg_metrics = defaultdict(list)
            for pred_d, gt_d, mask_d in zip(all_pred_depths, all_gt_depths, all_masks):
                sample_metrics = eval_depth(
                    gts = gt_d.unsqueeze(0),
                    preds = pred_d.unsqueeze(0),
                    masks = mask_d.unsqueeze(0),
                    max_depth = args.max_depth,
                )
                for name, vals in sample_metrics.items():
                    agg_metrics[name].append(vals.mean())

            # Aggregate metrics across samples
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
            metrics_path = out_dir / "additional_image" / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(results, f, indent = 2)
            print(f"\n\033[92mMetrics saved to {metrics_path}\033[0m")
        else:
            # No GT depths found — skip metrics
            pass

        print(f"Predictions saved to {pred_dir}")
        print(f"Visualisations saved to {vis_dir}")
    else:
        # No additional image folder — only dataset evaluation was run
        pass

    print(f"\n\033[1;32mDone.\033[0m")


if __name__ == "__main__":
    main()
