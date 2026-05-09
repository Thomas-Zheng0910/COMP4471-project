"""
visualize_depth.py — Visualize depth predictions with automatic or manual scaling.

Usage:
    python visualize_depth.py --checkpoint <path> --image_dir <path> --dataset <nyu|todd|kitti>
    python visualize_depth.py --checkpoint <path> --image_dir <path> --max_depth 80.0

Examples:
    # KITTI (outdoor, up to 80m)
    python visualize_depth.py --checkpoint runs/.../epoch_60.pth \
        --image_dir runs/visualizations/kitti_samples \
        --dataset kitti --cuda 4

    # NYU/TODD (indoor, up to 10m)
    python visualize_depth.py --checkpoint runs/.../epoch_60.pth \
        --image_dir datasets/todd/test/0000-00-00-00-00-00/image.jpg \
        --dataset todd --cuda 4

    # Manual depth range
    python visualize_depth.py --checkpoint runs/.../epoch_60.pth \
        --image_dir <path> --max_depth 50.0 --cuda 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model.unidepthv1.unidepthv1 import UniDepthV1
from utils.visualization import colorize, image_grid

# Dataset depth ranges
DATASET_RANGES = {
    "nyu": {"min": 0.01, "max": 10.0, "name": "NYUv2 (indoor)"},
    "todd": {"min": 0.01, "max": 10.0, "name": "TODD (indoor)"},
    "kitti": {"min": 0.01, "max": 80.0, "name": "KITTI (outdoor)"},
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize depth predictions with proper scaling"
    )
    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--encoder_name", type=str, default="convnextv2_large",
                        help="Encoder architecture")
    
    # Input
    parser.add_argument("--image_path", type=str, default=None,
                        help="Single image file")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory containing images/ subfolder")
    
    # Depth range (auto or manual)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["nyu", "todd", "kitti"],
                        help="Dataset type for auto depth range")
    parser.add_argument("--min_depth", type=float, default=None,
                        help="Minimum depth for colormap (default: 0.01)")
    parser.add_argument("--max_depth", type=float, default=None,
                        help="Maximum depth for colormap (auto if --dataset set)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="runs/visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--cmap", type=str, default="magma_r",
                        help="Colormap for depth visualization")
    
    # Model config
    parser.add_argument("--cuda", type=int, default=0,
                        help="GPU device index")
    parser.add_argument("--image_size", type=int, nargs=2, default=[480, 640],
                        help="Input image size (H W)")
    
    return parser.parse_args()


def resolve_depth_range(args) -> tuple:
    """Determine min/max depth for visualization."""
    if args.dataset and args.dataset in DATASET_RANGES:
        info = DATASET_RANGES[args.dataset]
        dmin = args.min_depth if args.min_depth is not None else info["min"]
        dmax = args.max_depth if args.max_depth is not None else info["max"]
        print(f"Using {info['name']} depth range: [{dmin}, {dmax}]m")
        return dmin, dmax
    
    # Manual fallback
    dmin = args.min_depth if args.min_depth is not None else 0.01
    dmax = args.max_depth if args.max_depth is not None else 10.0
    print(f"Using manual depth range: [{dmin}, {dmax}]m")
    return dmin, dmax


def load_model(checkpoint_path: str, encoder_name: str, device: torch.device):
    """Load UniDepthV1 model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Build config
    config = {
        "model": {
            "name": "UniDepthV1",
            "pixel_encoder": {
                "name": encoder_name,
                "use_checkpoint": True,
            },
            "pixel_decoder": {
                "name": "Decoder",
                "hidden_dim": 512,
                "dropout": 0.0,
                "depths": [3, 2, 1],
                "use_lidar_fusion": True,
                "lidar_fusion_type": "token",
            },
            "num_heads": 8,
            "expansion": 4,
        },
        "training": {"lr": 1e-4, "wd": 0.01, "losses": {}},
        "data": {"image_shape": [480, 640]},
    }
    
    # Build model
    model = UniDepthV1(config)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device).eval()
    print("Model loaded and set to eval mode.")
    return model


def preprocess_image(image_path: str, image_size: list) -> tuple:
    """Load and preprocess image."""
    from torchvision import transforms
    
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    
    # Resize and normalize
    transform = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_pil).unsqueeze(0)  # (1, 3, H, W)
    
    # Build intrinsics
    H, W = image_size
    fx = fy = float(max(W, H))
    cx, cy = float(W) / 2.0, float(H) / 2.0
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    
    return image_tensor, K, np.array(image_pil), (orig_h, orig_w)


def run_inference(model, image_tensor: torch.Tensor, K: torch.Tensor, device: torch.device):
    """Run depth inference."""
    # Denormalize for model input
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    image_tensor = image_tensor.to(device)
    rgb_denorm = image_tensor * imagenet_std + imagenet_mean
    rgb_uint8 = (rgb_denorm * 255).clamp(0, 255).to(torch.uint8)
    
    K = K.to(device).unsqueeze(0) if K is not None else None
    
    with torch.no_grad():
        pred = model.infer(rgb_uint8[0], intrinsics=K[0] if K is not None else None)["depth"]
    
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    
    return pred.squeeze().cpu().numpy()


def visualize_single(image_path: str, model, device: torch.device, 
                     image_size: list, depth_range: tuple, 
                     cmap: str, output_dir: Path):
    """Visualize depth for a single image."""
    dmin, dmax = depth_range
    
    # Load and preprocess
    image_tensor, K, rgb_np, orig_size = preprocess_image(image_path, image_size)
    
    # Inference
    depth_pred = run_inference(model, image_tensor, K, device)
    
    # Create visualization
    depth_col = colorize(depth_pred, vmin=dmin, vmax=dmax, cmap=cmap)
    grid = image_grid([rgb_np, depth_col], rows=1, cols=2)
    
    # Save
    stem = Path(image_path).stem
    out_path = output_dir / f"{stem}_vis.png"
    Image.fromarray(grid).save(str(out_path))
    
    # Also save raw depth
    depth_out = output_dir / f"{stem}_depth.png"
    depth_mm = (depth_pred / 0.001).clip(0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(str(depth_out))
    
    print(f"Saved: {out_path}")
    return out_path


def visualize_directory(image_dir: str, model, device: torch.device,
                        image_size: list, depth_range: tuple,
                        cmap: str, output_dir: Path):
    """Visualize depth for all images in a directory."""
    dmin, dmax = depth_range
    
    # Find images
    image_root = Path(image_dir)
    images_subdir = image_root / "images"
    
    if images_subdir.exists():
        image_paths = sorted(images_subdir.glob("*.png")) + sorted(images_subdir.glob("*.jpg"))
    else:
        image_paths = sorted(image_root.glob("*.png")) + sorted(image_root.glob("*.jpg"))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} image(s)")
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Visualizing"):
        try:
            visualize_single(str(img_path), model, device, image_size,
                             depth_range, cmap, output_dir)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    args = get_args()
    
    # Resolve depth range
    depth_range = resolve_depth_range(args)
    
    # Setup device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.encoder_name, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Run visualization
    if args.image_path:
        visualize_single(args.image_path, model, device, args.image_size,
                        depth_range, args.cmap, output_dir)
    elif args.image_dir:
        visualize_directory(args.image_dir, model, device, args.image_size,
                           depth_range, args.cmap, output_dir)
    else:
        print("Error: Must specify --image_path or --image_dir")
        sys.exit(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
