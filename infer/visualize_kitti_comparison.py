"""
visualize_kitti_comparison.py — Run all models on KITTI samples and create comparison grid.

Usage:
    python visualize_kitti_comparison.py --cuda 4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model.unidepthv1.unidepthv1 import UniDepthV1
from utils.visualization import colorize, image_grid

# Model configurations
MODELS = {
    "ViT Vanilla": {
        "checkpoint": "runs/experiments/vit-without-guide-30-epoches/checkpoints/epoch_30.pth",
        "encoder": "dinov3_vitl16",
        "output_idx": [5, 12, 18, 24],
    },
    "ConvNeXt 50ep": {
        "checkpoint": "runs/experiments/convnext-50-epoch/checkpoints/epoch_50.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "ConvNeXt 60ep": {
        "checkpoint": "runs/experiments/convnext-60-epoch/checkpoints/epoch_60.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Teacher/LiDAR": {
        "checkpoint": "datasets/teacher/train_depth_1776839443668_3116335/checkpoints/epoch_50.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Teacher NYU": {
        "checkpoint": "runs/experiments/teacher-finetuned-nyu/checkpoints/epoch_90.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Self-Distill 60ep": {
        "checkpoint": "runs/experiments/self-distillation-60-epoch-best/checkpoints/epoch_60.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Self-Distill 100ep": {
        "checkpoint": "runs/experiments/self-distillation-100-epoch-better?/checkpoints/epoch_100.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--image_dir", type=str,
                        default="runs/visualizations/kitti_samples")
    parser.add_argument("--output_dir", type=str,
                        default="runs/visualizations/kitti_comparison")
    return parser.parse_args()


def load_custom_model(name, config, device):
    """Load a custom UniDepthV1 model."""
    ckpt_path = config["checkpoint"]
    encoder = config["encoder"]
    output_idx = config.get("output_idx")
    
    if not Path(ckpt_path).exists():
        print(f"  ! Checkpoint not found: {ckpt_path}")
        return None
    
    # Build config
    cfg = {
        "model": {
            "name": "UniDepthV1",
            "pixel_encoder": {
                "name": encoder,
                "use_checkpoint": True,
                **({"output_idx": output_idx} if output_idx else {}),
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
    
    model = UniDepthV1(cfg)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    
    return model.to(device).eval()


def run_custom_inference(model, image_path, device, image_size=(480, 640)):
    """Run inference with custom model."""
    from torchvision import transforms
    
    # Load and preprocess
    image_pil = Image.open(image_path).convert("RGB")
    rgb_np = np.array(image_pil)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # Build intrinsics
    H, W = image_size
    fx = fy = float(max(W, H))
    K = torch.tensor([[fx, 0.0, W/2], [0.0, fy, H/2], [0.0, 0.0, 1.0]], device=device)
    
    # Denormalize
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
    std = IMAGENET_STD.view(1, 3, 1, 1).to(device)
    rgb_uint8 = ((image_tensor * std + mean) * 255).clamp(0, 255).to(torch.uint8)
    
    # Inference
    with torch.no_grad():
        pred = model.infer(rgb_uint8[0], intrinsics=K)["depth"]
    
    # Properly squeeze to [H, W]
    if pred.ndim == 4:
        pred = pred.squeeze(0).squeeze(0)  # [1, 1, H, W] -> [H, W]
    elif pred.ndim == 3:
        pred = pred.squeeze(0)  # [1, H, W] -> [H, W]
    
    return rgb_np, pred.cpu().numpy()


def run_baseline(baseline, image_path, output_dir, device_id):
    """Run baseline inference using eval_baselines script."""
    from torchvision import transforms
    
    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    rgb_np = np.array(image_pil)
    
    # Use eval_baselines to get prediction
    cmd = [
        "conda", "run", "-n", "DepthSense", "python", "-m", "infer.eval_baselines",
        "--baseline", baseline,
        "--image_path", str(image_path),
        "--output_dir", str(output_dir),
        "--cuda", str(device_id),
    ]
    
    # For now, import and use baseline directly
    from model.baselines import build_baseline
    
    model = build_baseline(baseline, device=f"cuda:{device_id}")
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(f"cuda:{device_id}")
    
    with torch.no_grad():
        if baseline == "marigold":
            pred = model.predict_depth(image_tensor, num_inference_steps=4, ensemble_size=1)
        else:
            pred = model.predict_depth(image_tensor)
    
    pred_np = pred.squeeze().cpu().numpy()
    
    # Scale to metric depth (baselines output relative depth)
    # Use median scaling as a simple approximation
    pred_np = pred_np * 20.0  # Rough scale factor for KITTI
    
    return rgb_np, pred_np


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_root = Path(args.image_dir)
    images_subdir = image_root / "images"
    if images_subdir.exists():
        image_paths = sorted(images_subdir.glob("*.png")) + sorted(images_subdir.glob("*.jpg"))
    else:
        image_paths = sorted(image_root.glob("*.png")) + sorted(image_root.glob("*.jpg"))
    
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_paths)} image(s)")
    
    # Depth range for KITTI
    vmin, vmax = 0.01, 80.0
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        results = {"RGB": None}
        
        # Run custom models
        for name, config in MODELS.items():
            try:
                model = load_custom_model(name, config, device)
                if model is None:
                    continue
                rgb, depth = run_custom_inference(model, str(img_path), device)
                results[name] = depth
                if results["RGB"] is None:
                    results["RGB"] = rgb
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        # Create comparison figure - 1 column vertical layout like NYUv2
        import matplotlib.pyplot as plt
        
        model_names = [name for name in MODELS.keys() if name in results]
        n_models = len(model_names)
        n_cols = 2  # RGB | Depth
        n_rows = n_models + 1  # +1 for RGB row
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        
        # Row 0: RGB (span both columns)
        axes[0, 0].imshow(results["RGB"])
        axes[0, 0].set_title("RGB Image", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')  # Hide second column in RGB row
        
        # Remaining rows: each model
        for i, name in enumerate(model_names):
            row = i + 1
            pred_np = results[name]
            
            # Raw depth on left
            axes[row, 0].imshow(pred_np, vmin=vmin, vmax=vmax, cmap='magma_r')
            axes[row, 0].set_title(f"{name} (raw)", fontsize=10)
            axes[row, 0].axis('off')
            
            # Colorized on right
            depth_col = colorize(pred_np, vmin=vmin, vmax=vmax, cmap="magma_r")
            axes[row, 1].imshow(depth_col)
            axes[row, 1].set_title(f"{name} (colorized)", fontsize=10)
            axes[row, 1].axis('off')
        
        # Add overall title
        fig.suptitle(f'KITTI Depth Comparison - Range: 0.01-80m', fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save
        stem = img_path.stem
        out_path = output_dir / f"{stem}_comparison.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {out_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
