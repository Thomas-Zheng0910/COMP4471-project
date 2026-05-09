#!/usr/bin/env python3
"""
Visualize depth predictions from multiple models on a specific NYUv2 image.

Usage:
    python visualize_nyuv2_comparison.py --image_idx 42 --output_dir runs/visualizations
    python visualize_nyuv2_comparison.py --image_idx 100 --split test --use_ssi

The script loads a specific NYUv2 image and runs inference with:
    - ViT Vanilla (30 epoch)
    - ConvNeXt Vanilla (50 epoch)
    - Teacher/LiDAR-guided (50 epoch)
    - Self-Distillation (60 epoch)
    - Depth Anything V2 (baseline)
    - Marigold (baseline)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data.nyuv2_dataset import NYUv2Dataset, IMAGENET_MEAN, IMAGENET_STD, MIN_DEPTH, MAX_DEPTH
from model.unidepthv1.unidepthv1 import UniDepthV1
from model.baselines import build_baseline
from utils.visualization import colorize
from utils.evaluation_depth import ssi


def load_checkpoint(model: UniDepthV1, ckpt_path: str, device: torch.device) -> None:
    """Load model weights from checkpoint."""
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
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    info = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded {ckpt_path}")
    if info.missing_keys:
        print(f"    Missing: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"    Unexpected: {info.unexpected_keys}")


def build_unidepth_config(encoder_name: str, use_lidar_fusion: bool = True) -> dict:
    """Build config dict for UniDepthV1."""
    return {
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
                "use_lidar_fusion": use_lidar_fusion,
                "lidar_fusion_type": "token",
            },
            "num_heads": 8,
            "expansion": 4,
        },
        "training": {"lr": 1e-4, "wd": 0.01, "losses": {}},
        "data": {"image_shape": [480, 640]},
    }


@torch.no_grad()
def run_unidepth_inference(
    model: UniDepthV1,
    rgb_uint8: torch.Tensor,  # [3, H, W] uint8 [0, 255]
    K: torch.Tensor,  # [3, 3]
    device: torch.device,
) -> torch.Tensor:
    """Run inference with UniDepthV1 model."""
    model.eval()
    pred = model.infer(rgb_uint8, intrinsics=K)["depth"]  # [1, H, W]
    return pred.squeeze(0).cpu()  # [H, W]


@torch.no_grad()
def run_baseline_inference(
    model,
    rgb_01: torch.Tensor,  # [3, H, W] float [0, 1]
    K: torch.Tensor,
    device: torch.device,
    model_name: str,
) -> torch.Tensor:
    """Run inference with baseline model."""
    model.eval()
    rgb_batched = rgb_01.unsqueeze(0).to(device)  # [1, 3, H, W]
    
    extra = {}
    if model_name == "unidepthv2":
        extra["intrinsics"] = K.unsqueeze(0).to(device)
    
    pred = model.predict_depth(rgb_batched, **extra)  # [1, 1, H, W] or [1, H, W]
    if pred.ndim == 4:
        pred = pred.squeeze(0).squeeze(0)  # [H, W]
    elif pred.ndim == 3:
        pred = pred.squeeze(0)  # [H, W]
    return pred.cpu()


def align_prediction_ssi(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply SSI alignment to prediction."""
    # Ensure all tensors are on the same device
    device = pred.device
    gt = gt.to(device)
    mask = mask.to(device)
    
    if mask.sum() < 10:
        return pred
    pred_aligned = ssi(gt[mask], pred[mask])
    result = pred.clone()
    result_flat = result.view(-1)
    mask_flat = mask.view(-1)
    result_flat[mask_flat] = pred_aligned
    return result.view(pred.shape)


def create_comparison_figure(
    rgb_np: np.ndarray,  # [H, W, 3] uint8
    gt_depth_np: np.ndarray,  # [H, W]
    predictions: Dict[str, np.ndarray],  # name -> [H, W]
    use_ssi: bool = False,
) -> plt.Figure:
    """Create a figure comparing all predictions."""
    n_models = len(predictions)
    n_cols = 3  # RGB, GT, Model
    n_rows = (n_models + 2) // 3  # +2 for RGB and GT rows
    
    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(18, 6 * (n_rows + 1)))
    if n_rows == 1:
        axes = axes.reshape(-1, n_cols)
    
    # Row 0: RGB and GT
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title("RGB Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(colorize(gt_depth_np, vmin=MIN_DEPTH, vmax=MAX_DEPTH, cmap="magma_r"))
    axes[0, 1].set_title("Ground Truth Depth", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Compute error metrics for each model
    mask = (gt_depth_np > MIN_DEPTH) & (gt_depth_np <= MAX_DEPTH)
    gt_masked = torch.from_numpy(gt_depth_np[mask])
    
    metrics_text = []
    for name, pred_np in predictions.items():
        pred_t = torch.from_numpy(pred_np)
        
        if use_ssi:
            pred_aligned = align_prediction_ssi(
                pred_t,
                torch.from_numpy(gt_depth_np),
                torch.from_numpy(mask)
            )
            pred_masked = pred_aligned.view(-1)[torch.from_numpy(mask).view(-1)]
        else:
            pred_masked = pred_t.view(-1)[torch.from_numpy(mask).view(-1)]
        
        absrel = (torch.abs(gt_masked - pred_masked) / gt_masked).mean().item()
        rmse = torch.sqrt(torch.pow(gt_masked - pred_masked, 2).mean()).item()
        metrics_text.append(f"{name}:\n  AbsRel: {absrel:.4f}, RMSE: {rmse:.4f}")
    
    axes[0, 2].text(0.1, 0.5, "\n\n".join(metrics_text), fontsize=11, family='monospace',
                   verticalalignment='center', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title("Metrics", fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Remaining rows: model predictions
    model_names = list(predictions.keys())
    for i, (name, pred_np) in enumerate(predictions.items()):
        row = (i // n_cols) + 1
        col = i % n_cols
        
        # Squeeze if needed and colorize depth
        if pred_np.ndim == 3 and pred_np.shape[0] == 1:
            pred_np = pred_np.squeeze(0)
        axes[row, col].imshow(colorize(pred_np, vmin=MIN_DEPTH, vmax=MAX_DEPTH, cmap="magma_r"))
        axes[row, col].set_title(name, fontsize=12)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(predictions), n_rows * n_cols):
        row = (i // n_cols) + 1
        col = i % n_cols
        if row < axes.shape[0]:
            axes[row, col].axis('off')
    
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize NYUv2 predictions")
    parser.add_argument("--image_idx", type=int, default=0, help="Index in dataset to visualize")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, default="runs/visualizations")
    parser.add_argument("--no_ssi", action="store_true", help="Disable SSI alignment (show raw predictions)")
    parser.add_argument("--cuda", type=int, default=4)
    parser.add_argument("--include_baselines", action="store_true", help="Also run DA v2 and Marigold")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Loading NYUv2 {args.split} dataset...")
    
    # Load dataset
    dataset = NYUv2Dataset(
        root="datasets/nyu_depth_v2_labeled.mat",
        split=args.split,
        image_shape=(480, 640),
        flip_aug=False,
        return_intrinsics=True,
    )
    
    if args.image_idx >= len(dataset):
        print(f"Error: index {args.image_idx} out of range (dataset has {len(dataset)} samples)")
        return
    
    # Get sample
    sample = dataset[args.image_idx]
    
    # Extract data
    image_tensor = sample["image"]  # [3, H, W] normalized
    depth_tensor = sample["depth"]  # [1, H, W]
    K_tensor = sample["K"]  # [3, 3]
    
    # Convert to numpy for visualization
    H, W = image_tensor.shape[1:]
    
    # Undo ImageNet normalization for RGB display
    image_np = image_tensor.permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    rgb_display = (image_np * std + mean)
    rgb_display = (np.clip(rgb_display, 0, 1) * 255).astype(np.uint8)
    
    # Get uint8 version for model inference
    rgb_uint8 = torch.from_numpy(rgb_display).permute(2, 0, 1)  # [3, H, W]
    
    gt_depth_np = depth_tensor.squeeze(0).numpy()  # [H, W]
    
    print(f"Image shape: {H}x{W}")
    print(f"Running inference with models...")
    
    predictions = {}
    
    # ImageNet stats for un-normalizing
    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 1. ViT Vanilla (DINOv3 ViT-L)
    # ═══════════════════════════════════════════════════════════════════════
    print("  [1/6] ViT Vanilla...")
    config_vit = build_unidepth_config("dinov3_vitl16", use_lidar_fusion=True)
    config_vit["model"]["pixel_encoder"]["output_idx"] = [5, 12, 18, 24]
    model_vit = UniDepthV1(config_vit).to(device)
    load_checkpoint(model_vit, 
        "runs/experiments/vit-without-guide-30-epoches/checkpoints/epoch_30.pth", device)
    pred_vit = run_unidepth_inference(model_vit, rgb_uint8.to(device), K_tensor.to(device), device)
    predictions["ViT Vanilla (30ep)"] = pred_vit
    del model_vit
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2. ConvNeXt Vanilla (50ep)
    # ═══════════════════════════════════════════════════════════════════════
    print("  [2/8] ConvNeXt Vanilla (50ep)...")
    config_convnext = build_unidepth_config("convnextv2_large", use_lidar_fusion=True)
    model_convnext = UniDepthV1(config_convnext).to(device)
    load_checkpoint(model_convnext,
        "runs/experiments/convnext-50-epoch/checkpoints/epoch_50.pth", device)
    pred_convnext = run_unidepth_inference(model_convnext, rgb_uint8.to(device), K_tensor.to(device), device)
    predictions["ConvNeXt Vanilla (50ep)"] = pred_convnext
    del model_convnext
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 3. ConvNeXt Vanilla (60ep)
    # ═══════════════════════════════════════════════════════════════════════
    print("  [3/8] ConvNeXt Vanilla (60ep)...")
    config_convnext60 = build_unidepth_config("convnextv2_large", use_lidar_fusion=True)
    model_convnext60 = UniDepthV1(config_convnext60).to(device)
    load_checkpoint(model_convnext60,
        "runs/experiments/convnext-60-epoch/checkpoints/epoch_60.pth", device)
    pred_convnext60 = run_unidepth_inference(model_convnext60, rgb_uint8.to(device), K_tensor.to(device), device)
    predictions["ConvNeXt Vanilla (60ep)"] = pred_convnext60
    del model_convnext60
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 4. Teacher / LiDAR-guided
    # ═══════════════════════════════════════════════════════════════════════
    print("  [4/8] Teacher/LiDAR...")
    config_teacher = build_unidepth_config("convnextv2_large", use_lidar_fusion=True)
    model_teacher = UniDepthV1(config_teacher).to(device)
    load_checkpoint(model_teacher,
        "datasets/teacher/train_depth_1776839443668_3116335/checkpoints/epoch_50.pth", device)
    pred_teacher = run_unidepth_inference(model_teacher, rgb_uint8.to(device), K_tensor.to(device), device)
    predictions["Teacher/LiDAR (50ep)"] = pred_teacher
    del model_teacher
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5. Teacher Fine-tuned NYU (90ep)
    # ═══════════════════════════════════════════════════════════════════════
    print("  [5/8] Teacher Fine-tuned NYU (90ep)...")
    config_teacher_nyu = build_unidepth_config("convnextv2_large", use_lidar_fusion=True)
    model_teacher_nyu = UniDepthV1(config_teacher_nyu).to(device)
    load_checkpoint(model_teacher_nyu,
        "runs/experiments/teacher-finetuned-nyu/checkpoints/epoch_90.pth", device)
    pred_teacher_nyu = run_unidepth_inference(model_teacher_nyu, rgb_uint8.to(device), K_tensor.to(device), device)
    predictions["Teacher Fine-tuned NYU (90ep)"] = pred_teacher_nyu
    del model_teacher_nyu
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 6. Self-Distillation (60ep)
    # ═══════════════════════════════════════════════════════════════════════
    print("  [6/8] Self-Distillation (60ep)...")
    config_distill = build_unidepth_config("convnextv2_large", use_lidar_fusion=True)
    model_distill = UniDepthV1(config_distill).to(device)
    load_checkpoint(model_distill,
        "runs/experiments/self-distillation-60-epoch-best/checkpoints/epoch_60.pth", device)
    pred_distill = run_unidepth_inference(model_distill, rgb_uint8.to(device), K_tensor.to(device), device)
    predictions["Self-Distill (60ep)"] = pred_distill
    del model_distill
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 7. Self-Distillation (100ep)
    # ═══════════════════════════════════════════════════════════════════════
    print("  [7/8] Self-Distillation (100ep)...")
    config_distill100 = build_unidepth_config("convnextv2_large", use_lidar_fusion=True)
    model_distill100 = UniDepthV1(config_distill100).to(device)
    load_checkpoint(model_distill100,
        "runs/experiments/self-distillation-100-epoch-better?/checkpoints/epoch_100.pth", device)
    pred_distill100 = run_unidepth_inference(model_distill100, rgb_uint8.to(device), K_tensor.to(device), device)
    predictions["Self-Distill (100ep)"] = pred_distill100
    del model_distill100
    torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 8. Baselines (optional) - these are relative depth models
    # ═══════════════════════════════════════════════════════════════════════
    use_ssi = not args.no_ssi  # Apply SSI to ALL models by default for fair comparison
    
    if args.include_baselines:
        print("  [8/8] Depth Anything V2...")
        model_da = build_baseline("depth_anything_v2", device=device)
        rgb_01 = (rgb_uint8.float() / 255.0).unsqueeze(0).to(device)  # [1, 3, H, W]
        # Resize to 518x518 for DA v2
        rgb_da = F.interpolate(rgb_01, size=(518, 518), mode='bilinear', align_corners=False)
        pred_da = run_baseline_inference(model_da, rgb_da.squeeze(0), K_tensor, device, "depth_anything_v2")
        # Resize back to original size
        pred_da = F.interpolate(pred_da.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
        predictions["DA v2 (baseline)"] = pred_da
        del model_da
        torch.cuda.empty_cache()
        
        print("  [6/6] Marigold...")
        model_marigold = build_baseline("marigold", device=device, num_inference_steps=4, ensemble_size=1)
        pred_marigold = run_baseline_inference(model_marigold, rgb_01.squeeze(0), K_tensor, device, "marigold")
        # Resize to original size
        pred_marigold = F.interpolate(pred_marigold.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
        predictions["Marigold (baseline)"] = pred_marigold
        del model_marigold
        torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════
    # Apply SSI alignment to ALL predictions (custom + baselines)
    # ═══════════════════════════════════════════════════════════════════════
    if use_ssi:
        print("Applying SSI alignment to all predictions...")
        gt_t = torch.from_numpy(gt_depth_np).to(device)
        mask_t = torch.from_numpy((gt_depth_np > MIN_DEPTH) & (gt_depth_np <= MAX_DEPTH)).to(device)
        for name in predictions:
            pred = predictions[name]
            # Squeeze if needed (some predictions may be [1, H, W])
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            predictions[name] = align_prediction_ssi(pred, gt_t, mask_t)
    
    # Convert all tensors to numpy for visualization
    for name in predictions:
        pred = predictions[name]
        if isinstance(pred, torch.Tensor):
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred.squeeze(0)
            predictions[name] = pred.cpu().numpy()
        elif pred.ndim == 3 and pred.shape[0] == 1:
            predictions[name] = pred.squeeze(0)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Create visualization
    # ═══════════════════════════════════════════════════════════════════════
    print("Creating visualization...")
    
    fig = create_comparison_figure(rgb_display, gt_depth_np, predictions, use_ssi=use_ssi)
    
    # Save
    ssi_tag = "_raw" if args.no_ssi else ""
    output_path = output_dir / f"nyuv2_{args.split}_idx{args.image_idx:04d}{ssi_tag}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved visualization to {output_path}")
    
    # Also save individual depth maps
    depth_dir = output_dir / f"depth_maps_{args.split}_idx{args.image_idx:04d}"
    depth_dir.mkdir(exist_ok=True)
    
    # Save GT
    gt_colored = colorize(gt_depth_np, vmin=MIN_DEPTH, vmax=MAX_DEPTH, cmap="magma_r")
    Image.fromarray(gt_colored).save(depth_dir / "ground_truth.png")
    
    # Save predictions
    for name, pred_np in predictions.items():
        if pred_np.ndim == 3 and pred_np.shape[0] == 1:
            pred_np = pred_np.squeeze(0)
        pred_colored = colorize(pred_np, vmin=MIN_DEPTH, vmax=MAX_DEPTH, cmap="magma_r")
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        Image.fromarray(pred_colored).save(depth_dir / f"{safe_name}.png")
    
    print(f"Saved individual depth maps to {depth_dir}")


if __name__ == "__main__":
    main()
