"""
Visualize UniDepth V2 data augmentations on NYUv2 images.

This script loads NYUv2 RGB images and applies the augmentation pipeline
(as configured in the training config) to generate side-by-side comparisons.

Usage:
    python visualize_augmentations.py --num_samples 5 --output_dir ./augmentation_vis
"""

import argparse
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from data.augmentations import build_train_augmentation, DEFAULT_AUG_CONFIG


def load_nyu_image(mat_path: str, idx: int) -> Image.Image:
    """Load a single RGB image from NYUv2 .mat file as PIL Image."""
    with h5py.File(mat_path, "r") as h5:
        # HDF5 layout: (N, 3, W, H) uint8 -> transpose to (H, W, 3)
        image_raw = h5["images"][idx]  # (3, W, H)
        image_np = np.transpose(image_raw, (2, 1, 0))  # (H, W, 3)
        return Image.fromarray(image_np.astype(np.uint8), mode="RGB")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denormed = tensor * std + mean
    return denormed.permute(1, 2, 0).clamp(0, 1).numpy()


def apply_augmentation_multiple_times(
    image_pil: Image.Image,
    aug_transform,
    to_tensor_norm,
    num_augmented: int = 4,
    seed_base: int = 42,
) -> list:
    """Apply augmentation multiple times with different random seeds."""
    augmented = []
    for i in range(num_augmented):
        random.seed(seed_base + i)
        np.random.seed(seed_base + i)
        torch.manual_seed(seed_base + i)
        
        aug_img = aug_transform(image_pil)
        tensor = to_tensor_norm(aug_img)
        augmented.append(denormalize(tensor))
    return augmented


def visualize_augmentations(
    original_pil: Image.Image,
    augmented_tensors: list,
    idx: int,
    aug_config: dict,
    output_path: Path,
):
    """Create a visualization grid: original + augmented variants."""
    num_aug = len(augmented_tensors)
    fig, axes = plt.subplots(1, num_aug + 1, figsize=(4 * (num_aug + 1), 4))
    
    if num_aug == 0:
        axes = [axes]
    
    # Original image
    axes[0].imshow(original_pil)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # Augmented images
    for i, aug_img in enumerate(augmented_tensors):
        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(f"Augmented #{i+1}", fontsize=12)
        axes[i + 1].axis("off")
    
    # Config text as suptitle
    config_str = (
        f"jitter={aug_config['jitter']}, hue={aug_config['jitter_hue']}, p={aug_config['jitter_p']} | "
        f"blur_sigma={aug_config['blur_sigma']}, p={aug_config['blur_p']} | "
        f"gamma={aug_config['gamma']}, p={aug_config['gamma_p']} | "
        f"grayscale_p={aug_config['grayscale_p']}"
    )
    fig.suptitle(f"NYUv2 Sample {idx} - Augmentation Config: {config_str}", fontsize=10, y=1.02)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_individual_effects(
    original_pil: Image.Image,
    idx: int,
    output_dir: Path,
):
    """Show each augmentation type applied individually for clarity."""
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Build augmentation for each effect
    effects = [
        ("ColorJitter", build_train_augmentation({
            "jitter": 0.4, "jitter_hue": 0.1, "jitter_p": 1.0,
            "blur_p": 0, "gamma": 0, "grayscale_p": 0,
        })),
        ("GaussianBlur", build_train_augmentation({
            "jitter": 0, "blur_sigma": 2.0, "blur_p": 1.0,
            "gamma": 0, "grayscale_p": 0,
        })),
        ("RandomGamma", build_train_augmentation({
            "jitter": 0, "blur_p": 0,
            "gamma": 0.2, "gamma_p": 1.0,
            "grayscale_p": 0,
        })),
        ("RandomGrayscale", build_train_augmentation({
            "jitter": 0, "blur_p": 0, "gamma": 0,
            "grayscale_p": 1.0,
        })),
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Original
    axes[0].imshow(original_pil)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis("off")
    
    # Each effect
    for i, (name, aug) in enumerate(effects):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        if aug is not None:
            aug_img = aug(original_pil)
        else:
            aug_img = original_pil
        tensor = to_tensor_norm(aug_img)
        axes[i + 1].imshow(denormalize(tensor))
        axes[i + 1].set_title(name, fontsize=12)
        axes[i + 1].axis("off")
    
    fig.suptitle(f"NYUv2 Sample {idx} - Individual Augmentation Effects", fontsize=12, y=1.02)
    fig.tight_layout()
    
    output_path = output_dir / f"sample_{idx:04d}_individual_effects.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize NYUv2 augmentations")
    parser.add_argument("--mat_path", type=str, default="datasets/nyu_depth_v2_labeled.mat",
                        help="Path to NYUv2 .mat file")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="./augmentation_vis",
                        help="Output directory for visualizations")
    parser.add_argument("--num_augmented", type=int, default=4,
                        help="Number of augmented variants per sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--skip_individual", action="store_true",
                        help="Skip individual effect visualization")
    
    # Allow overriding augmentation params
    parser.add_argument("--jitter", type=float, default=DEFAULT_AUG_CONFIG["jitter"])
    parser.add_argument("--jitter_hue", type=float, default=DEFAULT_AUG_CONFIG["jitter_hue"])
    parser.add_argument("--jitter_p", type=float, default=DEFAULT_AUG_CONFIG["jitter_p"])
    parser.add_argument("--blur_sigma", type=float, default=DEFAULT_AUG_CONFIG["blur_sigma"])
    parser.add_argument("--blur_p", type=float, default=DEFAULT_AUG_CONFIG["blur_p"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_AUG_CONFIG["gamma"])
    parser.add_argument("--gamma_p", type=float, default=DEFAULT_AUG_CONFIG["gamma_p"])
    parser.add_argument("--grayscale_p", type=float, default=DEFAULT_AUG_CONFIG["grayscale_p"])
    
    args = parser.parse_args()
    
    # Build augmentation config
    aug_config = {
        "jitter": args.jitter,
        "jitter_hue": args.jitter_hue,
        "jitter_p": args.jitter_p,
        "blur_sigma": args.blur_sigma,
        "blur_p": args.blur_p,
        "gamma": args.gamma,
        "gamma_p": args.gamma_p,
        "grayscale_p": args.grayscale_p,
    }
    
    print("=" * 60)
    print("NYUv2 Data Augmentation Visualizer")
    print("=" * 60)
    print(f"Augmentation Config:")
    for k, v in aug_config.items():
        print(f"  {k}: {v}")
    print(f"\nDataset: {args.mat_path}")
    print(f"Samples: {args.num_samples}, Augmented variants: {args.num_augmented}")
    print("=" * 60)
    
    # Check file exists
    if not os.path.exists(args.mat_path):
        print(f"\nError: NYUv2 .mat file not found at '{args.mat_path}'")
        print("Download from: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get total number of images
    with h5py.File(args.mat_path, "r") as h5:
        total_images = h5["images"].shape[0]
    print(f"Total images in dataset: {total_images}")
    
    # Select random sample indices
    random.seed(args.seed)
    sample_indices = random.sample(range(total_images), min(args.num_samples, total_images))
    print(f"Selected indices: {sample_indices}")
    
    # Build augmentation pipeline
    aug_transform = build_train_augmentation(aug_config)
    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"\nGenerating visualizations in: {output_dir.absolute()}")
    
    for idx in sample_indices:
        print(f"\nProcessing sample {idx}...")
        
        # Load original image
        original_pil = load_nyu_image(args.mat_path, idx)
        
        # Generate augmented variants
        augmented_tensors = apply_augmentation_multiple_times(
            original_pil, aug_transform, to_tensor_norm,
            num_augmented=args.num_augmented,
            seed_base=idx,  # Use different seed per sample for variety
        )
        
        # Save combined visualization
        output_path = output_dir / f"sample_{idx:04d}_augmentations.png"
        visualize_augmentations(
            original_pil, augmented_tensors, idx, aug_config, output_path
        )
        
        # Save individual effects visualization
        if not args.skip_individual:
            visualize_individual_effects(original_pil, idx, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Done! Generated {args.num_samples} visualization(s) in:")
    print(f"  {output_dir.absolute()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
