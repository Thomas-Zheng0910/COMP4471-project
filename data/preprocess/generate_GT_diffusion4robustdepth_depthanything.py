"""
Generate pseudo-GT depth maps for the Diffusion4RobustDepth collection using
Depth-Anything. The script walks the extracted dataset tree (see the layout in
get_diffusion4robustdepth.sh comments), runs Depth-Anything on every RGB image it
finds, and saves 16-bit depth PNGs next to the source images.

Example:
    python -m data.preprocess.generate_GT_diffusion4robustdepth_depthanything \
        --dataset_root datasets/Diffusion4RobustDepth \
        --model_id depth-anything/Depth-Anything-V2-Large \
        --save_vis

Notes:
- Existing *_depth.png files are kept unless --overwrite is passed.
- You can point --model_id to a local directory containing the model weights.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.visualization import colorize

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Macros
IMAGE_EXTS: Sequence[str] = (".jpg", ".jpeg", ".png")
DEPTH_COLORMAP_RANGE = (0.01, 10.0)  # metres
UINT16_MAX = 65534  # avoid potential wrap-around at 65535

# Allow loading of truncated images (some D4RD images are slightly corrupted)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Arg-parser
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Generate Depth-Anything depth maps for Diffusion4RobustDepth."
    )
    parser.add_argument(
        "--dataset_root",
        type = str,
        default = "datasets/Diffusion4RobustDepth",
        help = "Path to the extracted Diffusion4RobustDepth directory. (you can also specify a sub-dir inside)",
    )
    parser.add_argument(
        "--model_id",
        type = str,
        default = "depth-anything/Depth-Anything-V2-Large-hf",
        help = "HF model id or local directory for Depth-Anything.",
    )
    parser.add_argument(
        "--output_suffix",
        type = str,
        default = "_depth",
        help = "Suffix for saved depth maps placed next to RGB images.",
    )
    parser.add_argument(
        "--depth_scale",
        type = float,
        default = 1.0,
        help = "Scale to convert model output to stored units (1.0 keeps metres).",
    )
    parser.add_argument(
        "--save_vis",
        action = "store_true",
        help = "Also save colorized depth visualisations alongside predictions.",
    )
    parser.add_argument(
        "--overwrite",
        action = "store_true",
        help = "Regenerate depth maps even if the output file already exists.",
    )
    parser.add_argument(
        "--max_images",
        type = int,
        default = None,
        help = "Optional cap on number of images to process (for quick testing)",
    )
    parser.add_argument(
        "--cuda",
        type = int,
        default = 0,
        help = "CUDA device index to use (if available).",
    )
    parser.add_argument(
        "--amp",
        action = "store_true",
        help = "Use torch.autocast for faster inference on GPU.",
    )
    return parser.parse_args()

# Helper: collect images under dataset root
def collect_images(dataset_root: Path, output_suffix: str) -> List[Path]:
    
    """
    Collect all RGB images while skipping weight/cache folders and already-generated depths.
    """

    # We don't need to process some folders
    skip_dir_prefixes = tuple(pref.lower() for pref in ("weights", "hf_cache", ".cache"))
    
    output_suffix = output_suffix.lower()
    vis_suffix = output_suffix + "_vis"
    images: List[Path] = []

    # Walk the dataset tree
    for root, dirs, files in os.walk(dataset_root):
        
        # Skip the dirs we don't care
        # Note: modifying dirs in-place prevents os.walk from descending
        # into them
        dirs[:] = [
            d for d in dirs if not any(d.lower().startswith(pref) for pref in skip_dir_prefixes)
        ]

        # Convert to Path for easier handling
        root_path = Path(root)

        # Check files in the current dir
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            if fname.lower().endswith(output_suffix + ext) or \
                fname.lower().endswith(vis_suffix + ext):
                # Generated depth - we don't want to treat it as input image
                continue
            images.append(root_path / fname)

    images.sort()
    return images

# Helper: make output path for depth map based on image path and suffix
def make_output_path(image_path: Path, suffix: str) -> Path:
    suffix = suffix if suffix.startswith(("_", ".")) else f"_{suffix}"
    # NOTE: we assum we don't have ext names in suffix
    if not any(suffix.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
        suffix = f"{suffix}.png"
    return image_path.with_name(image_path.stem + suffix)

# Helper: build model and processor
def build_depth_anything(model_id: str, device: torch.device)\
    -> Tuple[AutoImageProcessor, AutoModelForDepthEstimation]:
    processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model

# Helper: run inference and save depth map (and optionally visualization)
def run_inference_depth_anything(
    processor: AutoImageProcessor,
    model: AutoModelForDepthEstimation,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    depth_scale: float,
    save_vis: bool,
    amp: bool,
) -> None:
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images = image, return_tensors = "pt").to(device)
    # Inference with optional autocast for speedup on GPU
    with torch.no_grad():
        with torch.autocast(device_type = device.type, enabled = amp and device.type == "cuda"):
            outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    # Resize to original resolution
    predicted_depth = F.interpolate(
        predicted_depth.unsqueeze(1),
        size = image.size[::-1],
        mode = "bicubic",
        align_corners = False,
    ).squeeze(1)

    # Convert to uint16 and save
    depth_np = predicted_depth.squeeze().cpu().numpy()
    depth_uint16 = (depth_np * depth_scale).clip(0, UINT16_MAX).astype(np.uint16)
    Image.fromarray(depth_uint16).save(output_path)

    # Optionally save a colorized visualization for quick sanity check
    if save_vis:
        vis_path = output_path.with_name(output_path.stem + "_vis.png")
        vmin, vmax = DEPTH_COLORMAP_RANGE
        # Use matplotlib to save the visualization
        # Previously BAD BAD
        fig, axes = plt.subplots(1, 2, figsize = (12, 6))
        axes[0].imshow(image)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")
        im = axes[1].imshow(depth_np, cmap = "magma", vmin = vmin, vmax = vmax)
        axes[1].set_title("Depth Map (colorized)")
        axes[1].axis("off")
        fig.colorbar(im, ax = axes[1], fraction = 0.046, pad = 0.04, label = "Depth (m)")
        plt.tight_layout()
        plt.savefig(vis_path)
        plt.close(fig)

# Main function
def main() -> None:

    # parse args
    args = parse_args()

    # fetch dataroot
    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # setup device and model
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor, model = build_depth_anything(args.model_id, device)

    # Collect images to processs
    image_paths = collect_images(dataset_root, args.output_suffix)

    # Optionally limit the number of images for quick testings
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    # Check if we found any imagess
    if not image_paths:
        raise RuntimeError(
            f"No RGB images found under {dataset_root}. Check that archives are extracted."
        )

    # verbose
    print(f"\033[92mFound {len(image_paths)} image(s); saving depths with suffix '{args.output_suffix}'.\n\033[0m")

    # Process each image
    for img_path in tqdm(image_paths, desc = "Generating depth maps"):
        # Determine output path for depth map
        out_path = make_output_path(img_path, args.output_suffix)
        # Skip if output already exists and we're not overwriting
        if out_path.exists() and not args.overwrite:
            continue
        # Ensure output directory exists
        out_path.parent.mkdir(parents = True, exist_ok = True)
        # inference and save
        run_inference_depth_anything(
            processor = processor,
            model = model,
            image_path = img_path,
            output_path = out_path,
            device = device,
            depth_scale = args.depth_scale,
            save_vis = args.save_vis,
            amp = args.amp,
        )

    print("\nDone. Depth maps saved next to the corresponding RGB images.")


if __name__ == "__main__":
    main()