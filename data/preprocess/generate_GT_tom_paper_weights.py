"""
Generate pseudo-GT depth maps for ToM using the paper's fine-tuned
Depth Anything v1 (ViT-B) weights.

Usage:
    python -m data.preprocess.generate_GT_tom_paper_weights [--depth_scale 256] [--overwrite]

Requires: torch, torchvision, numpy, Pillow, tqdm, cv2 (opencv-python)
The Depth Anything v1 code is loaded from the torch.hub cache
(downloaded once via torch.hub.download_url_to_file if not present).
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

WEIGHTS_PATH = "datasets/Diffusion4RobustDepth/weights_Table/weights/Table1/depth_anything/weights.pth"
INPUT_DIR = "datasets/Diffusion4RobustDepth/ToM"
DEPTH_SUFFIX = "_depth_anything"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
UINT16_MAX = 65534

# Path where torch.hub cached the Depth-Anything repo
DA_REPO_DIR = Path(torch.hub.get_dir()) / "LiheYoung_Depth-Anything_main"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    p.add_argument("--input_dir", type=str, default=INPUT_DIR)
    p.add_argument("--depth_scale", type=float, default=256.0,
                   help="Multiply depth by this before saving as uint16")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--max_images", type=int, default=None)
    return p.parse_args()


def build_model(encoder="vitb"):
    """Build Depth Anything v1 model by importing directly from the cached repo."""
    # Add repo root to sys.path so `depth_anything` and `torchhub/` are importable
    repo = str(DA_REPO_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from depth_anything.dpt import DPT_DINOv2
    model = DPT_DINOv2(
        encoder=encoder,
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        use_clstoken=False,
        localhub=True,
    )
    return model


def build_transform(input_size=518):
    """Build the standard Depth Anything v1 preprocessing transform."""
    repo = str(DA_REPO_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    return Compose([
        Resize(
            width=input_size, height=input_size,
            resize_target=False, keep_aspect_ratio=True,
            ensure_multiple_of=14, resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def collect_images(root: Path) -> list:
    skip_suffixes = (DEPTH_SUFFIX + ".png", DEPTH_SUFFIX + "_vis.png")
    images = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() not in IMAGE_EXTS:
                continue
            if any(fname.lower().endswith(s) for s in skip_suffixes):
                continue
            images.append(Path(dirpath) / fname)
    images.sort()
    return images


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not DA_REPO_DIR.exists():
        print(f"Depth Anything repo not found at {DA_REPO_DIR}")
        print("Downloading via torch.hub...")
        torch.hub.load("LiheYoung/Depth-Anything", "__invalid__",
                        trust_repo=True, skip_validation=True)

    # Build model and load paper's weights
    model = build_model(encoder="vitb")
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded weights from {args.weights}")

    transform = build_transform()

    # Collect images
    input_dir = Path(args.input_dir)
    images = collect_images(input_dir)
    if args.max_images:
        images = images[: args.max_images]
    print(f"Found {len(images)} images")

    # Process
    for img_path in tqdm(images, desc="Generating depth (paper weights)"):
        out_path = img_path.with_name(img_path.stem + DEPTH_SUFFIX + ".png")
        if out_path.exists() and not args.overwrite:
            continue

        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"  SKIP (cannot read): {img_path}")
            continue
        h, w = raw.shape[:2]

        # Preprocess: BGR→RGB, uint8→float [0,1], apply transform
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        with torch.no_grad():
            depth = model(image)  # [1, H', W']

        # Resize back to original resolution
        depth = F.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True
        ).squeeze().cpu().numpy()

        # Save as uint16
        depth_uint16 = (depth * args.depth_scale).clip(0, UINT16_MAX).astype(np.uint16)
        Image.fromarray(depth_uint16).save(out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
