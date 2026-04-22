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
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Device: 'cuda:0' etc. (xformers attention requires CUDA)")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers for parallel image loading")
    return p.parse_args()


def _ensure_dinov2_local(da_repo: Path):
    """Ensure DINOv2 is available under <DA_repo>/torchhub/facebookresearch_dinov2_main.

    DPT_DINOv2 with localhub=True loads DINOv2 via a relative path
    'torchhub/facebookresearch_dinov2_main' from cwd. We resolve this by
    ensuring it exists inside the DA repo itself (symlink from torch.hub cache).
    """
    target = da_repo / "torchhub" / "facebookresearch_dinov2_main"
    if target.exists():
        return
    # Look in torch.hub cache (the standard location)
    hub_dir = Path(torch.hub.get_dir())
    dinov2_cached = hub_dir / "facebookresearch_dinov2_main"
    if not dinov2_cached.exists():
        # Try to download it
        print("DINOv2 not found in torch.hub cache, downloading...")
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                        trust_repo=True, pretrained=False)
    if dinov2_cached.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(dinov2_cached)
        print(f"Symlinked DINOv2: {target} -> {dinov2_cached}")
    else:
        raise FileNotFoundError(
            f"Cannot find DINOv2 repo at {dinov2_cached}. "
            "Please run: python -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')\""
        )


def build_model(encoder="vitb"):
    """Build Depth Anything v1 model by importing directly from the cached repo."""
    # Add repo root to sys.path so `depth_anything` package is importable
    repo = str(DA_REPO_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # Ensure DINOv2 is available for localhub loading
    _ensure_dinov2_local(DA_REPO_DIR)

    # DPT_DINOv2 uses torch.hub.load(..., source='local') with a relative
    # path 'torchhub/facebookresearch_dinov2_main', so we must chdir to the
    # DA repo directory during model construction.
    orig_cwd = os.getcwd()
    try:
        os.chdir(DA_REPO_DIR)
        from depth_anything.dpt import DPT_DINOv2
        model = DPT_DINOv2(
            encoder=encoder,
            features=128,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            use_clstoken=False,
            localhub=True,
        )
    finally:
        os.chdir(orig_cwd)
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


def collect_easy_images(root: Path, overwrite: bool = True) -> list:
    """Collect only images under */easy/ subdirectories (paper generates depth on easy only)."""
    skip_suffixes = (DEPTH_SUFFIX + ".png", DEPTH_SUFFIX + "_vis.png")
    images = []
    for dirpath, _, filenames in os.walk(root):
        if Path(dirpath).name != "easy":
            continue
        for fname in filenames:
            if Path(fname).suffix.lower() not in IMAGE_EXTS:
                continue
            if any(fname.lower().endswith(s) for s in skip_suffixes):
                continue
            fpath = Path(dirpath) / fname
            if not overwrite:
                out = fpath.with_name(fpath.stem + DEPTH_SUFFIX + ".png")
                if out.exists():
                    continue
            images.append(fpath)
    images.sort()
    return images


class ImagePathDataset(Dataset):
    """Lightweight dataset that loads + preprocesses images for batched inference."""
    def __init__(self, paths: list, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        raw = cv2.imread(str(img_path))
        if raw is None:
            # Return a dummy; we'll skip in post-processing
            return torch.zeros(3, 518, 518), str(img_path), 0, 0, False
        h, w = raw.shape[:2]
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({"image": image})["image"]
        return torch.from_numpy(image), str(img_path), h, w, True


def main():
    args = parse_args()
    device = torch.device(args.device)
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

    # Collect only easy/ images
    input_dir = Path(args.input_dir)
    easy_images = collect_easy_images(input_dir, overwrite=args.overwrite)
    if args.max_images:
        easy_images = easy_images[: args.max_images]
    print(f"Found {len(easy_images)} easy images to process")

    # Multi-worker DataLoader for parallel image loading/preprocessing on CPU
    dataset = ImagePathDataset(easy_images, transform)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    copied = 0
    saved = 0

    for image, (img_path_str,), (orig_h,), (orig_w,), (valid,) in tqdm(
        loader, desc="Generating depth (paper weights)"
    ):
        if not valid:
            continue
        img_path = Path(img_path_str)
        h, w = int(orig_h), int(orig_w)

        with torch.no_grad():
            disp = model(image.to(device))  # [1, H', W'] — inverse depth

        # Resize to original resolution
        disp = F.interpolate(
            disp.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True
        ).squeeze().cpu().numpy()

        # Invert: disparity → depth (larger = farther)
        depth = 1.0 / (disp + 1e-6)

        # Normalize per-image to [0, 1] then scale to uint16
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-8:
            depth_norm = np.zeros_like(depth)
        else:
            depth_norm = (depth - d_min) / (d_max - d_min)

        depth_uint16 = (depth_norm * UINT16_MAX).clip(0, UINT16_MAX).astype(np.uint16)
        out_path = img_path.with_name(img_path.stem + DEPTH_SUFFIX + ".png")
        Image.fromarray(depth_uint16).save(out_path)
        saved += 1

        # Copy to corresponding challenging/ directory
        challenging_dir = img_path.parent.parent / "challenging"
        if challenging_dir.is_dir():
            challenge_depth = challenging_dir / (img_path.stem + DEPTH_SUFFIX + ".png")
            shutil.copy2(out_path, challenge_depth)
            copied += 1

    print(f"\nDone. Saved {saved} depth maps, copied {copied} to challenging/.")


if __name__ == "__main__":
    main()
