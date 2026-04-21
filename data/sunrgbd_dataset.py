"""
SUN-RGBD dataset loader for depth estimation training.

Expected directory layout (after unzipping SUNRGBD.zip):
    datasets/SUNRGBD/
        kv1/          — Kinect v1 captures
        kv2/          — Kinect v2 captures
        realsense/    — RealSense captures
        xtion/        — Asus Xtion captures

Each scene folder contains:
    image/  *.jpg      — RGB image
    depth_bfx/  *.png  — Hole-filled depth (uint16, mm)
    intrinsics.txt     — Camera intrinsics (3x3 matrix, row-major)

Usage:
    dataset = SUNRGBDDataset(root="datasets/SUNRGBD", split="train")
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Depth range (indoor, metres)
MIN_DEPTH: float = 0.005
MAX_DEPTH: float = 10.0

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def _default_image_transform(shape: Optional[Tuple[int, int]]) -> Callable:
    ops = []
    if shape is not None:
        ops.append(transforms.Resize(shape))
    ops += [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(ops)


def _default_depth_transform(shape: Optional[Tuple[int, int]]) -> Callable:
    def transform(depth_np: np.ndarray) -> torch.Tensor:
        if shape is not None:
            pil = Image.fromarray(depth_np.astype(np.float32), mode="F")
            pil = pil.resize((shape[1], shape[0]), resample=RESAMPLE_NEAREST)
            depth_np = np.array(pil, dtype=np.float32)
        return torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
    return transform


def _collect_sunrgbd_samples(root: Path) -> List[Dict]:
    """Recursively find all (image, depth, intrinsics) tuples."""
    samples = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        image_dir = dirpath / "image"
        depth_dir = dirpath / "depth_bfx"
        intrinsics_file = dirpath / "intrinsics.txt"

        if not image_dir.is_dir() or not depth_dir.is_dir():
            continue

        rgb_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        depth_files = sorted([f for f in depth_dir.iterdir() if f.suffix.lower() == ".png"])

        if not rgb_files or not depth_files:
            continue

        # Use the first matched pair (SUN-RGBD has one image per scene folder)
        K = None
        if intrinsics_file.exists():
            try:
                K = np.loadtxt(intrinsics_file, dtype=np.float32).reshape(3, 3)
            except Exception:
                K = None

        samples.append({
            "rgb": rgb_files[0],
            "depth": depth_files[0],
            "K": K,
        })

    samples.sort(key=lambda s: str(s["rgb"]))
    return samples


def _build_default_intrinsics(width: int, height: int) -> torch.Tensor:
    fx = fy = float(max(width, height))
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


class SUNRGBDDataset(Dataset):
    """PyTorch Dataset for SUN-RGBD indoor depth data."""

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = "datasets/SUNRGBD",
        split: str = "train",
        image_shape: Tuple[int, int] = (480, 640),
        depth_scale: float = 1.0,
        flip_aug: bool = False,
        return_intrinsics: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.flip_aug = flip_aug
        self.depth_scale = depth_scale
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.return_intrinsics = return_intrinsics

        self.image_transform = _default_image_transform(self.image_shape)
        self.depth_transform = _default_depth_transform(self.image_shape)

        all_samples = _collect_sunrgbd_samples(self.root)
        if not all_samples:
            raise RuntimeError(f"No SUN-RGBD samples found under {self.root}")

        # 90/10 train/test split (deterministic)
        n = len(all_samples)
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        split_idx = int(0.9 * n)
        if split == "train":
            self.samples = [all_samples[i] for i in indices[:split_idx]]
        elif split in ("test", "val"):
            self.samples = [all_samples[i] for i in indices[split_idx:]]
        else:
            self.samples = all_samples

        print(f"SUNRGBDDataset: {len(self.samples)} samples ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def _make_sample(self, image_t: torch.Tensor, depth_t: torch.Tensor,
                     mask_t: torch.Tensor, K: torch.Tensor, flip: bool) -> Dict:
        if flip:
            image_t = torch.flip(image_t, dims=[-1])
            depth_t = torch.flip(depth_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])

        sample = {"image": image_t, "depth": depth_t, "depth_mask": mask_t,
                  "flip": flip, "si": False}
        if self.return_intrinsics:
            K_out = K.clone()
            if flip:
                W = depth_t.shape[-1]
                K_out[0, 2] = W - K_out[0, 2]
            sample["K"] = K_out
        return sample

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        image_pil = Image.open(s["rgb"]).convert("RGB")
        image_tensor = self.image_transform(image_pil)

        # SUN-RGBD depth_bfx is uint16 with 3-bit left-shift encoding.
        # Decode: bitor(raw >> 3, raw << 13) then divide by 1000 → metres.
        depth_uint16 = np.asarray(Image.open(s["depth"]), dtype=np.uint16)
        depth_decoded = np.bitwise_or(
            depth_uint16 >> 3,
            (depth_uint16 << 13).astype(np.uint16),
        ).astype(np.float32)
        depth_m = depth_decoded / 1000.0 * self.depth_scale
        depth_tensor = self.depth_transform(depth_m)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)
        depth_mask = (depth_tensor > MIN_DEPTH) & torch.isfinite(depth_tensor)

        if s["K"] is not None:
            K = torch.from_numpy(s["K"]).float()
            # Scale intrinsics to match resized image
            if self.image_shape is not None:
                orig_h, orig_w = depth_uint16.shape[:2]
                sh = self.image_shape[0] / orig_h
                sw = self.image_shape[1] / orig_w
                K[0, :] *= sw
                K[1, :] *= sh
        else:
            H, W = depth_tensor.shape[-2:]
            K = _build_default_intrinsics(W, H)

        if self.flip_aug:
            return (self._make_sample(image_tensor, depth_tensor, depth_mask, K, False),
                    self._make_sample(image_tensor, depth_tensor, depth_mask, K, True))
        return self._make_sample(image_tensor, depth_tensor, depth_mask, K, False)

    @classmethod
    def collate_fn(cls, batch):
        if batch and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            flat = []
            for orig, flipped in batch:
                flat.append(orig)
                flat.append(flipped)
            batch = flat
        img_metas = [{k: item[k] for k in cls.META_KEYS if k in item} for item in batch]
        data_keys = [k for k in batch[0].keys() if k not in cls.META_KEYS]
        collated = {}
        for key in data_keys:
            vals = [item[key] for item in batch]
            collated[key] = torch.stack(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
        return {"data": collated, "img_metas": img_metas}
