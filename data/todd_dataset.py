"""
TODD (Toronto Transparent Object Depth Dataset) loader.

Expected layout:
    datasets/todd/
        test/
            <timestamp>/
                image.jpg          — RGB image
                depth.exr          — Raw depth (may contain holes)
                detp_GroundTruth.exr — Cleaned depth ground truth
                instance_segment.png — Instance segmentation mask
                apriltag.pkl       — AprilTag metadata

Usage:
    dataset = TODDDataset(root="datasets/todd", split="test")
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import OpenEXR
import Imath
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Depth range
MIN_DEPTH: float = 0.01
MAX_DEPTH: float = 10.0

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


def _build_default_intrinsics(width: int, height: int) -> torch.Tensor:
    fx = fy = float(max(width, height))
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def read_exr_depth(path: str) -> np.ndarray:
    """Read depth from EXR file."""
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Read RGB channels (depth is stored in all three channels)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr_file.channel('R', FLOAT)
    
    # Convert to numpy array
    depth = np.frombuffer(r_str, dtype=np.float32)
    depth = depth.reshape((height, width))
    
    return depth


def _collect_todd_samples(root: Path, split: str) -> List[Dict]:
    """Find all TODD samples under the root directory."""
    samples = []
    split_dir = root / split
    
    if not split_dir.exists():
        raise RuntimeError(f"TODD {split} directory not found: {split_dir}")
    
    # Each subdirectory is a sample
    for sample_dir in sorted(split_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        # Look for image and depth files
        image_file = sample_dir / "image.jpg"
        
        # Try detp_GroundTruth.exr first (cleaned depth), fall back to depth.exr
        depth_file = sample_dir / "detp_GroundTruth.exr"
        if not depth_file.exists():
            depth_file = sample_dir / "depth.exr"
        
        if image_file.exists() and depth_file.exists():
            samples.append({
                "rgb": image_file,
                "depth": depth_file,
                "stem": sample_dir.name,
            })
    
    samples.sort(key=lambda s: s["stem"])
    return samples


class TODDDataset(Dataset):
    """PyTorch Dataset for TODD evaluation."""

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = "datasets/todd",
        split: str = "test",
        image_shape: Tuple[int, int] = (480, 640),
        depth_scale: float = 1.0,
        flip_aug: bool = False,
        return_intrinsics: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.depth_scale = depth_scale
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.flip_aug = flip_aug
        self.return_intrinsics = return_intrinsics

        self.image_transform = _default_image_transform(self.image_shape)
        self.depth_transform = _default_depth_transform(self.image_shape)

        self.samples = _collect_todd_samples(self.root, split)
        
        if not self.samples:
            raise RuntimeError(f"No TODD samples found under {self.root}/{split}")
        
        print(f"TODDDataset: {len(self.samples)} samples ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image_pil = Image.open(s["rgb"]).convert("RGB")
        orig_w, orig_h = image_pil.size
        image_tensor = self.image_transform(image_pil)

        # TODD depth: float32 EXR, in metres
        depth_np = read_exr_depth(str(s["depth"]))
        depth_np = depth_np * self.depth_scale

        # Resize depth to target shape if needed
        if self.image_shape is not None:
            target_h, target_w = self.image_shape
            if depth_np.shape != (target_h, target_w):
                pil = Image.fromarray(depth_np.astype(np.float32), mode="F")
                pil = pil.resize((target_w, target_h), resample=RESAMPLE_NEAREST)
                depth_np = np.array(pil, dtype=np.float32)

        depth_tensor = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)

        # Valid depth mask
        depth_mask = (depth_tensor > MIN_DEPTH) & torch.isfinite(depth_tensor)
        depth_mask = depth_mask & (depth_tensor <= MAX_DEPTH)

        H, W = depth_tensor.shape[-2:]
        K = _build_default_intrinsics(W, H)

        sample = {"image": image_tensor, "depth": depth_tensor, "depth_mask": depth_mask,
                  "flip": False, "si": False}
        if self.return_intrinsics:
            sample["K"] = K
        return sample

    @classmethod
    def collate_fn(cls, batch):
        img_metas = [{k: item[k] for k in cls.META_KEYS if k in item} for item in batch]
        data_keys = [k for k in batch[0].keys() if k not in cls.META_KEYS]
        collated = {}
        for key in data_keys:
            vals = [item[key] for item in batch]
            collated[key] = torch.stack(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
        return {"data": collated, "img_metas": img_metas}
