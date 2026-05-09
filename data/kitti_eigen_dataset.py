"""
KITTI Eigen Split dataset loader.

Expected layout:
    datasets/kitti_eigen/
        test/
            2011_09_26/
                2011_09_26_drive_XXXX_sync/
                    image_02/data/           - RGB images (0000000000.png, etc.)
                    proj_depth/groundtruth/image_02/ - Ground truth depth
        train/
            ...

Usage:
    dataset = KITTIEigenDataset(root="datasets/kitti_eigen", split="test")
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Depth range (metres) - KITTI outdoor
MIN_DEPTH: float = 0.01
MAX_DEPTH: float = 80.0

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


def _collect_kitti_eigen_samples(root: Path, split: str) -> List[Dict]:
    """Find all (rgb, depth) pairs under the KITTI Eigen tree."""
    samples = []
    split_dir = root / split
    
    if not split_dir.exists():
        raise RuntimeError(f"KITTI {split} directory not found: {split_dir}")
    
    # Walk through date folders
    for date_dir in sorted(split_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        
        # Walk through drive folders
        for drive_dir in sorted(date_dir.iterdir()):
            if not drive_dir.is_dir():
                continue
            
            # Image and depth paths
            image_dir = drive_dir / "image_02" / "data"
            depth_dir = drive_dir / "proj_depth" / "groundtruth" / "image_02"
            
            if not image_dir.exists() or not depth_dir.exists():
                continue
            
            # Find all images with corresponding depth
            for img_path in sorted(image_dir.glob("*.png")):
                depth_path = depth_dir / img_path.name
                if depth_path.exists():
                    samples.append({
                        "rgb": img_path,
                        "depth": depth_path,
                        "stem": f"{date_dir.name}/{drive_dir.name}/{img_path.stem}",
                    })
    
    samples.sort(key=lambda s: s["stem"])
    return samples


class KITTIEigenDataset(Dataset):
    """PyTorch Dataset for KITTI Eigen evaluation."""

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = "datasets/kitti_eigen",
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

        self.samples = _collect_kitti_eigen_samples(self.root, split)
        
        if not self.samples:
            raise RuntimeError(f"No KITTI Eigen samples found under {self.root}/{split}")
        
        print(f"KITTIEigenDataset: {len(self.samples)} samples ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image_pil = Image.open(s["rgb"]).convert("RGB")
        image_tensor = self.image_transform(image_pil)

        # KITTI depth: stored as uint16 PNG where value / 256.0 = depth in meters
        depth_pil = Image.open(s["depth"])
        depth_np = np.array(depth_pil, dtype=np.float32) / 256.0
        depth_np = depth_np * self.depth_scale

        depth_tensor = self.depth_transform(depth_np)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)

        # Valid depth mask
        depth_mask = (depth_tensor > MIN_DEPTH) & (depth_tensor <= MAX_DEPTH) & torch.isfinite(depth_tensor)

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
