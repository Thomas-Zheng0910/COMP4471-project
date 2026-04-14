"""
DIODE Indoor evaluation dataset loader.

DIODE (Dense Indoor/Outdoor Depth) provides high-precision LiDAR ground truth.
This loader handles the indoor validation subset.

Expected layout after download & extraction:
    datasets/diode_indoor/
        val/
            indoors/
                scene_00000/
                    scan_00000/
                        00000.png          — RGB image (1024×768)
                        00000_depth.npy    — depth (float64, metres)
                        00000_depth_mask.npy — validity mask (bool)

Usage:
    dataset = DIODEIndoorDataset(root="datasets/diode_indoor", split="val")
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Depth range (LiDAR)
MIN_DEPTH: float = 0.01
MAX_DEPTH: float = 50.0

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


def _collect_diode_indoor_samples(root: Path) -> List[Dict]:
    """Find all (rgb, depth, mask) triplets under the DIODE indoor tree."""
    samples = []

    # Search recursively for PNG images paired with _depth.npy
    for dirpath, _, filenames in os.walk(root):
        dirpath = Path(dirpath)
        for fname in filenames:
            if fname.endswith(".png") and not fname.endswith("_depth.png"):
                stem = Path(fname).stem
                depth_file = dirpath / f"{stem}_depth.npy"
                mask_file = dirpath / f"{stem}_depth_mask.npy"

                if depth_file.exists():
                    samples.append({
                        "rgb": dirpath / fname,
                        "depth": depth_file,
                        "mask": mask_file if mask_file.exists() else None,
                    })

    samples.sort(key=lambda s: str(s["rgb"]))
    return samples


class DIODEIndoorDataset(Dataset):
    """PyTorch Dataset for DIODE Indoor evaluation."""

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = "datasets/diode_indoor",
        split: str = "val",
        image_shape: Tuple[int, int] = (480, 640),
        depth_scale: float = 1.0,
        return_intrinsics: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.depth_scale = depth_scale
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.return_intrinsics = return_intrinsics

        self.image_transform = _default_image_transform(self.image_shape)
        self.depth_transform = _default_depth_transform(self.image_shape)

        # DIODE indoor val typically lives under val/indoors/ or just indoors/
        samples = _collect_diode_indoor_samples(self.root)

        # Filter to indoor only if both indoor/outdoor are present
        indoor_samples = [s for s in samples if "indoor" in str(s["rgb"]).lower()]
        if indoor_samples:
            samples = indoor_samples

        if not samples:
            raise RuntimeError(f"No DIODE samples found under {self.root}")

        self.samples = samples
        print(f"DIODEIndoorDataset: {len(self.samples)} samples ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image_pil = Image.open(s["rgb"]).convert("RGB")
        image_tensor = self.image_transform(image_pil)

        # DIODE depth: float64 .npy, in metres
        depth_np = np.load(s["depth"]).astype(np.float32).squeeze()
        depth_np = depth_np * self.depth_scale

        depth_tensor = self.depth_transform(depth_np)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)

        # Use provided validity mask if available
        if s["mask"] is not None:
            mask_np = np.load(s["mask"]).astype(bool).squeeze()
            if self.image_shape is not None:
                mask_pil = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")
                mask_pil = mask_pil.resize((self.image_shape[1], self.image_shape[0]),
                                           resample=RESAMPLE_NEAREST)
                mask_np = np.array(mask_pil) > 127
            depth_mask = torch.from_numpy(mask_np).unsqueeze(0) & (depth_tensor > MIN_DEPTH)
        else:
            depth_mask = (depth_tensor > MIN_DEPTH) & torch.isfinite(depth_tensor)

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
