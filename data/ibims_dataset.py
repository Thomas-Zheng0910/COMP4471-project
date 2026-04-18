"""
iBims-1 evaluation dataset loader (from UniDepth Google Drive HDF5).

iBims-1 is a small (100 images) indoor benchmark for depth estimation with
high-quality ground truth.  This loader reads the UniDepth-provided HDF5.

Expected layout after download:
    datasets/unidepth_data/
        ibims.h5  (or similar name — auto-discovered)

Usage:
    dataset = IBims1Dataset(root="datasets/unidepth_data")
"""

import io
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Depth range (indoor)
MIN_DEPTH: float = 0.005
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


def _find_h5_file(root: Path, keywords: Tuple[str, ...] = ("ibims",)) -> Path:
    for h5 in sorted(root.glob("*.h5")) + sorted(root.glob("*.hdf5")):
        name_lower = h5.name.lower()
        if any(kw in name_lower for kw in keywords):
            return h5
    all_h5 = sorted(root.glob("*.h5")) + sorted(root.glob("*.hdf5"))
    if all_h5:
        return all_h5[0]
    raise FileNotFoundError(f"No HDF5 files found in {root}")


def _discover_h5_entries(h5path: Path) -> List[Dict]:
    """Discover image/depth entries in a UniDepth-style HDF5."""
    entries = []
    with h5py.File(h5path, "r") as f:
        if "images" in f and "depths" in f:
            n = len(f["images"])
            for i in range(n):
                entry = {"image_key": f"images/{i}", "depth_key": f"depths/{i}"}
                entries.append(entry)
            return entries

        image_paths = []
        depth_paths = []

        def _visitor(name, obj):
            name_lower = name.lower()
            if isinstance(obj, h5py.Dataset):
                if "image" in name_lower or "rgb" in name_lower:
                    image_paths.append(name)
                elif "depth" in name_lower:
                    depth_paths.append(name)

        f.visititems(_visitor)
        if image_paths and depth_paths:
            image_paths.sort()
            depth_paths.sort()
            n = min(len(image_paths), len(depth_paths))
            for i in range(n):
                entries.append({"image_key": image_paths[i], "depth_key": depth_paths[i]})

    return entries


class IBims1Dataset(Dataset):
    """PyTorch Dataset for iBims-1 evaluation (UniDepth HDF5 format)."""

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = "datasets/unidepth_data",
        split: str = "test",
        image_shape: Tuple[int, int] = (480, 640),
        depth_scale: float = 1.0,
        return_intrinsics: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split  # iBims-1 is eval-only; split is ignored
        self.depth_scale = depth_scale
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.return_intrinsics = return_intrinsics

        self.image_transform = _default_image_transform(self.image_shape)
        self.depth_transform = _default_depth_transform(self.image_shape)

        self.h5_path = _find_h5_file(self.root, keywords=("ibims",))
        self.entries = _discover_h5_entries(self.h5_path)
        if not self.entries:
            raise RuntimeError(f"No entries found in {self.h5_path}")

        print(f"IBims1Dataset: {len(self.entries)} samples from {self.h5_path.name}")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image_from_h5(self, entry: Dict) -> Image.Image:
        with h5py.File(self.h5_path, "r") as f:
            data = f[entry["image_key"]][()]
        if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
            return Image.open(io.BytesIO(data.tobytes())).convert("RGB")
        if data.ndim == 3:
            if data.shape[0] == 3:
                data = np.transpose(data, (1, 2, 0))
            return Image.fromarray(data.astype(np.uint8), mode="RGB")
        return Image.fromarray(data.astype(np.uint8)).convert("RGB")

    def _load_depth_from_h5(self, entry: Dict) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            data = f[entry["depth_key"]][()]
        if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
            depth_img = Image.open(io.BytesIO(data.tobytes()))
            data = np.array(depth_img, dtype=np.float32)
            # uint16 PNG (mode I;16): values are in millimetres → convert to metres
            if depth_img.mode == "I;16":
                data = data / 1000.0
        if data.ndim == 3:
            if data.shape[0] in (1, 3, 4):
                data = np.transpose(data, (1, 2, 0))
            if data.shape[-1] >= 3:
                # UniDepth 24-bit little-endian: R=LSB, G=middle, B=MSB
                R = data[..., 0].astype(np.float32)
                G = data[..., 1].astype(np.float32)
                B = data[..., 2].astype(np.float32)
                data = (R + G * 256.0 + B * 65536.0) / 1000.0
            else:
                data = data[..., 0].astype(np.float32)
        return data.astype(np.float32)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        image_pil = self._load_image_from_h5(entry)
        image_tensor = self.image_transform(image_pil)

        depth_np = self._load_depth_from_h5(entry) * self.depth_scale
        depth_tensor = self.depth_transform(depth_np)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)
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
