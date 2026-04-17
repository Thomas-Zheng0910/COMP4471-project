"""
Sintel depth dataset loader (from UniDepth Google Drive HDF5).

The UniDepth-provided HDF5 stores images and depth maps as compressed byte
blobs inside a hierarchical structure.  This loader discovers the structure
at __init__ time and exposes a flat list of (image, depth) entries.

Expected layout after download:
    datasets/unidepth_data/
        sintel.h5   (or similar name — auto-discovered)

Usage:
    dataset = SintelDataset(root="datasets/unidepth_data", split="train")
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

# Depth range (synthetic, wide range)
MIN_DEPTH: float = 0.01
MAX_DEPTH: float = 100.0

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


def _find_h5_file(root: Path, keywords: Tuple[str, ...] = ("sintel",)) -> Path:
    """Find the first HDF5 file matching any keyword under root."""
    for h5 in sorted(root.glob("*.h5")) + sorted(root.glob("*.hdf5")):
        name_lower = h5.name.lower()
        if any(kw in name_lower for kw in keywords):
            return h5
    # Fallback: return any HDF5 file
    all_h5 = sorted(root.glob("*.h5")) + sorted(root.glob("*.hdf5"))
    if all_h5:
        return all_h5[0]
    raise FileNotFoundError(f"No HDF5 files found in {root}")


def _discover_h5_entries(h5path: Path) -> List[Dict]:
    """
    Discover image/depth entries in a UniDepth-style HDF5 file.
    
    Supports two common layouts:
      1. Flat: /images/000, /depths/000
      2. Hierarchical: /sequence_name/rgb/000, /sequence_name/depth/000
      3. UniDepth encoded: paths stored in metadata, images as byte blobs
    """
    entries = []
    with h5py.File(h5path, "r") as f:
        # Try flat layout first
        if "images" in f and "depths" in f:
            n = len(f["images"])
            for i in range(n):
                entry = {"h5path": h5path, "image_key": f"images/{i}", "depth_key": f"depths/{i}"}
                if "intrinsics" in f:
                    entry["K_key"] = f"intrinsics/{i}" if f"intrinsics/{i}" in f else "intrinsics"
                entries.append(entry)
            return entries

        # Try to find image/depth groups recursively
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
                entries.append({
                    "h5path": h5path,
                    "image_key": image_paths[i],
                    "depth_key": depth_paths[i],
                })
            return entries

        # If nothing found with clear names, list all leaf datasets
        all_datasets = []
        def _list_all(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_datasets.append(name)
        f.visititems(_list_all)
        print(f"[SintelDataset] Could not auto-discover structure. Found datasets: {all_datasets[:20]}")

    return entries


class SintelDataset(Dataset):
    """PyTorch Dataset for MPI Sintel depth (UniDepth HDF5 format)."""

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = "datasets/unidepth_data",
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

        h5_path = _find_h5_file(self.root, keywords=("sintel",))
        self.h5_path = h5_path
        all_entries = _discover_h5_entries(h5_path)
        if not all_entries:
            raise RuntimeError(f"No image/depth entries found in {h5_path}")

        # 90/10 split
        n = len(all_entries)
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        split_idx = int(0.9 * n)
        if split == "train":
            self.entries = [all_entries[i] for i in indices[:split_idx]]
        elif split in ("test", "val"):
            self.entries = [all_entries[i] for i in indices[split_idx:]]
        else:
            self.entries = all_entries

        # Lazy per-worker file handle (avoid reopening h5 every sample)
        self._h5_cache: Optional[h5py.File] = None
        self._h5_cache_path: Optional[Path] = None

        print(f"SintelDataset: {len(self.entries)} samples ({split}) from {h5_path.name}")

    def _get_h5(self, h5path: Path) -> h5py.File:
        """Return a cached per-worker h5py file handle."""
        if self._h5_cache is None or self._h5_cache_path != h5path:
            if self._h5_cache is not None:
                self._h5_cache.close()
            self._h5_cache = h5py.File(h5path, "r")
            self._h5_cache_path = h5path
        return self._h5_cache

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image_from_h5(self, entry: Dict) -> Image.Image:
        f = self._get_h5(entry["h5path"])
        data = f[entry["image_key"]][()]
        if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
            return Image.open(io.BytesIO(data.tobytes())).convert("RGB")
        if data.ndim == 3:
            if data.shape[0] == 3:  # C, H, W
                data = np.transpose(data, (1, 2, 0))
            return Image.fromarray(data.astype(np.uint8), mode="RGB")
        if data.ndim == 2:
            return Image.fromarray(data.astype(np.uint8), mode="L").convert("RGB")
        return Image.fromarray(data.astype(np.uint8)).convert("RGB")

    def _load_depth_from_h5(self, entry: Dict) -> np.ndarray:
        f = self._get_h5(entry["h5path"])
        data = f[entry["depth_key"]][()]
        if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
            depth_img = Image.open(io.BytesIO(data.tobytes()))
            data = np.array(depth_img, dtype=np.float32)
        if data.ndim == 3:
            # Encoded depth: R*65536 + G*256 + B
            if data.shape[0] in (1, 3, 4):
                data = np.transpose(data, (1, 2, 0))
            if data.shape[-1] == 3:
                data = data[..., 2].astype(np.float32) + data[..., 1].astype(np.float32) * 255 + data[..., 0].astype(np.float32) * 255 * 255
            else:
                data = data[..., 0].astype(np.float32)
        return data.astype(np.float32)

    def _make_sample(self, image_t, depth_t, mask_t, K, flip):
        if flip:
            image_t = torch.flip(image_t, dims=[-1])
            depth_t = torch.flip(depth_t, dims=[-1])
            mask_t = torch.flip(mask_t, dims=[-1])
        sample = {"image": image_t, "depth": depth_t, "depth_mask": mask_t,
                  "flip": flip, "si": False}
        if self.return_intrinsics:
            K_out = K.clone()
            if flip:
                K_out[0, 2] = depth_t.shape[-1] - K_out[0, 2]
            sample["K"] = K_out
        return sample

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
