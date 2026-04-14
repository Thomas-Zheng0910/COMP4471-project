"""
Virtual KITTI 2 dataset loader for depth estimation training.

Expected directory layout (after extracting tarballs):
    datasets/virtual_kitti_2/
        vkitti_2.0.3_rgb/
            Scene01/
                clone/frames/rgb/Camera_0/  rgb_00000.jpg ...
            Scene02/ ...
        vkitti_2.0.3_depth/
            Scene01/
                clone/frames/depth/Camera_0/  depth_00000.png ...
            Scene02/ ...

Depth is stored as 16-bit PNG: pixel_value / 100.0 = depth in metres.
Camera intrinsics are constant across all Virtual KITTI 2 scenes.

Usage:
    dataset = VirtualKITTI2Dataset(root="datasets/virtual_kitti_2", split="train")
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Depth range (outdoor, metres)
MIN_DEPTH: float = 0.01
MAX_DEPTH: float = 80.0

# ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

# Virtual KITTI 2 camera intrinsics (constant across all scenes)
# From the dataset documentation: fx=725.0, fy=725.0, cx=620.5, cy=187.0
# Image resolution: 1242 x 375
VKITTI2_INTRINSICS = torch.tensor([
    [725.0087,   0.0,     620.5],
    [  0.0,    725.0087,  187.0],
    [  0.0,      0.0,       1.0],
], dtype=torch.float32)


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


def _collect_vkitti2_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    Find all (rgb, depth) pairs across all scenes/variations/cameras.
    """
    rgb_base = root / "vkitti_2.0.3_rgb"
    depth_base = root / "vkitti_2.0.3_depth"

    if not rgb_base.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_base}")
    if not depth_base.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_base}")

    pairs = []
    for rgb_path in sorted(rgb_base.rglob("rgb_*.jpg")):
        # Convert rgb path to depth path:
        #   .../vkitti_2.0.3_rgb/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
        #   → .../vkitti_2.0.3_depth/Scene01/clone/frames/depth/Camera_0/depth_00000.png
        rel = rgb_path.relative_to(rgb_base)
        depth_rel = Path(str(rel).replace("/rgb/", "/depth/").replace("rgb_", "depth_").replace(".jpg", ".png"))
        depth_path = depth_base / depth_rel

        if depth_path.exists():
            pairs.append((rgb_path, depth_path))

    if not pairs:
        raise RuntimeError(f"No Virtual KITTI 2 pairs found under {root}")
    return pairs


class VirtualKITTI2Dataset(Dataset):
    """PyTorch Dataset for Virtual KITTI 2 depth estimation."""

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = "datasets/virtual_kitti_2",
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

        all_pairs = _collect_vkitti2_pairs(self.root)

        # 90/10 train/test split (deterministic)
        n = len(all_pairs)
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        split_idx = int(0.9 * n)
        if split == "train":
            self.pairs = [all_pairs[i] for i in indices[:split_idx]]
        elif split in ("test", "val"):
            self.pairs = [all_pairs[i] for i in indices[split_idx:]]
        else:
            self.pairs = all_pairs

        print(f"VirtualKITTI2Dataset: {len(self.pairs)} samples ({split})")

    def __len__(self) -> int:
        return len(self.pairs)

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
        rgb_path, depth_path = self.pairs[idx]

        image_pil = Image.open(rgb_path).convert("RGB")
        image_tensor = self.image_transform(image_pil)

        # Virtual KITTI 2 depth: uint16, value / 100 = metres
        depth_raw = np.asarray(Image.open(depth_path), dtype=np.float32)
        depth_m = depth_raw / 100.0 * self.depth_scale
        depth_tensor = self.depth_transform(depth_m)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)
        depth_mask = (depth_tensor > MIN_DEPTH) & torch.isfinite(depth_tensor)

        # Scale intrinsics to resized image
        K = VKITTI2_INTRINSICS.clone()
        if self.image_shape is not None:
            orig_h, orig_w = 375, 1242  # VKITTI2 native resolution
            sh = self.image_shape[0] / orig_h
            sw = self.image_shape[1] / orig_w
            K[0, :] *= sw
            K[1, :] *= sh

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
