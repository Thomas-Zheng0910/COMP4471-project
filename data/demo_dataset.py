"""
DemoImageDataset: loads plain RGB images + depth maps from PNG files.
No HDF5 required. For demonstration purposes only.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DemoImageDataset(Dataset):
    """
    Simple dataset that loads RGB images and depth maps from a folder.

    Expected folder structure:
        root/
            images/   -> *.png RGB images
            depths/   -> *.png depth maps (16-bit, depth in mm, divide by 1000 for meters)
            intrinsics.json  -> (optional) {"stem": [fx, fy, cx, cy], ...}

    Args:
        root: Root directory of the dataset.
        image_shape: (H, W) target resolution for images and depth maps.
        depth_scale: Scale factor to convert raw depth values to meters.
                     Default 0.001 converts millimeters (16-bit PNG) to meters.
    """

    # ImageNet normalization constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_shape: Tuple[int, int] = (384, 384),
        depth_scale: float = 0.001,
    ):
        self.root = Path(root)
        self.image_shape = tuple(image_shape)  # (H, W)
        self.depth_scale = depth_scale

        # Discover image files
        images_dir = self.root / "images"
        depths_dir = self.root / "depths"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not depths_dir.exists():
            raise FileNotFoundError(f"Depths directory not found: {depths_dir}")

        # Collect matching image/depth pairs
        image_files = sorted(images_dir.glob("*.png"))
        self.samples: List[Dict] = []

        for img_path in image_files:
            stem = img_path.stem
            depth_path = depths_dir / f"{stem}.png"
            if depth_path.exists():
                self.samples.append({"image": img_path, "depth": depth_path, "stem": stem})

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matching image/depth pairs found in {root}. "
                "Ensure images/ and depths/ contain matching *.png files."
            )

        # Load optional intrinsics JSON
        intrinsics_path = self.root / "intrinsics.json"
        self.intrinsics_map: Optional[Dict] = None
        if intrinsics_path.exists():
            with open(intrinsics_path) as f:
                self.intrinsics_map = json.load(f)

        # Image transform: resize + normalize
        H, W = self.image_shape
        self.image_transform = transforms.Compose([
            transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        H, W = self.image_shape

        # Load RGB image
        img_pil = Image.open(sample["image"]).convert("RGB")
        orig_w, orig_h = img_pil.size  # PIL: (W, H)

        # Apply transform (resize + normalize)
        image = self.image_transform(img_pil)  # [3, H, W] float32 normalized

        # Load depth map (16-bit PNG, values in mm)
        depth_pil = Image.open(sample["depth"])
        depth_np = np.array(depth_pil, dtype=np.float32)  # [H_orig, W_orig]

        # Convert to meters using depth_scale
        depth_np = depth_np * self.depth_scale  # [H_orig, W_orig] in meters

        # Resize depth to target shape using nearest-neighbour to avoid blending invalid values
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H_orig, W_orig]
        depth_tensor = F.interpolate(depth_tensor, size=(H, W), mode="nearest").squeeze(0)  # [1, H, W]

        # Valid depth mask
        depth_mask = (depth_tensor > 0).bool()  # [1, H, W]

        # Build camera intrinsics matrix
        K = self._get_intrinsics(sample["stem"], orig_h, orig_w, H, W)

        return {
            "image": image,            # [3, H, W] float32, ImageNet-normalized
            "depth": depth_tensor,     # [1, H, W] float32, meters
            "depth_mask": depth_mask,  # [1, H, W] bool
            "K": K,                    # [3, 3] float32
            "filename": sample["stem"],
        }

    def _get_intrinsics(
        self, stem: str, orig_h: int, orig_w: int, tgt_h: int, tgt_w: int
    ) -> torch.Tensor:
        """Return a 3×3 intrinsics tensor, scaled to the target resolution."""
        if self.intrinsics_map is not None and stem in self.intrinsics_map:
            fx, fy, cx, cy = self.intrinsics_map[stem]
        else:
            # Default: simple pinhole with 60° horizontal FoV
            fx = orig_w / (2.0 * np.tan(np.radians(30)))
            fy = fx
            cx = orig_w / 2.0
            cy = orig_h / 2.0

        # Scale intrinsics to match the resized image
        scale_x = tgt_w / orig_w
        scale_y = tgt_h / orig_h
        fx = fx * scale_x
        fy = fy * scale_y
        cx = cx * scale_x
        cy = cy * scale_y

        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        return K

    @classmethod
    def collate_fn(cls, batch: List[Dict]) -> Dict:
        """Default collate function — stacks tensors, keeps filenames as list."""
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                collated[key] = torch.stack(vals, dim=0)
            else:
                collated[key] = vals  # e.g., list of filenames
        return collated
