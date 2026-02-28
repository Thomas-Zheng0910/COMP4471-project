"""
Dummy Dataset:
will generate random RGB images and depth maps on the fly, 
without loading from disk.
"""

# ------------- Start of configuration ------------- #

# LENGTH of the dataset (number of samples)
DATASET_LENGTH = 1000

# -------------- END of configuration -------------- #

# import necessary libraries
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DummyDataset(Dataset):
    """
    Dummy dataset that generates random RGB images and depth maps on the fly.
    """

    # ImageNet normalization constants
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str = "DummyDataset",
        image_shape: Tuple[int, int] = (384, 384),
        depth_scale: float = 0.001,
    ):
        
        """
        Initialize the dummy dataset.
        Args:
            root: Dummy root directory (not used for loading, just for naming).
            image_shape: (H, W) target resolution for images and depth maps.
            depth_scale: Scale factor to convert raw depth values to meters.
                         Default 0.001 converts millimeters (16-bit PNG) to meters.
        NOTE: Actually, you don't need to specify anything
              Put them here for compatibility
        """

        self.root = Path(root)
        self.image_shape = tuple(image_shape)  # (H, W)
        self.depth_scale = depth_scale

        # Image transform: resize + normalize
        H, W = self.image_shape
        self.image_transform = transforms.Compose([
            transforms.Resize((H, W), interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self) -> int:
        return DATASET_LENGTH

    def __getitem__(self, idx: int) -> Dict:

        # Generate a dummy sample for each index
        H, W = self.image_shape
        
        # Create a dummy RGB image (random values)
        img_np = np.random.rand(H, W, 3).astype(np.float32)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

        # Create a dummy depth map (random values)
        depth_np = np.random.rand(H, W).astype(np.float32) * 1000  # Random depth in mm

        # Build camera intrinsics matrix (dummy values)
        K = torch.tensor(
            [[W / 2.0, 0.0, W / 2.0], [0.0, H / 2.0, H / 2.0], [0.0, 0.0, 1.0]],
            dtype = torch.float32,
        )

        # Apply image transform
        image = self.image_transform(img_pil)  # [3, H, W] float32 normalized

        # Convert depth to meters and create a valid mask
        depth_tensor = torch.from_numpy(depth_np * self.depth_scale).unsqueeze(0)
        depth_mask = (depth_tensor > 0).bool()  # [1, H, W]

        # Return the sample as a dictionary
        return {
            "image": image,            # [3, H, W] float32, ImageNet-normalized
            "depth": depth_tensor,     # [1, H, W] float32, meters
            "depth_mask": depth_mask,  # [1, H, W] bool
            "K": K,                    # [3, 3] float32
        }

    @classmethod
    def collate_fn(cls, batch: List[Dict]) -> Dict:

        """
        Default collate function — stacks tensors, keeps filenames as list.
        """

        keys = batch[0].keys()
        # Get image data
        collated = {}
        for key in keys:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                collated[key] = torch.stack(vals, dim=0)
            else:
                collated[key] = vals  # e.g., list of filenames
        # Get metadata
        meta = { "val": torch.rand(1, 1024, dtype = torch.float32) }
        # Output dict
        output_dict = {
            'data': collated,
            'img_metas': meta,
        }
        return output_dict

# Test the dataset
if __name__ == "__main__":
    dataset = DummyDataset()
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample["image"].shape)
    print("Depth shape:", sample["depth"].shape)
    print("Depth mask shape:", sample["depth_mask"].shape)
    print("K shape:", sample["K"].shape)