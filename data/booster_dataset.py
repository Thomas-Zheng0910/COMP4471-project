"""
Booster MONO Dataset loader for inference.
Dataset: https://amsacta.unibo.it/id/eprint/7161/
"""

import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.nyuv2_dataset import IMAGENET_MEAN, IMAGENET_STD


class BoosterMonoDataset(Dataset):
    """
    Booster MONO dataset - test set only (no ground truth depth).
    Structure:
        booster_mono/
            test_scene/
                calib_00-02.xml
                camera_00/
                    0000.png, 0001.png, ...
    """

    def __init__(
        self,
        root: str,
        image_shape: Tuple[int, int] = (480, 640),
        return_intrinsics: bool = True,
    ):
        self.root = Path(root)
        self.image_shape = image_shape  # (H, W)
        self.return_intrinsics = return_intrinsics

        # Find all scenes (subdirectories)
        self.scenes = sorted([d for d in self.root.iterdir() if d.is_dir()])
        
        # Build list of all images with their scene info
        self.samples = []
        for scene_dir in self.scenes:
            camera_dir = scene_dir / "camera_00"
            if camera_dir.exists():
                images = sorted(camera_dir.glob("*.png"))
                calib_file = scene_dir / "calib_00-02.xml"
                for img_path in images:
                    self.samples.append({
                        "image": img_path,
                        "scene": scene_dir.name,
                        "calib": calib_file if calib_file.exists() else None,
                    })

        print(f"Booster MONO: Found {len(self.scenes)} scenes, {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def _parse_intrinsics(self, calib_file: Path) -> np.ndarray:
        """Parse left camera intrinsics from XML calibration file."""
        try:
            tree = ET.parse(calib_file)
            root = tree.getroot()
            
            # Get mtxL (left camera intrinsics)
            mtxL = root.find("mtxL")
            if mtxL is not None:
                data_elem = mtxL.find("data")
                if data_elem is not None:
                    data_text = data_elem.text.strip().split()
                    values = [float(x) for x in data_text if x]
                    K = np.array(values).reshape(3, 3)
                    return K
            
            # Fallback: use default Booster intrinsics
            return np.array([
                [4713.34, 0, 2050.13],
                [0, 4712.07, 1514.33],
                [0, 0, 1]
            ], dtype=np.float32)
        except Exception as e:
            print(f"Warning: Failed to parse {calib_file}: {e}")
            return np.array([
                [4713.34, 0, 2050.13],
                [0, 4712.07, 1514.33],
                [0, 0, 1]
            ], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image"]).convert("RGB")
        orig_w, orig_h = image.size  # PIL returns (W, H)
        
        # Resize to target shape
        image = image.resize((self.image_shape[1], self.image_shape[0]), Image.BILINEAR)
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)
        image_norm = (image_np - mean) / std
        
        # To tensor [3, H, W]
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)
        
        result = {
            "image": image_tensor,
            "filename": str(sample["image"]),
            "scene": sample["scene"],
        }
        
        if self.return_intrinsics and sample["calib"]:
            K = self._parse_intrinsics(sample["calib"])
            
            # Adjust intrinsics for resized image
            scale_x = self.image_shape[1] / orig_w
            scale_y = self.image_shape[0] / orig_h
            K_scaled = K.copy()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy
            
            result["intrinsics"] = torch.from_numpy(K_scaled).float()
        
        return result
