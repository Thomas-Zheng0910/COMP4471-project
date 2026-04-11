from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict, Any, Callable

import random
import numpy as np

try:
    import cv2
except Exception:
    raise RuntimeError("opencv-python not installed. Run: pip install opencv-python or opencv-python-headless")
try:
    import albumentations as A
except Exception:
    raise RuntimeError("albumentations not installed. Run: pip install albumentations")

def get_albumentations_weather(prob: float = 0.5) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a callable(img) -> img that applies weather-style augmentations
    using albumentations with an internal sampling probability.
    """
    aug = A.Compose([
        A.RandomRain(brightness_coefficient = 0.9, drop_length = 20, drop_width = 1, p = 0.4),
        A.RandomSnow(p = 0.3),
        A.RandomFog(p = 0.4),
        A.RandomBrightnessContrast(brightness_limit = 0.3, contrast_limit = 0.3, p = 0.5),
        A.MotionBlur(blur_limit = 7, p = 0.2),
        A.CLAHE(p = 0.2),
    ], p = 1.0)
    return lambda img: aug(image=img)['image'] if random.random() < prob else img

def depth_based_fog(image: np.ndarray, depth: np.ndarray, 
                    beta_range: Tuple[float, float] = (0.01, 0.04), 
                    A_range: Tuple[int, int] = (200, 255), 
                    prob: float = 1.0) -> np.ndarray:
    """
    Depth-dependent fog using atmospheric scattering model:
      I = J * t + A * (1 - t), t = exp(-beta * depth)
    - image: HxWx3 uint8
    - depth: HxW float (meters) or same unit used in training
    - returns uint8 image
    """
    if random.random() > prob:
        return image
    img = image.astype(np.float32)
    dep = depth.astype(np.float32).copy()
    # replace zeros/invalid with a high value to avoid infinite transmission
    valid_mask = (dep > 0) & np.isfinite(dep)
    if not np.any(valid_mask):
        return image
    maxvalid = np.nanmax(dep[valid_mask])
    dep[~valid_mask] = maxvalid
    beta = float(np.random.uniform(*beta_range))
    A_val = float(np.random.uniform(*A_range))
    t = np.exp(-beta * dep)
    if t.ndim == 2:
        t = t[:, :, None]
    fog_img = img * t + A_val * (1.0 - t)
    fog_img = np.clip(fog_img, 0, 255).astype(np.uint8)
    return fog_img