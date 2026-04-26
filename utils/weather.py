from __future__ import annotations

from typing import Tuple, Union, List, Dict, Callable

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

# Type aliases for augmentation function signatures
ImgAugFn = Callable[[np.ndarray], np.ndarray]
DepthAugFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Depth Augmentation functions

# Depth-based fog augmentation using the atmospheric scattering model
def _depth_aug_fog(
    image: np.ndarray,
    depth: np.ndarray,
    beta_range: Tuple[float, float] = (0.01, 0.04),
    A_range: Tuple[int, int] = (200, 255),
) -> np.ndarray:
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
    return np.clip(fog_img, 0, 255).astype(np.uint8)

# Image Augmentation functions

# Random crop with scale between 0.72 and 0.95,
# then resize back to original size
def _aug_random_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    crop_scale = float(np.random.uniform(0.72, 0.95))
    new_h = max(1, int(h * crop_scale))
    new_w = max(1, int(w * crop_scale))
    y0 = random.randint(0, h - new_h)
    x0 = random.randint(0, w - new_w)
    crop = image[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(crop, (w, h), interpolation = cv2.INTER_LINEAR)

# Localized Gaussian blur: apply a blurred version of the image to
# random circular spots
def _aug_local_gaussian_blur(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX = float(np.random.uniform(1.0, 3.2)))
    out = image.copy()
    n_spots = random.randint(1, 4)
    for _ in range(n_spots):
        r = int(np.random.uniform(min(h, w) * 0.04, min(h, w) * 0.16))
        cx = random.randint(r, max(r, w - r))
        cy = random.randint(r, max(r, h - r))
        yy, xx = np.ogrid[:h, :w]
        spot_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)
        out[spot_mask] = blurred[spot_mask]
    return out

# Contrast jitter using albumentations' RandomBrightnessContrast
def _aug_contrast_jitter(image: np.ndarray) -> np.ndarray:
    aug = A.RandomBrightnessContrast(
        brightness_limit = 0.25,
        contrast_limit = 0.35,
        brightness_by_max = True,
        p = 1.0,
    )
    return aug(image = image)["image"]

# Weather effects using albumentations' RandomRain, RandomSnow,
# and RandomFog
def _aug_weather(image: np.ndarray) -> np.ndarray:
    aug = A.OneOf(
        [
            A.RandomRain(brightness_coefficient = 0.9, drop_length = 20, drop_width = 1, p = 1.0),
            A.RandomSnow(p = 1.0),
            A.RandomFog(p = 1.0),
        ],
        p = 1.0,
    )
    return aug(image = image)["image"]

# Motion blur using albumentations' MotionBlur with a random blur limit
def _aug_motion_blur(image: np.ndarray) -> np.ndarray:
    aug = A.MotionBlur(blur_limit = 7, p = 1.0)
    return aug(image = image)["image"]

# Fog on lens effect: add random circular fog spots with varying alpha,
# then apply a strong Gaussian blur to simulate fog on the camera lens
def _aug_fog_on_lens(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype = np.float32)
    n_spots = random.randint(1, 4)
    for _ in range(n_spots):
        radius = int(np.random.uniform(min(h, w) * 0.08, min(h, w) * 0.20))
        cx = random.randint(radius, max(radius, w - radius))
        cy = random.randint(radius, max(radius, h - radius))
        alpha = float(np.random.uniform(0.20, 0.48))
        cv2.circle(overlay, (cx, cy), radius, (255.0 * alpha, 255.0 * alpha, 255.0 * alpha), thickness = -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX = float(np.random.uniform(6.0, 14.0)))
    out = image.astype(np.float32) + overlay
    return np.clip(out, 0, 255).astype(np.uint8)

# Registries for augmentation functions
IMG_AUG_REGISTRY: Dict[str, ImgAugFn] = {
    "random_crop": _aug_random_crop,
    "local_gaussian_blur": _aug_local_gaussian_blur,
    "contrast_jitter": _aug_contrast_jitter,
    "weather": _aug_weather,
    "motion_blur": _aug_motion_blur,
    "fog_on_lens": _aug_fog_on_lens,
}
DEPTH_AUG_REGISTRY: Dict[str, DepthAugFn] = {
    "depth_fog": _depth_aug_fog,
}

# Helper functions to build augmentation pipelines based on user-specified names
def _parse_aug_names(aug_names: Union[str, List[str], None]) -> List[str]:
    if aug_names is None:
        return []
    if isinstance(aug_names, str):
        raw = aug_names.strip()
        if raw == "" or raw.lower() == "none":
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(x).strip() for x in aug_names if str(x).strip()]

# Factory functions to create augmentation pipelines based on specified augmentation names
def build_img_aug_pipeline(aug_names: Union[str, List[str], None]) -> Union[ImgAugFn, None]:
    names = _parse_aug_names(aug_names)
    if not names:
        return None
    funcs = []
    for name in names:
        if name not in IMG_AUG_REGISTRY:
            raise ValueError(f"Unknown img augmentation: '{name}'. Available: {sorted(IMG_AUG_REGISTRY.keys())}")
        funcs.append(IMG_AUG_REGISTRY[name])

    # !!! This is the callable augmentation function constructed
    def _apply(image: np.ndarray) -> np.ndarray:
        out = image
        for fn in funcs:
            out = fn(out)
        return out

    return _apply

# Similar factory for depth augmentations, 
# which take both image and depth as input
def build_depth_aug_pipeline(aug_names: Union[str, List[str], None]) -> Union[DepthAugFn, None]:
    names = _parse_aug_names(aug_names)
    if not names:
        return None
    funcs = []
    for name in names:
        if name not in DEPTH_AUG_REGISTRY:
            raise ValueError(f"Unknown depth augmentation: '{name}'. Available: {sorted(DEPTH_AUG_REGISTRY.keys())}")
        funcs.append(DEPTH_AUG_REGISTRY[name])

    def _apply(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
        out = image
        for fn in funcs:
            out = fn(out, depth)
        return out

    return _apply