"""
Shared data augmentation transforms for depth estimation training.

These augmentations match the original UniDepth V2 training pipeline:
  - ColorJitter (brightness, contrast, saturation, hue)
  - GaussianBlur
  - RandomGamma (custom)
  - RandomGrayscale

All transforms operate on **PIL Images** and should be applied BEFORE
ToTensor() and Normalize(). They only affect the RGB image — depth and
masks are NOT augmented.

Usage:
    from data.augmentations import build_train_augmentation
    aug = build_train_augmentation({"jitter": 0.4, "blur_p": 0.2, ...})
    # aug is a transforms.Compose or None
"""

import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class RandomGamma:
    """Apply random gamma correction to a PIL image.

    gamma_range controls the deviation: the actual gamma is sampled from
    [1 - gamma_range, 1 + gamma_range].  gamma < 1 brightens, gamma > 1
    darkens.
    """

    def __init__(self, gamma_range: float = 0.2):
        self.gamma_range = gamma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        gamma = random.uniform(1.0 - self.gamma_range, 1.0 + self.gamma_range)
        gamma = max(gamma, 0.01)  # safety
        inv_gamma = 1.0 / gamma
        table = [int((i / 255.0) ** inv_gamma * 255) for i in range(256)]
        if img.mode == "RGB":
            table = table * 3
        return img.point(table)

    def __repr__(self):
        return f"RandomGamma(gamma_range={self.gamma_range})"


class RandomApply:
    """Apply a transform with a given probability."""

    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img

    def __repr__(self):
        return f"RandomApply(p={self.p}, transform={self.transform})"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

DEFAULT_AUG_CONFIG = {
    "jitter": 0.4,        # ColorJitter brightness/contrast/saturation
    "jitter_hue": 0.1,    # ColorJitter hue
    "jitter_p": 0.8,      # probability of applying color jitter
    "blur_sigma": 2.0,    # GaussianBlur max sigma
    "blur_p": 0.2,        # probability of applying blur
    "gamma": 0.2,         # RandomGamma range
    "gamma_p": 0.8,       # probability of applying gamma
    "grayscale_p": 0.2,   # probability of converting to grayscale
}


def build_train_augmentation(config: dict = None) -> transforms.Compose:
    """Build a training augmentation pipeline from a config dict.

    Parameters
    ----------
    config : dict or None
        Keys matching DEFAULT_AUG_CONFIG override defaults.
        If None or empty, returns None (no augmentation).

    Returns
    -------
    transforms.Compose or None
    """
    if config is None:
        return None

    cfg = {**DEFAULT_AUG_CONFIG, **config}

    aug_list = []

    # Color jitter
    jitter = float(cfg["jitter"])
    if jitter > 0 and float(cfg["jitter_p"]) > 0:
        aug_list.append(
            RandomApply(
                transforms.ColorJitter(
                    brightness=jitter,
                    contrast=jitter,
                    saturation=jitter,
                    hue=float(cfg["jitter_hue"]),
                ),
                p=float(cfg["jitter_p"]),
            )
        )

    # Gaussian blur
    blur_p = float(cfg["blur_p"])
    blur_sigma = float(cfg["blur_sigma"])
    if blur_p > 0 and blur_sigma > 0:
        aug_list.append(
            RandomApply(
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, blur_sigma)),
                p=blur_p,
            )
        )

    # Random gamma
    gamma_range = float(cfg["gamma"])
    gamma_p = float(cfg["gamma_p"])
    if gamma_range > 0 and gamma_p > 0:
        aug_list.append(
            RandomApply(RandomGamma(gamma_range=gamma_range), p=gamma_p)
        )

    # Random grayscale
    grayscale_p = float(cfg["grayscale_p"])
    if grayscale_p > 0:
        aug_list.append(transforms.RandomGrayscale(p=grayscale_p))

    if not aug_list:
        return None

    return transforms.Compose(aug_list)
