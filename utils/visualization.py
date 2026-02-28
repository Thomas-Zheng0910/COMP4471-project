"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from utils.misc import ssi_helper


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img


def image_grid(imgs: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    if not len(imgs):
        return None
    assert len(imgs) == rows * cols
    h, w = imgs[0].shape[:2]
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize(
                (w, h), resample=Image.BILINEAR
            ),
            box=(i % cols * w, i // cols * h),
        )

    return np.array(grid)


def log_train_artifacts(*args, **kwargs):
    # Replaced wandb logging with a no-op.
    # Use TensorBoard SummaryWriter in train.py for visualization.
    pass
