# NYU Depth V2 Dataset
# Adapted from: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat


# --------------- Start of configuration --------------- #

# Path to the NYU Depth V2 .mat file (MATLAB v7.3 / HDF5 format).
NYUV2_MAT_PATH = "datasets/nyu_depth_v2_labeled.mat"

# ---------------- End of configuration ---------------- #

# import necessary libraries
import os
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Camera intrinsics (from the NYU Depth V2 toolbox / UniDepth reference)
# ---------------------------------------------------------------------------
# fmt: off
NYUV2_INTRINSICS = torch.tensor([
    [5.1885790117450188e+02, 0.0,                    3.2558244941119034e+02],
    [0.0,                    5.1946961112127485e+02,  2.5373616633400465e+02],
    [0.0,                    0.0,                    1.0                    ],
], dtype=torch.float32)
# fmt: on

# Depth range (metres) consistent with UniDepth / standard NYUv2 evaluation
MIN_DEPTH: float = 0.005
MAX_DEPTH: float = 10.0

# Raw depth values in the .mat file are in metres (float32 already scaled)
DEPTH_SCALE: float = 1.0

# Eigen split – 654 test images (standard benchmark, 0-based indices)
EIGEN_TEST_INDICES_PATH: Optional[str] = None  # set to a .txt of indices if available


# ---------------------------------------------------------------------------
# Helper: load the HDF5-backed .mat file (v7.3 format)
# ---------------------------------------------------------------------------

def _load_mat(mat_path: str) -> h5py.File:

    """
    Open the NYU Depth V2 .mat file (MATLAB v7.3 / HDF5 format).
    """
    
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(
            f"NYU Depth V2 .mat file not found at '{mat_path}'.\n"
            "Download it from:\n"
            "  http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat\n"
            "and update the NYUV2_MAT_PATH macro at the top of data/nyuv2.py."
        )
    return h5py.File(mat_path, "r")


# ---------------------------------------------------------------------------
# Default image transforms
# ---------------------------------------------------------------------------

def _default_image_transform() -> Callable:

    """
    Standard ImageNet-normalised transform used by depth models.
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
        ),
    ])


def _default_depth_transform(depth_np: np.ndarray) -> torch.Tensor:

    """
    Convert a raw HxW float32 numpy depth array to a (1, H, W) float32 tensor.
    """

    return torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# NYUv2Dataset
# ---------------------------------------------------------------------------

class NYUv2Dataset(Dataset):
    """
    PyTorch Dataset for NYU Depth V2 (labeled split).

    Reads directly from the official ``nyu_depth_v2_labeled.mat`` file
    (MATLAB v7.3 / HDF5 format) without any pre-processing step.

    The dataset contains 1449 densely labelled RGBD pairs captured in
    indoor scenes.  The standard Eigen et al. train/test split uses
    654 test images; the remaining 795 are used for training.

    Args:
        mat_path (str): Path to ``nyu_depth_v2_labeled.mat``.
            Defaults to the ``NYUV2_MAT_PATH`` macro defined at the top
            of this file.
        split (str): One of ``"train"``, ``"test"``, or ``"all"``.
            Uses the standard 654-image Eigen test split.
        image_transform (callable, optional): Transform applied to the
            RGB image (PIL Image → tensor).  Defaults to
            ``_default_image_transform()``.
        depth_transform (callable, optional): Transform applied to the
            raw depth numpy array.  Defaults to
            ``_default_depth_transform()``.
        return_intrinsics (bool): If ``True``, each sample also returns
            the 3×3 camera intrinsic matrix ``K``.

    Returns (per sample):
        image  (torch.Tensor): ``(3, H, W)`` float32, normalised.
        depth  (torch.Tensor): ``(1, H, W)`` float32, depth in **metres**,
                               clipped to [MIN_DEPTH, MAX_DEPTH].
        K      (torch.Tensor, optional): ``(3, 3)`` float32 intrinsics.
    """

    # Eigen et al. 654 test indices (1-based -> converted to 0-based)
    _EIGEN_TEST_INDICES = [i - 1 for i in range(1, 655)]

    def __init__(
        self,
        mat_path: str = NYUV2_MAT_PATH,
        split: str = "train",
        image_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        return_intrinsics: bool = False,
    ) -> None:
        super().__init__()

        assert split in ("train", "test", "all"), (
            f"split must be one of 'train', 'test', 'all', got '{split}'."
        )

        self.mat_path = mat_path
        self.split = split
        self.return_intrinsics = return_intrinsics

        self.image_transform = (
            image_transform if image_transform is not None else _default_image_transform()
        )
        self.depth_transform = (
            depth_transform if depth_transform is not None else _default_depth_transform
        )

        # Read total count once, then close (fork-safety for DataLoader)
        h5 = _load_mat(self.mat_path)
        total = h5["images"].shape[0]  # (N, 3, W, H)
        h5.close()

        test_set = set(self._EIGEN_TEST_INDICES)
        all_indices = list(range(total))

        if split == "train":
            self.indices = [i for i in all_indices if i not in test_set]
        elif split == "test":
            self.indices = [i for i in all_indices if i in test_set]
        else:
            self.indices = all_indices

        # Lazy per-worker file handle
        self._h5: Optional[h5py.File] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_h5(self) -> h5py.File:

        """
        Return a per-process/worker file handle (lazy open).
        """

        if self._h5 is None:
            self._h5 = _load_mat(self.mat_path)
        return self._h5

    @staticmethod
    def _apply_eval_mask(depth: torch.Tensor) -> torch.Tensor:
        """
        Zero out the border crop used in the Eigen benchmark.

        Mirrors UniDepth's ``eval_mask``:  border_mask[..., 45:-9, 41:-39] = 1
        """

        mask = torch.zeros_like(depth, dtype = torch.bool)
        mask[..., 45:-9, 41:-39] = True
        return depth * mask.float()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple:
        mat_idx = self.indices[idx]
        h5 = self._get_h5()

        # RGB image
        # HDF5 layout: (N, 3, W, H) uint8  ->  transpose to (H, W, 3)
        image_raw = h5["images"][mat_idx]                  # (3, W, H)
        image_np = np.transpose(image_raw, (2, 1, 0))      # (H, W, 3)
        image_pil = Image.fromarray(image_np.astype(np.uint8), mode = "RGB")

        # Depth map
        # HDF5 layout: (N, W, H) float32, metres  ->  transpose to (H, W)
        depth_raw = h5["depths"][mat_idx]                  # (W, H)
        depth_np = np.transpose(depth_raw, (1, 0))         # (H, W)

        # Transforms
        image_tensor: torch.Tensor = self.image_transform(image_pil)

        depth_tensor: torch.Tensor = self.depth_transform(depth_np)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)

        if self.split == "test":
            depth_tensor = self._apply_eval_mask(depth_tensor)

        if self.return_intrinsics:
            return image_tensor, depth_tensor, NYUV2_INTRINSICS.clone()
        return image_tensor, depth_tensor

    def __repr__(self) -> str:
        return (
            f"NYUv2Dataset(split='{self.split}', "
            f"n_samples={len(self)}, "
            f"mat_path='{self.mat_path}')"
        )


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def get_nyuv2_train(
    mat_path: str = NYUV2_MAT_PATH,
    image_transform: Optional[Callable] = None,
    depth_transform: Optional[Callable] = None,
    return_intrinsics: bool = False,
) -> NYUv2Dataset:
    
    """
    Return the NYU Depth V2 training split (Eigen split, ~795 images).
    """

    return NYUv2Dataset(
        mat_path = mat_path,
        split = "train",
        image_transform = image_transform,
        depth_transform = depth_transform,
        return_intrinsics = return_intrinsics,
    )


def get_nyuv2_test(
    mat_path: str = NYUV2_MAT_PATH,
    image_transform: Optional[Callable] = None,
    depth_transform: Optional[Callable] = None,
    return_intrinsics: bool = False,
) -> NYUv2Dataset:
    
    """
    Return the NYU Depth V2 test split (Eigen split, 654 images).
    """

    return NYUv2Dataset(
        mat_path = mat_path,
        split = "test",
        image_transform = image_transform,
        depth_transform = depth_transform,
        return_intrinsics = return_intrinsics,
    )

# Test code
if __name__ == "__main__":

    # Quick test: load the first sample from the training split
    dataset = get_nyuv2_train()
    print(dataset)
    sample = dataset[0]
    print("Sample keys:", sample.keys() if isinstance(sample, dict) else "N/A")
    print("Image shape:", sample[0].shape)
    print("Depth shape:", sample[1].shape)

    # Plot the first sample
    from matplotlib import pyplot as plt
    image_tensor, depth_tensor = sample[:2]
    image_np = image_tensor.permute(1, 2, 0).numpy()
    depth_np = depth_tensor.squeeze(0).numpy()
    # Normalize image for display (undo ImageNet normalization)
    image_undo_norm = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image_undo_norm = np.clip(image_undo_norm, 0, 1)
    # Show the RGB image, normed image and depth map side by side
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    axes[0].imshow(image_undo_norm)
    axes[0].set_title("RGB Image (undo norm)")
    axes[0].axis("off")
    im1 = axes[1].imshow(image_np)
    axes[1].set_title("RGB Image (normed)")
    axes[1].axis("off")
    fig.colorbar(im1, ax = axes[1], fraction = 0.046, pad = 0.04)
    im2 = axes[2].imshow(depth_np, cmap = "plasma", vmin = MIN_DEPTH, vmax = MAX_DEPTH)
    axes[2].set_title("Depth Map (metres)")
    axes[2].axis("off")
    fig.colorbar(im2, ax = axes[2], fraction = 0.046, pad = 0.04)
    fig.tight_layout()
    fig.savefig("./datasets/nyuv2_sample.png")