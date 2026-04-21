# KITTI Dataset (from - Diffusion4RobustDepth subset)
# NOTE: we use 2012/2015 folder with specified split.

# --------------- Start of configuration --------------- #

# Default path to the KITTI challenging subset inside Diffusion4RobustDepth.
KITTI_ROOT = "datasets/Diffusion4RobustDepth/kitti/driving/kitti/challenging"

# Depth map filename suffix used during preprocessing (e.g., OUTPUT_SUFFIX in
# the Depth-Anything generation script). The dataset will look for files like:
#   <image_stem> + DEPTH_SUFFIX + ".png"
DEPTH_SUFFIX = "_depth_anything"

# ---------------- End of configuration ---------------- #

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Acceptable image extensions
IMAGE_EXTS: Sequence[str] = (".jpg", ".jpeg", ".png")
RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

# Depth range (metres)
# NOTE: this is an outdoor scene dataset
MIN_DEPTH: float = 0.01
MAX_DEPTH: float = 80.0

# Raw depth values in the saved PNGs are assumed to be in metres (float32 after
# conversion), optionally scaled by ``depth_scale`` at load time.
DEPTH_SCALE: float = 1.0

# ImageNet normalisation stats (from torchvision.transforms)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Default transforms
# ---------------------------------------------------------------------------

def _default_image_transform(resample_shape: Optional[Tuple[int, int]]) -> Callable:
    if resample_shape is not None:
        t = [
            transforms.Resize(resample_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ]
    else:
        t = [
            transforms.ToTensor(),
            transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ]
    return transforms.Compose(t)


def _default_depth_transform(resample_shape: Optional[Tuple[int, int]]) -> Callable:
    def transform(depth_np: np.ndarray) -> torch.Tensor:
        if resample_shape is not None:
            depth_pil = Image.fromarray(depth_np.astype(np.float32), mode = "F")
            depth_pil = depth_pil.resize(resample_shape, resample = RESAMPLE_NEAREST)
            depth_np_resampled = np.array(depth_pil, dtype = np.float32)
            return torch.from_numpy(depth_np_resampled).unsqueeze(0)
        return torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)

    return transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_suffix(depth_suffix: str) -> str:
    """
    Normalize the provided depth suffix to include a leading separator and an
    image extension (defaults to .png).
    """
    suffix = depth_suffix if depth_suffix.startswith(("_", ".")) else f"_{depth_suffix}"
    has_ext = any(suffix.lower().endswith(ext) for ext in IMAGE_EXTS)
    if not has_ext:
        suffix = f"{suffix}.png"
    return suffix


def _make_depth_path(image_path: Path, depth_suffix: str) -> Path:
    """
    Build the expected depth-map path for a given image.
    """
    suffix = _normalize_suffix(depth_suffix)
    return image_path.with_name(image_path.stem + suffix)


def _collect_pairs(root: Path, depth_suffix: str, split: str) -> List[Tuple[Path, Path]]:
    """
    Collect (image, depth) pairs for the requested split.  Only the KITTI 2012
    and 2015 subfolders are considered.  ``training`` subdirs feed the training
    split; ``testing`` subdirs feed the testing/val split.
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    split_to_subdirs = {
        "train": ["2012/training/colored_0", "2015/training/image_2", "2015/training/image_3"],
        "test": ["2012/testing/colored_0", "2015/testing/image_2", "2015/testing/image_3"],
    }
    split_to_subdirs["all"] = split_to_subdirs["train"] + split_to_subdirs["test"]

    if split not in split_to_subdirs:
        raise ValueError("split must be one of 'train', 'test', or 'all'.")

    normalized_suffix = _normalize_suffix(depth_suffix)
    suffix_lower = normalized_suffix.lower()
    vis_suffix = suffix_lower.replace(".png", "_vis.png")
    pairs: List[Tuple[Path, Path]] = []

    for rel_subdir in split_to_subdirs[split]:
        base_dir = root / rel_subdir
        if not base_dir.exists():
            continue
        for dirpath, _, filenames in os.walk(base_dir):
            for fname in filenames:
                fpath = Path(dirpath) / fname
                ext = fpath.suffix.lower()
                fname_lower = fname.lower()

                if ext not in IMAGE_EXTS:
                    continue
                if fname_lower.endswith(suffix_lower) or fname_lower.endswith(vis_suffix):
                    continue

                depth_path = _make_depth_path(fpath, depth_suffix)
                if depth_path.exists():
                    pairs.append((fpath, depth_path))

    pairs.sort(key = lambda p: str(p[0]))
    if not pairs:
        raise RuntimeError(f"No image/depth pairs found under {root} for split '{split}'.")
    return pairs


def _build_default_intrinsics(width: int, height: int) -> torch.Tensor:
    """
    Build a simple pinhole intrinsic matrix using the image dimensions.
    """
    fx = fy = float(max(width, height))
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype = torch.float32,
    )


# ---------------------------------------------------------------------------
# KITTIDataset
# ---------------------------------------------------------------------------

class KITTIDataset(Dataset):

    """
    PyTorch Dataset for the KITTI portion of Diffusion4RobustDepth (challenging
    split) with generated pseudo-GT depth maps.
    """

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = KITTI_ROOT,
        depth_suffix: str = DEPTH_SUFFIX,
        split: str = "train",
        image_shape: Tuple[int, int] = (384, 384),
        depth_scale: float = DEPTH_SCALE,
        flip_aug: bool = False,
        image_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        return_intrinsics: bool = True,
    ) -> None:

        super().__init__()

        # Store config
        self.root = Path(root)
        self.depth_suffix = depth_suffix
        self.split = split
        self.flip_aug = flip_aug
        self.depth_scale = depth_scale
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        self.return_intrinsics = return_intrinsics

        # Transforms
        self.image_transform = (
            image_transform if image_transform is not None else _default_image_transform(self.image_shape)
        )
        self.depth_transform = (
            depth_transform if depth_transform is not None else _default_depth_transform(self.image_shape)
        )

        # Collect pairs
        pairs = _collect_pairs(self.root, self.depth_suffix, split)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _make_sample(
        self,
        image_tensor: torch.Tensor,
        depth_tensor: torch.Tensor,
        depth_mask: torch.Tensor,
        flip: bool,
    ) -> Dict:

        if flip:
            image_tensor = torch.flip(image_tensor, dims = [-1])
            depth_tensor = torch.flip(depth_tensor, dims = [-1])
            depth_mask = torch.flip(depth_mask, dims = [-1])

        sample: Dict = {
            "image": image_tensor,
            "depth": depth_tensor,
            "depth_mask": depth_mask,
            "flip": flip,
            "si": False,
        }

        if self.return_intrinsics:
            H, W = depth_tensor.shape[-2:]
            K = _build_default_intrinsics(W, H)
            if flip:
                K = K.clone()
                K[0, 2] = (W - 1) - K[0, 2]
            sample["K"] = K

        return sample

    def __getitem__(self, idx: int):

        image_path, depth_path = self.pairs[idx]

        image_pil = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.image_transform(image_pil)

        depth_np = np.asarray(Image.open(depth_path), dtype = np.float32) * float(self.depth_scale)

        depth_tensor: torch.Tensor = self.depth_transform(depth_np)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)

        depth_mask = depth_tensor > 0

        if self.flip_aug:
            original = self._make_sample(image_tensor, depth_tensor, depth_mask, flip = False)
            flipped = self._make_sample(image_tensor, depth_tensor, depth_mask, flip = True)
            return original, flipped
        else:
            return self._make_sample(image_tensor, depth_tensor, depth_mask, flip = False)

    def __repr__(self) -> str:
        return (
            f"KITTIDataset(split='{self.split}', "
            f"flip_aug={self.flip_aug}, "
            f"n_samples={len(self)}, "
            f"root='{self.root}', "
            f"depth_suffix='{self.depth_suffix}')"
        )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    @classmethod
    def collate_fn(cls, batch: List) -> Dict:

        if batch and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            flat: List[Dict] = []
            for orig, flipped in batch:
                flat.append(orig)
                flat.append(flipped)
            batch = flat

        img_metas: List[Dict] = [
            {k: item[k] for k in cls.META_KEYS if k in item}
            for item in batch
        ]

        data_keys = [k for k in batch[0].keys() if k not in cls.META_KEYS]
        collated: Dict = {}
        for key in data_keys:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                collated[key] = torch.stack(vals, dim = 0)
            else:
                collated[key] = vals

        return {
            "data": collated,
            "img_metas": img_metas,
        }


# Test code
if __name__ == "__main__":

    # Quick test: load the first sample from the training split
    dataset = KITTIDataset(split = "train", flip_aug = True)

    # Print dataset info
    print(dataset)

    # Create a DataLoader and fetch one batch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, collate_fn = KITTIDataset.collate_fn)
    sample = next(iter(dataloader))["data"]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample["image"].shape)
    print("Depth shape:", sample["depth"].shape)
    print("Depth mask shape:", sample["depth_mask"].shape)
    print("Intrinsics shape:", sample["K"].shape)

    # Plot a sample
    from matplotlib import pyplot as plt
    image_tensor, depth_tensor = sample["image"][0], sample["depth"][0]  # (3, H, W), (1, H, W)
    image_np = image_tensor.permute(1, 2, 0).numpy()
    depth_np = depth_tensor.squeeze(0).numpy()
    # Normalize image for display (undo ImageNet normalization)
    image_undo_norm = (image_np * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
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
    fig.savefig("./datasets/kitti_sample.png")