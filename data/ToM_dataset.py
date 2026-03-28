# ToM Dataset (from - Diffusion4RobustDepth subset)

# --------------- Start of configuration --------------- #

# Default path to the ToM subset inside Diffusion4RobustDepth.
TOM_ROOT = "datasets/Diffusion4RobustDepth/ToM"

# Depth map filename suffix used during preprocessing (e.g., OUTPUT_SUFFIX in the
# Depth-Anything generation script). The dataset will look for files like:
#   <image_stem> + DEPTH_SUFFIX + ".png"
DEPTH_SUFFIX = "_depth_anything"

# Train split ratio (reproducible split). Remaining samples go to the "test" split.
TRAIN_SPLIT_RATIO: float = 0.9

# Random seed for reproducible shuffling when creating the split.
SPLIT_SEED: int = 42

# ---------------- End of configuration ---------------- #

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Acceptable image extensions
IMAGE_EXTS: Sequence[str] = (".jpg", ".jpeg", ".png")

# Depth range (metres)
# NOTE: loosely based on the Depth-Anything visualization range
MIN_DEPTH: float = 0.01
MAX_DEPTH: float = 10.0

# Raw depth values in the .mat file are in metres (float32 already scaled)
DEPTH_SCALE: float = 1.0

# Image net normalisation stats (from torchvision.transforms)
# NOTE: These are used in inferencing (revert normalised images)
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
            depth_pil = depth_pil.resize(resample_shape, resample = Image.NEAREST)
            depth_np_resampled = np.array(depth_pil, dtype = np.float32)
            return torch.from_numpy(depth_np_resampled).unsqueeze(0)
        return torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)

    return transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_suffix(depth_suffix: str) -> str:
    """
    Normalize the provided depth suffix to include a leading separator and
    an image extension (defaults to .png).
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


def _collect_pairs(root: Path, depth_suffix: str) -> List[Tuple[Path, Path]]:
    """
    Walk the dataset tree and collect (image, depth) pairs.
    Skips already-generated depth maps and visualization files.
    """
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    normalized_suffix = _normalize_suffix(depth_suffix)
    suffix_lower = normalized_suffix.lower()
    pairs: List[Tuple[Path, Path]] = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            fname_lower = fname.lower()

            # Only consider RGB images (skip non-image files)
            if ext not in IMAGE_EXTS:
                continue

            # Skip generated depth maps and their visualizations
            vis_suffix = suffix_lower.replace(".png", "_vis.png")
            if fname_lower.endswith(suffix_lower) or fname_lower.endswith(vis_suffix):
                continue

            depth_path = _make_depth_path(fpath, depth_suffix)
            if depth_path.exists():
                pairs.append((fpath, depth_path))

    pairs.sort(key = lambda p: str(p[0]))
    if not pairs:
        raise RuntimeError(f"No image/depth pairs found under {root}")

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
# ToMDataset
# ---------------------------------------------------------------------------

class ToMDataset(Dataset):

    """
    PyTorch Dataset for the ToM subset of Diffusion4RobustDepth with generated
    Depth-Anything pseudo-GT maps.
    """

    META_KEYS = {"flip", "si"}

    def __init__(
        self,
        root: str = TOM_ROOT,
        depth_suffix: str = DEPTH_SUFFIX,
        train_split_ratio: float = TRAIN_SPLIT_RATIO,
        split: str = "train",
        image_shape: Tuple[int, int] = (384, 384),
        depth_scale: float = DEPTH_SCALE,
        flip_aug: bool = False,
        seed: int = SPLIT_SEED,
        image_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        return_intrinsics: bool = True,
    ) -> None:
        
        # Init
        super().__init__()

        # Assertions and validations
        assert split in ("train", "test", "all"), "split must be 'train', 'test', or 'all'."
        if not (0.0 < train_split_ratio < 1.0):
            raise ValueError("train_split_ratio must be greater than 0 and less than 1 (exclusive).")

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

        # Collect pairs and create reproducible split
        pairs = _collect_pairs(self.root, self.depth_suffix)

        # Reproducibly shuffle and split the dataset
        rng = random.Random(seed)
        rng.shuffle(pairs)
        split_idx = int(len(pairs) * train_split_ratio)

        if split == "train":
            self.pairs = pairs[:split_idx]
        elif split == "test":
            self.pairs = pairs[split_idx:]
        else:
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
        
        # Apply horizontal flip if specified
        if flip:
            image_tensor = torch.flip(image_tensor, dims = [-1])
            depth_tensor = torch.flip(depth_tensor, dims = [-1])
            depth_mask = torch.flip(depth_mask, dims = [-1])

        # Build the sample dictionary
        sample: Dict = {
            "image": image_tensor,
            "depth": depth_tensor,
            "depth_mask": depth_mask,
            "flip": flip,
            "si": False,
        }

        # Optionally include intrinsics
        # (same for all samples, but depends on flip)
        if self.return_intrinsics:
            H, W = depth_tensor.shape[-2:]
            K = _build_default_intrinsics(W, H)
            if flip:
                K = K.clone()
                K[0, 2] = W - K[0, 2]
            sample["K"] = K

        return sample

    def __getitem__(self, idx: int):

        # Get the image and depth paths for the given index
        image_path, depth_path = self.pairs[idx]

        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_tensor: torch.Tensor = self.image_transform(image_pil)

        # Load depth
        depth_np = np.asarray(Image.open(depth_path), dtype = np.float32) * float(self.depth_scale)

        # Apply depth transform and clamp to valid range
        depth_tensor: torch.Tensor = self.depth_transform(depth_np)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)

        # Create a mask of valid depth pixels (where depth > 0)
        depth_mask = depth_tensor > 0

        # Return the sample(s), applying flip augmentation if specified
        if self.flip_aug:
            original = self._make_sample(image_tensor, depth_tensor, depth_mask, flip = False)
            flipped = self._make_sample(image_tensor, depth_tensor, depth_mask, flip = True)
            return original, flipped
        else:
            return self._make_sample(image_tensor, depth_tensor, depth_mask, flip = False)

    def __repr__(self) -> str:
        return (
            f"ToMDataset(split='{self.split}', "
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

        # Unpack paired samples produced by flip_aug = True
        if batch and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            flat: List[Dict] = []
            for orig, flipped in batch:
                flat.append(orig)
                flat.append(flipped)
            batch = flat

        # Extract metadata (keys in META_KEYS) into a separate list of dicts
        img_metas: List[Dict] = [
            {k: item[k] for k in cls.META_KEYS if k in item}
            for item in batch
        ]

        # Extract data keys (those not in META_KEYS) and collate them into tensors or lists
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
    dataset = ToMDataset(split = "train", flip_aug = True)
    
    # Print dataset info
    print(dataset)

    # Create a DataLoader and fetch one batch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, collate_fn = ToMDataset.collate_fn)
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
    image_undo_norm = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image_undo_norm = np.clip(image_undo_norm, 0, 1)
    print(f"[DEBUG] un-normed image shape: {image_undo_norm.shape}, dtype: {image_undo_norm.dtype}")
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
    fig.savefig("./datasets/ToM_sample.png")