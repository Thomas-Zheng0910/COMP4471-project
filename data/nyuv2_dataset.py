# NYU Depth V2 Dataset
# Adapted from: https://github.com/lpiccinelli-eth/UniDepth/blob/main/unidepth/datasets/nyuv2.py


# --------------- Start of configuration --------------- #

# Path to the NYU Depth V2 .mat file (MATLAB v7.3 / HDF5 format).
NYUV2_MAT_PATH = "datasets/nyu_depth_v2_labeled.mat"

# ---------------- End of configuration ---------------- #

# import necessary libraries
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

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
LIDAR_DEPTH_SCALE: float = 1.0

# Image net normalisation stats (from torchvision.transforms)
# NOTE: These are used in inferencing (revert normalised images)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

def _default_image_transform(resample_shape: Tuple[int, int] = None) -> Callable:

    """
    Standard ImageNet-normalised transform used by depth models.
    """

    if resample_shape is not None:
        image_transforms = [
            transforms.Resize(resample_shape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]
    else:
        image_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]

    return transforms.Compose(image_transforms)


def _default_depth_transform(resample_shape: Tuple[int, int] = None) -> Callable:

    """
    Convert a raw HxW float32 numpy depth array to a (1, H, W) float32 tensor.
    """

    def transform(depth_np: np.ndarray) -> torch.Tensor:
        if resample_shape is not None:
            # Resample depth map using PIL (nearest neighbour to preserve values)
            depth_pil = Image.fromarray(depth_np.astype(np.float32), mode = "F")
            depth_pil_resampled = depth_pil.resize((resample_shape[1], resample_shape[0]), resample = Image.NEAREST)
            depth_np_resampled = np.array(depth_pil_resampled, dtype = np.float32)
            return torch.from_numpy(depth_np_resampled).unsqueeze(0)
        return torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)

    return transform


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
        root (str): Path to ``nyu_depth_v2_labeled.mat``.
            Defaults to the ``NYUV2_MAT_PATH`` macro defined at the top
            of this file.
        image_shape (Tuple[int, int]): Target resolution (H, W) for
            images and depth maps.  Defaults to (480, 640) (the original
            resolution).
        depth_scale (float): Scale factor to convert raw depth values to
            metres.  Defaults to 1.0 since the .mat file already contains
            depth in metres (float32).
        split (str): One of ``"train"``, ``"test"``, or ``"all"``.
            Uses the standard 654-image Eigen test split.
        flip_aug (bool): If ``True``, ``__getitem__`` returns a
            ``(original, flipped)`` tuple so the DataLoader / collate_fn
            can build interleaved batches required by the SelfDistill
            invariance loss.  Should be ``True`` for training only.
            Defaults to ``False``.
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
        depth_mask (torch.Tensor): ``(1, H, W)`` bool, valid depth pixels.
        K      (torch.Tensor, optional): ``(3, 3)`` float32 intrinsics.

    When ``flip_aug=True`` the return is a 2-tuple
    ``(original_dict, flipped_dict)`` where the flipped dict has the image,
    depth and mask horizontally mirrored and the principal-point cx updated
    to ``W - cx`` in K.
    """

    # Eigen et al. 654 test indices (0-based) from the official splits.mat
    # Source: https://github.com/VainF/nyuv2-python-toolkit/blob/master/splits.mat
    _EIGEN_TEST_INDICES = [
        0, 1, 8, 13, 14, 15, 16, 17, 20, 27, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 55, 56, 58, 59, 60, 61, 62,
        75, 76, 77, 78, 83, 84, 85, 86, 87, 88, 89, 90, 116, 117, 118, 124,
        125, 126, 127, 128, 130, 131, 132, 133, 136, 152, 153, 154, 166, 167,
        168, 170, 171, 172, 173, 174, 175, 179, 180, 181, 182, 183, 184, 185,
        186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 206, 207, 208, 209, 210, 211, 219, 220, 221, 249, 263, 270,
        271, 272, 278, 279, 280, 281, 282, 283, 284, 295, 296, 297, 298, 299,
        300, 301, 309, 310, 311, 314, 315, 316, 324, 325, 326, 327, 328, 329,
        330, 331, 332, 333, 334, 350, 351, 354, 355, 356, 357, 358, 359, 360,
        361, 362, 363, 383, 384, 385, 386, 387, 388, 389, 394, 395, 396, 410,
        411, 412, 413, 429, 430, 431, 432, 433, 434, 440, 441, 442, 443, 444,
        445, 446, 447, 461, 462, 463, 464, 465, 468, 469, 470, 471, 472, 473,
        474, 475, 476, 507, 508, 509, 510, 511, 512, 514, 515, 516, 517, 518,
        519, 520, 521, 522, 523, 524, 525, 530, 531, 532, 536, 537, 538, 548,
        549, 550, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565,
        566, 567, 568, 569, 570, 578, 579, 580, 581, 582, 590, 591, 592, 593,
        602, 603, 604, 605, 606, 611, 612, 616, 617, 618, 619, 620, 632, 633,
        634, 635, 636, 637, 643, 644, 649, 650, 655, 656, 657, 662, 663, 667,
        668, 669, 670, 671, 672, 675, 676, 677, 678, 679, 680, 685, 686, 687,
        688, 689, 692, 693, 696, 697, 698, 705, 706, 707, 708, 709, 710, 711,
        712, 716, 717, 723, 724, 725, 726, 727, 730, 731, 732, 733, 742, 743,
        758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771,
        772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785,
        786, 799, 800, 801, 802, 803, 809, 810, 811, 812, 813, 820, 821, 822,
        832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845,
        849, 850, 851, 856, 857, 858, 859, 860, 861, 868, 869, 870, 905, 906,
        907, 916, 917, 918, 925, 926, 927, 931, 932, 933, 934, 944, 945, 946,
        958, 959, 960, 961, 964, 965, 966, 969, 970, 971, 972, 973, 974, 975,
        976, 990, 991, 992, 993, 994, 1000, 1001, 1002, 1003, 1009, 1010,
        1011, 1020, 1021, 1022, 1031, 1032, 1033, 1037, 1038, 1047, 1048,
        1051, 1052, 1056, 1057, 1074, 1075, 1076, 1077, 1078, 1079, 1080,
        1081, 1082, 1083, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094,
        1095, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1105, 1106, 1107,
        1108, 1116, 1117, 1118, 1122, 1123, 1124, 1125, 1126, 1127, 1128,
        1129, 1130, 1134, 1135, 1143, 1144, 1145, 1146, 1147, 1148, 1149,
        1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1161, 1162, 1163,
        1164, 1165, 1166, 1169, 1170, 1173, 1174, 1175, 1178, 1179, 1180,
        1181, 1182, 1183, 1191, 1192, 1193, 1194, 1195, 1200, 1201, 1202,
        1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1215, 1216,
        1217, 1218, 1219, 1225, 1226, 1227, 1228, 1229, 1232, 1233, 1234,
        1246, 1247, 1248, 1249, 1253, 1254, 1255, 1256, 1257, 1258, 1259,
        1260, 1261, 1262, 1263, 1264, 1274, 1275, 1276, 1277, 1278, 1279,
        1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294,
        1296, 1297, 1298, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1313,
        1314, 1328, 1329, 1330, 1331, 1334, 1335, 1336, 1337, 1338, 1339,
        1346, 1347, 1348, 1352, 1353, 1354, 1355, 1363, 1364, 1367, 1368,
        1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1393, 1394, 1395,
        1396, 1397, 1398, 1399, 1400, 1406, 1407, 1408, 1409, 1410, 1411,
        1412, 1413, 1420, 1421, 1422, 1423, 1429, 1430, 1431, 1432, 1440,
        1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448,
    ]

    def __init__(
        self,
        root: str = NYUV2_MAT_PATH,
        image_shape: Tuple[int, int] = (480, 640),
        depth_scale: float = DEPTH_SCALE,
        use_lidar: bool = False,
        lidar_root: Optional[str] = None,
        lidar_depth_scale: float = LIDAR_DEPTH_SCALE,
        lidar_h5_key: Optional[str] = None,
        lidar_confidence_h5_key: Optional[str] = None,
        split: str = "train",
        split_indices_file: Optional[str] = None,
        val_ratio: float = 0.1,
        val_seed: int = 42,
        flip_aug: bool = False,
        image_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        return_intrinsics: bool = True,
    ) -> None:
        super().__init__()

        assert split in ("train", "val", "test", "all"), (
            f"split must be one of 'train', 'val', 'test', 'all', got '{split}'."
        )

        self.mat_path = root
        self.split = split
        self.flip_aug = flip_aug
        self.return_intrinsics = return_intrinsics
        self.image_shape = tuple(image_shape)
        self.depth_scale = depth_scale
        self.use_lidar = use_lidar
        self.lidar_root = Path(lidar_root) if lidar_root is not None else None
        self.lidar_depth_scale = lidar_depth_scale
        self.lidar_h5_key = lidar_h5_key
        self.lidar_confidence_h5_key = lidar_confidence_h5_key
        self.split_indices_file = split_indices_file
        self.val_ratio = float(val_ratio)
        self.val_seed = int(val_seed)

        self.image_transform = (
            image_transform if image_transform is not None else _default_image_transform(self.image_shape)
        )
        self.depth_transform = (
            depth_transform if depth_transform is not None else _default_depth_transform(self.image_shape)
        )

        # Read total count once, then close (fork-safety for DataLoader)
        h5 = _load_mat(self.mat_path)
        total = h5["images"].shape[0]  # (N, 3, W, H)

        if self.use_lidar:
            if self.lidar_h5_key is None:
                if "lidar_depths" in h5:
                    self.lidar_h5_key = "lidar_depths"
                elif "lidar" in h5:
                    self.lidar_h5_key = "lidar"

            if self.lidar_confidence_h5_key is None:
                if "lidar_confidence" in h5:
                    self.lidar_confidence_h5_key = "lidar_confidence"
                elif "lidar_confidences" in h5:
                    self.lidar_confidence_h5_key = "lidar_confidences"

            if self.lidar_h5_key is None and self.lidar_root is None:
                h5.close()
                raise ValueError(
                    "use_lidar=True but no LiDAR source found. "
                    "Provide lidar_root or a valid lidar_h5_key in the .mat file."
                )

            if self.lidar_h5_key is not None and self.lidar_h5_key not in h5:
                h5.close()
                raise KeyError(
                    f"LiDAR HDF5 key '{self.lidar_h5_key}' not found in {self.mat_path}."
                )

            if self.lidar_confidence_h5_key is not None and self.lidar_confidence_h5_key not in h5:
                h5.close()
                raise KeyError(
                    f"LiDAR confidence HDF5 key '{self.lidar_confidence_h5_key}' not found in {self.mat_path}."
                )
        h5.close()

        if split_indices_file is not None:
            self.indices = self._load_indices_from_file(split_indices_file, total)
        else:
            split_map = self.build_standard_train_val_test_splits(
                total=total,
                val_ratio=self.val_ratio,
                seed=self.val_seed,
            )
            if split == "all":
                self.indices = list(range(total))
            elif split in split_map:
                self.indices = split_map[split]
            else:
                raise ValueError(f"Unsupported split '{split}'.")

        # Lazy per-worker file handle
        self._h5: Optional[h5py.File] = None

    def _load_lidar_from_file(self, mat_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load LiDAR depth and optional confidence from external files.

        Supported file patterns under ``lidar_root``:
            - ``{idx:05d}.npy`` / ``{idx:04d}.npy`` / ``{idx}.npy``
            - ``{idx:05d}.npz`` / ``{idx:04d}.npz`` / ``{idx}.npz``

        For NPZ files, expected keys are one of:
            - depth: ``lidar_depth`` | ``depth`` | ``sparse_depth``
            - confidence: ``lidar_confidence`` | ``confidence``
        """
        if self.lidar_root is None:
            raise RuntimeError("LiDAR root is not configured.")

        candidates = [
            self.lidar_root / f"{mat_idx:05d}.npy",
            self.lidar_root / f"{mat_idx:04d}.npy",
            self.lidar_root / f"{mat_idx}.npy",
            self.lidar_root / f"{mat_idx:05d}.npz",
            self.lidar_root / f"{mat_idx:04d}.npz",
            self.lidar_root / f"{mat_idx}.npz",
        ]
        src = next((path for path in candidates if path.is_file()), None)
        if src is None:
            raise FileNotFoundError(
                f"No LiDAR file found for index {mat_idx} under '{self.lidar_root}'."
            )

        if src.suffix == ".npy":
            lidar_depth = np.load(src).astype(np.float32)
            lidar_conf = None
        else:
            obj = np.load(src)
            if "lidar_depth" in obj:
                lidar_depth = obj["lidar_depth"].astype(np.float32)
            elif "depth" in obj:
                lidar_depth = obj["depth"].astype(np.float32)
            elif "sparse_depth" in obj:
                lidar_depth = obj["sparse_depth"].astype(np.float32)
            else:
                raise KeyError(
                    f"NPZ file '{src}' does not contain one of ['lidar_depth', 'depth', 'sparse_depth']."
                )

            if "lidar_confidence" in obj:
                lidar_conf = obj["lidar_confidence"].astype(np.float32)
            elif "confidence" in obj:
                lidar_conf = obj["confidence"].astype(np.float32)
            else:
                lidar_conf = None

        return lidar_depth, lidar_conf

    @classmethod
    def build_standard_train_val_test_splits(
        cls,
        total: int,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, List[int]]:
        """
        Build deterministic train/val/test indices.

        - test: official Eigen split (654 images)
        - train/val: partition of non-test subset with fixed RNG seed
        """

        if total <= 0:
            return {"train": [], "val": [], "test": []}

        ratio = float(np.clip(val_ratio, 0.0, 0.5))
        test_set = {i for i in cls._EIGEN_TEST_INDICES if 0 <= i < total}
        all_indices = list(range(total))
        train_pool = [i for i in all_indices if i not in test_set]
        test_indices = sorted(i for i in all_indices if i in test_set)

        if len(train_pool) <= 1 or ratio <= 0.0:
            return {
                "train": sorted(train_pool),
                "val": [],
                "test": test_indices,
            }

        rng = np.random.default_rng(seed)
        shuffled = np.array(train_pool, dtype=np.int64)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * ratio))
        n_val = max(1, min(len(shuffled) - 1, n_val))

        val_indices = sorted(shuffled[:n_val].tolist())
        train_indices = sorted(shuffled[n_val:].tolist())
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

    @staticmethod
    def _load_indices_from_file(split_file: str, total: int) -> List[int]:
        path = Path(split_file)
        if not path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        indices: List[int] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            idx = int(line)
            if idx < 0 or idx >= total:
                raise ValueError(
                    f"Split index {idx} in '{split_file}' is out of range [0, {total - 1}]"
                )
            indices.append(idx)

        deduped = sorted(set(indices))
        if len(deduped) != len(indices):
            raise ValueError(f"Split file '{split_file}' contains duplicate indices.")
        return deduped

    def _load_lidar(self, h5: h5py.File, mat_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load LiDAR depth and optional confidence from HDF5 or external files.
        """
        if self.lidar_h5_key is not None:
            lidar_raw = h5[self.lidar_h5_key][mat_idx]
            if lidar_raw.ndim != 2:
                raise ValueError(
                    f"LiDAR depth at key '{self.lidar_h5_key}' must be 2D, got shape {lidar_raw.shape}."
                )

            # Align with NYUv2 HDF5 layout (W, H) -> (H, W) when needed.
            if lidar_raw.shape == (640, 480):
                lidar_depth = np.transpose(lidar_raw, (1, 0)).astype(np.float32)
            else:
                lidar_depth = lidar_raw.astype(np.float32)

            lidar_conf = None
            if self.lidar_confidence_h5_key is not None:
                lidar_conf_raw = h5[self.lidar_confidence_h5_key][mat_idx]
                if lidar_conf_raw.shape == (640, 480):
                    lidar_conf = np.transpose(lidar_conf_raw, (1, 0)).astype(np.float32)
                else:
                    lidar_conf = lidar_conf_raw.astype(np.float32)
            return lidar_depth, lidar_conf

        return self._load_lidar_from_file(mat_idx)

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

    def _make_sample(
        self,
        image_tensor: torch.Tensor,
        depth_tensor: torch.Tensor,
        depth_mask: torch.Tensor,
        lidar_depth_tensor: Optional[torch.Tensor],
        lidar_mask_tensor: Optional[torch.Tensor],
        lidar_confidence_tensor: Optional[torch.Tensor],
        flip: bool,
    ) -> Dict:
        
        """
        Build a single sample dict, optionally horizontal-flipped.
        """

        if flip:
            image_tensor = torch.flip(image_tensor, dims = [-1])
            depth_tensor = torch.flip(depth_tensor, dims = [-1])
            depth_mask   = torch.flip(depth_mask,   dims = [-1])
            if lidar_depth_tensor is not None:
                lidar_depth_tensor = torch.flip(lidar_depth_tensor, dims = [-1])
            if lidar_mask_tensor is not None:
                lidar_mask_tensor = torch.flip(lidar_mask_tensor, dims = [-1])
            if lidar_confidence_tensor is not None:
                lidar_confidence_tensor = torch.flip(lidar_confidence_tensor, dims = [-1])

        sample: Dict = {
            "image":      image_tensor,  # [3, H, W] float32
            "depth":      depth_tensor,  # [1, H, W] float32
            "depth_mask": depth_mask,    # [1, H, W] bool
            "flip":       flip,          # bool – consumed by collate_fn → img_metas
            "si":         False,         # scale-invariant flag (always False for NYUv2)
            "has_lidar":  lidar_depth_tensor is not None,
        }

        if lidar_depth_tensor is not None and lidar_mask_tensor is not None:
            sample["lidar_depth"] = lidar_depth_tensor
            sample["lidar_mask"] = lidar_mask_tensor
            if lidar_confidence_tensor is not None:
                sample["lidar_confidence"] = lidar_confidence_tensor

        if self.return_intrinsics:
            K = NYUV2_INTRINSICS.clone()  # [3, 3] float32
            if flip:
                # Horizontal flip maps pixel x -> W - x, so cx -> W - cx.
                # W is the image width AFTER transforms (may differ from 640
                # if image_shape was set to a non-default value).
                W_img = image_tensor.shape[-1]
                K = K.clone()
                K[0, 2] = W_img - K[0, 2]
            sample["K"] = K

        return sample

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
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
        # Scaling
        depth_np = depth_np * self.depth_scale if self.depth_scale != 1.0 else depth_np

        # Transforms
        image_tensor: torch.Tensor = self.image_transform(image_pil)

        depth_tensor: torch.Tensor = self.depth_transform(depth_np)
        depth_tensor = torch.clamp(depth_tensor, MIN_DEPTH, MAX_DEPTH)

        # Optional LiDAR sparse depth
        lidar_depth_tensor: Optional[torch.Tensor] = None
        lidar_mask_tensor: Optional[torch.Tensor] = None
        lidar_confidence_tensor: Optional[torch.Tensor] = None
        if self.use_lidar:
            lidar_depth_np, lidar_conf_np = self._load_lidar(h5, mat_idx)
            lidar_depth_np = (
                lidar_depth_np * self.lidar_depth_scale
                if self.lidar_depth_scale != 1.0
                else lidar_depth_np
            )

            lidar_depth_tensor_raw = self.depth_transform(lidar_depth_np)
            lidar_mask_tensor = (
                torch.isfinite(lidar_depth_tensor_raw)
                & (lidar_depth_tensor_raw > MIN_DEPTH)
            )
            lidar_depth_tensor = torch.clamp(lidar_depth_tensor_raw, MIN_DEPTH, MAX_DEPTH)
            lidar_depth_tensor = torch.where(
                lidar_mask_tensor,
                lidar_depth_tensor,
                torch.zeros_like(lidar_depth_tensor),
            )

            if lidar_conf_np is not None:
                lidar_confidence_tensor = self.depth_transform(lidar_conf_np)
                lidar_confidence_tensor = torch.clamp(lidar_confidence_tensor, min = 0.0)
                lidar_confidence_tensor = torch.where(
                    lidar_mask_tensor,
                    lidar_confidence_tensor,
                    torch.zeros_like(lidar_confidence_tensor),
                )
            else:
                lidar_confidence_tensor = lidar_mask_tensor.float()

        if self.split == "test":
            depth_tensor = self._apply_eval_mask(depth_tensor)

        # Depth mask (valid pixels only). For test split this naturally excludes
        # regions outside eval crop because they are zeroed by _apply_eval_mask.
        depth_mask = (
            torch.isfinite(depth_tensor)
            & (depth_tensor > MIN_DEPTH)
            & (depth_tensor <= MAX_DEPTH)
        )

        if self.flip_aug:
            # Return (original, horizontally-flipped) pair so the collate_fn
            # can interleave them into [orig0, flip0, orig1, flip1, ...] batches
            # required by the SelfDistill invariance loss.
            original = self._make_sample(
                image_tensor,
                depth_tensor,
                depth_mask,
                lidar_depth_tensor,
                lidar_mask_tensor,
                lidar_confidence_tensor,
                flip = False,
            )
            flipped  = self._make_sample(
                image_tensor,
                depth_tensor,
                depth_mask,
                lidar_depth_tensor,
                lidar_mask_tensor,
                lidar_confidence_tensor,
                flip = True,
            )
            return original, flipped
        else:
            return self._make_sample(
                image_tensor,
                depth_tensor,
                depth_mask,
                lidar_depth_tensor,
                lidar_mask_tensor,
                lidar_confidence_tensor,
                flip = False,
            )

    def __repr__(self) -> str:
        return (
            f"NYUv2Dataset(split='{self.split}', "
            f"flip_aug={self.flip_aug}, "
            f"n_samples={len(self)}, "
            f"mat_path='{self.mat_path}')"
        )
    
    @classmethod
    def collate_fn(cls, batch: List) -> Dict:
        """
        Collate function that handles both:

        * **Regular samples** - ``batch`` is a list of dicts (``flip_aug=False``).
        * **Paired samples** - ``batch`` is a list of ``(original_dict, flipped_dict)``
          tuples (``flip_aug=True``).  The originals and their flips are interleaved
          so that consecutive batch positions are always ``(orig_i, flip_i)`` from
          the same scene - the layout expected by the SelfDistill invariance loss.

        ``flip`` and ``si`` fields are extracted from each sample and returned
        in ``img_metas`` as a list of per-sample dicts; they are NOT included in
        the ``data`` tensor dict.
        """
        # unpack paired samples
        if batch and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            flat: List[Dict] = []
            for orig, flipped in batch:
                flat.append(orig)
                flat.append(flipped)
            batch = flat

        # separate metadata fields from tensor fields
        META_KEYS = {"flip", "si"}
        img_metas: List[Dict] = [
            {k: item[k] for k in META_KEYS if k in item}
            for item in batch
        ]

        # stack tensor fields
        data_keys = [k for k in batch[0].keys() if k not in META_KEYS]
        collated: Dict = {}
        for key in data_keys:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                collated[key] = torch.stack(vals, dim = 0)
            else:
                collated[key] = vals

        return {
            "data":      collated,
            "img_metas": img_metas,
        }


# Test code
if __name__ == "__main__":

    # Quick test: load the first sample from the training split
    dataset = NYUv2Dataset(split = "train", return_intrinsics = True)
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = True, 
                            collate_fn = NYUv2Dataset.collate_fn)
    sample = next(iter(dataloader))["data"]
    print("Sample keys:", sample.keys())
    print("Image shape:", sample["image"].shape)
    print("Depth shape:", sample["depth"].shape)
    print("Depth mask shape:", sample["depth_mask"].shape)
    print("K shape:", sample["K"].shape)

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
    fig.savefig("./datasets/nyuv2_sample.png")