"""
train_depth.py - Single-GPU training script for UniDepthV1.

Usage:
    python -m train.train_depth <args>
    (see run_script/run_train_demo.sh for full configuration)

Features:
    - No distributed training (single GPU or CPU fallback)
    - TensorBoard logging (loss curves, LR, sample depth images)
    - AdamW optimizer + CosineAnnealingLR scheduler
    - Checkpointing every N epochs
    - Experiment outputs saved under ./runs/
"""

import argparse
import os
import shutil
from time import time
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from data.nyuv2_dataset import NYUv2Dataset
from model.unidepthv1.unidepthv1 import UniDepthV1
from utils.camera import Pinhole
from utils.visualization import colorize


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UniDepthV1 on a demo dataset.")

    # General training settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5,
                        help='Encoder LR (default: 10x lower than decoder)')
    parser.add_argument('--layer_decay', type=float, default=0.9,
                        help='Layer-wise LR decay for encoder')
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Linear LR warmup steps (0 to disable)')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0,
                        help='Freeze encoder for first N epochs')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable mixed precision (AMP) to reduce GPU memory usage')

    # Model architecture
    parser.add_argument('--encoder_name', type=str, default='convnextv2_large')
    parser.add_argument('--pretrained', type=str, default="")
    parser.add_argument('--output_idx', type=int, nargs='+', default=None)
    parser.add_argument('--use_checkpoint', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--use_lidar_fusion', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--lidar_fusion_type', type=str, default='late', choices=['late', 'token'], help='Fusion type: late (1/16 scale) or token (multi-scale)')
    parser.add_argument('--lidar_dropout_prob', type=float, default=0.0)
    parser.add_argument('--phase4_eval_fallback', type=lambda x: x.lower() == 'true', default=True)
    
    # Phase 5 ablation configuration
    parser.add_argument('--phase5_ablation', type=str, default=None, 
                        choices=['rgb_only', 'supervision_only', 'late_fusion', 'token_fusion'],
                        help='Phase 5 ablation: train specific variant')

    # Loss configuration
    parser.add_argument('--depth_loss_name', type=str, default='SILog')
    parser.add_argument('--depth_loss_weight', type=float, default=10.0)
    parser.add_argument('--camera_loss_name', type=str, default='Regression')
    parser.add_argument('--camera_loss_weight', type=float, default=0.5)
    parser.add_argument('--invariance_loss_name', type=str, default='SelfDistill')
    parser.add_argument('--invariance_loss_weight', type=float, default=0.1)
    parser.add_argument('--lidar_loss_weight', type=float, default=0.5)

    # Data configuration
    parser.add_argument('--train_root', type=str, default=None)
    parser.add_argument('--val_root', type=str, default=None)
    parser.add_argument('--train_split', type=str, default='train', help='Training split name')
    parser.add_argument('--val_split', type=str, default='val', help='Validation split name (falls back to test if val does not exist)')
    parser.add_argument('--test_split', type=str, default='test', help='Test split name (for final evaluation only, not used during training)')
    parser.add_argument('--image_shape', type=int, nargs=2, default=[384, 384])
    parser.add_argument('--depth_scale', type=float, default=1.0)
    parser.add_argument('--use_lidar', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--lidar_root', type=str, default=None)
    parser.add_argument('--lidar_depth_scale', type=float, default=1.0)
    parser.add_argument('--lidar_h5_key', type=str, default=None)
    parser.add_argument('--lidar_confidence_h5_key', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=0, help='Use first N training samples (0=all)')
    parser.add_argument('--max_val_samples', type=int, default=0, help='Use first N validation samples (0=all)')
    parser.add_argument('--datasets', type=str, default='nyuv2',
                        help='Comma-separated training dataset names (e.g. "nyuv2,sunrgbd,vkitti2,sintel")')
    parser.add_argument('--dataset_roots', type=str, default=None,
                        help='Comma-separated roots matching --datasets (uses defaults if omitted)')

    # Checkpoint resume
    parser.add_argument('--resume', type=str, default=None)

    # Script path for copying to experiment folder
    parser.add_argument('--script_path', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None, help='Optional output run folder name under runs/')

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:

    """
    Build the nested config dict expected by UniDepthV1 from 
    flat argparse args.
    """

    return {
        "model": {
            "name": "UniDepthV1",
            "pixel_encoder": {
                "name": args.encoder_name,
                "pretrained": args.pretrained if hasattr(args, "pretrained") and args.pretrained else None,
                # If output_idx is None, don't set this key
                **({"output_idx": args.output_idx} if args.output_idx is not None else {}),
                "use_checkpoint": args.use_checkpoint,
                "lr": args.encoder_lr,
            },
            "pixel_decoder": {
                "name": "Decoder",
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "depths": args.depths,
                "use_lidar_fusion": args.use_lidar_fusion,
                "lidar_fusion_type": args.lidar_fusion_type,
            },
            "num_heads": args.num_heads,
            "expansion": args.expansion,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "ld": args.layer_decay,
            "lr_min": args.lr_min,
            "wd": args.weight_decay,
            "log_every": args.log_every,
            "save_every": args.save_every,
            "lidar_loss_weight": args.lidar_loss_weight,
            "losses": {
                "depth": {
                    "name": args.depth_loss_name,
                    "weight": args.depth_loss_weight,
                    "output_fn": "sqrt",
                    "input_fn": "log",
                    "dims": [-2, -1],
                    "integrated": 0.15,
                },
                "camera": {
                    "name": args.camera_loss_name,
                    "weight": args.camera_loss_weight,
                    "output_fn": "sqrt",
                    "input_fn": "linear",
                    "dims": [-1],
                    "fn": "charbonnier",
                    "alpha": 1.0,
                    "gamma": 0.01,
                },
                "invariance": {
                    "name": args.invariance_loss_name,
                    "weight": args.invariance_loss_weight,
                    "output_fn": "sqrt",
                },
            },
        },
        "data": {
            "train_root": args.train_root,
            "val_root": args.val_root,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "image_shape": args.image_shape,
            "depth_scale": args.depth_scale,
            "use_lidar": args.use_lidar,
            "lidar_root": args.lidar_root,
            "lidar_depth_scale": args.lidar_depth_scale,
            "lidar_h5_key": args.lidar_h5_key,
            "lidar_confidence_h5_key": args.lidar_confidence_h5_key,
            "num_workers": args.num_workers,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "lidar_dropout_prob": args.lidar_dropout_prob,
            "phase4_eval_fallback": args.phase4_eval_fallback,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────────────────────────

DATASET_DEFAULT_ROOTS = {
    "nyuv2": "datasets/nyu_depth_v2_labeled.mat",
    "sunrgbd": "datasets/SUNRGBD",
    "vkitti2": "datasets/virtual_kitti_2",
    "sintel": "datasets/unidepth_data",
}


def _unified_collate_fn(batch):
    """Collate function compatible with all dataset outputs."""
    # Unpack paired (original, flipped) tuples from flip_aug=True datasets
    if batch and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
        flat = []
        for orig, flipped in batch:
            flat.append(orig)
            flat.append(flipped)
        batch = flat

    META_KEYS = {"flip", "si"}
    img_metas = [{k: item[k] for k in META_KEYS if k in item} for item in batch]
    data_keys = [k for k in batch[0].keys() if k not in META_KEYS]
    collated = {}
    for key in data_keys:
        vals = [item[key] for item in batch]
        collated[key] = torch.stack(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
    return {"data": collated, "img_metas": img_metas}


def build_dataset(name: str, split: str, data_cfg: dict, root_override: str = None,
                  flip_aug: bool = False, extra_kwargs: dict = None):
    """Build a dataset by name with common parameters."""
    root = root_override or DATASET_DEFAULT_ROOTS.get(name)
    image_shape = data_cfg["image_shape"]
    depth_scale = data_cfg.get("depth_scale", 1.0)

    if name == "nyuv2":
        return NYUv2Dataset(
            root=root or data_cfg.get("train_root"),
            split=split,
            image_shape=image_shape,
            depth_scale=depth_scale,
            use_lidar=data_cfg.get("use_lidar", False),
            lidar_root=data_cfg.get("lidar_root"),
            lidar_depth_scale=data_cfg.get("lidar_depth_scale", 1.0),
            lidar_h5_key=data_cfg.get("lidar_h5_key"),
            lidar_confidence_h5_key=data_cfg.get("lidar_confidence_h5_key"),
            flip_aug=flip_aug,
        )
    elif name == "sunrgbd":
        from data.sunrgbd_dataset import SUNRGBDDataset
        return SUNRGBDDataset(root=root, split=split, image_shape=image_shape,
                              depth_scale=depth_scale, flip_aug=flip_aug)
    elif name == "vkitti2":
        from data.vkitti2_dataset import VirtualKITTI2Dataset
        return VirtualKITTI2Dataset(root=root, split=split, image_shape=image_shape,
                                    depth_scale=depth_scale, flip_aug=flip_aug)
    elif name == "sintel":
        from data.sintel_dataset import SintelDataset
        return SintelDataset(root=root, split=split, image_shape=image_shape,
                             depth_scale=depth_scale, flip_aug=flip_aug)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_DEFAULT_ROOTS.keys())}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_camera_from_batch(K: torch.Tensor) -> Pinhole:

    """
    Create a Pinhole camera object from a [B, 3, 3] intrinsics batch.
    """

    params = torch.stack(
        [K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], dim=-1
    )
    return Pinhole(params=params, K=K)


def log_depth_images(writer: SummaryWriter, tag: str, depth: torch.Tensor, step: int, n: int = 4):
    
    """
    Write up to *n* colorized depth images to TensorBoard.
    """

    depth_np = depth[:n, 0].detach().cpu().float().numpy()  # [n, H, W]
    for i, d in enumerate(depth_np):
        colored = colorize(d)  # [H, W, 3] uint8
        # TensorBoard expects [C, H, W] or [H, W, C] depending on format
        writer.add_image(f"{tag}/{i}", colored.transpose(2, 0, 1), step)


# ImageNet normalization constants (used to denormalize for TensorBoard display)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def log_rgb_images(writer: SummaryWriter, tag: str, image: torch.Tensor, step: int, n: int = 4):
    """Write up to *n* RGB images (ImageNet-normalized) to TensorBoard."""
    imgs = image[:n].detach().cpu().float()
    imgs = imgs * _IMAGENET_STD + _IMAGENET_MEAN  # denormalize
    imgs = imgs.clamp(0, 1)
    for i, img in enumerate(imgs):
        writer.add_image(f"{tag}/{i}", img, step)  # [3, H, W] float 0-1


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    torch.save(state, path)
    print(f"  Checkpoint saved -> {path}")


def compute_lidar_sparse_loss(
    pred_depth: torch.Tensor,
    lidar_depth: torch.Tensor,
    lidar_mask: torch.Tensor,
    lidar_confidence: torch.Tensor = None,
    eps: float = 1e-6,
):

    """
    Phase 2 sparse LiDAR supervision:
        weighted mean(|log(pred) - log(lidar)|) on valid sparse pixels.
    """

    valid = (
        lidar_mask.bool()
        & torch.isfinite(pred_depth)
        & torch.isfinite(lidar_depth)
        & (pred_depth > eps)
        & (lidar_depth > eps)
    )
    if not torch.any(valid):
        return None, {
            "valid_ratio": 0.0,
            "valid_pixels": 0,
        }

    if lidar_confidence is not None:
        weights = torch.clamp(lidar_confidence, min = 0.0) * valid.float()
    else:
        weights = valid.float()

    weight_sum = weights.sum()
    if weight_sum <= 0:
        return None, {
            "valid_ratio": float(valid.float().mean().item()),
            "valid_pixels": int(valid.sum().item()),
        }

    pred_log = torch.log(torch.clamp(pred_depth, min = eps))
    lidar_log = torch.log(torch.clamp(lidar_depth, min = eps))
    abs_log_diff = torch.abs(pred_log - lidar_log)

    sparse_loss = (abs_log_diff * weights).sum() / weight_sum
    stats = {
        "valid_ratio": float(valid.float().mean().item()),
        "valid_pixels": int(valid.sum().item()),
    }
    return sparse_loss, stats


def compute_depth_rmse(pred_depth: torch.Tensor, gt_depth: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    valid = gt_mask.bool()
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred_depth.device)
    mse = ((pred_depth[valid] - gt_depth[valid]) ** 2).mean()
    return torch.sqrt(mse)


def compute_depth_abs_rel(pred_depth: torch.Tensor, gt_depth: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    valid = gt_mask.bool() & torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred_depth.device)
    denom = torch.clamp(gt_depth[valid], min=eps)
    abs_rel = torch.abs(pred_depth[valid] - gt_depth[valid]) / denom
    return abs_rel.mean()


def compute_delta1(pred_depth: torch.Tensor, gt_depth: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute delta1 (% of pixels where max(pred/gt, gt/pred) < 1.25)."""
    valid = gt_mask.bool() & (pred_depth > eps) & (gt_depth > eps)
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred_depth.device)
    ratio = torch.max(pred_depth[valid] / gt_depth[valid],
                      gt_depth[valid] / pred_depth[valid])
    return (ratio < 1.25).float().mean()


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    config = build_config(args)
    
    # Apply Phase 5 ablation configurations
    if args.phase5_ablation:
        print(f"\n🔄 Applying Phase 5 ablation: {args.phase5_ablation}")
        if args.phase5_ablation == 'rgb_only':
            # RGB only: no LiDAR
            config["data"]["use_lidar"] = False
            config["model"]["pixel_decoder"]["use_lidar_fusion"] = False
        elif args.phase5_ablation == 'supervision_only':
            # RGB + LiDAR supervision (no fusion)
            config["model"]["pixel_decoder"]["use_lidar_fusion"] = False
        elif args.phase5_ablation == 'late_fusion':
            # RGB + LiDAR with late fusion
            config["model"]["pixel_decoder"]["use_lidar_fusion"] = True
            config["model"]["pixel_decoder"]["lidar_fusion_type"] = "late"
        elif args.phase5_ablation == 'token_fusion':
            # RGB + LiDAR with token fusion
            config["model"]["pixel_decoder"]["use_lidar_fusion"] = True
            config["model"]["pixel_decoder"]["lidar_fusion_type"] = "token"

    print("Arguments:")
    for arg in vars(args):
        print(f"  \033[1m{arg}:\033[0m {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set Device
    device = torch.device(
        f"cuda:{args.cuda}" if (args.cuda is not None and args.cuda >= 0 and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    # Experiment output directory under ./runs/
    if args.run_name:
        log_dir = args.run_name if args.run_name.startswith("runs/") else f"runs/{args.run_name}"
    else:
        log_dir = f"runs/train_depth_{int(time() * 1000)}_{os.getpid()}"
    os.makedirs(log_dir, exist_ok = True)
    print(f"\n\033[1mLogging to {log_dir}\033[0m")
    tensorboard_dir = f"{log_dir}/tensorboard"
    ckpt_dir = f"{log_dir}/checkpoints"
    os.makedirs(tensorboard_dir, exist_ok = True)
    os.makedirs(ckpt_dir, exist_ok = True)

    # Copy run script for reproducibility
    if args.script_path and os.path.isfile(args.script_path):
        shutil.copy(args.script_path, f"{log_dir}/run_script.sh")
        print(f"\033[1mSaved launch script to {log_dir}/run_script.sh\033[0m")

    # Set up model
    model = UniDepthV1(config)
    model.to(device)

    # Set up Datasets & DataLoaders
    print("\n>>> Setting up datasets and dataloaders >>>")

    train_cfg = config["training"]
    data_cfg = config["data"]

    # Multi-dataset training support
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    dataset_roots_override = {}
    if args.dataset_roots:
        roots = [r.strip() for r in args.dataset_roots.split(",")]
        for name, root in zip(dataset_names, roots):
            if root:
                dataset_roots_override[name] = root
    # Use train_root as override for nyuv2 if specified
    if data_cfg.get("train_root") and "nyuv2" in dataset_names:
        dataset_roots_override.setdefault("nyuv2", data_cfg["train_root"])

    train_datasets = []
    for ds_name in dataset_names:
        ds = build_dataset(
            ds_name,
            split="train",
            data_cfg=data_cfg,
            root_override=dataset_roots_override.get(ds_name),
            flip_aug=True,
        )
        train_datasets.append(ds)
        print(f"  {ds_name}: {len(ds)} train samples")

    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
        print(f"  Combined: {len(train_dataset)} total train samples")

    max_train = data_cfg.get("max_train_samples", 0)
    train_sampler = None
    train_shuffle = True
    if max_train and max_train > 0 and max_train < len(train_dataset):
        train_sampler = RandomSampler(train_dataset, replacement=False,
                                      num_samples=max_train)
        train_shuffle = False  # sampler and shuffle are mutually exclusive
        print(f"  Sampling {max_train}/{len(train_dataset)} per epoch (RandomSampler)")
    num_workers = data_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size = train_cfg["batch_size"],
        shuffle = train_shuffle,
        sampler = train_sampler,
        num_workers = num_workers,
        pin_memory = device.type == "cuda",
        drop_last = True,
        collate_fn = _unified_collate_fn,
        persistent_workers = num_workers > 0,
    )

    # Optional validation loader (always NYUv2 for consistency)
    val_loader = None
    if data_cfg.get("val_root") is not None:
        val_split_name = data_cfg.get("val_split", "val")
        try:
            val_dataset = NYUv2Dataset(
                root = data_cfg["val_root"],
                split = val_split_name,
                image_shape = data_cfg["image_shape"],
                depth_scale = data_cfg.get("depth_scale", 1.0),
                use_lidar = data_cfg.get("use_lidar", False),
                lidar_root = data_cfg.get("lidar_root", None),
                lidar_depth_scale = data_cfg.get("lidar_depth_scale", 1.0),
                lidar_h5_key = data_cfg.get("lidar_h5_key", None),
                lidar_confidence_h5_key = data_cfg.get("lidar_confidence_h5_key", None),
            )
        except ValueError:
            test_split_name = data_cfg.get("test_split", "test")
            val_dataset = NYUv2Dataset(
                root = data_cfg["val_root"],
                split = test_split_name,
                image_shape = data_cfg["image_shape"],
                depth_scale = data_cfg.get("depth_scale", 1.0),
                use_lidar = data_cfg.get("use_lidar", False),
                lidar_root = data_cfg.get("lidar_root", None),
                lidar_depth_scale = data_cfg.get("lidar_depth_scale", 1.0),
                lidar_h5_key = data_cfg.get("lidar_h5_key", None),
                lidar_confidence_h5_key = data_cfg.get("lidar_confidence_h5_key", None),
            )
            print(f"\033[93m[WARN] val_split '{val_split_name}' not available; fell back to test_split '{test_split_name}'\033[0m")
        if data_cfg.get("max_val_samples", 0) and data_cfg["max_val_samples"] > 0:
            val_dataset = Subset(val_dataset, range(min(data_cfg["max_val_samples"], len(val_dataset))))
        val_loader = DataLoader(
            val_dataset,
            batch_size = train_cfg["batch_size"],
            shuffle = False,
            num_workers = num_workers,
            pin_memory = device.type == "cuda",
            collate_fn = _unified_collate_fn,
            persistent_workers = num_workers > 0,
        )

    # verbose
    print(f"\033[1mTrain samples:\033[0m {len(train_dataset)}")
    print(f"\033[1mLiDAR enabled:\033[0m {data_cfg.get('use_lidar', False)}")
    print(f"\033[1mLiDAR fusion enabled:\033[0m {config['model']['pixel_decoder'].get('use_lidar_fusion', False)}")
    if val_loader:
        print(f"\033[1mVal samples:\033[0m   {len(val_dataset)}")

    # Set up Optimizer
    # Use model.get_params() for layer-wise LR decay (encoder vs decoder)
    try:
        param_groups = model.get_params(config)
        print(f"\033[92m[OK] model.get_params() succeeded — using separate encoder/decoder LRs\033[0m")
        for i, pg in enumerate(param_groups):
            print(f"  param_group[{i}]: lr={pg.get('lr', 'default')}, #params={len(pg['params'])}, wd={pg.get('weight_decay', 'default')}")
    except Exception as e:
        print(f"\033[93m[WARN] model.get_params() failed: {e}")
        print(f"  Falling back to uniform LR for all parameters — encoder may overfit!\033[0m")
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr = train_cfg["lr"],
        weight_decay = train_cfg["wd"],
    )

    # Set up LR Scheduler (warmup + cosine annealing)
    num_epochs = train_cfg["epochs"]
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = num_epochs,
        eta_min = train_cfg.get("lr_min", 1e-6),
    )
    warmup_steps = args.warmup_steps
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_steps,
        )
        # Warmup operates per-step; cosine operates per-epoch.
        # We'll step warmup manually and switch to cosine after warmup.
        using_warmup = True
        warmup_done = False
        warmup_step_count = 0
        print(f"\033[1mLR warmup: {warmup_steps} steps\033[0m")
    else:
        using_warmup = False
        warmup_done = True
    scheduler = cosine_scheduler

    # Mixed precision (AMP) — enable with --amp when GPU memory is limited
    use_amp = args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"\033[1mMixed precision (AMP): {use_amp}\033[0m")

    # Gradient accumulation
    accum_steps = args.accum_steps
    if accum_steps > 1:
        print(f"\033[1mGradient accumulation: {accum_steps} steps (effective batch = {train_cfg['batch_size'] * accum_steps * 2})\033[0m")

    # Encoder freezing
    freeze_encoder_epochs = args.freeze_encoder_epochs

    # OPTIONAL: Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        print(f"\n>>> Resuming from checkpoint: {args.resume} >>>")
        ckpt = torch.load(args.resume, map_location = device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"\033[92mResumed at epoch {start_epoch}, step {global_step}\033[0m")

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir = tensorboard_dir)
    print(f"\n\033[1mTensorBoard logs -> {tensorboard_dir}\033[0m")

    # Config logging and checkpointing intervals
    log_every = train_cfg.get("log_every", 50)
    save_every = train_cfg.get("save_every", 1)

    # ############# #
    # Training loop #
    # ############# #
    for epoch in range(start_epoch, num_epochs):

        # Encoder freezing for first N epochs
        if freeze_encoder_epochs > 0:
            if epoch < freeze_encoder_epochs:
                if epoch == start_epoch:
                    for p in model.pixel_encoder.parameters():
                        p.requires_grad = False
                    print(f"\033[93m[FREEZE] Encoder frozen for epochs 0-{freeze_encoder_epochs - 1}\033[0m")
            elif epoch == freeze_encoder_epochs:
                for p in model.pixel_encoder.parameters():
                    p.requires_grad = True
                print(f"\033[92m[UNFREEZE] Encoder unfrozen at epoch {epoch}\033[0m")

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        lidar_epoch_valid_ratio_sum = 0.0
        lidar_epoch_steps = 0
        lidar_dropout_ratio_sum = 0.0
        lidar_gate_mean_sum = 0.0
        lidar_fusion_steps = 0
        current_lr = optimizer.param_groups[0]["lr"]

        for batch_idx, batch in tqdm(enumerate(train_loader), 
                                     total = len(train_loader),
                                     desc = f"Epoch {epoch + 1}/{num_epochs}",
                                     unit = "batch"):
            
            # Move tensors to device.
            # With flip_aug=True the collate_fn has already interleaved
            # (original, flipped) pairs: [orig0, flip0, orig1, flip1, ...].
            # Each consecutive pair is the same scene under different flips,
            # which is what SelfDistill expects.
            image = batch['data']["image"].to(device)            # [2B, 3, H, W]
            depth = batch['data']["depth"].to(device)            # [2B, 1, H, W]
            depth_mask = batch['data']["depth_mask"].to(device)  # [2B, 1, H, W]
            K = batch['data']["K"].to(device)                    # [2B, 3, 3]

            lidar_depth = batch['data'].get("lidar_depth", None)
            lidar_mask = batch['data'].get("lidar_mask", None)
            lidar_confidence = batch['data'].get("lidar_confidence", None)
            if lidar_depth is not None:
                lidar_depth = lidar_depth.to(device)
            if lidar_mask is not None:
                lidar_mask = lidar_mask.to(device)
            if lidar_confidence is not None:
                lidar_confidence = lidar_confidence.to(device)

            # Phase 3: LiDAR dropout (sample-level), improves RGB-only robustness.
            lidar_dropout_prob = float(data_cfg.get("lidar_dropout_prob", 0.0))
            if lidar_depth is not None and lidar_mask is not None and lidar_dropout_prob > 0.0:
                keep = (torch.rand((image.shape[0], 1, 1, 1), device=device) >= lidar_dropout_prob)
                keep_f = keep.float()
                lidar_depth = lidar_depth * keep_f
                lidar_mask = lidar_mask & keep
                if lidar_confidence is not None:
                    lidar_confidence = lidar_confidence * keep_f
                lidar_dropout_ratio_sum += float((~keep).float().mean().item())

            # Build Pinhole camera with per-sample intrinsics (cx already updated
            # for flipped samples by the dataset's _make_sample method).
            camera = build_camera_from_batch(K)

            # Prepare inputs dict as expected by UniDepthV1
            inputs = {
                "image": image,
                "depth": depth,
                "depth_mask": depth_mask,
                "camera": camera,
            }
            if lidar_depth is not None and lidar_mask is not None:
                inputs["lidar_depth"] = lidar_depth
                inputs["lidar_mask"] = lidar_mask
                if lidar_confidence is not None:
                    inputs["lidar_confidence"] = lidar_confidence

            # image_metas carry the flip / si flags set per-sample by the dataset
            image_metas = batch['img_metas']

            # Forward pass
            if accum_steps <= 1:
                optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs, losses = model.forward(inputs, image_metas)

            # Compute total loss
            lidar_raw_loss = None
            lidar_stats = None
            lidar_weight = train_cfg.get("lidar_loss_weight", 0.0)
            if lidar_weight > 0.0 and lidar_depth is not None and lidar_mask is not None and "depth" in outputs:
                lidar_raw_loss, lidar_stats = compute_lidar_sparse_loss(
                    pred_depth = outputs["depth"],
                    lidar_depth = lidar_depth,
                    lidar_mask = lidar_mask,
                    lidar_confidence = lidar_confidence,
                )
                if lidar_raw_loss is not None:
                    losses["opt"]["LiDARSparse"] = lidar_weight * lidar_raw_loss

            total_loss = sum(losses["opt"].values())
            if not torch.isfinite(total_loss):
                print(f"  [WARNING] Non-finite loss at step {global_step}, skipping.")
                continue

            # Backward (with gradient accumulation)
            if accum_steps > 1:
                total_loss = total_loss / accum_steps
            scaler.scale(total_loss).backward()

            if accum_steps <= 1 or (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Step warmup scheduler per optimizer step
                if using_warmup and not warmup_done:
                    warmup_scheduler.step()
                    warmup_step_count += 1
                    if warmup_step_count >= warmup_steps:
                        warmup_done = True
                        print(f"\033[92m[WARMUP] Warmup complete at step {global_step}\033[0m")

            epoch_loss += total_loss.item() * (accum_steps if accum_steps > 1 else 1)
            num_batches += 1
            global_step += 1

            # ── Logging ───────────────────────────────────────────────────
            if global_step % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                # Log total and per-loss values
                writer.add_scalar("train/loss_total", total_loss.item(), global_step)
                for loss_name, loss_val in losses["opt"].items():
                    writer.add_scalar(f"train/loss_{loss_name}", loss_val.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                if lidar_raw_loss is not None and lidar_stats is not None:
                    writer.add_scalar("train/lidar_loss_raw", lidar_raw_loss.item(), global_step)
                    writer.add_scalar("train/lidar_valid_ratio", lidar_stats["valid_ratio"], global_step)
                    writer.add_scalar("train/lidar_valid_pixels", lidar_stats["valid_pixels"], global_step)
                    lidar_epoch_valid_ratio_sum += lidar_stats["valid_ratio"]
                    lidar_epoch_steps += 1

                fusion_stats = outputs.get("fusion_stats", None)
                if fusion_stats is not None:
                    writer.add_scalar("train/fusion_lidar_used", float(fusion_stats["lidar_used"].item()), global_step)
                    writer.add_scalar("train/fusion_lidar_valid_ratio", float(fusion_stats["lidar_valid_ratio"].item()), global_step)
                    writer.add_scalar("train/fusion_lidar_gate_mean", float(fusion_stats["lidar_gate_mean"].item()), global_step)
                    if float(fusion_stats["lidar_used"].item()) > 0.0:
                        lidar_gate_mean_sum += float(fusion_stats["lidar_gate_mean"].item())
                        lidar_fusion_steps += 1

                # Log sample predicted and GT depth images, and original RGB
                if "depth" in outputs:
                    log_depth_images(writer, "train/pred_depth", outputs["depth"], global_step)
                log_depth_images(writer, "train/gt_depth", depth, global_step)
                log_rgb_images(writer, "train/input_rgb", image, global_step)

        # ── End-of-epoch ─────────────────────────────────────────────────
        avg_loss = epoch_loss / max(num_batches, 1)
        writer.add_scalar("epoch/train_loss", avg_loss, epoch + 1)
        if lidar_epoch_steps > 0:
            writer.add_scalar(
                "epoch/train_lidar_valid_ratio",
                lidar_epoch_valid_ratio_sum / lidar_epoch_steps,
                epoch + 1,
            )
        if num_batches > 0:
            writer.add_scalar(
                "epoch/train_lidar_dropout_ratio",
                lidar_dropout_ratio_sum / num_batches,
                epoch + 1,
            )
        if lidar_fusion_steps > 0:
            writer.add_scalar(
                "epoch/train_fusion_lidar_gate_mean",
                lidar_gate_mean_sum / lidar_fusion_steps,
                epoch + 1,
            )
        print(f"\033[1mEpoch [{epoch+1}/{num_epochs}] avg loss: {avg_loss:.4f} - LR: {current_lr:.6f}\033[0m")

        # Step the LR scheduler
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                val_lidar_valid_ratio_sum = 0.0
                val_lidar_steps = 0
                val_fusion_gate_mean_sum = 0.0
                val_fusion_steps = 0
                val_rmse_sum = 0.0
                val_abs_rel_sum = 0.0
                val_delta1_sum = 0.0
                val_metric_steps = 0
                val_rmse_with_lidar_sum = 0.0
                val_rmse_rgb_only_sum = 0.0
                val_rmse_compare_steps = 0
                for batch in val_loader:
                    image = batch["data"]["image"].to(device)
                    depth = batch["data"]["depth"].to(device)
                    depth_mask = batch["data"]["depth_mask"].to(device)
                    K = batch["data"]["K"].to(device)
                    lidar_depth = batch["data"].get("lidar_depth", None)
                    lidar_mask = batch["data"].get("lidar_mask", None)
                    lidar_confidence = batch["data"].get("lidar_confidence", None)
                    if lidar_depth is not None:
                        lidar_depth = lidar_depth.to(device)
                    if lidar_mask is not None:
                        lidar_mask = lidar_mask.to(device)
                    if lidar_confidence is not None:
                        lidar_confidence = lidar_confidence.to(device)
                    camera = build_camera_from_batch(K)
                    inputs = {
                        "image": image,
                        "depth": depth,
                        "depth_mask": depth_mask,
                        "camera": camera,
                    }
                    if lidar_depth is not None and lidar_mask is not None:
                        inputs["lidar_depth"] = lidar_depth
                        inputs["lidar_mask"] = lidar_mask
                        if lidar_confidence is not None:
                            inputs["lidar_confidence"] = lidar_confidence
                    image_metas = batch["img_metas"]
                    # model is in eval() mode: forward() dispatches to forward_test
                    # which does NOT compute losses. We use forward_train explicitly
                    # so we can still get loss values for monitoring.
                    outputs_val, losses_val = model.forward_train(inputs, image_metas, force_compute_losses = True)
                    lidar_val_weight = train_cfg.get("lidar_loss_weight", 0.0)
                    if lidar_val_weight > 0.0 and lidar_depth is not None and lidar_mask is not None and "depth" in outputs_val:
                        lidar_val_raw, lidar_val_stats = compute_lidar_sparse_loss(
                            pred_depth = outputs_val["depth"],
                            lidar_depth = lidar_depth,
                            lidar_mask = lidar_mask,
                            lidar_confidence = lidar_confidence,
                        )
                        if lidar_val_raw is not None:
                            losses_val["opt"]["LiDARSparse"] = lidar_val_weight * lidar_val_raw
                            val_lidar_valid_ratio_sum += lidar_val_stats["valid_ratio"]
                            val_lidar_steps += 1

                    fusion_stats_val = outputs_val.get("fusion_stats", None)
                    if fusion_stats_val is not None and float(fusion_stats_val["lidar_used"].item()) > 0.0:
                        val_fusion_gate_mean_sum += float(fusion_stats_val["lidar_gate_mean"].item())
                        val_fusion_steps += 1

                    if "depth" in outputs_val:
                        rmse_val = compute_depth_rmse(outputs_val["depth"], depth, depth_mask)
                        abs_rel_val = compute_depth_abs_rel(outputs_val["depth"], depth, depth_mask)
                        delta1_val = compute_delta1(outputs_val["depth"], depth, depth_mask)
                        val_rmse_sum += float(rmse_val.item())
                        val_abs_rel_sum += float(abs_rel_val.item())
                        val_delta1_sum += float(delta1_val.item())
                        val_metric_steps += 1

                    # Phase 4: fallback check (RGB-only) during validation.
                    if (
                        data_cfg.get("phase4_eval_fallback", True)
                        and config["model"]["pixel_decoder"].get("use_lidar_fusion", False)
                        and lidar_depth is not None
                        and lidar_mask is not None
                    ):
                        inputs_rgb_only = {
                            "image": image,
                            "depth": depth,
                            "depth_mask": depth_mask,
                            "camera": camera,
                        }
                        outputs_rgb_only, _ = model.forward_train(
                            inputs_rgb_only,
                            image_metas,
                            force_compute_losses = False,
                        )
                        rmse_with_lidar = compute_depth_rmse(outputs_val["depth"], depth, depth_mask)
                        rmse_rgb_only = compute_depth_rmse(outputs_rgb_only["depth"], depth, depth_mask)
                        val_rmse_with_lidar_sum += float(rmse_with_lidar.item())
                        val_rmse_rgb_only_sum += float(rmse_rgb_only.item())
                        val_rmse_compare_steps += 1
                    val_loss += sum(losses_val["opt"].values())
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)
            if val_metric_steps > 0:
                writer.add_scalar(
                    "epoch/val_rmse",
                    val_rmse_sum / val_metric_steps,
                    epoch + 1,
                )
                writer.add_scalar(
                    "epoch/val_abs_rel",
                    val_abs_rel_sum / val_metric_steps,
                    epoch + 1,
                )
                writer.add_scalar(
                    "epoch/val_delta1",
                    val_delta1_sum / val_metric_steps,
                    epoch + 1,
                )
            if val_lidar_steps > 0:
                writer.add_scalar(
                    "epoch/val_lidar_valid_ratio",
                    val_lidar_valid_ratio_sum / val_lidar_steps,
                    epoch + 1,
                )
            if val_fusion_steps > 0:
                writer.add_scalar(
                    "epoch/val_fusion_lidar_gate_mean",
                    val_fusion_gate_mean_sum / val_fusion_steps,
                    epoch + 1,
                )
            if val_rmse_compare_steps > 0:
                writer.add_scalar(
                    "epoch/val_rmse_with_lidar",
                    val_rmse_with_lidar_sum / val_rmse_compare_steps,
                    epoch + 1,
                )
                writer.add_scalar(
                    "epoch/val_rmse_rgb_only_fallback",
                    val_rmse_rgb_only_sum / val_rmse_compare_steps,
                    epoch + 1,
                )
                writer.add_scalar(
                    "epoch/val_rmse_gap_rgb_only_minus_lidar",
                    (val_rmse_rgb_only_sum - val_rmse_with_lidar_sum) / val_rmse_compare_steps,
                    epoch + 1,
                )
            if val_metric_steps > 0:
                avg_val_rmse = val_rmse_sum / val_metric_steps
                avg_val_abs_rel = val_abs_rel_sum / val_metric_steps
                avg_val_delta1 = val_delta1_sum / val_metric_steps
                print(f"\033[1mVal loss: {avg_val_loss:.4f} | Val RMSE: {avg_val_rmse:.4f} | Val AbsRel: {avg_val_abs_rel:.4f} | Val δ1: {avg_val_delta1:.4f}\033[0m")
            else:
                print(f"\033[1mVal loss: {avg_val_loss:.4f}\033[0m")

        # ── Save checkpoint ───────────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )

    writer.close()
    print(f"\n\033[1;32mTraining complete. Tensorboard logs saved to: {tensorboard_dir}\033[0m")


if __name__ == "__main__":
    main()