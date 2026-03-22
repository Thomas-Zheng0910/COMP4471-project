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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.nyuv2_dataset import NYUv2Dataset as ImageDataset
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
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1)

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
    parser.add_argument('--image_shape', type=int, nargs=2, default=[384, 384])
    parser.add_argument('--depth_scale', type=float, default=0.001)
    parser.add_argument('--use_lidar', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--lidar_root', type=str, default=None)
    parser.add_argument('--lidar_depth_scale', type=float, default=1.0)
    parser.add_argument('--lidar_h5_key', type=str, default=None)
    parser.add_argument('--lidar_confidence_h5_key', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)

    # Checkpoint resume
    parser.add_argument('--resume', type=str, default=None)

    # Script path for copying to experiment folder
    parser.add_argument('--script_path', type=str, default=None)

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
            },
            "pixel_decoder": {
                "name": "Decoder",
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "depths": args.depths,
            },
            "num_heads": args.num_heads,
            "expansion": args.expansion,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_min": args.lr_min,
            "wd": args.weight_decay,
            "log_every": args.log_every,
            "save_every": args.save_every,
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
                "lidar": {
                    "name": "LiDARSparse",
                    "weight": args.lidar_loss_weight,
                    "fn": "log_l1",
                },
            },
        },
        "data": {
            "train_root": args.train_root,
            "val_root": args.val_root,
            "image_shape": args.image_shape,
            "depth_scale": args.depth_scale,
            "use_lidar": args.use_lidar,
            "lidar_root": args.lidar_root,
            "lidar_depth_scale": args.lidar_depth_scale,
            "lidar_h5_key": args.lidar_h5_key,
            "lidar_confidence_h5_key": args.lidar_confidence_h5_key,
            "num_workers": args.num_workers,
        },
    }


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

    valid = lidar_mask.bool()
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


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    config = build_config(args)

    print("Arguments:")
    for arg in vars(args):
        print(f"  \033[1m{arg}:\033[0m {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment output directory under ./runs/
    log_dir = f"runs/train_depth_{int(time())}"
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

    train_dataset = ImageDataset(
        root = data_cfg["train_root"],
        split = "train",
        image_shape = data_cfg["image_shape"],
        depth_scale = data_cfg.get("depth_scale", 0.001),
        use_lidar = data_cfg.get("use_lidar", False),
        lidar_root = data_cfg.get("lidar_root", None),
        lidar_depth_scale = data_cfg.get("lidar_depth_scale", 1.0),
        lidar_h5_key = data_cfg.get("lidar_h5_key", None),
        lidar_confidence_h5_key = data_cfg.get("lidar_confidence_h5_key", None),
        flip_aug = True,   # produce (original, flipped) pairs for SelfDistill
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size = train_cfg["batch_size"],
        shuffle = True,
        num_workers = data_cfg.get("num_workers", 4),
        pin_memory = device.type == "cuda",
        drop_last = True,
        collate_fn = ImageDataset.collate_fn,
    )

    # Optional validation loader
    # NOTE: We only have 2 splits, originally was train/test
    #       we call it val for now.
    val_loader = None
    if data_cfg.get("val_root") is not None:
        val_dataset = ImageDataset(
            root = data_cfg["val_root"],
            split = "test",
            image_shape = data_cfg["image_shape"],
            depth_scale = data_cfg.get("depth_scale", 0.001),
            use_lidar = data_cfg.get("use_lidar", False),
            lidar_root = data_cfg.get("lidar_root", None),
            lidar_depth_scale = data_cfg.get("lidar_depth_scale", 1.0),
            lidar_h5_key = data_cfg.get("lidar_h5_key", None),
            lidar_confidence_h5_key = data_cfg.get("lidar_confidence_h5_key", None),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size = train_cfg["batch_size"],
            shuffle = False,
            num_workers = data_cfg.get("num_workers", 4),
            pin_memory = device.type == "cuda",
            collate_fn = ImageDataset.collate_fn,
        )

    # verbose
    print(f"\033[1mTrain samples:\033[0m {len(train_dataset)}")
    print(f"\033[1mLiDAR enabled:\033[0m {data_cfg.get('use_lidar', False)}")
    if val_loader:
        print(f"\033[1mVal samples:\033[0m   {len(val_dataset)}")

    # Set up Optimizer
    # Use model.get_params() for layer-wise LR decay (encoder vs decoder)
    try:
        param_groups = model.get_params(config)
    except Exception:
        # Fallback: treat all parameters uniformly
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(
        param_groups,
        lr = train_cfg["lr"],
        weight_decay = train_cfg["wd"],
    )

    # Set up LR Scheduler
    num_epochs = train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = num_epochs,
        eta_min = train_cfg.get("lr_min", 1e-6),
    )

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

        model.train()
        epoch_loss = 0.0
        num_batches = 0
        lidar_epoch_valid_ratio_sum = 0.0
        lidar_epoch_steps = 0
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

            # image_metas carry the flip / si flags set per-sample by the dataset
            image_metas = batch['img_metas']

            # Forward pass
            optimizer.zero_grad()
            outputs, losses = model.forward(inputs, image_metas)

            # Compute total loss
            lidar_raw_loss = None
            lidar_stats = None
            lidar_weight = train_cfg["losses"].get("lidar", {}).get("weight", 0.0)
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

            # Backward + optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
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

                # Log sample predicted and GT depth images
                if "depth" in outputs:
                    log_depth_images(writer, "train/pred_depth", outputs["depth"], global_step)
                log_depth_images(writer, "train/gt_depth", depth, global_step)

        # ── End-of-epoch ─────────────────────────────────────────────────
        avg_loss = epoch_loss / max(num_batches, 1)
        writer.add_scalar("epoch/train_loss", avg_loss, epoch + 1)
        if lidar_epoch_steps > 0:
            writer.add_scalar(
                "epoch/train_lidar_valid_ratio",
                lidar_epoch_valid_ratio_sum / lidar_epoch_steps,
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
                    image_metas = batch["img_metas"]
                    # model is in eval() mode: forward() dispatches to forward_test
                    # which does NOT compute losses. We use forward_train explicitly
                    # so we can still get loss values for monitoring.
                    outputs_val, losses_val = model.forward_train(inputs, image_metas, force_compute_losses = True)
                    lidar_val_weight = train_cfg["losses"].get("lidar", {}).get("weight", 0.0)
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
                    val_loss += sum(losses_val["opt"].values())
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)
            if val_lidar_steps > 0:
                writer.add_scalar(
                    "epoch/val_lidar_valid_ratio",
                    val_lidar_valid_ratio_sum / val_lidar_steps,
                    epoch + 1,
                )
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