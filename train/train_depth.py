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
    parser.add_argument('--output_idx', type=int, nargs='+', default=[3, 6, 33, 36])
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

    # Data configuration
    parser.add_argument('--train_root', type=str, default=None)
    parser.add_argument('--val_root', type=str, default=None)
    parser.add_argument('--image_shape', type=int, nargs=2, default=[384, 384])
    parser.add_argument('--depth_scale', type=float, default=0.001)
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
                "output_idx": args.output_idx,
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
            },
        },
        "data": {
            "train_root": args.train_root,
            "val_root": args.val_root,
            "image_shape": args.image_shape,
            "depth_scale": args.depth_scale,
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


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    config = build_config(args)

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment output directory under ./runs/
    log_dir = f"runs/train_depth_{int(time())}"
    os.makedirs(log_dir, exist_ok = True)
    print(f"Logging to {log_dir}")
    tensorboard_dir = f"{log_dir}/tensorboard"
    ckpt_dir = f"{log_dir}/checkpoints"
    os.makedirs(tensorboard_dir, exist_ok = True)
    os.makedirs(ckpt_dir, exist_ok = True)

    # Copy run script for reproducibility
    if args.script_path and os.path.isfile(args.script_path):
        shutil.copy(args.script_path, f"{log_dir}/run_script.sh")
        print(f"Saved launch script to {log_dir}/run_script.sh")

    # Set up model
    print("Building UniDepthV1 ...")
    model = UniDepthV1(config)
    model.to(device)

    # Set up Datasets & DataLoaders
    print("Setting up datasets and dataloaders ...")

    train_cfg = config["training"]
    data_cfg = config["data"]

    train_dataset = ImageDataset(
        root = data_cfg["train_root"],
        split = "train",
        image_shape = data_cfg["image_shape"],
        depth_scale = data_cfg.get("depth_scale", 0.001),
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
    print(f"Train samples: {len(train_dataset)}")
    if val_loader:
        print(f"Val samples:   {len(val_dataset)}")

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
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location = device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir = tensorboard_dir)
    print(f"TensorBoard logs -> {tensorboard_dir}")

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

        for batch_idx, batch in tqdm(enumerate(train_loader), 
                                     total = len(train_loader),
                                     desc = f"Epoch {epoch + 1}/{num_epochs}",
                                     unit = "batch"):
            
            # Move tensors to device
            image = batch['data']["image"].to(device)            # [B, 3, H, W]
            depth = batch['data']["depth"].to(device)            # [B, 1, H, W]
            depth_mask = batch['data']["depth_mask"].to(device)  # [B, 1, H, W]
            K = batch['data']["K"].to(device)                    # [B, 3, 3]

            # Build camera object for this batch
            camera = build_camera_from_batch(K)

            # Prepare inputs dict as expected by UniDepthV1
            inputs = {
                "image": image,
                "depth": depth,
                "depth_mask": depth_mask,
                "camera": camera,
            }
            img_metas = batch['img_metas']

            # image_metas: per-sample metadata (empty dicts for simplicity)
            image_metas = [{} for _ in range(image.shape[0])]

            # Forward pass
            optimizer.zero_grad()
            outputs, losses = model.forward(inputs, image_metas)

            # Compute total loss
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

                # Log sample predicted and GT depth images
                if "depth" in outputs:
                    log_depth_images(writer, "train/pred_depth", outputs["depth"], global_step)
                log_depth_images(writer, "train/gt_depth", depth, global_step)

                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step {global_step} "
                    f"Loss: {total_loss.item():.4f} "
                    f"LR: {current_lr:.2e}"
                )

        # ── End-of-epoch ─────────────────────────────────────────────────
        avg_loss = epoch_loss / max(num_batches, 1)
        writer.add_scalar("epoch/train_loss", avg_loss, epoch + 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] avg loss: {avg_loss:.4f}")

        # Step the LR scheduler
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device)
                    depth = batch["depth"].to(device)
                    depth_mask = batch["depth_mask"].to(device)
                    K = batch["K"].to(device)
                    camera = build_camera_from_batch(K)
                    inputs = {
                        "image": image,
                        "depth": depth,
                        "depth_mask": depth_mask,
                        "camera": camera,
                    }
                    image_metas = [{} for _ in range(image.shape[0])]
                    # Run in train mode temporarily to get losses
                    model.train()
                    with torch.no_grad():
                        _, losses_val = model(inputs, image_metas)
                    model.eval()
                    val_loss += sum(losses_val["opt"].values()).item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)
            print(f"  Val loss: {avg_val_loss:.4f}")

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
    print(f"Training complete. Tensorboard logs saved to: {tensorboard_dir}")


if __name__ == "__main__":
    main()
