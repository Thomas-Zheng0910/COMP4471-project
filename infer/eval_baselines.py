"""
eval_baselines.py — Evaluate baseline depth models on NYUv2.

Usage:
    python -m infer.eval_baselines --baseline marigold --data_root datasets/nyu_depth_v2_labeled.mat
    python -m infer.eval_baselines --baseline depth_anything_v2
    python -m infer.eval_baselines --baseline unidepthv2

Loads a pretrained baseline via the registry, runs inference on the
NYUv2 test split, and reports the same metrics as infer_depth.py.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.nyuv2_dataset import NYUv2Dataset, IMAGENET_MEAN, IMAGENET_STD
from model.baselines import build_baseline, BASELINE_REGISTRY
from utils.evaluation_depth import eval_depth


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline depth model on NYUv2.",
    )
    parser.add_argument(
        "--baseline", type=str, required=True,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Baseline model name.",
    )
    parser.add_argument(
        "--data_root", type=str, default="datasets/nyu_depth_v2_labeled.mat",
        help="Path to NYUv2 .mat file.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, default="runs/baselines")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument(
        "--image_shape", type=int, nargs=2, default=[480, 640],
        help="Resolution to feed the dataset (H W).",
    )
    # Marigold-specific
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--ensemble_size", type=int, default=1)
    # Model ID override
    parser.add_argument(
        "--model_id", type=str, default=None,
        help="Override the default HuggingFace model ID.",
    )
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build baseline model
    extra_kwargs = {}
    if args.model_id:
        extra_kwargs["model_id"] = args.model_id
    if args.baseline == "marigold":
        extra_kwargs["num_inference_steps"] = args.num_inference_steps
        extra_kwargs["ensemble_size"] = args.ensemble_size

    print(f"\n>>> Loading baseline: {args.baseline} >>>")
    model = build_baseline(args.baseline, device=device, **extra_kwargs)
    print(f"\033[92m{args.baseline} loaded.\033[0m")

    # Dataset
    print(f"\n>>> Loading NYUv2 {args.split}-set from {args.data_root} >>>")
    dataset = NYUv2Dataset(
        root=args.data_root,
        split=args.split,
        image_shape=args.image_shape,
        flip_aug=False,
        return_intrinsics=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=NYUv2Dataset.collate_fn,
    )
    print(f"\033[92mLoaded {len(dataset)} samples.\033[0m")

    # ImageNet stats for un-normalising
    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # Inference + evaluation
    agg = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.baseline}"):
            images = batch["data"]["image"].to(device)   # (B, 3, H, W) ImageNet-normed
            gts = batch["data"]["depth"].to(device)       # (B, 1, H, W)

            # Undo ImageNet normalisation -> [0, 1] float
            rgb_01 = images * imagenet_std + imagenet_mean  # (B, 3, H, W) [0,1]

            # Predict
            pred = model.predict_depth(rgb_01)  # (B, 1, H, W)

            # Ensure same spatial size as GT
            if pred.shape[-2:] != gts.shape[-2:]:
                pred = F.interpolate(
                    pred, size=gts.shape[-2:], mode="bilinear", align_corners=False
                )

            # Per-sample metrics
            B = gts.shape[0]
            for i in range(B):
                gt_i = gts[i]       # (1, H, W)
                pred_i = pred[i]    # (1, H, W)
                mask_i = (gt_i > 0)
                if args.max_depth is not None:
                    mask_i = mask_i & (gt_i <= args.max_depth)

                sample_m = eval_depth(
                    gts=gt_i.unsqueeze(0),
                    preds=pred_i.unsqueeze(0),
                    masks=mask_i.unsqueeze(0),
                    max_depth=args.max_depth,
                )
                for name, vals in sample_m.items():
                    agg[name].append(vals.mean().item())

    # Print results
    print(f"\n{args.baseline} — NYUv2 {args.split}-Set Metrics")
    results = {name: float(np.mean(v)) for name, v in agg.items()}
    acc_keys = sorted(k for k in results if k.startswith("d"))
    err_keys = sorted(k for k in results if k not in acc_keys)
    for key in acc_keys + err_keys:
        print(f"  \033[1m{key:20s}:\033[0m {results[key]:.4f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / f"metrics_{args.baseline}_{args.split}.json"
    with open(m_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\033[92mMetrics saved to {m_path}\033[0m")


if __name__ == "__main__":
    main()
