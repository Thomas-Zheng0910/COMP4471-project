"""
eval_baselines.py — Evaluate baseline depth models on multiple benchmarks.

Usage:
    python -m infer.eval_baselines --baseline marigold --data_root datasets/nyu_depth_v2_labeled.mat
    python -m infer.eval_baselines --baseline depth_anything_v2 --eval_datasets nyuv2,ibims1,diode_indoor,todd

Loads a pretrained baseline via the registry, runs inference on the
selected evaluation benchmarks, and reports metrics.
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

from data.nyuv2_dataset import NYUv2Dataset, IMAGENET_MEAN, IMAGENET_STD, MIN_DEPTH, NYUV2_INTRINSICS
from model.baselines import build_baseline, BASELINE_REGISTRY
from utils.evaluation_depth import eval_depth, ssi


# ──────────────────────────────────────────────────────────────────────────────
# Eval dataset factory
# ──────────────────────────────────────────────────────────────────────────────

EVAL_DATASET_DEFAULTS = {
    "nyuv2": {"root": "datasets/nyu_depth_v2_labeled.mat", "max_depth": 10.0},
    "ibims1": {"root": "datasets/unidepth_data", "max_depth": 10.0},
    "diode_indoor": {"root": "datasets/diode_indoor", "max_depth": 50.0},
    "todd": {"root": "datasets/todd", "max_depth": 10.0},
    "kitti": {"root": "datasets/kitti_eigen", "max_depth": 80.0},
}

# Baselines that output relative (non-metric) depth and need alignment.
# DA v2 outputs disparity (higher = closer); Marigold outputs relative depth [0,1].
RELATIVE_BASELINES = {"depth_anything_v2", "marigold"}


def _unified_collate_fn(batch):
    META_KEYS = {"flip", "si"}
    img_metas = [{k: item[k] for k in META_KEYS if k in item} for item in batch]
    data_keys = [k for k in batch[0].keys() if k not in META_KEYS]
    collated = {}
    for key in data_keys:
        vals = [item[key] for item in batch]
        collated[key] = torch.stack(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
    return {"data": collated, "img_metas": img_metas}


def build_eval_dataset(name: str, image_shape, root_override: str = None):
    """Build an evaluation dataset by name."""
    defaults = EVAL_DATASET_DEFAULTS.get(name, {})
    root = root_override or defaults.get("root")

    if name == "nyuv2":
        return NYUv2Dataset(root=root, split="test", image_shape=image_shape,
                            flip_aug=False, return_intrinsics=True)
    elif name == "ibims1":
        from data.ibims_dataset import IBims1Dataset
        return IBims1Dataset(root=root, image_shape=image_shape)
    elif name == "diode_indoor":
        from data.diode_dataset import DIODEIndoorDataset
        return DIODEIndoorDataset(root=root, split="val", image_shape=image_shape)
    elif name == "todd":
        from data.todd_dataset import TODDDataset
        return TODDDataset(root=root, split="test", image_shape=image_shape)
    elif name == "kitti":
        from data.kitti_eigen_dataset import KITTIEigenDataset
        return KITTIEigenDataset(root=root, split="test", image_shape=image_shape,
                                 flip_aug=False, return_intrinsics=True)
    else:
        raise ValueError(f"Unknown eval dataset: {name}. Available: {list(EVAL_DATASET_DEFAULTS.keys())}")


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
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    parser.add_argument("--output_dir", type=str, default="runs/baselines")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument(
        "--image_shape", type=int, nargs=2, default=[480, 640],
        help="Resolution to feed the dataset (H W).",
    )
    parser.add_argument(
        "--eval_datasets", type=str, default="nyuv2",
        help="Comma-separated eval benchmark names (e.g. 'nyuv2,ibims1,diode_indoor').",
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

    # ImageNet stats for un-normalising
    imagenet_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # Loop over eval datasets
    eval_dataset_names = [d.strip() for d in args.eval_datasets.split(",")]
    all_results = {}

    for ds_name in eval_dataset_names:
        ds_defaults = EVAL_DATASET_DEFAULTS.get(ds_name, {})
        max_depth = ds_defaults.get("max_depth", args.max_depth)

        # Build dataset — use --data_root override only for nyuv2
        root_override = args.data_root if ds_name == "nyuv2" else None
        print(f"\n>>> Loading {ds_name} eval dataset >>>")
        dataset = build_eval_dataset(ds_name, args.image_shape, root_override=root_override)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_unified_collate_fn,
        )
        print(f"\033[92mLoaded {len(dataset)} samples.\033[0m")

        # Inference + evaluation
        agg = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {args.baseline} on {ds_name}"):
                images = batch["data"]["image"].to(device)
                gts = batch["data"]["depth"].to(device)

                # Undo ImageNet normalisation -> [0, 1] float
                rgb_01 = images * imagenet_std + imagenet_mean

                # Predict
                pred = model.predict_depth(rgb_01)

                # Ensure same spatial size as GT
                if pred.shape[-2:] != gts.shape[-2:]:
                    pred = F.interpolate(
                        pred, size=gts.shape[-2:], mode="bilinear", align_corners=False
                    )

                is_relative = args.baseline in RELATIVE_BASELINES

                # Per-sample metrics
                B = gts.shape[0]
                for i in range(B):
                    gt_i = gts[i]
                    pred_i = pred[i]
                    mask_i = (gt_i > MIN_DEPTH)
                    if max_depth is not None:
                        mask_i = mask_i & (gt_i <= max_depth)

                    if mask_i.sum() < 10:
                        continue

                    # For relative-depth models, apply SSI alignment before
                    # computing ALL metrics (standard protocol for DA v2,
                    # Marigold, etc.).
                    if is_relative:
                        gt_masked = gt_i[mask_i]
                        pred_masked = pred_i[mask_i]
                        pred_aligned_masked = ssi(gt_masked, pred_masked)
                        # Write aligned values back into a full tensor
                        pred_i = pred_i.clone()
                        pred_i[mask_i] = pred_aligned_masked

                    # Clamp to valid depth range
                    pred_i = pred_i.clamp(min=MIN_DEPTH, max=max_depth)

                    sample_m = eval_depth(
                        gts=gt_i.unsqueeze(0),
                        preds=pred_i.unsqueeze(0),
                        masks=mask_i.unsqueeze(0),
                        max_depth=max_depth,
                    )
                    for name, vals in sample_m.items():
                        agg[name].append(vals.mean().item())

        # Print results for this dataset
        print(f"\n{args.baseline} — {ds_name} Metrics")
        results = {name: float(np.mean(v)) for name, v in agg.items()}
        acc_keys = sorted(k for k in results if k.startswith("d"))
        err_keys = sorted(k for k in results if k not in acc_keys)
        for key in acc_keys + err_keys:
            print(f"  \033[1m{key:20s}:\033[0m {results[key]:.4f}")

        all_results[ds_name] = results

        # Save per-dataset metrics
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        m_path = out_dir / f"metrics_{args.baseline}_{ds_name}.json"
        with open(m_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\033[92mMetrics saved to {m_path}\033[0m")

    # Summary across all datasets
    if len(eval_dataset_names) > 1:
        print(f"\n{'='*60}")
        print(f"  {args.baseline} — Summary Across Benchmarks")
        print(f"{'='*60}")
        header = f"{'Dataset':20s} {'d1':>8s} {'AbsRel':>8s} {'RMSE':>8s}"
        print(header)
        print("-" * len(header))
        for ds_name, res in all_results.items():
            d1 = res.get("d1", res.get("d1_ssi", float("nan")))
            arel = res.get("arel", res.get("arel_ssi", float("nan")))
            rmse = res.get("rmse", res.get("rmse_ssi", float("nan")))
            print(f"{ds_name:20s} {d1:8.4f} {arel:8.4f} {rmse:8.4f}")
        print(f"{'='*60}")

        # Save combined results
        combined_path = Path(args.output_dir) / f"metrics_{args.baseline}_all.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\033[92mCombined metrics saved to {combined_path}\033[0m")


if __name__ == "__main__":
    main()
