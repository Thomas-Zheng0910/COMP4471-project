#!/usr/bin/env python3
"""
Run full evaluation on all models and collect metrics.

Usage:
    python infer/run_full_evaluation.py              # Run all evaluations
    python infer/run_full_evaluation.py --nyu        # Run only NYU
    python infer/run_full_evaluation.py --todd       # Run only TODD
    python infer/run_full_evaluation.py --nyu --todd # Run both (same as default)
"""

import subprocess
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Model configurations
MODELS = {
    "ViT Vanilla (30ep)": {
        "checkpoint": "runs/experiments/vit-without-guide-30-epoches/checkpoints/epoch_30.pth",
        "encoder": "dinov3_vitl16",
        "output_idx": [5, 12, 18, 24],
    },
    "ConvNeXt Vanilla (50ep)": {
        "checkpoint": "runs/experiments/convnext-50-epoch/checkpoints/epoch_50.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "ConvNeXt Vanilla (60ep)": {
        "checkpoint": "runs/experiments/convnext-60-epoch/checkpoints/epoch_60.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Teacher/LiDAR (50ep)": {
        "checkpoint": "datasets/teacher/train_depth_1776839443668_3116335/checkpoints/epoch_50.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Teacher Fine-tuned NYU (90ep)": {
        "checkpoint": "runs/experiments/teacher-finetuned-nyu/checkpoints/epoch_90.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Self-Distill (60ep Best)": {
        "checkpoint": "runs/experiments/self-distillation-60-epoch-best/checkpoints/epoch_60.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
    "Self-Distill (100ep Better)": {
        "checkpoint": "runs/experiments/self-distillation-100-epoch-better?/checkpoints/epoch_100.pth",
        "encoder": "convnextv2_large",
        "output_idx": None,
    },
}

BASELINES = ["depth_anything_v2", "marigold"]


def run_command(cmd, desc):
    """Run command and print status."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ Error: {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
        return False
    
    print(f"  ✓ Success")
    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
    return True


def evaluate_custom_model(name, config, output_dir, cuda=4):
    """Evaluate a custom UniDepthV1 model."""
    
    checkpoint = config["checkpoint"]
    encoder = config["encoder"]
    output_idx = config.get("output_idx")
    
    # Check if checkpoint exists
    if not Path(checkpoint).exists():
        print(f"  ! Checkpoint not found: {checkpoint}")
        return None
    
    # Build command
    cmd = f"""conda run -n DepthSense python -m infer.infer_depth \
        --checkpoint {checkpoint} \
        --data_root datasets/nyu_depth_v2_labeled.mat \
        --split test \
        --output_dir "{output_dir}/{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}" \
        --cuda {cuda} \
        --encoder_name {encoder} \
        --use_checkpoint true \
        --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
        --num_heads 8 --expansion 4 \
        --use_lidar_fusion true --lidar_fusion_type token \
        --image_shape 480 640 \
        --max_depth 10.0 \
        --batch_size 4 --num_workers 4"""
    
    if output_idx:
        idx_str = " ".join(map(str, output_idx))
        cmd += f" --output_idx {idx_str}"
    
    success = run_command(cmd, f"Evaluating: {name}")
    
    # Try to read metrics
    metrics_file = Path(output_dir) / name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '') / "metrics_nyuv2dataset_test.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def evaluate_baseline(baseline, output_dir, cuda=4):
    """Evaluate a baseline model."""
    
    extra_args = ""
    if baseline == "marigold":
        extra_args = "--num_inference_steps 4 --ensemble_size 1"
    
    cmd = f"""conda run -n DepthSense python -m infer.eval_baselines \
        --baseline {baseline} \
        --data_root datasets/nyu_depth_v2_labeled.mat \
        --split test \
        --output_dir {output_dir}/baselines \
        --cuda {cuda} \
        --batch_size 4 --num_workers 4 \
        --max_depth 10.0 \
        --image_shape 480 640 \
        --eval_datasets nyuv2 {extra_args}"""
    
    success = run_command(cmd, f"Evaluating baseline: {baseline}")
    
    # Try to read metrics
    metrics_file = Path(output_dir) / "baselines" / f"metrics_{baseline}_nyuv2.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def print_results_table(all_results):
    """Print results in a formatted table."""
    
    print("\n" + "="*80)
    print("  EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Model':<35} {'d1':>8} {'d2':>8} {'d3':>8} {'AbsRel':>8} {'RMSE':>8} {'log10':>8}")
    print("-"*80)
    
    for name, metrics in all_results.items():
        if metrics:
            d1 = metrics.get('d1', metrics.get('d1_ssi', 0))
            d2 = metrics.get('d2', metrics.get('d2_ssi', 0))
            d3 = metrics.get('d3', metrics.get('d3_ssi', 0))
            absrel = metrics.get('arel', metrics.get('arel_ssi', 0))
            rmse = metrics.get('rmse', metrics.get('rmse_ssi', 0))
            log10 = metrics.get('log10', 0)
            
            print(f"{name:<35} {d1:>8.4f} {d2:>8.4f} {d3:>8.4f} {absrel:>8.4f} {rmse:>8.4f} {log10:>8.4f}")
        else:
            print(f"{name:<35} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    
    print("="*80)


def evaluate_custom_model_todd(name, config, output_dir, cuda=4):
    """Evaluate a custom model on TODD dataset."""
    
    checkpoint = config["checkpoint"]
    encoder = config["encoder"]
    output_idx = config.get("output_idx")
    
    if not Path(checkpoint).exists():
        print(f"  ! Checkpoint not found: {checkpoint}")
        return None
    
    cmd = f"""conda run -n DepthSense python -m infer.infer_depth \
        --checkpoint {checkpoint} \
        --data_root datasets/todd \
        --split test \
        --output_dir "{output_dir}/{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}_todd" \
        --cuda {cuda} \
        --encoder_name {encoder} \
        --use_checkpoint true \
        --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
        --num_heads 8 --expansion 4 \
        --use_lidar_fusion true --lidar_fusion_type token \
        --image_shape 480 640 \
        --max_depth 10.0 \
        --batch_size 4 --num_workers 4 \
        --eval_datasets todd"""
    
    if output_idx:
        idx_str = " ".join(map(str, output_idx))
        cmd += f" --output_idx {idx_str}"
    
    success = run_command(cmd, f"Evaluating on TODD: {name}")
    
    metrics_file = Path(output_dir) / f"{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}_todd" / "metrics_todddataset_test.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def evaluate_baseline_todd(baseline, output_dir, cuda=4):
    """Evaluate a baseline on TODD dataset."""
    
    extra_args = ""
    if baseline == "marigold":
        extra_args = "--num_inference_steps 4 --ensemble_size 1"
    
    cmd = f"""conda run -n DepthSense python -m infer.eval_baselines \
        --baseline {baseline} \
        --data_root datasets/todd \
        --split test \
        --output_dir {output_dir}/baselines_todd \
        --cuda {cuda} \
        --batch_size 4 --num_workers 4 \
        --max_depth 10.0 \
        --image_shape 480 640 \
        --eval_datasets todd {extra_args}"""
    
    success = run_command(cmd, f"Evaluating baseline on TODD: {baseline}")
    
    metrics_file = Path(output_dir) / "baselines_todd" / f"metrics_{baseline}_todd.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def evaluate_custom_model_kitti(name, config, output_dir, cuda=4):
    """Evaluate a custom model on KITTI dataset."""
    
    checkpoint = config["checkpoint"]
    encoder = config["encoder"]
    output_idx = config.get("output_idx")
    
    if not Path(checkpoint).exists():
        print(f"  ! Checkpoint not found: {checkpoint}")
        return None
    
    cmd = f"""conda run -n DepthSense python -m infer.infer_depth \
        --checkpoint {checkpoint} \
        --data_root datasets/kitti_eigen \
        --split test \
        --output_dir "{output_dir}/{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}_kitti" \
        --cuda {cuda} \
        --encoder_name {encoder} \
        --use_checkpoint true \
        --hidden_dim 512 --dropout 0.0 --depths 3 2 1 \
        --num_heads 8 --expansion 4 \
        --use_lidar_fusion true --lidar_fusion_type token \
        --image_shape 480 640 \
        --max_depth 80.0 \
        --batch_size 4 --num_workers 4 \
        --eval_datasets kitti"""
    
    if output_idx:
        idx_str = " ".join(map(str, output_idx))
        cmd += f" --output_idx {idx_str}"
    
    success = run_command(cmd, f"Evaluating on KITTI: {name}")
    
    metrics_file = Path(output_dir) / f"{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}_kitti" / "metrics_kittidataset_test.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def evaluate_baseline_kitti(baseline, output_dir, cuda=4):
    """Evaluate a baseline on KITTI dataset."""
    
    extra_args = ""
    if baseline == "marigold":
        extra_args = "--num_inference_steps 4 --ensemble_size 1"
    
    cmd = f"""conda run -n DepthSense python -m infer.eval_baselines \
        --baseline {baseline} \
        --data_root datasets/kitti_eigen \
        --split test \
        --output_dir {output_dir}/baselines_kitti \
        --cuda {cuda} \
        --batch_size 4 --num_workers 4 \
        --max_depth 80.0 \
        --image_shape 480 640 \
        --eval_datasets kitti {extra_args}"""
    
    success = run_command(cmd, f"Evaluating baseline on KITTI: {baseline}")
    
    metrics_file = Path(output_dir) / "baselines_kitti" / f"metrics_{baseline}_kitti.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


def get_args():
    parser = argparse.ArgumentParser(description="Run full evaluation on all models")
    parser.add_argument("--nyu", action="store_true", help="Run NYU evaluations")
    parser.add_argument("--todd", action="store_true", help="Run TODD evaluations")
    parser.add_argument("--kitti", action="store_true", help="Run KITTI evaluations")
    parser.add_argument("--cuda", type=int, default=4, help="GPU device index")
    return parser.parse_args()


def main():
    args = get_args()
    
    # If no flags specified, run all
    no_flags = not args.nyu and not args.todd and not args.kitti
    run_nyu = args.nyu or no_flags
    run_todd = args.todd or no_flags
    run_kitti = args.kitti or no_flags
    
    output_base = "runs/full_eval"
    Path(output_base).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Evaluate custom models on NYU
    if run_nyu:
        print("\n" + "="*80)
        print("  CUSTOM MODEL EVALUATIONS - NYU")
        print("="*80)
        
        for name, config in MODELS.items():
            metrics = evaluate_custom_model(name, config, output_base, cuda=args.cuda)
            all_results[name] = metrics
        
        # Evaluate baselines on NYU
        print("\n" + "="*80)
        print("  BASELINE MODEL EVALUATIONS - NYU")
        print("="*80)
        
        for baseline in BASELINES:
            metrics = evaluate_baseline(baseline, output_base, cuda=args.cuda)
            all_results[baseline] = metrics
    
    # Evaluate custom models on TODD
    if run_todd:
        print("\n" + "="*80)
        print("  CUSTOM MODEL EVALUATIONS - TODD")
        print("="*80)
        
        for name, config in MODELS.items():
            metrics = evaluate_custom_model_todd(name, config, output_base, cuda=args.cuda)
            all_results[f"{name} (TODD)"] = metrics
        
        # Evaluate baselines on TODD
        print("\n" + "="*80)
        print("  BASELINE MODEL EVALUATIONS - TODD")
        print("="*80)
        
        for baseline in BASELINES:
            metrics = evaluate_baseline_todd(baseline, output_base, cuda=args.cuda)
            all_results[f"{baseline} (TODD)"] = metrics
    
    # Evaluate custom models on KITTI
    if run_kitti:
        print("\n" + "="*80)
        print("  CUSTOM MODEL EVALUATIONS - KITTI")
        print("="*80)
        
        for name, config in MODELS.items():
            metrics = evaluate_custom_model_kitti(name, config, output_base, cuda=args.cuda)
            all_results[f"{name} (KITTI)"] = metrics
        
        # Evaluate baselines on KITTI
        print("\n" + "="*80)
        print("  BASELINE MODEL EVALUATIONS - KITTI")
        print("="*80)
        
        for baseline in BASELINES:
            metrics = evaluate_baseline_kitti(baseline, output_base, cuda=args.cuda)
            all_results[f"{baseline} (KITTI)"] = metrics
    
    # Print summary table
    print_results_table(all_results)
    
    # Save all results
    results_file = Path(output_base) / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ All results saved to {results_file}")


if __name__ == "__main__":
    main()
