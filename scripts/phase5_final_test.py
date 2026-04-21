#!/usr/bin/env python3
"""
Phase 5 Final Test Evaluation Script

Reads the run mapping JSON (produced by run_phase5_ablation.sh),
selects the best checkpoint per method based on validation metrics,
then evaluates on the final test split.

Outputs:
  - Per-method final test metrics JSON
  - Merged comparison JSON
  - Summary table
"""

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 final test evaluation (test split only)."
    )
    parser.add_argument(
        "--runs_json",
        type=str,
        default="runs/phase5_fullscale_latest.json",
        help="JSON file with run metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/phase5_final_test",
        help="Directory to save final test metrics.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/nyu_depth_v2_labeled.mat",
        help="NYU Depth V2 .mat file path.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="GPU device index.",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default="python",
        help="Python binary path.",
    )
    return parser.parse_args()


def find_event_file(run_dir: Path) -> Optional[Path]:
    """Locate TensorBoard event file in run directory."""
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        return None
    files = sorted(tb_dir.glob("events.out.tfevents.*"))
    return files[0] if files else None


def get_best_checkpoint_by_val_rmse(run_dir: Path) -> Optional[Path]:
    """
    Find the checkpoint with lowest validation RMSE.
    Uses TensorBoard event file.
    Returns checkpoint path or None if not found.
    """
    event_file = find_event_file(run_dir)
    if event_file is None:
        return None

    try:
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        scalars = ea.Scalars("epoch/val_rmse")
        if not scalars:
            return None

        # Find epoch with minimum val_rmse
        best_epoch = min(scalars, key=lambda x: x.value).step
        ckpt_dir = run_dir / "checkpoints"
        ckpt_path = ckpt_dir / f"epoch_{best_epoch}.pth"
        if ckpt_path.is_file():
            return ckpt_path
    except Exception as e:
        print(f"[WARN] Could not parse event file: {e}")

    # Fallback: use latest checkpoint
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
        if ckpts:
            return ckpts[-1]

    return None


def run_inference(
    checkpoint: Path,
    data_root: str,
    split: str,
    output_dir: Path,
    cuda: int,
    python_bin: str = "python",
) -> Optional[Dict]:
    """
    Run infer_depth.py on checkpoint, return metrics dict or None if failed.
    """
    temp_out = Path(tempfile.mkdtemp())

    cmd = [
        python_bin,
        "-m",
        "infer.infer_depth",
        "--checkpoint",
        str(checkpoint),
        "--data_root",
        data_root,
        "--split",
        split,
        "--output_dir",
        str(temp_out),
        "--cuda",
        str(cuda),
        "--image_shape",
        "384",
        "384",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[ERROR] Inference failed: {result.stderr}")
            return None

        # Look for metrics JSON in temp output
        metrics_file = temp_out / f"metrics_NyuvDataset_{split}.json"
        if not metrics_file.exists():
            print(f"[WARN] No metrics file found at {metrics_file}")
            return None

        with open(metrics_file) as f:
            metrics = json.load(f)

        return metrics
    except subprocess.TimeoutExpired:
        print("[ERROR] Inference timeout")
        return None
    except Exception as e:
        print(f"[ERROR] Inference exception: {e}")
        return None
    finally:
        import shutil
        if temp_out.exists():
            shutil.rmtree(temp_out, ignore_errors=True)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_json = Path(args.runs_json)
    if not runs_json.exists():
        print(f"[ERROR] Runs JSON not found: {runs_json}")
        return

    with open(runs_json) as f:
        payload = json.load(f)

    runs = payload.get("runs", [])
    if not runs:
        print("[ERROR] No runs in JSON")
        return

    print("\n" + "=" * 70)
    print("PHASE 5 FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Evaluating {len(runs)} training runs on test split\n")

    method_results = {}

    for run_entry in runs:
        method = run_entry.get("method", "unknown")
        run_path = run_entry.get("run_path", "")
        exit_code = run_entry.get("exit_code", -1)

        print(f"\n>>> Processing: {method} <<<")
        print(f"    Run path: {run_path}")
        print(f"    Exit code: {exit_code}")

        if exit_code != 0:
            print(f"    [SKIP] Training failed (exit={exit_code})")
            continue

        run_dir = Path(run_path)
        if not run_dir.exists():
            print(f"    [SKIP] Directory not found")
            continue

        # Find best checkpoint by validation RMSE
        best_ckpt = get_best_checkpoint_by_val_rmse(run_dir)
        if best_ckpt is None:
            print(f"    [SKIP] No checkpoint found")
            continue

        print(f"    Best checkpoint: {best_ckpt.name}")

        # Run inference on test split
        print(f"    Running inference on test split...")
        metrics = run_inference(
            checkpoint=best_ckpt,
            data_root=args.data_root,
            split="test",
            output_dir=out_dir,
            cuda=args.cuda,
            python_bin=args.python_bin,
        )

        if metrics is None:
            print(f"    [WARN] Inference failed, skipping")
            continue

        method_results[method] = {
            "run_path": run_path,
            "checkpoint": str(best_ckpt),
            "test_metrics": metrics,
        }

        # Print key metrics
        rmse = metrics.get("rmse", None)
        abs_rel = metrics.get("rel", None)
        d1 = metrics.get("d1_ssi", None)
        print(f"    ✅ Test RMSE: {rmse:.4f}" if rmse else "    ✅ Inference complete")
        if abs_rel:
            print(f"       Test Abs.Rel: {abs_rel:.4f}")
        if d1:
            print(f"       Test δ₁ (SSI): {d1:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Build comparison table
    if method_results:
        methods_list = sorted(method_results.keys())
        print(f"\n{'Method':<25} | {'RMSE':<10} | {'Abs.Rel':<10} | {'δ₁ (SSI)':<10}")
        print("-" * 60)

        rows = []
        for method in methods_list:
            metrics = method_results[method]["test_metrics"]
            rmse_val = metrics.get("rmse", None)
            rel_val = metrics.get("rel", None)
            d1_val = metrics.get("d1_ssi", None)

            rmse_str = f"{rmse_val:.4f}" if rmse_val else "N/A"
            rel_str = f"{rel_val:.4f}" if rel_val else "N/A"
            d1_str = f"{d1_val:.4f}" if d1_val else "N/A"

            print(f"{method:<25} | {rmse_str:<10} | {rel_str:<10} | {d1_str:<10}")

            rows.append(
                {
                    "method": method,
                    "rmse": rmse_val,
                    "abs_rel": rel_val,
                    "d1_ssi": d1_val,
                }
            )

        # Save per-method JSON files
        print("\n📁 Saving per-method JSON files...")
        for method, result in method_results.items():
            method_file = out_dir / f"phase5_final_test_{method}.json"
            with open(method_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  ✓ {method_file}")

        # Save merged summary
        summary_file = out_dir / "phase5_final_test_summary.json"
        summary = {
            "schema_version": "1.0",
            "type": "phase5_final_test_summary",
            "timestamp": str(Path(args.runs_json).stat().st_mtime),
            "runs_json_source": str(args.runs_json),
            "test_split": "test",
            "results": rows,
            "method_details": method_results,
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Merged summary: {summary_file}")

    else:
        print("[WARN] No successful evaluations completed.")

    print("\n" + "=" * 70)
    print("✅ Final test evaluation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
