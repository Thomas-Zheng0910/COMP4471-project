#!/usr/bin/env python3
"""
Phase 5 Multi-Seed Statistics Aggregation

Reads final test metrics from multiple runs (same method, different seeds),
computes per-method aggregated statistics (mean, std, 95% CI),
and generates comparison figure with error bars.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate P5 multi-seed statistics.")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="runs/phase5_final_test",
        help="Directory containing per-method final test JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/figures",
        help="Directory to save figures and summary.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["RGB-only", "Supervision-only", "Late fusion", "Token fusion"],
        help="Method names to aggregate.",
    )
    return parser.parse_args()


def collect_metrics_by_method(
    metrics_dir: Path,
    method_names: List[str],
) -> Dict[str, List[Dict]]:
    """
    Scan metrics_dir for phase5_final_test_*.json files,
    group by method name, return per-method list of metric dicts.
    """
    all_results = defaultdict(list)

    for json_file in sorted(metrics_dir.glob("phase5_final_test_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # If it's a single-method result file (has 'method' from wrapper)
            if "method" in data:
                method = data["method"]
                if "test_metrics" in data:
                    metrics = data["test_metrics"]
                elif "results" in data:
                    # Could be a row from summary
                    metrics = data["results"][0] if data["results"] else {}
                else:
                    metrics = {}
            else:
                # Try to infer from filename
                stem = json_file.stem
                method = stem.replace("phase5_final_test_", "").replace("_", " ")
                if "test_metrics" in data:
                    metrics = data["test_metrics"]
                else:
                    metrics = data

            # Filter to recognized methods
            found_method = None
            for m in method_names:
                if m.lower() in method.lower():
                    found_method = m
                    break

            if found_method:
                all_results[found_method].append(metrics)

        except Exception as e:
            print(f"[WARN] Could not parse {json_file}: {e}")

    return all_results


def compute_statistics(
    metric_values: List[float],
) -> Dict[str, float]:
    """
    Compute mean, std, 95% CI for a list of values.
    """
    arr = np.array(metric_values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "ci_lower": None,
            "ci_upper": None,
            "count": 0,
        }

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    median_val = float(np.median(arr))

    # 95% CI using bootstrap (simple t-based for small N)
    if len(arr) >= 2:
        se = std_val / np.sqrt(len(arr))
        ci_margin = 1.96 * se
        ci_lower = mean_val - ci_margin
        ci_upper = mean_val + ci_margin
    else:
        ci_lower = mean_val
        ci_upper = mean_val

    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "count": len(arr),
    }


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_dir.exists():
        print(f"[ERROR] Metrics directory not found: {metrics_dir}")
        return

    print("\n" + "=" * 70)
    print("PHASE 5 MULTI-SEED STATISTICS AGGREGATION")
    print("=" * 70)

    # Collect metrics by method
    method_metrics = collect_metrics_by_method(metrics_dir, args.methods)

    if not method_metrics:
        print("[WARN] No metrics found to aggregate")
        return

    # Compute statistics per method
    print(f"\nAggregating {len(method_metrics)} methods...\n")

    stats_summary = {}

    for method in args.methods:
        if method not in method_metrics:
            print(f"  [SKIP] {method}: no data")
            continue

        metrics_list = method_metrics[method]
        n_runs = len(metrics_list)
        print(f"  {method}: {n_runs} run(s)")

        # Collect per-metric values
        metric_names = ["rmse", "rel", "d1_ssi", "silog"]
        method_stats = {"runs_count": n_runs}

        for metric_name in metric_names:
            values = []
            for m in metrics_list:
                if metric_name in m and m[metric_name] is not None:
                    values.append(float(m[metric_name]))

            if values:
                stats = compute_statistics(values)
                method_stats[metric_name] = stats
                mean_str = f"{stats['mean']:.4f}" if stats["mean"] else "N/A"
                std_str = f"{stats['std']:.4f}" if stats["std"] else "N/A"
                print(f"      {metric_name}: {mean_str} ± {std_str}")

        stats_summary[method] = method_stats

    # Save aggregated stats as JSON
    stats_file = out_dir / "phase5_stats_summary.json"
    with open(stats_file, "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"\n📄 Saved stats: {stats_file}")

    # Generate comparison bar chart with error bars
    if "rmse" in stats_summary.get(args.methods[0], {}) if args.methods else False:
        print("\nGenerating comparison figure...")

        methods = []
        rmse_means = []
        rmse_errs = []

        for method in args.methods:
            if method in stats_summary and "rmse" in stats_summary[method]:
                stats = stats_summary[method]["rmse"]
                if stats.get("mean") is not None:
                    methods.append(method)
                    rmse_means.append(stats["mean"])
                    err = stats.get("std", 0) or stats.get("ci_upper", stats["mean"]) - stats.get("ci_lower", stats["mean"])
                    rmse_errs.append(err)

        if methods:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(methods))
            ax.bar(x, rmse_means, yerr=rmse_errs, capsize=5, alpha=0.7, color="steelblue")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=15, ha="right")
            ax.set_ylabel("RMSE (metres)", fontsize=12)
            ax.set_title("Phase 5 Multi-Seed Comparison - Test RMSE with Uncertainty", fontsize=13)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            fig_path = out_dir / "phase5_multiseed_rmse_comparison.png"
            fig.savefig(fig_path, dpi=220)
            print(f"  ✓ Saved figure: {fig_path}")
            plt.close(fig)

    print("\n" + "=" * 70)
    print("✅ Statistics aggregation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
