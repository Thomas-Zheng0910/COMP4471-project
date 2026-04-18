"""
Compare ablation experiment results.
Reads TensorBoard event files and prints a comparison table.
"""
import os
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed")
    sys.exit(1)

runs_dir = "runs"
experiments = ["ablation_baseline", "ablation_grad_matching", "ablation_edge_ssi",
               "ablation_color_jitter", "ablation_deep_sup", "ablation_combined"]

# Metrics we care about
epoch_metrics = [
    "epoch/val_loss",
    "epoch/val_abs_rel",
    "epoch/val_rmse",
    "epoch/val_delta1",
]

results = {}

for exp in experiments:
    tb_dir = os.path.join(runs_dir, exp, "tensorboard")
    if not os.path.exists(tb_dir):
        print(f"  {exp}: NOT FOUND")
        continue
    
    print(f"Loading {exp}...")
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    
    tags = ea.Tags().get("scalars", [])
    
    exp_results = {}
    for metric in epoch_metrics:
        if metric in tags:
            events = ea.Scalars(metric)
            if events:
                # Get the best value
                values = [(e.step, e.value) for e in events]
                exp_results[metric] = values
    
    # Also get training loss curve (final value)
    if "train/total_loss" in tags:
        events = ea.Scalars("train/total_loss")
        if events:
            exp_results["train/total_loss_final"] = events[-1].value
            exp_results["train/total_loss_step"] = events[-1].step
    
    results[exp] = exp_results

print("\n" + "=" * 100)
print("ABLATION EXPERIMENT COMPARISON")
print("=" * 100)

# Print header
print(f"\n{'Experiment':<30} {'Best AbsRel':>12} {'Best RMSE':>12} {'Best δ1':>12} {'Best ValLoss':>12} {'@ Epoch':>8}")
print("-" * 88)

for exp in experiments:
    if exp not in results:
        print(f"{exp:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>8}")
        continue
    
    r = results[exp]
    
    # Best AbsRel (lower is better)
    best_abs_rel = "N/A"
    best_abs_rel_epoch = ""
    if "epoch/val_abs_rel" in r:
        vals = r["epoch/val_abs_rel"]
        best = min(vals, key=lambda x: x[1])
        best_abs_rel = f"{best[1]:.6f}"
        best_abs_rel_epoch = str(best[0])
    
    # Best RMSE (lower is better)
    best_rmse = "N/A"
    if "epoch/val_rmse" in r:
        vals = r["epoch/val_rmse"]
        best = min(vals, key=lambda x: x[1])
        best_rmse = f"{best[1]:.4f}"
    
    # Best δ1 (higher is better)
    best_d1 = "N/A"
    if "epoch/val_delta1" in r:
        vals = r["epoch/val_delta1"]
        best = max(vals, key=lambda x: x[1])
        best_d1 = f"{best[1]:.6f}"
    
    # Best val loss
    best_val = "N/A"
    if "epoch/val_loss" in r:
        vals = r["epoch/val_loss"]
        best = min(vals, key=lambda x: x[1])
        best_val = f"{best[1]:.4f}"
    
    print(f"{exp:<30} {best_abs_rel:>12} {best_rmse:>12} {best_d1:>12} {best_val:>12} {best_abs_rel_epoch:>8}")

# Print per-epoch detail for each experiment
print("\n\nPER-EPOCH VALIDATION RESULTS:")
print("=" * 100)

for exp in experiments:
    if exp not in results:
        continue
    r = results[exp]
    if "epoch/val_abs_rel" not in r:
        continue
    
    print(f"\n{exp}:")
    print(f"  {'Epoch':>6} {'AbsRel':>10} {'RMSE':>10} {'δ1':>10} {'ValLoss':>10}")
    print(f"  {'-'*48}")
    
    abs_rel_vals = {e: v for e, v in r.get("epoch/val_abs_rel", [])}
    rmse_vals = {e: v for e, v in r.get("epoch/val_rmse", [])}
    d1_vals = {e: v for e, v in r.get("epoch/val_delta1", [])}
    loss_vals = {e: v for e, v in r.get("epoch/val_loss", [])}
    
    all_epochs = sorted(set(abs_rel_vals.keys()) | set(rmse_vals.keys()) | set(d1_vals.keys()) | set(loss_vals.keys()))
    
    for ep in all_epochs:
        ar = f"{abs_rel_vals[ep]:.6f}" if ep in abs_rel_vals else "N/A"
        rm = f"{rmse_vals[ep]:.4f}" if ep in rmse_vals else "N/A"
        d1 = f"{d1_vals[ep]:.6f}" if ep in d1_vals else "N/A"
        lv = f"{loss_vals[ep]:.4f}" if ep in loss_vals else "N/A"
        print(f"  {ep:>6} {ar:>10} {rm:>10} {d1:>10} {lv:>10}")

# Summary with relative changes vs baseline
print("\n\nIMPROVEMENT vs BASELINE (final epoch):")
print("=" * 100)

if "ablation_baseline" in results and "epoch/val_abs_rel" in results["ablation_baseline"]:
    base_r = results["ablation_baseline"]
    base_abs_rel = base_r["epoch/val_abs_rel"][-1][1] if "epoch/val_abs_rel" in base_r else None
    base_rmse = base_r["epoch/val_rmse"][-1][1] if "epoch/val_rmse" in base_r else None
    base_d1 = base_r["epoch/val_delta1"][-1][1] if "epoch/val_delta1" in base_r else None
    
    print(f"\n{'Experiment':<30} {'ΔAbsRel%':>12} {'ΔRMSE%':>12} {'Δδ1%':>12}")
    print("-" * 66)
    
    for exp in experiments:
        if exp not in results or exp == "ablation_baseline":
            continue
        r = results[exp]
        
        dabs = "N/A"
        if "epoch/val_abs_rel" in r and base_abs_rel:
            val = r["epoch/val_abs_rel"][-1][1]
            dabs = f"{(val - base_abs_rel) / base_abs_rel * 100:+.2f}%"
        
        drmse = "N/A"
        if "epoch/val_rmse" in r and base_rmse:
            val = r["epoch/val_rmse"][-1][1]
            drmse = f"{(val - base_rmse) / base_rmse * 100:+.2f}%"
        
        dd1 = "N/A"
        if "epoch/val_delta1" in r and base_d1:
            val = r["epoch/val_delta1"][-1][1]
            dd1 = f"{(val - base_d1) / base_d1 * 100:+.2f}%"
        
        print(f"{exp:<30} {dabs:>12} {drmse:>12} {dd1:>12}")
    
    print("\n  (Negative ΔAbsRel% / ΔRMSE% = improvement, Positive Δδ1% = improvement)")
else:
    print("  Baseline results not yet available.")

print("\nDone.")
