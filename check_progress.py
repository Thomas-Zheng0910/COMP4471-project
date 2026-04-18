"""Check training progress by reading TensorBoard event files."""
import os
import sys

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("tensorboard not installed, trying tbparse")
    sys.exit(1)

runs_dir = "runs"
experiments = ["ablation_baseline", "ablation_grad_matching", "ablation_edge_ssi",
               "ablation_color_jitter", "ablation_deep_sup", "ablation_combined"]

for exp in experiments:
    tb_dir = os.path.join(runs_dir, exp, "tensorboard")
    if not os.path.exists(tb_dir):
        print(f"{exp}: NOT STARTED")
        continue
    
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"{exp}: No scalar data yet")
        continue
    
    # Check for epoch-level metrics
    epoch_tags = [t for t in tags if t.startswith("epoch/")]
    train_tags = [t for t in tags if t.startswith("train/")]
    
    # Get latest training step
    max_step = 0
    for t in train_tags[:1]:
        events = ea.Scalars(t)
        if events:
            max_step = max(e.step for e in events)
    
    # Get epoch-level validation
    val_results = {}
    for t in epoch_tags:
        events = ea.Scalars(t)
        if events:
            val_results[t] = [(e.step, e.value) for e in events]
    
    print(f"\n{'='*60}")
    print(f"  {exp}")
    print(f"{'='*60}")
    print(f"  Latest training step: {max_step}")
    print(f"  Available tags: {len(tags)} ({len(epoch_tags)} epoch, {len(train_tags)} train)")
    
    if val_results:
        print(f"  Validation results:")
        for tag, vals in sorted(val_results.items()):
            for step, val in vals:
                print(f"    {tag} @ epoch {step}: {val:.6f}")
    else:
        print(f"  No validation results yet (epoch not complete)")
        # Show latest train loss
        if "train/total_loss" in tags:
            events = ea.Scalars("train/total_loss")
            if events:
                latest = events[-1]
                print(f"  Latest train loss @ step {latest.step}: {latest.value:.4f}")
