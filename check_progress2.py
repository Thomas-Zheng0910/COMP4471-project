"""Quick progress check - just latest step and any val metrics."""
import os, glob, struct

runs_dir = "runs"
experiments = ["ablation_baseline", "ablation_grad_matching", "ablation_edge_ssi",
               "ablation_color_jitter", "ablation_deep_sup", "ablation_combined"]

for exp in experiments:
    tb_dir = os.path.join(runs_dir, exp, "tensorboard")
    ckpt_dir = os.path.join(runs_dir, exp, "checkpoints")
    
    if not os.path.exists(tb_dir):
        print(f"{exp}: NOT STARTED")
        continue
    
    # Check TB file size
    tb_files = glob.glob(os.path.join(tb_dir, "events*"))
    tb_size = sum(os.path.getsize(f) for f in tb_files) if tb_files else 0
    
    # Check checkpoints
    ckpts = []
    if os.path.exists(ckpt_dir):
        ckpts = sorted(os.listdir(ckpt_dir))
    
    # Check latest log line by reading the log file with binary mode
    log_file = os.path.join("runs/ablation_logs", exp.replace("ablation_", "") + ".log")
    last_epoch_line = ""
    val_lines = []
    if os.path.exists(log_file):
        with open(log_file, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")
            # Split on \r or \n to get real lines
            lines = content.replace("\r", "\n").split("\n")
            for line in lines:
                if "Epoch" in line and "batch/s" in line:
                    last_epoch_line = line.strip()
                if "Val loss" in line or "Val RMSE" in line or "median-scaled" in line:
                    val_lines.append(line.strip())
    
    print(f"{exp}:")
    print(f"  TB size: {tb_size/1024/1024:.1f} MB | Checkpoints: {ckpts if ckpts else 'none'}")
    if last_epoch_line:
        print(f"  Progress: {last_epoch_line[:100]}")
    if val_lines:
        for v in val_lines:
            print(f"  {v[:120]}")
    print()
