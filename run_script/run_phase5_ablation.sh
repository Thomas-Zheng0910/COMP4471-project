#!/bin/bash

# Phase 5 Ablation Study Runner
# Run 4 variants in parallel on GPU 3/6/7/0.
# Auto-generates comparison plots after all training finishes.
# Supports tmux mode for persistent runs (won't disconnect on SSH close).

set -e

# Configuration
MAT_PATH="${1:-datasets/nyu_depth_v2_labeled.mat}"
LIDAR_H5_KEY="${2:-auto}"
LIDAR_ROOT="${3:-}"
EPOCHS="${4:-20}"
SAVE_EVERY=20
BATCH_SIZE="${5:-2}"
GPU_LIST_RAW="${6:-4,6,7,0}"
BATCH_SIZE_FUSION="${7:-$BATCH_SIZE}"
USE_TMUX="${8:-true}"
DEPTH_SCALE="${DEPTH_SCALE:-1.0}"
LIDAR_DEPTH_SCALE="${LIDAR_DEPTH_SCALE:-1.0}"
LIDAR_DROPOUT_PROB="${LIDAR_DROPOUT_PROB:-0.0}"

PYTHON_BIN="${PYTHON_BIN:-/localdata/yhip/COMP4471-project/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

RESOLVED_LIDAR_H5_KEY=""

# Tmux wrapper support
if [ "$USE_TMUX" = "true" ] && [ -z "$TMUX" ]; then
  SESSION_NAME="phase5_ablation_$(date +%s)"
  echo "Launching in tmux session: $SESSION_NAME"
  echo "Tmux mode will persist even if SSH disconnects."
  echo "Re-attach with: tmux attach-session -t $SESSION_NAME"
  echo ""
  tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" "$0" "$MAT_PATH" "$LIDAR_H5_KEY" "$LIDAR_ROOT" "$EPOCHS" "$BATCH_SIZE" "$GPU_LIST_RAW" "$BATCH_SIZE_FUSION" false
  exit 0
fi

echo "=========================================="
echo "Phase 5 Ablation Study (4-GPU Parallel)"
echo "=========================================="
echo "MAT_PATH: $MAT_PATH"
echo "LIDAR_H5_KEY (requested): $LIDAR_H5_KEY"
echo "LIDAR_ROOT: $LIDAR_ROOT"
echo "EPOCHS: $EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GPU_LIST_RAW: $GPU_LIST_RAW"
echo "BATCH_SIZE_FUSION (legacy arg): $BATCH_SIZE_FUSION"
echo "DEPTH_SCALE: $DEPTH_SCALE"
echo "LIDAR_DEPTH_SCALE: $LIDAR_DEPTH_SCALE"
echo "LIDAR_DROPOUT_PROB: $LIDAR_DROPOUT_PROB"
echo "=========================================="
echo ""

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Parse GPU list (supports comma or space separated)
GPU_LIST_CSV="${GPU_LIST_RAW// /,}"
IFS=',' read -r -a GPU_POOL <<< "$GPU_LIST_CSV"
if [ "${#GPU_POOL[@]}" -lt 4 ]; then
  echo "❌ Need at least 4 GPUs for 4-way parallel run. Got: ${GPU_POOL[*]}"
  exit 1
fi

echo "Using GPUs: ${GPU_POOL[*]}"
echo ""

# Common arguments (array form avoids eval/string escaping issues)
COMMON_ARGS=(
  --train_root "$MAT_PATH"
  --val_root "$MAT_PATH"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --depth_scale "$DEPTH_SCALE"
  --lidar_depth_scale "$LIDAR_DEPTH_SCALE"
  --image_shape 384 384
  --encoder_name convnextv2_large
  --hidden_dim 512
  --num_heads 8
  --expansion 4
  --depth_loss_weight 10.0
  --camera_loss_weight 0.5
  --invariance_loss_weight 0.1
  --lidar_loss_weight 0.5
  --lr 1e-4
  --weight_decay 0.01
  --save_every "$SAVE_EVERY"
  --log_every 50
  --num_workers 4
  --lidar_dropout_prob "$LIDAR_DROPOUT_PROB"
  --phase4_eval_fallback true
)

# Resolve/validate LiDAR source configuration up front
if [ -n "$LIDAR_ROOT" ]; then
  COMMON_ARGS+=(--use_lidar true --lidar_root "$LIDAR_ROOT")

  # If user explicitly specifies a non-auto key, keep it and validate.
  if [ -n "$LIDAR_H5_KEY" ] && [ "$LIDAR_H5_KEY" != "auto" ]; then
    KEY_EXISTS="$($PYTHON_BIN - <<PY
import h5py
mat_path = "$MAT_PATH"
key = "$LIDAR_H5_KEY"
try:
    with h5py.File(mat_path, "r") as h5:
        print("1" if key in h5 else "0")
except Exception:
    print("0")
PY
)"
    if [ "$KEY_EXISTS" = "1" ]; then
      RESOLVED_LIDAR_H5_KEY="$LIDAR_H5_KEY"
      COMMON_ARGS+=(--lidar_h5_key "$RESOLVED_LIDAR_H5_KEY")
    else
      echo "❌ Explicit lidar_h5_key '$LIDAR_H5_KEY' not found in $MAT_PATH"
      echo "   Remove the key (use 'auto') or provide a valid key."
      exit 1
    fi
  fi
else
  RESOLVED_LIDAR_H5_KEY="$($PYTHON_BIN - <<PY
import h5py
mat_path = "$MAT_PATH"
requested = "$LIDAR_H5_KEY"
candidates = ["lidar_depths", "rawDepths", "lidar"]
try:
    with h5py.File(mat_path, "r") as h5:
        if requested and requested != "auto":
            print(requested if requested in h5 else "")
        else:
            for key in candidates:
                if key in h5:
                    print(key)
                    break
            else:
                print("")
except Exception:
    print("")
PY
)"

  if [ -z "$RESOLVED_LIDAR_H5_KEY" ]; then
    echo "❌ No valid LiDAR HDF5 key found in $MAT_PATH"
    echo "   Tried: lidar_depths, rawDepths, lidar"
    echo "   Provide a valid key as arg2 or pass LIDAR_ROOT as arg3."
    exit 1
  fi

  COMMON_ARGS+=(--use_lidar true --lidar_h5_key "$RESOLVED_LIDAR_H5_KEY")
fi

echo "LIDAR_H5_KEY (resolved): ${RESOLVED_LIDAR_H5_KEY:-<none>}"

# Timestamp for grouping
TIMESTAMP=$(date +%s)
MAP_FILE="runs/phase5_fullscale_${TIMESTAMP}.json"
MAP_LATEST="runs/phase5_fullscale_latest.json"
SUMMARY_LOG="runs/phase5_fullscale_${TIMESTAMP}.log"
PROGRESS_LOG="runs/phase5_progress_${TIMESTAMP}.log"
MONITOR_STOP_FILE="runs/.phase5_monitor_stop_${TIMESTAMP}"

mkdir -p runs

RESULT_RGB="runs/.phase5_result_${TIMESTAMP}_rgb_only.txt"
RESULT_SUP="runs/.phase5_result_${TIMESTAMP}_supervision_only.txt"
RESULT_LATE="runs/.phase5_result_${TIMESTAMP}_late_fusion.txt"
RESULT_TOKEN="runs/.phase5_result_${TIMESTAMP}_token_fusion.txt"

echo "Experiment timestamp: $TIMESTAMP"
echo "Python binary: $PYTHON_BIN"
echo "Run mapping file: $MAP_FILE"
echo "Summary log: $SUMMARY_LOG"
echo "Progress log (1-min): $PROGRESS_LOG"
echo ""

declare -A PID_TO_GPU
declare -A PID_TO_METHOD
declare -A PID_TO_RESULT_FILE

start_variant_bg() {
  local method_name="$1"
  local ablation="$2"
  local gpu_id="$3"
  local result_file="$4"
  shift 4

  local method_slug
  method_slug="$(echo "$ablation" | tr '-' '_' )"
  local run_name="train_depth_${TIMESTAMP}_${method_slug}_gpu${gpu_id}"
  local run_path="runs/${run_name}"
  local run_log="runs/phase5_${TIMESTAMP}_${method_slug}_gpu${gpu_id}.log"

  echo "[START] $method_name on GPU $gpu_id"
  (
    set +e
    "$PYTHON_BIN" -m train.train_depth "${COMMON_ARGS[@]}" --cuda "$gpu_id" --run_name "$run_name" --phase5_ablation "$ablation" "$@" 2>&1 | tee "$run_log"
    local code=${PIPESTATUS[0]}
    if [ ! -d "$run_path" ]; then
      detected_run_path="$(grep -o 'runs/train_depth_[^ ]\+' "$run_log" | tail -1 || true)"
      if [ -n "$detected_run_path" ]; then
        run_path="$detected_run_path"
      fi
    fi
    echo "$method_name|$ablation|$gpu_id|${run_path}|$code|$run_log" > "$result_file"
    exit "$code"
  ) &

  local pid=$!
  PID_TO_GPU[$pid]="$gpu_id"
  PID_TO_METHOD[$pid]="$method_name"
  PID_TO_RESULT_FILE[$pid]="$result_file"
}

collect_result() {
  local result_file="$1"
  if [ ! -f "$result_file" ]; then
    return 1
  fi
  local line
  line="$(cat "$result_file")"
  echo "$line" >> "$SUMMARY_LOG"
  return 0
}

organize_current_run() {
  local timestamp="$1"
  local exp_folder="runs/phase5_experiment_${timestamp}"
  
  # Skip if already organized
  if [ -d "$exp_folder" ]; then
    echo "⏭️  Already organized: $exp_folder"
    return 0
  fi
  
  echo ""
  echo "📦 Organizing experiment files for timestamp: $timestamp"
  mkdir -p "$exp_folder"
  
  local moved_count=0
  
  # Move training directories
  for train_dir in runs/train_depth_${timestamp}_*; do
    if [ -d "$train_dir" ]; then
      mv "$train_dir" "$exp_folder/" && ((moved_count+=1))
    fi
  done
  
  # Move individual experiment log files
  for log_file in runs/phase5_${timestamp}_*.log; do
    if [ -f "$log_file" ]; then
      mv "$log_file" "$exp_folder/" && ((moved_count+=1))
    fi
  done
  
  # Move JSON mapping files
  for json_file in runs/phase5_fullscale_${timestamp}*.json; do
    if [ -f "$json_file" ]; then
      mv "$json_file" "$exp_folder/" && ((moved_count+=1))
    fi
  done
  
  # Move summary and progress logs
  for summary_file in runs/phase5_fullscale_${timestamp}.log runs/phase5_progress_${timestamp}.log; do
    if [ -f "$summary_file" ]; then
      mv "$summary_file" "$exp_folder/" && ((moved_count+=1))
    fi
  done
  
  # Clean up temp result files
  rm -f runs/.phase5_result_${timestamp}_*.txt
  
  echo "✅ Organized into: $exp_folder ($moved_count items grouped)"
}

echo "════════════════════════════════════════════════════════════════"
echo "[Launch] 4-way parallel on GPU ${GPU_POOL[0]}/${GPU_POOL[1]}/${GPU_POOL[2]}/${GPU_POOL[3]}"
echo "════════════════════════════════════════════════════════════════"

# Start minute-level monitor
rm -f "$MONITOR_STOP_FILE"
"$PYTHON_BIN" -m scripts.monitor_phase5_progress \
  --timestamp "$TIMESTAMP" \
  --epochs "$EPOCHS" \
  --interval 60 \
  --stop_file "$MONITOR_STOP_FILE" \
  --output_log "$PROGRESS_LOG" &
MONITOR_PID=$!
echo "[MONITOR] Started (PID=$MONITOR_PID), report every 1 minute"
echo ""

# Launch all 4 variants in parallel
start_variant_bg "RGB-only" "rgb_only" "${GPU_POOL[0]}" "$RESULT_RGB"
start_variant_bg "Supervision-only" "supervision_only" "${GPU_POOL[1]}" "$RESULT_SUP"
start_variant_bg "Late fusion" "late_fusion" "${GPU_POOL[2]}" "$RESULT_LATE" --lidar_fusion_type late
start_variant_bg "Token fusion" "token_fusion" "${GPU_POOL[3]}" "$RESULT_TOKEN" --lidar_fusion_type token

while [ "${#PID_TO_GPU[@]}" -gt 0 ]; do
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      set +e
      wait "$pid"
      status=$?
      set -e

      method="${PID_TO_METHOD[$pid]}"
      gpu_id="${PID_TO_GPU[$pid]}"
      result_file="${PID_TO_RESULT_FILE[$pid]}"

      echo "[DONE] $method on GPU $gpu_id (exit=$status)"
      collect_result "$result_file" || true

      unset PID_TO_GPU[$pid]
      unset PID_TO_METHOD[$pid]
      unset PID_TO_RESULT_FILE[$pid]
    fi
  done
  if [ "${#PID_TO_GPU[@]}" -gt 0 ]; then
    sleep 10
  fi
done

echo ""
echo "All 4 training jobs finished."

# Stop monitor gracefully
touch "$MONITOR_STOP_FILE"
set +e
wait "$MONITOR_PID" 2>/dev/null
set -e
rm -f "$MONITOR_STOP_FILE"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

rows = []
result_files = [
    Path("$RESULT_RGB"),
    Path("$RESULT_SUP"),
    Path("$RESULT_LATE"),
    Path("$RESULT_TOKEN"),
]
for rf in result_files:
  if not rf.exists():
    continue
  line = rf.read_text(encoding="utf-8").strip()
  if not line or "|" not in line:
    continue
  method, ablation, gpu_id, run_path, exit_code, run_log = line.split("|", 5)
  rows.append({
      "method": method,
      "ablation": ablation,
      "gpu": int(gpu_id),
      "run_path": run_path,
      "exit_code": int(exit_code),
      "run_log": run_log,
  })

payload = {
  "type": "phase5_fullscale_parallel",
  "timestamp": "$TIMESTAMP",
  "epochs": int("$EPOCHS"),
  "batch_size": int("$BATCH_SIZE"),
  "gpus": [int(x) for x in "$GPU_LIST_CSV".split(",") if x],
  "lidar_h5_key": "${RESOLVED_LIDAR_H5_KEY:-}",
  "runs": rows,
}

Path("$MAP_FILE").write_text(json.dumps(payload, indent=2), encoding="utf-8")
Path("$MAP_LATEST").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Saved run mapping to $MAP_FILE")
print(f"Updated latest mapping: $MAP_LATEST")
PY

rm -f "$RESULT_RGB" "$RESULT_SUP" "$RESULT_LATE" "$RESULT_TOKEN"

echo ""
echo "Generating plots automatically..."
VALID_TB_RUNS="$($PYTHON_BIN - <<PY
import json
from pathlib import Path

payload_path = Path("$MAP_FILE")
if not payload_path.exists():
    print(0)
    raise SystemExit(0)

payload = json.loads(payload_path.read_text(encoding="utf-8"))
count = 0
for run in payload.get("runs", []):
    if run.get("exit_code") != 0:
        continue
    run_path = run.get("run_path") or ""
    if not run_path:
        continue
    tb_dir = Path(run_path) / "tensorboard"
    if any(tb_dir.glob("events.out.tfevents.*")):
        count += 1
print(count)
PY
)"

if [ "${VALID_TB_RUNS:-0}" -gt 0 ]; then
  "$PYTHON_BIN" -m scripts.compare_v2_methods \
    --runs_json "$MAP_FILE" \
    --output_dir docs/figures \
    --title_prefix "NYUv2 V2 Full-scale Phase5"
else
  echo "[WARN] No successful runs with TensorBoard events found; skipping plot generation."
  echo "[HINT] Check per-run logs under runs/phase5_${TIMESTAMP}_*.log"
fi

# Automatically organize this run's files (after plotting to keep run_path valid)
organize_current_run "$TIMESTAMP"

echo "Plots saved under docs/figures"

echo "════════════════════════════════════════════════════════════════"
echo "✅ Phase 5 Ablation Complete"
echo "════════════════════════════════════════════════════════════════"
echo "Organized output folder: runs/phase5_experiment_${TIMESTAMP}/"
echo "  ├── All 4 training directories"
echo "  ├── All experiment logs"
echo "  └── Metadata JSON files"
echo ""
echo "Comparison figures: docs/figures/"
echo "  ├── v2_training_error_lineplot.png"
echo "  └── v2_testing_error_lineplot.png"
echo ""
