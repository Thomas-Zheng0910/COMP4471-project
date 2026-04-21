#!/bin/bash

# Organize Phase 5 Ablation Runs by experiment timestamp
# Groups training folders, logs, and result files into organized experiment folders
# Usage: ./organize_phase5_runs.sh [runs_directory]

RUNS_DIR="${1:-runs}"

if [ ! -d "$RUNS_DIR" ]; then
  echo "❌ Directory not found: $RUNS_DIR"
  exit 1
fi

echo "=========================================="
echo "Organizing Phase 5 Ablation Runs"
echo "=========================================="
echo "Source directory: $RUNS_DIR"
echo ""

# Find all unique timestamps from training directories and logs
TIMESTAMPS=$(find "$RUNS_DIR" -maxdepth 1 -type d -name "train_depth_*" | sed 's/.*train_depth_//' | sed 's/_[a-z_]*$//' | sort -u)
TIMESTAMPS="$TIMESTAMPS $(find "$RUNS_DIR" -maxdepth 1 -type f -name "phase5_*" | sed 's|.*/phase5_||' | sed 's/_.*//' | sort -u)"
TIMESTAMPS=$(echo "$TIMESTAMPS" | tr ' ' '\n' | sort -u)

if [ -z "$TIMESTAMPS" ]; then
  echo "✅ No Phase 5 runs found to organize."
  exit 0
fi

echo "Found experiment timestamps:"
echo "$TIMESTAMPS" | while read ts; do
  echo "  - $ts"
done
echo ""

# Process each timestamp
for TIMESTAMP in $TIMESTAMPS; do
  if [ -z "$TIMESTAMP" ]; then
    continue
  fi
  
  # Create experiment folder
  EXP_FOLDER="$RUNS_DIR/phase5_experiment_${TIMESTAMP}"
  
  # Check if already organized
  if [ -d "$EXP_FOLDER" ]; then
    echo "⏭️  Already organized: $EXP_FOLDER"
    continue
  fi
  
  echo "📦 Processing timestamp: $TIMESTAMP"
  mkdir -p "$EXP_FOLDER"
  
  # Move training directories
  for train_dir in "$RUNS_DIR"/train_depth_${TIMESTAMP}_*; do
    if [ -d "$train_dir" ]; then
      mv "$train_dir" "$EXP_FOLDER/" && echo "  ✓ Moved $(basename "$train_dir")"
    fi
  done
  
  # Move log files
  for log_file in "$RUNS_DIR"/phase5_${TIMESTAMP}_*.log; do
    if [ -f "$log_file" ]; then
      mv "$log_file" "$EXP_FOLDER/" && echo "  ✓ Moved $(basename "$log_file")"
    fi
  done
  
  # Move JSON mapping files
  for json_file in "$RUNS_DIR"/phase5_fullscale_${TIMESTAMP}*.json; do
    if [ -f "$json_file" ]; then
      mv "$json_file" "$EXP_FOLDER/" && echo "  ✓ Moved $(basename "$json_file")"
    fi
  done
  
  # Move summary and progress logs
  for summary_file in "$RUNS_DIR"/phase5_fullscale_${TIMESTAMP}.log "$RUNS_DIR"/phase5_progress_${TIMESTAMP}.log; do
    if [ -f "$summary_file" ]; then
      mv "$summary_file" "$EXP_FOLDER/" && echo "  ✓ Moved $(basename "$summary_file")"
    fi
  done
  
  # Clean up temp result files if any exist
  rm -f "$RUNS_DIR"/.phase5_result_${TIMESTAMP}_*.txt
  
  echo "  ✅ Organized into: $EXP_FOLDER"
  echo ""
done

echo "=========================================="
echo "✅ Organization complete!"
echo ""
echo "Directory structure:"
find "$RUNS_DIR" -maxdepth 2 -type d -name "phase5_experiment_*" | sort | while read folder; do
  echo "  $folder"
  ls -la "$folder" | tail -n +4 | sed 's/^/    /'
done
