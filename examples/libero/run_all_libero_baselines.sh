#!/bin/bash
# Run Pi0.5 FP baseline evaluation on all 4 LIBERO task suites
#
# This script runs the baseline (no quantization) evaluation on:
#   - libero_spatial (10 tasks)
#   - libero_object (10 tasks)
#   - libero_goal (10 tasks)
#   - libero_10 (10 tasks)
#
# Total: 40 tasks x 20 trials each = 800 trials

set -euo pipefail

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_SCRIPT="$SCRIPT_DIR/run_fp_baseline.sh"

if [ ! -f "$BASELINE_SCRIPT" ]; then
  echo "Error: baseline script not found: $BASELINE_SCRIPT" >&2
  exit 1
fi

# Task suites to evaluate
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# Configuration
NUM_TRIALS=${NUM_TRIALS:-20}
SEED=${SEED:-42}

# Summary file
SUMMARY_LOG="logs/all_baselines_summary_$(date +%F_%H%M%S).log"
mkdir -p logs

echo "=============================================="
echo "Pi0.5 FP Baseline - All LIBERO Suites"
echo "=============================================="
echo "Running 4 task suites: ${TASK_SUITES[*]}"
echo "Trials per task: $NUM_TRIALS"
echo "Seed: $SEED"
echo "Summary will be saved to: $SUMMARY_LOG"
echo "=============================================="
echo ""

# Start time
START_TIME=$(date +%s)

# Run each task suite
for i in "${!TASK_SUITES[@]}"; do
  SUITE="${TASK_SUITES[$i]}"
  SUITE_NUM=$((i + 1))

  echo ""
  echo "=========================================="
  echo "[$SUITE_NUM/4] Running $SUITE"
  echo "=========================================="
  echo ""

  # Run baseline with specific task suite
  TASK_SUITE="$SUITE" \
  NUM_TRIALS="$NUM_TRIALS" \
  SEED="$SEED" \
  bash "$BASELINE_SCRIPT"

  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ $SUITE completed successfully"
  else
    echo "✗ $SUITE failed with exit code $EXIT_CODE"
    echo "  Continuing with remaining suites..."
  fi

  echo ""
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=============================================="
echo "All Baseline Evaluations Complete"
echo "=============================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Generate summary
{
  echo "=============================================="
  echo "Pi0.5 FP Baseline - Summary"
  echo "Date: $(date)"
  echo "=============================================="
  echo ""
  echo "Configuration:"
  echo "  Trials per task: $NUM_TRIALS"
  echo "  Seed: $SEED"
  echo "  Quantization: DISABLED (FP/BF16)"
  echo ""
  echo "Results by Task Suite:"
  echo "----------------------------------------------"

  for SUITE in "${TASK_SUITES[@]}"; do
    RESULT_DIR="results/libero_fp_baseline/${SUITE}"

    if [ -d "$RESULT_DIR" ]; then
      # Try to find success rate in results
      SUCCESS_FILE="$RESULT_DIR/success_rate.txt"
      if [ -f "$SUCCESS_FILE" ]; then
        SUCCESS_RATE=$(cat "$SUCCESS_FILE")
        echo "  $SUITE: $SUCCESS_RATE"
      else
        # Count result files
        NUM_FILES=$(find "$RESULT_DIR" -name "*.json" 2>/dev/null | wc -l)
        echo "  $SUITE: $NUM_FILES result files found"
      fi
    else
      echo "  $SUITE: No results found"
    fi
  done

  echo "----------------------------------------------"
  echo ""
  echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
  echo ""
  echo "Individual logs:"
  for SUITE in "${TASK_SUITES[@]}"; do
    LOG_FILE=$(ls -t logs/fp_baseline_${SUITE}_*.log 2>/dev/null | head -1)
    if [ -n "$LOG_FILE" ]; then
      echo "  $SUITE: $LOG_FILE"
    fi
  done
  echo ""
  echo "=============================================="
} | tee "$SUMMARY_LOG"

echo ""
echo "Summary saved to: $SUMMARY_LOG"
echo ""
echo "To view results:"
echo "  ls -lh results/libero_fp_baseline/*/"
echo ""
echo "To view logs:"
echo "  ls -lh logs/fp_baseline_*"
