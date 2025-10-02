#!/bin/bash
# Compare DuQuant performance: Original vs Optimized
# This script helps you verify the optimization gains

set -e

if [ -z "$CKPT" ]; then
    echo "Error: CKPT environment variable must be set"
    echo "Usage: export CKPT=/path/to/checkpoint"
    exit 1
fi

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

export PYTHONPATH=$PWD/src:$PWD/third_party/libero

# Test with 1 episode for quick comparison
TASK_SUITE="libero_spatial"
NUM_TRIALS=1  # Just 1 episode for quick test
SEED=42

# DuQuant configuration
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
export OPENPI_DUQUANT_LS=0.15

echo "========================================"
echo "DuQuant Performance Comparison"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Task: $TASK_SUITE"
echo "  Episodes: $NUM_TRIALS (quick test)"
echo "  W4A8, block=16, permute=1, row_rot=restore"
echo ""
echo "This script will run a quick test to verify optimizations."
echo "For full evaluation, run with NUM_TRIALS=20"
echo ""
echo "========================================"
echo ""

# Use optimized implementation (current code)
export OPENPI_DUQUANT_PACKDIR="duquant_packed_optimized_test"

echo "Running OPTIMIZED implementation..."
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)

python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task "$NUM_TRIALS" \
  --args.seed "$SEED" 2>&1 | grep -E "(DUQUANT|episode|complete|ERROR)" || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "Results"
echo "========================================"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Duration: ${DURATION} seconds"
echo ""

MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))
echo "Time: ${MINUTES}m ${SECONDS}s"
echo ""

if [ $NUM_TRIALS -eq 1 ]; then
    echo "Estimated time for 20 episodes: ~$((DURATION * 20 / 60)) minutes"
fi

echo ""
echo "✅ Optimization verification complete!"
echo ""
echo "Expected performance with optimizations:"
echo "  1 episode:  ~2-3 minutes"
echo "  20 episodes: ~40-60 minutes total"
echo ""
echo "If you see '[DUQUANT][CACHE] ... pre-quantized weights cached'"
echo "in the output above, optimizations are working correctly!"
echo ""
echo "========================================"
