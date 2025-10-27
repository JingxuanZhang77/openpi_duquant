#!/bin/bash
# Helper script to compare different DuQuant configurations
# Runs all configurations in sequence and compares results

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

# Configuration
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "DuQuant Configuration Comparison"
echo "========================================"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "This will run 4 experiments:"
echo "  1. Baseline (no DuQuant)"
echo "  2. Original (block_size=16, calib=32)"
echo "  3. Improved calib (block_size=16, calib=128)"
echo "  4. Head-aligned (block_size=64, calib=128)"
echo ""
echo "Total estimated time: 2-4 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create results directory
RESULTS_DIR="results/libero/duquant_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Experiment 1: Baseline (no DuQuant)
echo "========================================"
echo "Experiment 1/4: Baseline (no DuQuant)"
echo "========================================"
unset OPENPI_DUQUANT_WBITS_DEFAULT
unset OPENPI_DUQUANT_ABITS
unset OPENPI_DUQUANT_PACKDIR
unset OPENPI_DUQUANT_SCOPE
unset OPENPI_DUQUANT_INCLUDE
unset OPENPI_DUQUANT_EXCLUDE
unset OPENPI_DUQUANT_DEBUG
unset OPENPI_DUQUANT_PROFILE
unset OPENPI_DUQUANT_BLOCK
unset OPENPI_DUQUANT_CALIB_STEPS
unset OPENPI_DUQUANT_PERMUTE
unset OPENPI_DUQUANT_ROW_ROT

export TASK_SUITE="$TASK_SUITE"
export NUM_TRIALS="$NUM_TRIALS"
export SEED="$SEED"

bash examples/libero/run_headless.sh 2>&1 | tee "$RESULTS_DIR/1_baseline.log"

echo ""
echo "Experiment 1 complete! Check $RESULTS_DIR/1_baseline.log"
echo ""
sleep 2

# Experiment 2: Original (block_size=16, calib=32)
echo "========================================"
echo "Experiment 2/4: Original DuQuant"
echo "========================================"
export TASK_SUITE="$TASK_SUITE"
export NUM_TRIALS="$NUM_TRIALS"
export SEED="$SEED"

bash examples/libero/run_llm_dit_mlp_w4a8.sh 2>&1 | tee "$RESULTS_DIR/2_original_block16_calib32.log"

echo ""
echo "Experiment 2 complete! Check $RESULTS_DIR/2_original_block16_calib32.log"
echo ""
sleep 2

# Experiment 3: Improved calibration (block_size=16, calib=128)
echo "========================================"
echo "Experiment 3/4: Improved Calibration"
echo "========================================"
export TASK_SUITE="$TASK_SUITE"
export NUM_TRIALS="$NUM_TRIALS"
export SEED="$SEED"

bash examples/libero/run_llm_dit_mlp_w4a8_improved_calib.sh 2>&1 | tee "$RESULTS_DIR/3_improved_calib_block16_calib128.log"

echo ""
echo "Experiment 3 complete! Check $RESULTS_DIR/3_improved_calib_block16_calib128.log"
echo ""
sleep 2

# Experiment 4: Head-aligned (block_size=64, calib=128)
echo "========================================"
echo "Experiment 4/4: Head-Aligned Quantization"
echo "========================================"
export TASK_SUITE="$TASK_SUITE"
export NUM_TRIALS="$NUM_TRIALS"
export SEED="$SEED"

bash examples/libero/run_llm_dit_mlp_w4a8_head_aligned.sh 2>&1 | tee "$RESULTS_DIR/4_head_aligned_block64_calib128.log"

echo ""
echo "Experiment 4 complete! Check $RESULTS_DIR/4_head_aligned_block64_calib128.log"
echo ""

# Parse results and create summary
echo ""
echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo ""
echo "Parsing results..."
echo ""

# Function to extract success rate from log
extract_success_rate() {
    local log_file=$1
    # Look for lines like "Task X: Y/20 successes"
    # Average across all tasks
    grep -E "Task.*successes" "$log_file" | \
        awk -F'/' '{sum+=$1} END {if(NR>0) print sum/(NR*20)*100; else print "N/A"}'
}

# Create summary
SUMMARY_FILE="$RESULTS_DIR/SUMMARY.txt"

cat > "$SUMMARY_FILE" <<EOF
DuQuant Configuration Comparison Summary
=========================================
Date: $(date)
Task Suite: $TASK_SUITE
Num Trials: $NUM_TRIALS
Seed: $SEED

Results:
--------
EOF

for i in 1 2 3 4; do
    case $i in
        1) name="Baseline (no DuQuant)" ;;
        2) name="Original (block=16, calib=32)" ;;
        3) name="Improved calib (block=16, calib=128)" ;;
        4) name="Head-aligned (block=64, calib=128)" ;;
    esac

    log_file=$(ls "$RESULTS_DIR"/${i}_*.log 2>/dev/null | head -1)
    if [ -f "$log_file" ]; then
        # Try to extract success rate
        rate=$(extract_success_rate "$log_file")
        echo "$i. $name" >> "$SUMMARY_FILE"
        echo "   Success rate: ${rate}%" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" <<EOF

Expected Results (for reference):
----------------------------------
1. Baseline: ~92.4%
2. Original: ~80.5% (drop of -11.9%)
3. Improved calib: ~85-88% (drop of -4 to -7%)
4. Head-aligned: ~88-91% (drop of -1 to -4%)

Analysis:
---------
- If improved calib > original: Calibration distribution was an issue
- If head-aligned > improved calib: Head misalignment was the main issue
- If head-aligned ≈ baseline: Problem fully solved! ✅

Files:
------
$(ls -1 "$RESULTS_DIR")

For detailed logs, check: $RESULTS_DIR/
EOF

# Display summary
cat "$SUMMARY_FILE"

echo ""
echo "========================================"
echo "Summary saved to: $SUMMARY_FILE"
echo "All logs saved to: $RESULTS_DIR/"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review the summary above"
echo "  2. Check detailed logs in $RESULTS_DIR/"
echo "  3. If head-aligned still not satisfactory, consider:"
echo "     - Per-head quantization scales (see DUQUANT_LONG_TASK_FIX.md)"
echo "     - Mixed-precision quantization (W6A8 for attention)"
echo "========================================"
