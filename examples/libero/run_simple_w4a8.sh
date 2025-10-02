#!/bin/bash
# Simple W4A8 DuQuant configuration (no permutation, no rotation)
# This is the most basic quantization - faster but potentially lower accuracy
#
# Configuration comparison:
# ┌─────────────────┬──────────────────────┬─────────────────────────┐
# │   Configuration │ This Script (Simple) │ run_optimized_duquant.sh│
# ├─────────────────┼──────────────────────┼─────────────────────────┤
# │ Weight bits     │ 4                    │ 4                       │
# │ Activation bits │ 8                    │ 8                       │
# │ Permutation     │ OFF (0)              │ ON (1)                  │
# │ Row Rotation    │ OFF (0)              │ ON (restore)            │
# │ Lambda Smooth   │ N/A                  │ 0.15                    │
# │ Speed (no torch.│ ~1-2 min/episode     │ ~2-3 min/episode        │
# │    compile)     │                      │                         │
# │ Accuracy        │ Lower                │ Higher                  │
# │ Pack size       │ Smaller              │ Larger                  │
# └─────────────────┴──────────────────────┴─────────────────────────┘

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch

# Check CKPT
if [ -z "$CKPT" ]; then
    echo "Error: CKPT environment variable must be set"
    echo "Usage: export CKPT=/path/to/checkpoint"
    exit 1
fi

# Set environment
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

# ============================================
# Simple W4A8 DuQuant Configuration
# ============================================
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=0           # Disabled - no input permutation
export OPENPI_DUQUANT_ROW_ROT=0           # Disabled - no rotation matrices
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
# NOTE: lambda_smooth is not used when permutation is disabled

# Disable torch.compile for faster startup (can be re-enabled for better throughput)
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# Pack directory for simple W4A8 (no permutation/rotation)
# This will be generated on first run if it doesn't exist
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_w4a8_simple"

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant SIMPLE W4A8 (No Permutation/Rotation)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (Simple W4A8):"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "  BLOCK=$OPENPI_DUQUANT_BLOCK"
echo "  PERMUTE=$OPENPI_DUQUANT_PERMUTE (disabled - no permutation)"
echo "  ROW_ROT=$OPENPI_DUQUANT_ROW_ROT (disabled - no rotation)"
echo "  ACT_PCT=$OPENPI_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$OPENPI_DUQUANT_CALIB_STEPS"
echo "  PACKDIR=$OPENPI_DUQUANT_PACKDIR"
echo ""
echo "⚡ FEATURES:"
echo "  ✅ Basic W4A8 fake quantization"
echo "  ✅ Block-wise quantization (block_size=16)"
echo "  ✅ Pre-quantized weights (no repeated fake_quantize)"
echo "  ❌ Input permutation disabled (faster, may reduce accuracy)"
echo "  ❌ Rotation matrices disabled (faster, may reduce accuracy)"
echo "  ❌ torch.compile disabled (faster startup, slower per-episode)"
echo ""
echo "Expected speed:"
echo "  Episode 1: ~1-2 min (may include pack generation)"
echo "  Episode 2+: ~1-2 min each"
echo ""
echo "NOTE: This is the simplest quantization config for quick testing."
echo "      For better accuracy, use run_optimized_duquant.sh instead."
echo "========================================"
echo ""

# Run evaluation with timing
time python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task 5 \
  --args.seed "$SEED"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Check results in: results/libero/"
echo "========================================"
