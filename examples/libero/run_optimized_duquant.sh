#!/bin/bash
# Test optimized DuQuant implementation with W4A8 default configuration
# This uses the optimized code with pre-cached tensors and pre-quantized weights

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
# Full DuQuant Configuration (W4A8 + Permutation + Rotation)
# ============================================
# This uses all DuQuant optimizations for best accuracy:
# - Input permutation (PERMUTE=1)
# - Row rotation with output restoration (ROW_ROT=restore)
# - Lambda smoothing (LS=0.15)
#
# For simpler/faster config without permutation/rotation, use run_simple_w4a8.sh
# ============================================

export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1           # Enable input permutation
export OPENPI_DUQUANT_ROW_ROT=restore     # Enable rotation with output restoration
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
export OPENPI_DUQUANT_LS=0.15             # Lambda smooth for permutation

# torch.compile configuration
# OPTION 1 (RECOMMENDED): Enable torch.compile for best throughput
#   - First episode: ~15-20 min (compilation overhead)
#   - Subsequent episodes: ~30-60s each
# OPTION 2: Disable torch.compile for faster startup
#   - All episodes: ~2-3 min each
#
# Current setting: DISABLED (change to unset for OPTION 1)
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# Disable CUDA graphs to avoid empty graph warnings
# (Only relevant if torch.compile is enabled)
export TORCH_CUDA_GRAPH_DISABLE=1

# Pack directory
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_b16_p1_rrestore_a999"

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant FULL W4A8 (Permutation + Rotation)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (Full W4A8):"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "  BLOCK=$OPENPI_DUQUANT_BLOCK"
echo "  PERMUTE=$OPENPI_DUQUANT_PERMUTE (enabled)"
echo "  ROW_ROT=$OPENPI_DUQUANT_ROW_ROT (output restoration enabled)"
echo "  ACT_PCT=$OPENPI_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$OPENPI_DUQUANT_CALIB_STEPS"
echo "  LS=$OPENPI_DUQUANT_LS"
echo "  PACKDIR=$OPENPI_DUQUANT_PACKDIR"
echo ""
echo "⚡ FEATURES:"
echo "  ✅ Full W4A8 fake quantization"
echo "  ✅ Input permutation enabled (better accuracy)"
echo "  ✅ Rotation matrices with output restoration (better accuracy)"
echo "  ✅ Pre-cached rotation matrices (optimized)"
echo "  ✅ Pre-quantized weights (no repeated fake_quantize)"
echo "  ✅ Optimized forward pass (dict reuse)"

if [ "$OPENPI_DISABLE_TORCH_COMPILE" = "1" ]; then
    echo "  ❌ torch.compile DISABLED (faster startup, slower per-episode)"
    echo ""
    echo "Expected speed:"
    echo "  All episodes: ~2-3 min each"
else
    echo "  ✅ torch.compile ENABLED (CUDA kernel fusion)"
    echo ""
    echo "Expected speed:"
    echo "  Episode 1: ~15-20 min (torch.compile compilation)"
    echo "  Episode 2+: ~30-60s each (using cached compilation)"
fi

echo ""
echo "NOTE: For simpler config without permutation/rotation,"
echo "      use run_simple_w4a8.sh instead (faster but lower accuracy)."
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
