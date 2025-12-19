#!/bin/bash
# FP16 Baseline Evaluation for Pi0.5
#
# Standard FP16 precision without any quantization.
# This serves as the baseline for comparison with W8A8 and other quantization methods.
#
# Usage:
#   bash examples/libero/run_fp16_baseline.sh
#
# Run in parallel with W8A8 in another terminal:
#   Terminal 1: OPENPI_W8A8_ENABLE=1 bash examples/libero/run_w8a8.sh
#   Terminal 2: bash examples/libero/run_fp16_baseline.sh

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

# Fix CUDA library version mismatch
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch

if [ -z "$CKPT" ]; then
    echo "Error: CKPT environment variable must be set"
    exit 1
fi

# Set environment
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# FP16 Baseline Configuration
# ============================================
# Disable all quantization
unset OPENPI_W8A8_ENABLE
unset OPENPI_DUQUANT_WBITS_DEFAULT
unset OPENPI_DUQUANT_ABITS
unset OPENPI_BITBLAS_ENABLE

# Disable torch.compile for faster startup
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "FP16 Baseline (No Quantization)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "Configuration:"
echo "  W8A8: DISABLED"
echo "  DuQuant: DISABLED"
echo "  BitBLAS: DISABLED"
echo ""
echo "Expected Performance:"
echo "  Memory: ~18GB (full FP16)"
echo "  Speed: baseline (1.0x)"
echo "  Accuracy: ~76% (baseline)"
echo "========================================"
echo ""

# Run evaluation
echo "Starting evaluation..."
time python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task "$NUM_TRIALS" \
  --args.seed "$SEED"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
