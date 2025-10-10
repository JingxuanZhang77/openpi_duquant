#!/bin/bash
# Run LLM W4A8 quantization with FAKE quantization (no BitBLAS)
# This provides accuracy benefits of quantization-aware optimization
# without requiring INT4 kernel support

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch

export PYTHONPATH=$PWD/src:$PWD/third_party/libero

# DuQuant Configuration for LLM W4A8 (FAKE quantization)
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
export OPENPI_DUQUANT_LS=0.15

# Use FAKE quantization (no real INT4 kernels)
export OPENPI_DUQUANT_BACKEND=fake
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_llm_w4a8_fake"
export OPENPI_DUQUANT_PROFILE=1

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant: LLM Quantization (W4A8 FAKE)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (LLM W4A8 FAKE):"
echo "  TARGET: Language Model (Gemma LLM)"
echo "  BACKEND: FAKE (no INT4 kernels)"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "========================================"
echo ""

# Run evaluation
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
echo "========================================"
