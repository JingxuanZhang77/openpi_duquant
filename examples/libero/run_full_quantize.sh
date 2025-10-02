#!/bin/bash
# Full model quantization: LLM + DiT + Vision
# This applies W4A8 quantization to ALL transformer components
#
# Model Structure:
# ┌────────────────────────────────────────────────────────────┐
# │ OpenPI Model (PI0Pytorch)                                  │
# │                                                            │
# │ ├── paligemma_with_expert.paligemma                        │
# │ │   ├── vision_tower (SigLIP - 27 layers)    ← QUANTIZE   │
# │ │   └── language_model (Gemma LLM - 18 layers) ← QUANTIZE │
# │ │                                                          │
# │ └── paligemma_with_expert.gemma_expert (DiT - 18 layers)  │
# │     └── model                                  ← QUANTIZE │
# └────────────────────────────────────────────────────────────┘
#
# WARNING: This is the most aggressive quantization!
# - Maximum memory savings (~60%)
# - Potential accuracy loss across all components
# - Recommended for memory-constrained environments only

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
# Full Model W4A8 DuQuant Configuration
# ============================================
# CRITICAL: This quantizes ALL components!
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."  # No specific submodule - quantize ALL
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1           # Enable for better accuracy
export OPENPI_DUQUANT_ROW_ROT=restore     # Enable for better accuracy
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
export OPENPI_DUQUANT_LS=0.15

# Disable torch.compile for faster startup
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# Pack directory for full model quantization
# This will be LARGE (~3x the size of individual component packs)
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_full_w4a8"

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant: FULL MODEL Quantization (W4A8)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "⚠️  WARNING: FULL MODEL QUANTIZATION ACTIVE ⚠️"
echo ""
echo "DuQuant Config (Full W4A8):"
echo "  TARGET: ALL transformer components"
echo "  SCOPE: $OPENPI_DUQUANT_SCOPE"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "  BLOCK=$OPENPI_DUQUANT_BLOCK"
echo "  PERMUTE=$OPENPI_DUQUANT_PERMUTE (enabled)"
echo "  ROW_ROT=$OPENPI_DUQUANT_ROW_ROT (enabled)"
echo "  ACT_PCT=$OPENPI_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$OPENPI_DUQUANT_CALIB_STEPS"
echo "  LS=$OPENPI_DUQUANT_LS"
echo "  PACKDIR=$OPENPI_DUQUANT_PACKDIR"
echo ""
echo "⚡ QUANTIZATION TARGET:"
echo "  ✅ Vision Tower (SigLIP) - 27 layers × 4 linear = ~108 layers"
echo "  ✅ Language Model (Gemma LLM) - 18 layers × 7 linear = ~126 layers"
echo "  ✅ DiT (Action Expert) - 18 layers × 7 linear = ~126 layers"
echo "  📊 TOTAL: ~360 linear layers will be quantized!"
echo ""
echo "⚡ MEMORY & PERFORMANCE:"
echo "  📉 Memory savings: ~60% (maximum possible)"
echo "  ⚠️  Accuracy: May degrade across all tasks"
echo "  ⏱️  First run: May take 5-10 min to generate all packs"
echo "  ⏱️  Subsequent runs: ~3-5 min per episode (without torch.compile)"
echo ""
echo "NOTE: If accuracy is too low, try:"
echo "  - run_llm_w4a8.sh (LLM only)"
echo "  - run_optimized_duquant.sh (DiT only)"
echo "  - Or increase bits: WBITS=8, ABITS=8"
echo "========================================"
echo ""

# Confirmation prompt (optional - can comment out for automation)
read -p "⚠️  Proceed with FULL model quantization? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi

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
