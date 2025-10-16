#!/bin/bash
# Quantize the LLM (Gemma language model) instead of DiT
# This applies W4A8 quantization to the main language understanding component
#
# Model Structure:
# ┌────────────────────────────────────────────────────────────┐
# │ OpenPI Model (PI0Pytorch)                                  │
# │                                                            │
# │ ├── paligemma_with_expert.paligemma                        │
# │ │   ├── vision_tower (SigLIP - 27 layers)                  │
# │ │   └── language_model (Gemma LLM - 18 layers) ← QUANTIZE │
# │ │                                                          │
# │ └── paligemma_with_expert.gemma_expert (DiT - 18 layers)  │
# │     └── model                                              │
# └────────────────────────────────────────────────────────────┘
#
# Comparison with other scripts:
# ┌──────────────────────┬─────────────────┬──────────────────┐
# │ Script               │ Quantized       │ Memory Savings   │
# ├──────────────────────┼─────────────────┼──────────────────┤
# │ run_optimized_duquant│ DiT only        │ ~20%             │
# │ run_simple_w4a8      │ DiT only        │ ~20%             │
# │ THIS SCRIPT          │ LLM only        │ ~40% (largest!)  │
# │ run_full_quantize    │ LLM+DiT+Vision  │ ~60% (max)       │
# └──────────────────────┴─────────────────┴──────────────────┘

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
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ============================================
# LLM W4A8 DuQuant Configuration
# ============================================
# CRITICAL: This quantizes the LANGUAGE MODEL, not the DiT!
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1           # Enable input permutation
export OPENPI_DUQUANT_ROW_ROT=restore     # ← Now FAST with QR optimization! (was too slow with SVD)
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32      # Restored to default for better accuracy
export OPENPI_DUQUANT_LS=0.15             # Lambda smooth (only used when PERMUTE=1)

# Disable torch.compile for faster startup (can be enabled for better throughput)
export OPENPI_DISABLE_TORCH_COMPILE=1  # COMMENTED FOR SPEEDUP
export TORCH_COMPILE_DISABLE=1  # COMMENTED FOR SPEEDUP
export TORCHDYNAMO_DISABLE=1  # COMMENTED FOR SPEEDUP
unset CUBLAS_WORKSPACE_CONFIG

# Pack directory for LLM quantization
# IMPORTANT: This MUST be different from DiT pack directory!
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_llm_w4a8"




# export OPENPI_DUQUANT_PROFILE=1
echo "check check check"
echo $OPENPI_DUQUANT_PROFILE
# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant: LLM Quantization (W4A8)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (LLM W4A8):"
echo "  TARGET: Language Model (Gemma LLM)"
echo "  SCOPE: $OPENPI_DUQUANT_SCOPE"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "  BLOCK=$OPENPI_DUQUANT_BLOCK"
echo "  PERMUTE=$OPENPI_DUQUANT_PERMUTE"
echo "  ROW_ROT=$OPENPI_DUQUANT_ROW_ROT"
echo "  ACT_PCT=$OPENPI_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$OPENPI_DUQUANT_CALIB_STEPS"
if [ "$OPENPI_DUQUANT_PERMUTE" = "1" ]; then
    echo "  LS=$OPENPI_DUQUANT_LS"
fi
echo "  PACKDIR=$OPENPI_DUQUANT_PACKDIR"
echo ""
echo "⚡ QUANTIZATION TARGET:"
echo "  ✅ Language Model (Gemma LLM) - 18 layers × 7 linear = ~126 layers"
echo "  ❌ DiT (Action Expert) - NOT quantized (full precision)"
echo "  ❌ Vision Tower (SigLIP) - NOT quantized (full precision)"
echo ""
echo "⚡ FEATURES:"
echo "  ✅ W4A8 fake quantization"
if [ "$OPENPI_DUQUANT_PERMUTE" = "1" ]; then
    echo "  ✅ Input permutation enabled (better accuracy)"
else
    echo "  ❌ Input permutation disabled (faster)"
fi
if [ "$OPENPI_DUQUANT_ROW_ROT" = "restore" ]; then
    echo "  ✅ Row rotation with output restoration (better accuracy)"
elif [ "$OPENPI_DUQUANT_ROW_ROT" = "0" ]; then
    echo "  ❌ Row rotation disabled (MUCH faster - SVD very slow for 16384-dim)"
else
    echo "  ❌ Row rotation disabled"
fi
echo "  ✅ Pre-cached rotation matrices (optimized)"
echo "  ✅ Pre-quantized weights (optimized)"

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
echo "NOTE: This quantizes the LANGUAGE MODEL, not the DiT."
echo "      For DiT quantization, use run_optimized_duquant.sh"
echo "      For full quantization, use run_full_quantize.sh"
echo "========================================"
echo ""

# Run evaluation with timing
time python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task 20 \
  --args.seed "$SEED"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Check results in: results/libero/"
echo "========================================"
