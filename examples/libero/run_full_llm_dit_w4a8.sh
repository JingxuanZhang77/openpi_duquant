#!/bin/bash
# DuQuant W4A8 for ALL linear layers in LLM + DiT (full quantization)
# Quantizes ALL linear layers in both Gemma LLM and DiT
# Maximum quantization for testing accuracy degradation
#
# Model Structure:
# ┌────────────────────────────────────────────────────────────┐
# │ OpenPI Model (PI0Pytorch)                                  │
# │                                                            │
# │ ├── paligemma_with_expert.paligemma                        │
# │ │   ├── vision_tower (SigLIP - NOT quantized)             │
# │ │   └── language_model (Gemma LLM - 18 layers) ← QUANTIZE │
# │ │       ├── self_attn.{q,k,v,o}_proj (18×4 = 72)         │
# │ │       └── mlp.{gate,up,down}_proj (18×3 = 54)          │
# │ │                                                          │
# │ └── paligemma_with_expert.gemma_expert (DiT - 18 layers) ← QUANTIZE
# │     └── model                                              │
# │         ├── self_attn.{q,k,v,o}_proj (18×4 = 72)         │
# │         └── mlp.{gate,up,down}_proj (18×3 = 54)          │
# │                                                            │
# └────────────────────────────────────────────────────────────┘
#
# Expected layer count:
#   LLM: 18 layers × 7 linears (4 attn + 3 mlp) = 126 layers
#   DiT: 18 layers × 7 linears (4 attn + 3 mlp) = 126 layers
#   TOTAL: 252 layers (excluding vision tower and embeddings)

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# Full LLM+DiT W4A8 DuQuant Configuration
# ============================================
# Target: ALL linear layers in both LLM and DiT
# Includes: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# Excludes: Vision tower, embeddings, normalization layers
# ============================================
export OPENPI_DUQUANT_DEBUG=1
# CRITICAL: Use BOTH scopes - will match BOTH LLM and DiT
# We use a prefix that captures both branches
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1           # Enable input permutation
export OPENPI_DUQUANT_ROW_ROT=restore     # Enable rotation with output restoration
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32      # Calibration steps for activation quantization
export OPENPI_DUQUANT_LS=0.15             # Lambda smooth (only used when PERMUTE=1)

# Include ALL attention and MLP layers
export OPENPI_DUQUANT_INCLUDE='.*(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj).*'
# Exclude: embeddings, normalization, vision tower, and multi_modal_projector
# CRITICAL: Exclude vision_tower to prevent quantizing SigLIP vision encoder
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|vision|multi_modal_projector|lm_head)(?:\.|$)'

# Disable torch.compile for faster startup (recommended for full quantization)
export OPENPI_DISABLE_TORCH_COMPILE=1  # COMMENTED FOR SPEEDUP
export TORCH_COMPILE_DISABLE=1  # COMMENTED FOR SPEEDUP
export TORCHDYNAMO_DISABLE=1  # COMMENTED FOR SPEEDUP
unset CUBLAS_WORKSPACE_CONFIG

# Pack directory for full LLM+DiT quantization
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_full_llm_dit_w4a8"

# Enable profiling to measure fake quantization overhead
export OPENPI_DUQUANT_PROFILE=1

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant: FULL LLM+DiT Quantization (W4A8)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (Full LLM+DiT W4A8):"
echo "  TARGET: ALL linear layers in LLM + DiT"
echo "  SCOPE: $OPENPI_DUQUANT_SCOPE"
echo "  INCLUDE: $OPENPI_DUQUANT_INCLUDE"
echo "  EXCLUDE: $OPENPI_DUQUANT_EXCLUDE"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "  BLOCK=$OPENPI_DUQUANT_BLOCK"
echo "  PERMUTE=$OPENPI_DUQUANT_PERMUTE"
echo "  ROW_ROT=$OPENPI_DUQUANT_ROW_ROT"
echo "  ACT_PCT=$OPENPI_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$OPENPI_DUQUANT_CALIB_STEPS"
echo "  LS=$OPENPI_DUQUANT_LS"
echo "  PACKDIR=$OPENPI_DUQUANT_PACKDIR"
echo ""
echo "⚡ QUANTIZATION TARGET:"
echo "  ✅ LLM (Gemma) Attention + MLP - Expected: ~126 layers"
echo "     └─ 18 layers × (4 attn + 3 mlp) = 126"
echo "  ✅ DiT (Action Expert) Attention + MLP - Expected: ~126 layers"
echo "     └─ 18 layers × (4 attn + 3 mlp) = 126"
echo "  ❌ Vision Tower (SigLIP) - NOT quantized"
echo "  ❌ Embeddings - NOT quantized"
echo "  ❌ Normalization layers - NOT quantized"
echo ""
echo "  📊 TOTAL EXPECTED: ~252 linear layers quantized"
echo ""
echo "⚡ FEATURES:"
echo "  ✅ Full W4A8 fake quantization (max compression)"
echo "  ✅ Input permutation enabled (better accuracy)"
echo "  ✅ Row rotation with output restoration (better accuracy)"
echo "  ✅ Pre-cached rotation matrices (optimized)"
echo "  ✅ Pre-quantized weights (optimized)"
echo "  ✅ Profiling enabled (measure overhead)"
echo ""
echo "⚠️  WARNING: This is the MOST aggressive quantization!"
echo "   - Quantizes BOTH LLM and DiT simultaneously"
echo "   - May cause significant accuracy degradation"
echo "   - Use for testing worst-case DuQuant impact"
echo ""
if [ "$OPENPI_DISABLE_TORCH_COMPILE" = "1" ]; then
    echo "  ❌ torch.compile DISABLED (faster startup, slower per-episode)"
    echo ""
    echo "Expected speed:"
    echo "  Episode 1: ~3-5 min (first run with packing)"
    echo "  Episode 2+: ~3-5 min each"
else
    echo "  ✅ torch.compile ENABLED (CUDA kernel fusion)"
    echo ""
    echo "Expected speed:"
    echo "  Episode 1: ~20-30 min (torch.compile compilation)"
    echo "  Episode 2+: ~1-2 min each (using cached compilation)"
fi

echo ""
echo "COMPARISON:"
echo "┌────────────────────────┬──────────────┬──────────────┐"
echo "│ Script                 │ Layers       │ Memory Save  │"
echo "├────────────────────────┼──────────────┼──────────────┤"
echo "│ run_llm_w4a8.sh        │ LLM only     │ ~20-30%      │"
echo "│ run_optimized_duquant  │ DiT only     │ ~20-30%      │"
echo "│ run_dit_qkvo_w4a8.sh   │ DiT QKVO     │ ~15-20%      │"
echo "│ THIS SCRIPT            │ LLM+DiT ALL  │ ~40-50%      │"
echo "└────────────────────────┴──────────────┴──────────────┘"
echo ""
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
echo "Check profiling in stdout for [DUQUANT][PROFILE]"
echo ""
echo "Next steps:"
echo "  1. Compare accuracy with baseline (no quantization)"
echo "  2. Compare with partial quantization (LLM-only or DiT-only)"
echo "  3. Check [DUQUANT] Total layers replaced in logs"
echo "  4. Review profiling data for overhead analysis"
echo "========================================"
