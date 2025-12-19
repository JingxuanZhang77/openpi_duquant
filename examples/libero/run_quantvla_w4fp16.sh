#!/bin/bash
# BitBLAS W4FP16 True Quantization for PI0.5
# Quantizes:
#   - LLM: MLP layers (gate_proj, up_proj, down_proj)
#   - DiT: MLP layers (gate_proj, up_proj, down_proj)
#   - Uses TRUE INT4 weights (not fake quantization)
#
# Model Structure:
# ┌────────────────────────────────────────────────────────────┐
# │ OpenPI Model (PI0Pytorch)                                  │
# │                                                            │
# │ ├── paligemma_with_expert.paligemma                        │
# │ │   ├── vision_tower (SigLIP - NOT quantized)             │
# │ │   └── language_model (Gemma LLM - 18 layers)            │
# │ │       ├── self_attn.{q,k,v,o}_proj (NOT quantized)      │
# │ │       └── mlp.{gate,up,down}_proj (18×3 = 54)  ✅ INT4  │
# │ │                                                          │
# │ └── paligemma_with_expert.gemma_expert (DiT - 18 layers)  │
# │     └── model                                              │
# │         ├── self_attn.{q,k,v,o}_proj (NOT quantized)      │
# │         └── mlp.{gate,up,down}_proj (18×3 = 54)  ✅ INT4  │
# │                                                            │
# └────────────────────────────────────────────────────────────┘
#
# Expected layer count:
#   LLM MLP: 18 layers × 3 linears = 54 layers
#   DiT MLP: 18 layers × 3 linears = 54 layers
#   TOTAL: 108 layers quantized to TRUE INT4
#

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

# Fix CUDA library version mismatch (PyTorch 2.7.1 CUDA 12.6 vs System CUDA 12.8)
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

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
# BitBLAS W4FP16 Configuration
# ============================================
# Enable BitBLAS true INT4 quantization
export OPENPI_BITBLAS_ENABLE=1

# Quantization settings
export OPENPI_BITBLAS_WBITS=4
export OPENPI_BITBLAS_GROUP_SIZE=128
export OPENPI_BITBLAS_ENABLE_TUNING=0  # Set to 1 for first run to auto-tune
export OPENPI_BITBLAS_OPT_M="1,16,32,64"  # Optimize for these batch sizes

# Reuse DuQuant parameters (rotation matrices and scales)
export OPENPI_BITBLAS_DUQUANT_PACKDIR="duquant_packed_full_llm_dit_mlp_w4a8_atm"

# Layer selection (same as DuQuant quantvla script - MLP only)
export OPENPI_BITBLAS_SCOPE=""
export OPENPI_BITBLAS_INCLUDE='(.*language_model.*(gate_proj|up_proj|down_proj).*|.*gemma_expert.*(gate_proj|up_proj|down_proj).*)'
export OPENPI_BITBLAS_EXCLUDE='(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|multi_modal_projector|lm_head)(?:\.|$)'

# Debug and profiling
export OPENPI_BITBLAS_DEBUG=1
export OPENPI_BITBLAS_PROFILE=0

# CRITICAL: Disable DuQuant to avoid conflict with BitBLAS
unset OPENPI_DUQUANT_WBITS_DEFAULT
unset OPENPI_DUQUANT_ABITS

# Disable torch.compile for faster startup (recommended for initial testing)
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# ============================================
# ATM configuration (DiT only) - Optional
# ============================================
export ATM_ENABLE=${ATM_ENABLE:-0}  # Disabled by default for W4FP16 testing
export ATM_SCOPE=${ATM_SCOPE:-dit}
if [ -z "${ATM_ALPHA_PATH:-}" ] && [ "$ATM_ENABLE" = "1" ]; then
    export ATM_ALPHA_PATH="atm_alpha_llm_full_ditmlp.json"
fi

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "BitBLAS: TRUE W4FP16 Quantization"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "BitBLAS Config (W4FP16 True Quantization):"
echo "  ENABLED: $OPENPI_BITBLAS_ENABLE"
echo "  WBITS: $OPENPI_BITBLAS_WBITS (TRUE INT4)"
echo "  GROUP_SIZE: $OPENPI_BITBLAS_GROUP_SIZE"
echo "  TUNING: $OPENPI_BITBLAS_ENABLE_TUNING"
echo "  OPT_M: $OPENPI_BITBLAS_OPT_M"
echo "  DUQUANT_PACKDIR: $OPENPI_BITBLAS_DUQUANT_PACKDIR"
echo "  SCOPE: $OPENPI_BITBLAS_SCOPE"
echo "  INCLUDE: $OPENPI_BITBLAS_INCLUDE"
echo "  EXCLUDE: $OPENPI_BITBLAS_EXCLUDE"
echo ""
echo "⚡ QUANTIZATION TARGET:"
echo "  ✅ LLM (Gemma) MLP ONLY - Expected: ~54 layers"
echo "     └─ 18 layers × 3 mlp = 54"
echo "  ✅ DiT (Action Expert) MLP ONLY - Expected: ~54 layers"
echo "     └─ 18 layers × 3 mlp = 54"
echo "  ❌ Attention layers (Q/K/V/O) - NOT quantized (FP16)"
echo "  ❌ Vision Tower (SigLIP) - NOT quantized"
echo "  ❌ Embeddings - NOT quantized"
echo "  ❌ Normalization layers - NOT quantized"
echo ""
echo "  📊 TOTAL EXPECTED: ~108 MLP layers with TRUE INT4 weights"
echo ""
echo "⚡ KEY DIFFERENCES vs DuQuant W4A8:"
echo "  ✅ TRUE INT4 weight storage (not fake quantization)"
echo "  ✅ Weights stored as INT4 (50% memory vs FP16)"
echo "  ✅ Reuses DuQuant rotation matrices and scales"
echo "  ⚠️  Dequantizes to FP16 for matmul (standard PyTorch)"
echo "  ❌ Activations remain FP16 (W4FP16, not W4A8)"
echo ""
echo "⚡ PERFORMANCE EXPECTATIONS:"
echo "  Memory: Should use ~15GB (vs 18GB FP16, vs 18GB fake quant)"
echo "  Speed: Similar to DuQuant (dequant overhead)"
echo "  Accuracy: Similar to DuQuant W4A8 (~75-76% on libero_spatial)"
echo ""
echo "⚠️  NOTE: Current implementation uses INT4 storage + FP16 compute"
echo "   BitBLAS optimized kernels skipped due to CUDA compatibility"
echo ""
if [ "$OPENPI_DISABLE_TORCH_COMPILE" = "1" ]; then
    echo "  ❌ torch.compile DISABLED (faster startup, testing mode)"
    echo ""
    echo "Expected speed:"
    echo "  Episode 1: ~3-5 min (first run with weight conversion)"
    echo "  Episode 2+: ~2-3 min each"
else
    echo "  ✅ torch.compile ENABLED (CUDA kernel fusion)"
    echo ""
    echo "Expected speed:"
    echo "  Episode 1: ~20-30 min (torch.compile compilation + conversion)"
    echo "  Episode 2+: ~1-2 min each (using cached compilation)"
fi

echo ""
echo "COMPARISON:"
echo "┌────────────────────────┬──────────────┬──────────────┬──────────────┐"
echo "│ Method                 │ Layers       │ Memory       │ Weight Type  │"
echo "├────────────────────────┼──────────────┼──────────────┼──────────────┤"
echo "│ FP16 Baseline          │ None         │ ~18GB        │ FP16         │"
echo "│ DuQuant W4A8 (fake)    │ LLM+DiT MLP  │ ~18GB        │ FP16 (fake)  │"
echo "│ THIS SCRIPT (W4FP16)   │ LLM+DiT MLP  │ ~15GB        │ TRUE INT4    │"
echo "└────────────────────────┴──────────────┴──────────────┴──────────────┘"
echo ""
echo "========================================"
echo ""

# Run evaluation with timing
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
echo "Check results in: results/libero/"
echo ""
echo "Next steps:"
echo "  1. Verify TRUE INT4 weights: Check [BITBLAS] logs for weight conversion"
echo "  2. Compare memory usage: Should be ~15GB (vs 18GB FP16)"
echo "  3. Compare speed: Should be faster than DuQuant fake quant"
echo "  4. Compare accuracy: Should be similar to DuQuant W4A8"
echo "  5. If first run, consider enabling OPENPI_BITBLAS_ENABLE_TUNING=1"
echo "     for hardware-aware optimization (takes ~20-30min extra)"
echo "========================================"
