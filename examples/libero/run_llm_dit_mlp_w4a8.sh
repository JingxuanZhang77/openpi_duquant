#!/bin/bash
# Quantize LLM (all linear layers) + DiT MLP layers (gate_proj, up_proj, down_proj)
# This applies W4A8 quantization to both language model and DiT MLP components
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
# │     └── model.layers[*].mlp (MLP layers) ← QUANTIZE       │
# └────────────────────────────────────────────────────────────┘
#
# Quantization targets:
# 1. LLM: ALL linear layers in language_model (126 layers)
# 2. DiT MLP: gate_proj, up_proj, down_proj in each DiT layer (54 layers)
# Total: 180 layers quantized

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
# LLM + DiT MLP W4A8 DuQuant Configuration
# ============================================
export OPENPI_DUQUANT_DEBUG=1

# CRITICAL: Use custom layer selection to quantize:
# 1. All LLM layers: paligemma_with_expert.paligemma.model.language_model.*
# 2. DiT MLP layers: paligemma_with_expert.gemma_expert.model.layers.*.mlp.*
# We'll set SCOPE to empty and use INCLUDE/EXCLUDE regex for fine control
export OPENPI_DUQUANT_SCOPE=""  # Empty scope = search entire model

# INCLUDE: Match both LLM layers AND DiT MLP layers
# - LLM: language_model.*.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)
# - DiT MLP: gemma_expert.model.layers.*.mlp.(gate_proj|up_proj|down_proj)
export OPENPI_DUQUANT_INCLUDE='.*(language_model\.(.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|gemma_expert\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)).*'

# EXCLUDE: Exclude vision tower, embeddings, norms
# CRITICAL: Use negative lookahead to exclude self_attn ONLY for DiT, not for LLM
# This allows language_model.*.self_attn.* but blocks gemma_expert.*.self_attn.*
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\.|$)|gemma_expert\..*\.self_attn\.'

export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1           # Enable input permutation
export OPENPI_DUQUANT_ROW_ROT=restore     # Output rotation with restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32      # Calibration steps for activation quantization
export OPENPI_DUQUANT_LS=0.15             # Lambda smooth (only used when PERMUTE=1)

# Disable torch.compile for faster startup
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# CRITICAL: Disable CUDA graphs to avoid memory overwrite issues with DuQuant
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

# Pack directory for LLM + DiT MLP quantization
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_llm_dit_mlp_w4a8"

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant W4A8: LLM (ALL) + DiT MLP"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (LLM + DiT MLP W4A8):"
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
echo "  ✅ LLM (Gemma) ALL layers - Expected: ~126 layers"
echo "     └─ 18 layers × (4 attn + 3 mlp) = 126"
echo "  ✅ DiT MLP ONLY - Expected: ~54 layers"
echo "     └─ 18 layers × 3 mlp = 54"
echo "  ❌ DiT Attention (QKVO) - NOT quantized"
echo "  ❌ Vision Tower (SigLIP) - NOT quantized"
echo "  ❌ Embeddings - NOT quantized"
echo ""
echo "  📊 TOTAL EXPECTED: ~180 linear layers quantized"
echo ""
echo "⚡ FEATURES:"
echo "  ✅ W4A8 fake quantization"
echo "  ✅ Input permutation enabled"
echo "  ✅ Row rotation with output restoration"
echo ""
if [ "$OPENPI_DISABLE_TORCH_COMPILE" = "1" ]; then
    echo "  ❌ torch.compile DISABLED (faster startup, slower per-episode)"
else
    echo "  ✅ torch.compile ENABLED (CUDA kernel fusion)"
fi
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
echo "========================================"
