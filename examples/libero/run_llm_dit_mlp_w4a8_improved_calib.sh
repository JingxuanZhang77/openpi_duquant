#!/bin/bash
# Quantize LLM (all linear layers) + DiT MLP layers (gate_proj, up_proj, down_proj)
# IMPROVED VERSION: Increased calibration steps for better long-task performance
#
# Changes from run_llm_dit_mlp_w4a8.sh:
# - CALIB_STEPS: 32 → 128 (4x more calibration data)
# - This should better capture long-task (Libero-10) activation distributions
#
# Expected improvement: 80.5% → 85-88% on Libero-10

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
# IMPROVED: 4x more calibration steps
# ============================================
export OPENPI_DUQUANT_DEBUG=1

# Layer selection (same as original)
export OPENPI_DUQUANT_SCOPE=""  # Empty scope = search entire model
export OPENPI_DUQUANT_INCLUDE='.*(language_model\.(.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|gemma_expert\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\\.|$)|gemma_expert\\..*\\.self_attn\\.'

export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16
export OPENPI_DUQUANT_PERMUTE=1           # Enable input permutation
export OPENPI_DUQUANT_ROW_ROT=restore     # Output rotation with restore
export OPENPI_DUQUANT_ACT_PCT=99.9

# CRITICAL: Increased from 32 to 128 for better long-task calibration
export OPENPI_DUQUANT_CALIB_STEPS=128     # 4x more calibration data

export OPENPI_DUQUANT_LS=0.15             # Lambda smooth (only used when PERMUTE=1)

# Disable torch.compile for faster startup
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# CRITICAL: Disable CUDA graphs to avoid memory overwrite issues with DuQuant
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

# Pack directory for LLM + DiT MLP quantization with improved calibration
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_llm_dit_mlp_w4a8_calib128"

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant W4A8: LLM + DiT MLP"
echo "IMPROVED: 4x Calibration Steps"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (LLM + DiT MLP W4A8 - Improved):"
echo "  SCOPE: $OPENPI_DUQUANT_SCOPE"
echo "  INCLUDE: $OPENPI_DUQUANT_INCLUDE"
echo "  EXCLUDE: $OPENPI_DUQUANT_EXCLUDE"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "  BLOCK=$OPENPI_DUQUANT_BLOCK"
echo "  PERMUTE=$OPENPI_DUQUANT_PERMUTE"
echo "  ROW_ROT=$OPENPI_DUQUANT_ROW_ROT"
echo "  ACT_PCT=$OPENPI_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$OPENPI_DUQUANT_CALIB_STEPS  ⬆️ INCREASED FROM 32"
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
echo ""
echo "  📊 TOTAL EXPECTED: ~180 linear layers quantized"
echo ""
echo "🔧 IMPROVEMENTS:"
echo "  ✅ Calibration steps: 32 → 128 (4x more data)"
echo "  ✅ Better long-task activation distribution capture"
echo "  ✅ More robust percentile estimation"
echo ""
echo "📈 EXPECTED RESULTS:"
echo "  Baseline (no DuQuant): ~92.4% on Libero-10"
echo "  Original (CALIB=32):   ~80.5% on Libero-10 (-11.9%)"
echo "  Expected (CALIB=128):  ~85-88% on Libero-10 (-4 to -7%)"
echo ""
echo "⏱️  TIMING:"
echo "  First episode: ~4-6 min (128 calibration steps)"
echo "  Later episodes: ~3-5 min each"
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
echo ""
echo "📊 Compare with baseline:"
echo "  No DuQuant:       bash examples/libero/run_headless.sh"
echo "  Original (CALIB=32): bash examples/libero/run_llm_dit_mlp_w4a8.sh"
echo "  This run (CALIB=128): [results above]"
echo ""
echo "Check results in: results/libero/"
echo "Check [DUQUANT] logs for layer count and calibration progress"
echo "========================================"
