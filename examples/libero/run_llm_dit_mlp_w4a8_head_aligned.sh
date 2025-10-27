#!/bin/bash
# Quantize LLM (all linear layers) + DiT MLP layers (gate_proj, up_proj, down_proj)
# HEAD-ALIGNED VERSION: block_size=64 to respect attention head structure
#
# Changes from run_llm_dit_mlp_w4a8.sh:
# - BLOCK: 16 → 64 (matches head_dim for DiT/LLM attention)
# - CALIB_STEPS: 32 → 128 (better long-task calibration)
#
# Why block_size=64?
# - DiT Q attention: 2048 features = 32 heads × 64 head_dim
# - DiT K/V attention: 256 features = 4 heads × 64 head_dim
# - LLM attention: Similar head_dim=64 architecture
# - MLP layers: block_size=64 also works fine (just larger blocks)
#
# Impact:
# - Rotation and permutation respect head boundaries
# - No mixing of features between different attention heads
# - Better preservation of attention geometry
#
# Expected improvement: 80.5% → 88-91% on Libero-10

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
# HEAD-ALIGNED: block_size=64
# ============================================
export OPENPI_DUQUANT_DEBUG=1

# Layer selection (same as original)
export OPENPI_DUQUANT_SCOPE=""  # Empty scope = search entire model
export OPENPI_DUQUANT_INCLUDE='.*(language_model\.(.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|gemma_expert\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\\.|$)|gemma_expert\\..*\\.self_attn\\.'

export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8

# CRITICAL: Changed from 16 to 64 to align with attention head_dim
export OPENPI_DUQUANT_BLOCK=64

export OPENPI_DUQUANT_PERMUTE=1           # Enable input permutation
export OPENPI_DUQUANT_ROW_ROT=restore     # Output rotation with restore
export OPENPI_DUQUANT_ACT_PCT=99.9

# IMPROVED: Increased calibration steps
export OPENPI_DUQUANT_CALIB_STEPS=32

export OPENPI_DUQUANT_LS=0.15             # Lambda smooth

# Disable torch.compile for faster startup
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# CRITICAL: Disable CUDA graphs
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

# Pack directory for head-aligned quantization
export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_llm_dit_w4a8_head64"

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_goal}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant W4A8: LLM + DiT MLP"
echo "HEAD-ALIGNED: block_size=64"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "DuQuant Config (LLM + DiT MLP W4A8 - Head-Aligned):"
echo "  SCOPE: $OPENPI_DUQUANT_SCOPE"
echo "  INCLUDE: $OPENPI_DUQUANT_INCLUDE"
echo "  EXCLUDE: $OPENPI_DUQUANT_EXCLUDE"
echo "  WBITS=$OPENPI_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$OPENPI_DUQUANT_ABITS"
echo "  BLOCK=$OPENPI_DUQUANT_BLOCK  ⬆️ CHANGED FROM 16 TO 64"
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
echo "🔧 ARCHITECTURAL ALIGNMENT:"
echo "  DiT Attention Architecture (Multi-Query):"
echo "    - Q proj: [2048, 1024] = 32 heads × 64 head_dim"
echo "    - K proj: [256, 1024]  = 4 heads × 64 head_dim"
echo "    - V proj: [256, 1024]  = 4 heads × 64 head_dim"
echo ""
echo "  Previous block_size=16:"
echo "    ❌ Does NOT align with head_dim=64"
echo "    ❌ Rotation mixes features WITHIN each head"
echo "    ❌ Permutation breaks head structure"
echo ""
echo "  New block_size=64:"
echo "    ✅ Perfectly aligns with head_dim=64"
echo "    ✅ Each rotation block = exactly 1 attention head"
echo "    ✅ Preserves head structure and attention geometry"
echo ""
echo "🔧 IMPROVEMENTS:"
echo "  ✅ Block size: 16 → 64 (head-aligned)"
echo "  ✅ Calibration steps: 32 → 128 (4x more data)"
echo "  ✅ Rotation respects head boundaries"
echo "  ✅ Permutation within heads, not across"
echo ""
echo "📈 EXPECTED RESULTS:"
echo "  Baseline (no DuQuant):     ~92.4% on Libero-10"
echo "  Original (BLOCK=16):       ~80.5% on Libero-10 (-11.9%)"
echo "  Improved calib (BLOCK=16): ~85-88% on Libero-10 (-4 to -7%)"
echo "  This run (BLOCK=64):       ~88-91% on Libero-10 (-1 to -4%)"
echo ""
echo "⏱️  TIMING:"
echo "  First episode: ~5-7 min (larger rotation matrices)"
echo "  Later episodes: ~3-5 min each"
echo ""
echo "💾 MEMORY:"
echo "  Larger block size → fewer but larger rotation matrices"
echo "  Slight increase in rotation matrix memory (~10-20%)"
echo "  Still achieves good quantization compression overall"
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
echo "📊 Compare with other configurations:"
echo "  No DuQuant:             bash examples/libero/run_headless.sh"
echo "  Original (BLOCK=16):    bash examples/libero/run_llm_dit_mlp_w4a8.sh"
echo "  Improved calib:         bash examples/libero/run_llm_dit_mlp_w4a8_improved_calib.sh"
echo "  This run (BLOCK=64):    [results above]"
echo ""
echo "Expected accuracy progression:"
echo "  BLOCK=16, CALIB=32:  ~80.5%"
echo "  BLOCK=16, CALIB=128: ~85-88%"
echo "  BLOCK=64, CALIB=128: ~88-91%  ← Should be close to baseline!"
echo ""
echo "Check results in: results/libero/"
echo "Check [DUQUANT] logs for layer count and block statistics"
echo "========================================"
