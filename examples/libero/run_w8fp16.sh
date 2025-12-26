#!/bin/bash
# BitBLAS W8FP16 Quantization for PI0.5
#
# INT8 weights with FP16 activations using BitBLAS.
# No activation quantization - higher accuracy than W8A8.
#
# Architecture:
#   FP16 Input → FP16×INT8 kernel (BitBLAS dequant) → FP16 Output
#
# Comparison with W8A8:
#   W8A8:   INT8 weights × INT8 activations (dynamic quant) - fastest
#   W8FP16: INT8 weights × FP16 activations (no quant) - higher accuracy
#
# Benefits:
#   - 50% memory reduction (INT8 weights vs FP16)
#   - Higher accuracy than W8A8 (no activation quantization loss)
#   - Still uses BitBLAS optimized kernels
#
# Quantization targets (same as W8A8):
#   ✅ LLM (Gemma) MLP layers: gate_proj, up_proj, down_proj (54 layers)
#   ✅ DiT (Action Expert) MLP layers: gate_proj, up_proj, down_proj (54 layers)
#   ❌ Attention layers (Q/K/V/O) - NOT quantized (FP16)
#   ❌ Vision Tower (SigLIP) - NOT quantized
#   ❌ Embeddings - NOT quantized
#
# Expected: ~108 MLP layers with INT8 weights, FP16 activations

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
# W8FP16 Configuration
# ============================================
export OPENPI_W8A8_ENABLE=1
export OPENPI_W8A8_ACTIVATION_DTYPE=float16  # Key difference: FP16 activations
export OPENPI_W8A8_ENABLE_TUNING=${OPENPI_W8A8_ENABLE_TUNING:-0}  # Disable tuning by default
export OPENPI_W8A8_OPT_M="1,16,32,64"

# Layer selection (same as W8A8):
# - LLM (language_model): ALL linear layers (q,k,v,o + mlp)
# - DiT (gemma_expert): MLP only (gate_proj, up_proj, down_proj)
export OPENPI_W8A8_INCLUDE='(.*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*|.*gemma_expert.*(gate_proj|up_proj|down_proj).*)'
export OPENPI_W8A8_EXCLUDE='(?:^|\\.)(norm|ln|layernorm|emb|embed|vision_tower|multi_modal_projector|lm_head)(?:\\.|$)'

# Debug
export OPENPI_W8A8_DEBUG=${OPENPI_W8A8_DEBUG:-1}

# Disable other quantization methods
unset OPENPI_DUQUANT_WBITS_DEFAULT
unset OPENPI_DUQUANT_ABITS
unset OPENPI_BITBLAS_ENABLE

# Disable torch.compile for faster startup
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Default parameters
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "BitBLAS W8FP16 (INT8 Weights × FP16 Activations)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo ""
echo "W8FP16 Configuration:"
echo "  ENABLED: $OPENPI_W8A8_ENABLE"
echo "  ACTIVATION_DTYPE: $OPENPI_W8A8_ACTIVATION_DTYPE"
echo "  TUNING: $OPENPI_W8A8_ENABLE_TUNING"
echo "  OPT_M: $OPENPI_W8A8_OPT_M"
echo "  INCLUDE: $OPENPI_W8A8_INCLUDE"
echo "  EXCLUDE: $OPENPI_W8A8_EXCLUDE"
echo ""
echo "Architecture:"
echo "  FP16 Input → FP16×INT8 kernel → FP16 Output"
echo ""
echo "Comparison:"
echo "  W8A8:   INT8 × INT8 → INT32 → FP16 (fastest, some accuracy loss)"
echo "  W8FP16: FP16 × INT8 → FP16 (higher accuracy, same memory savings)"
echo ""
echo "Expected Performance:"
echo "  Memory: ~12GB (50% reduction from 18GB FP16)"
echo "  Speed: Moderate speedup (BitBLAS optimized kernels)"
echo "  Accuracy: Higher than W8A8 (no activation quantization)"
echo ""
echo "Quantization Targets:"
echo "  ✅ LLM MLP: 18 layers × 3 = 54 layers (INT8 weights)"
echo "  ✅ DiT MLP: 18 layers × 3 = 54 layers (INT8 weights)"
echo "  ❌ Attention layers (FP16)"
echo "  ❌ Vision Tower (FP16)"
echo "  📊 TOTAL: ~108 layers with INT8 weights, FP16 activations"
echo ""
if [ "$OPENPI_W8A8_ENABLE_TUNING" = "1" ]; then
    echo "NOTE: First run will take longer due to BitBLAS auto-tuning."
    echo "      Tuning results are cached for subsequent runs."
fi
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
