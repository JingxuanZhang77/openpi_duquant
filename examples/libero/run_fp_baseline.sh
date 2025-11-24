#!/bin/bash
# Pi0.5 LIBERO baseline evaluation - No quantization, pure FP/BF16
#
# This runs the model without any DuQuant, SmoothQuant, ATM, or other modifications.
# Use this as a baseline to compare against quantized versions.

set -euo pipefail

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

export CKPT=${CKPT:-~/VLM_REPO/openpi/ckpts/pi05_libero_torch}

if [ ! -d "$CKPT" ]; then
  echo "Error: checkpoint directory not found: $CKPT" >&2
  exit 1
fi

export PYTHONPATH=$PWD:$PWD/src:$PWD/third_party/libero
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========================= Disable ALL quantization =========================
# Disable DuQuant
unset OPENPI_DUQUANT_SCOPE
unset OPENPI_DUQUANT_INCLUDE
unset OPENPI_DUQUANT_EXCLUDE
unset OPENPI_DUQUANT_WBITS_DEFAULT
unset OPENPI_DUQUANT_ABITS
unset OPENPI_DUQUANT_PERMUTE
unset OPENPI_DUQUANT_CALIB_STEPS
unset OPENPI_DUQUANT_PACKDIR

# Disable SmoothQuant
export OPENPI_SMOOTHQUANT_ENABLE=0

# Disable ATM
export ATM_ENABLE=0

# Disable torch compile/dynamo
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# ============================= Result paths ===================================
RESULTS_OUT_PATH=${RESULTS_OUT_PATH:-$PWD/results/libero_fp_baseline}
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-$PWD/data/libero/videos_fp_baseline}
mkdir -p "$RESULTS_OUT_PATH" "$VIDEO_OUT_PATH"

TASK_SUITE=${TASK_SUITE:-libero_10}
NUM_TRIALS=${NUM_TRIALS:-20}
SEED=${SEED:-42}

LOGDIR=${LOGDIR:-logs}
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/fp_baseline_${TASK_SUITE}_$(date +%F_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=============================================="
echo "Pi0.5 FP Baseline LIBERO Evaluation"
echo "=============================================="
echo "Checkpoint        : $CKPT"
echo "Task suite        : $TASK_SUITE"
echo "Trials per task   : $NUM_TRIALS"
echo "Seed              : $SEED"
echo "Results directory : $RESULTS_OUT_PATH"
echo "Videos directory  : $VIDEO_OUT_PATH"
echo ""
echo "Configuration:"
echo "  Quantization    : DISABLED (FP/BF16 baseline)"
echo "  DuQuant         : OFF"
echo "  SmoothQuant     : OFF"
echo "  ATM             : OFF"
echo "=============================================="
echo ""

time python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task "$NUM_TRIALS" \
  --args.seed "$SEED" \
  --args.results-out-path "$RESULTS_OUT_PATH" \
  --args.video-out-path "$VIDEO_OUT_PATH"

echo ""
echo "Evaluation complete. Results saved under $RESULTS_OUT_PATH"
echo "Full log stored at $LOGFILE"
