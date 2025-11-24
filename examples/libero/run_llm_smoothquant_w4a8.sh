#!/bin/bash
# SmoothQuant-style W4A8 quantization for Pi0.5 LLM (language_model only).

set -euo pipefail

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

export CKPT=${CKPT:-~/VLM_REPO/openpi/ckpts/pi05_libero_torch}
if [ ! -d "$CKPT" ]; then
  echo "Error: checkpoint directory $CKPT not found" >&2
  exit 1
fi

export PYTHONPATH=$PWD:$PWD/src:$PWD/third_party/libero
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SmoothQuant-like DuQuant settings (LLM only, no permute/row-rot)
export OPENPI_DUQUANT_SCOPE=""
export OPENPI_DUQUANT_INCLUDE='.*paligemma_with_expert\.paligemma\.model\.language_model\.(?:.*\.)?(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|emb|embed|lm_head)(?:\.|$)'

export OPENPI_DUQUANT_WBITS_DEFAULT=${OPENPI_DUQUANT_WBITS_DEFAULT:-8}
export OPENPI_DUQUANT_ABITS=${OPENPI_DUQUANT_ABITS:-8}
export OPENPI_DUQUANT_BLOCK=${OPENPI_DUQUANT_BLOCK:-64}
export OPENPI_DUQUANT_PERMUTE=${OPENPI_DUQUANT_PERMUTE:-0}
export OPENPI_DUQUANT_ROW_ROT=${OPENPI_DUQUANT_ROW_ROT:-none}
export OPENPI_DUQUANT_ACT_PCT=${OPENPI_DUQUANT_ACT_PCT:-100}
export OPENPI_DUQUANT_CALIB_STEPS=${OPENPI_DUQUANT_CALIB_STEPS:-32}
export OPENPI_DUQUANT_LS=${OPENPI_DUQUANT_LS:-0.5}
export OPENPI_DUQUANT_PACKDIR=${OPENPI_DUQUANT_PACKDIR:-$PWD/duquant_packed_llm_smooth_w8a8}

export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# SmoothQuant runtime scales (generated via tools/smoothquant_llm.py)
export SMOOTHQUANT_ENABLE=${SMOOTHQUANT_ENABLE:-1}
export SMOOTHQUANT_ALPHA_PATH=${SMOOTHQUANT_ALPHA_PATH:-$PWD/smoothquant_llm.json}

# ATM disabled by default
export ATM_ENABLE=${ATM_ENABLE:-0}

TASK_SUITE=${TASK_SUITE:-libero_object}
NUM_TRIALS=${NUM_TRIALS:-10}
SEED=${SEED:-42}

RESULTS_OUT_PATH=${RESULTS_OUT_PATH:-$PWD/results/libero_llm_smoothquant_w4a8}
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-$PWD/data/libero/videos_llm_smoothquant_w4a8}
mkdir -p "$RESULTS_OUT_PATH" "$VIDEO_OUT_PATH"

LOGDIR=${LOGDIR:-logs}
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/llm_smoothquant_w4a8_${TASK_SUITE}_$(date +%F_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=============================================="
echo "Pi0.5 LLM SmoothQuant W4A8 evaluation"
echo "Checkpoint        : $CKPT"
echo "Task suite        : $TASK_SUITE"
echo "Trials per task   : $NUM_TRIALS"
echo "Seed              : $SEED"
echo "Results directory : $RESULTS_OUT_PATH"
echo "Videos directory  : $VIDEO_OUT_PATH"
echo "DuQuant pack dir  : $OPENPI_DUQUANT_PACKDIR"
echo "SmoothQuant map   : $SMOOTHQUANT_ALPHA_PATH"
echo "=============================================="

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
echo "Log stored at $LOGFILE"
