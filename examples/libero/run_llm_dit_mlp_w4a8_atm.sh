#!/bin/bash
# LLM (all linear layers) + DiT MLP (gate/up/down) W4A8 DuQuant
# ATM enabled for DiT attention (per-head temperature matching)

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch

if [ -z "$CKPT" ]; then
    echo "Error: CKPT environment variable must be set"
    echo "Usage: export CKPT=/path/to/checkpoint"
    exit 1
fi

export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================
# DuQuant configuration (same as run_llm_dit_mlp_w4a8.sh)
# ============================================
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE=""
export OPENPI_DUQUANT_INCLUDE='.*(language_model\.(.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|gemma_expert\.model\.layers\.\d+\.(gate_proj|up_proj|down_proj)).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\.|$)'

export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=64
export OPENPI_DUQUANT_PERMUTE=1
export OPENPI_DUQUANT_ROW_ROT=restore
export OPENPI_DUQUANT_ACT_PCT=99.9
export OPENPI_DUQUANT_CALIB_STEPS=32
export OPENPI_DUQUANT_LS=0.15

export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

export OPENPI_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/openpi/duquant_packed_llm_dit_w4a8_atm"

# ============================================
# ATM configuration (DiT only)
# ============================================
export ATM_ENABLE=${ATM_ENABLE:-1}
export ATM_SCOPE=${ATM_SCOPE:-dit}
if [ -z "${ATM_ALPHA_PATH:-}" ]; then
    export ATM_ALPHA_PATH="/home/jz97/VLM_REPO/openpi/atm_alpha_dit_new.json"
fi

# Defaults
TASK_SUITE="${TASK_SUITE:-libero_goal}"
NUM_TRIALS="${NUM_TRIALS:-20}"
SEED="${SEED:-42}"

echo "========================================"
echo "LIBERO Headless Evaluation"
echo "DuQuant W4A8 (LLM + DiT MLP) with DiT ATM"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "Task suite: $TASK_SUITE"
echo "Num trials: $NUM_TRIALS"
echo "Seed: $SEED"
echo "ATM_ENABLE=$ATM_ENABLE"
echo "ATM_SCOPE=$ATM_SCOPE"
echo "ATM_ALPHA_PATH=$ATM_ALPHA_PATH"
echo "========================================"

python - <<'PY'
import os
import torch
from openpi.training import config as train_config
from openpi.policies import policy_config

cfg = train_config.get_config("pi05_libero")
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = policy_config.create_trained_policy(cfg, os.environ["CKPT"], pytorch_device=device)

for name, mod in policy._model.named_modules():
    alpha = getattr(mod, "_atm_alpha_all", None)
    if "gemma_expert" in name and alpha is not None:
        alpha = alpha.detach().cpu()
        print(f"ATM(DiT) check: layer={name}, heads={alpha.numel()}, alpha[min]={alpha.min():.3f}, alpha[max]={alpha.max():.3f}")
        break
else:
    print("ATM(DiT) check: no DiT attention layers received ATM coefficients! Verify ATM_* env.")
PY


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
echo "Results in: results/libero/"
echo "========================================"
