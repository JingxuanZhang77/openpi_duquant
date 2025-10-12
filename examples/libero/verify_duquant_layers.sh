#!/bin/bash
# Verification script to check how many layers will be quantized
# Uses DRYRUN mode to list layers without actually quantizing
# This helps verify the layer counts before running full evaluation

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

echo "========================================"
echo "DuQuant Layer Count Verification"
echo "========================================"
echo ""

# ============================================
# Test 1: DiT QKVO Only
# ============================================
echo "📊 Test 1: DiT QKVO Only (q/k/v/o_proj)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export OPENPI_DUQUANT_DRYRUN=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
export OPENPI_DUQUANT_INCLUDE='.*(q_proj|k_proj|v_proj|o_proj).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(norm|ln|layernorm|emb|gate_proj|up_proj|down_proj)(?:\.|$)'
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8

python -c "
import sys
sys.path.insert(0, 'src')
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import load_config

config = load_config('$CKPT/config.json')
policy = create_trained_policy(config, '$CKPT', pytorch_device='cpu')
" 2>&1 | grep -E "DUQUANT.*layers (listed|replaced)" || echo "No output found"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================
# Test 2: DiT All Layers
# ============================================
echo "📊 Test 2: DiT ALL Layers (attention + MLP)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export OPENPI_DUQUANT_DRYRUN=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
export OPENPI_DUQUANT_INCLUDE='.*(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)'

python -c "
import sys
sys.path.insert(0, 'src')
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import load_config

config = load_config('$CKPT/config.json')
policy = create_trained_policy(config, '$CKPT', pytorch_device='cpu')
" 2>&1 | grep -E "DUQUANT.*layers (listed|replaced)" || echo "No output found"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================
# Test 3: LLM Only
# ============================================
echo "📊 Test 3: LLM Only (Gemma language model)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export OPENPI_DUQUANT_DRYRUN=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model."
export OPENPI_DUQUANT_INCLUDE='.*(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)'

python -c "
import sys
sys.path.insert(0, 'src')
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import load_config

config = load_config('$CKPT/config.json')
policy = create_trained_policy(config, '$CKPT', pytorch_device='cpu')
" 2>&1 | grep -E "DUQUANT.*layers (listed|replaced)" || echo "No output found"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================
# Test 4: Full LLM+DiT
# ============================================
echo "📊 Test 4: FULL LLM+DiT (all linear layers)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export OPENPI_DUQUANT_DRYRUN=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert."
export OPENPI_DUQUANT_INCLUDE='.*(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj).*'
export OPENPI_DUQUANT_EXCLUDE='(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|vision|multi_modal_projector|lm_head)(?:\.|$)'

python -c "
import sys
sys.path.insert(0, 'src')
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import load_config

config = load_config('$CKPT/config.json')
policy = create_trained_policy(config, '$CKPT', pytorch_device='cpu')
" 2>&1 | grep -E "DUQUANT.*layers (listed|replaced)" || echo "No output found"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "========================================"
echo "Summary of Expected Layers:"
echo "========================================"
echo "Test 1 (DiT QKVO):     72-144 layers"
echo "Test 2 (DiT All):      126-252 layers"
echo "Test 3 (LLM Only):     108-126 layers"
echo "Test 4 (LLM+DiT All):  234-252 layers"
echo ""
echo "Note: Range depends on whether model has"
echo "      dual attention layers or not."
echo "========================================"
