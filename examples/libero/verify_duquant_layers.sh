#!/bin/bash
# Quick verification script to check how many layers DuQuant will match
# This uses DRYRUN mode so it doesn't actually modify anything

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

echo "========================================"
echo "DuQuant Layer Matching Verification"
echo "========================================"
echo ""

echo "Testing DiT (Action Expert)..."
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_DRYRUN=1 \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
OPENPI_DUQUANT_WBITS_DEFAULT=4 \
OPENPI_DUQUANT_ABITS=8 \
python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.seed 42 2>&1 | grep -E "\[DUQUANT\] (SCOPE|Matched|Total)" | head -5

echo ""
echo "Testing LLM (Language Model)..."
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_DRYRUN=1 \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.paligemma.model.language_model." \
OPENPI_DUQUANT_WBITS_DEFAULT=4 \
OPENPI_DUQUANT_ABITS=8 \
python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.seed 42 2>&1 | grep -E "\[DUQUANT\] (SCOPE|Matched|Total)" | head -5

echo ""
echo "========================================"
echo "Expected results:"
echo "  DiT: 126 layers (18 layers × 7 linear each)"
echo "  LLM: 126 layers (18 layers × 7 linear each)"
echo "========================================"
