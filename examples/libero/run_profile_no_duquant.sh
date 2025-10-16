#!/bin/bash
# Profile Linear layers without DuQuant
# This script runs inference and measures each Linear layer's forward time

set -e

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch

# Set environment
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CRITICAL: Disable DuQuant completely
unset OPENPI_DUQUANT_WBITS_DEFAULT
unset OPENPI_DUQUANT_ABITS
unset OPENPI_DUQUANT_PACKDIR
unset OPENPI_DUQUANT_SCOPE
unset OPENPI_DUQUANT_INCLUDE
unset OPENPI_DUQUANT_EXCLUDE
unset OPENPI_DUQUANT_DEBUG
unset OPENPI_DUQUANT_PROFILE

# Disable torch.compile for consistent timing
export OPENPI_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
unset CUBLAS_WORKSPACE_CONFIG

# Enable linear layer profiling
export OPENPI_PROFILE_LINEAR=1

echo "========================================"
echo "Linear Layer Profiling (No DuQuant)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "DuQuant: DISABLED"
echo "Linear profiling: ENABLED"
echo "========================================"
echo ""

# Run profiling script
time python examples/libero/profile_linear_layers.py

echo ""
echo "========================================"
echo "Profiling complete!"
echo "========================================"
