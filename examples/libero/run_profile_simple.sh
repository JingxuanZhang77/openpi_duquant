#!/bin/bash
# Simple profiling script - minimal environment setup

cd ~/VLM_REPO/openpi

# Activate virtual environment
source examples/libero/.venv/bin/activate

# Set only essential variables
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch
export PYTHONPATH=$PWD/src:$PWD/third_party/libero

echo "========================================"
echo "Linear Layer Profiling (No DuQuant)"
echo "========================================"
echo "Checkpoint: $CKPT"
echo "========================================"
echo ""

# Run profiling script with minimal env
python examples/libero/profile_linear_layers.py

echo ""
echo "========================================"
echo "Profiling complete!"
echo "========================================"
