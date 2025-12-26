#!/bin/bash
# Setup script for W8A8 model environment
# This script creates a conda environment with all dependencies needed to run
# the W8A8 quantized model on Libero evaluation.

set -e

ENV_NAME="openpi_w8a8"
PYTHON_VERSION="3.10"

echo "========================================"
echo "Creating conda environment: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo "========================================"

# Step 1: Create conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Step 2: Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Step 3: Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install transformers and HuggingFace packages
echo "Installing transformers and related packages..."
pip install transformers accelerate safetensors huggingface_hub einops

# Step 5: Install BitBLAS and robot simulation packages
echo "Installing BitBLAS and robot simulation packages..."
pip install bitblas scipy robosuite mujoco

# Step 6: Install Libero (from local path or clone from github)
echo "Installing Libero..."
# Option A: From local path (if exists)
if [ -d "third_party/libero" ]; then
    pip install -e third_party/libero
else
    # Option B: Clone from github
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
    pip install -e third_party/libero
fi

# Step 7: Install openpi dependencies
echo "Installing openpi dependencies..."
pip install sentencepiece draccus imageio imageio-ffmpeg moviepy \
    orbax-checkpoint flax dm-pix optax matplotlib h5py boto3 tqdm

echo "========================================"
echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run Libero evaluation with W8A8 model:"
echo "  python scripts/run_libero_w8a8.py --task-suite libero_spatial --num-trials 20"
echo "========================================"
