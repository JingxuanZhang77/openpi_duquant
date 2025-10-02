#!/bin/bash
# Setup complete environment for LIBERO headless evaluation
# This script installs both LIBERO and OpenPI dependencies

set -e  # Exit on error

echo "========================================"
echo "LIBERO Headless Environment Setup"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "examples/libero/requirements.txt" ]; then
    echo "Error: Please run this script from the openpi root directory"
    echo "Usage: bash examples/libero/setup_headless_env.sh"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.9" ]; then
    echo "Warning: This script is designed for Python 3.9, but you have $PYTHON_VERSION"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
ENV_PATH="examples/libero/.venv-headless"
echo "Creating virtual environment at $ENV_PATH..."
python3 -m venv $ENV_PATH

# Activate environment
echo "Activating environment..."
source $ENV_PATH/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install LIBERO dependencies
echo "Installing LIBERO dependencies..."
pip install -r examples/libero/requirements.txt \
            -r third_party/libero/requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu113 \
            --index-strategy=unsafe-best-match

# Install OpenPI (from root directory)
echo "Installing OpenPI..."
pip install -e .

# Install openpi-client
echo "Installing openpi-client..."
pip install -e packages/openpi-client

# Install additional dependencies
echo "Installing additional dependencies..."
pip install sentencepiece

# Optional: Setup robosuite macros
echo "Setting up robosuite macros (optional)..."
python $(python -c "import robosuite, pathlib; print(pathlib.Path(robosuite.__file__).parent / 'scripts' / 'setup_macros.py')") || echo "Warning: robosuite macros setup failed (not critical)"

# Verify installation
echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"

# Check JAX
if python -c "import jax; print('✓ JAX:', jax.__version__)" 2>/dev/null; then
    echo "✓ JAX installed successfully"
else
    echo "✗ JAX installation failed"
    exit 1
fi

# Check PyTorch
if python -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>/dev/null; then
    echo "✓ PyTorch installed successfully"
    python -c "import torch; print('  CUDA available:', torch.cuda.is_available())"
else
    echo "✗ PyTorch installation failed"
    exit 1
fi

# Check OpenPI
if python -c "from openpi.policies import policy_config; print('✓ OpenPI imported successfully')" 2>/dev/null; then
    echo "✓ OpenPI installed successfully"
else
    echo "✗ OpenPI installation failed"
    exit 1
fi

# Check LIBERO
if python -c "from libero.libero import benchmark; print('✓ LIBERO imported successfully')" 2>/dev/null; then
    echo "✓ LIBERO installed successfully"
else
    echo "✗ LIBERO installation failed"
    exit 1
fi

# Check openpi-client
if python -c "from openpi_client import local_policy; print('✓ openpi-client installed successfully')" 2>/dev/null; then
    echo "✓ openpi-client installed successfully"
else
    echo "✗ openpi-client installation failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ Environment Setup Complete!"
echo "========================================"
echo ""
echo "To use this environment:"
echo "  source $ENV_PATH/bin/activate"
echo "  export PYTHONPATH=\$PWD/src:\$PWD/third_party/libero"
echo "  export CKPT=/path/to/your/checkpoint"
echo "  bash examples/libero/run_headless_eval.sh"
echo ""
echo "Quick test:"
echo "  export CKPT=/path/to/checkpoint"
echo "  export NUM_TRIALS=1"
echo "  bash examples/libero/run_headless_eval.sh"
echo ""