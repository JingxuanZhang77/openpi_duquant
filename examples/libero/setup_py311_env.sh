#!/bin/bash
# Setup Python 3.11 environment for LIBERO headless evaluation
# This resolves the Python version incompatibility between LIBERO (3.9) and OpenPI (3.11+)

set -e  # Exit on error

echo "========================================"
echo "Python 3.11 Environment Setup"
echo "========================================"

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: Python 3.11 not found"
    echo ""
    echo "Please install Python 3.11 first:"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo ""
    echo "Or using pyenv:"
    echo "  pyenv install 3.11.11"
    echo "  pyenv local 3.11.11"
    echo ""
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "examples/libero/requirements.txt" ]; then
    echo "❌ Error: Please run this script from the openpi root directory"
    echo "Usage: bash examples/libero/setup_py311_env.sh"
    exit 1
fi

# Show Python version
PY_VERSION=$(python3.11 --version)
echo "✓ Found: $PY_VERSION"
echo ""

# Create virtual environment
ENV_PATH="examples/libero/.venv-py311"
echo "Creating virtual environment at $ENV_PATH..."
python3.11 -m venv $ENV_PATH

# Activate environment
echo "Activating environment..."
source $ENV_PATH/bin/activate

# Verify Python version in venv
VENV_PY_VERSION=$(python --version)
echo "✓ Virtual environment Python: $VENV_PY_VERSION"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install LIBERO dependencies
echo ""
echo "Installing LIBERO dependencies..."
pip install -r examples/libero/requirements.txt \
            -r third_party/libero/requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu113 \
            --index-strategy=unsafe-best-match

# Install OpenPI (this will work now with Python 3.11)
echo ""
echo "Installing OpenPI..."
pip install -e .

# Install openpi-client
echo ""
echo "Installing openpi-client..."
pip install -e packages/openpi-client

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install sentencepiece

# Optional: Setup robosuite macros
echo ""
echo "Setting up robosuite macros (optional)..."
python $(python -c "import robosuite, pathlib; print(pathlib.Path(robosuite.__file__).parent / 'scripts' / 'setup_macros.py')") 2>/dev/null || echo "⚠️  Warning: robosuite macros setup failed (not critical)"

# Verify installation
echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"

# Check Python version
echo "Python version:"
python -c "import sys; print('  ', sys.version)"

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
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "🎯 To use this environment:"
echo ""
echo "  # Activate environment"
echo "  source $ENV_PATH/bin/activate"
echo ""
echo "  # Set environment variables"
echo "  export PYTHONPATH=\$PWD/src:\$PWD/third_party/libero"
echo "  export CKPT=/path/to/your/checkpoint"
echo ""
echo "  # Run headless evaluation"
echo "  bash examples/libero/run_headless_eval.sh"
echo ""
echo "🧪 Quick test (1 episode):"
echo ""
echo "  export CKPT=/path/to/checkpoint"
echo "  export NUM_TRIALS=1"
echo "  bash examples/libero/run_headless_eval.sh"
echo ""
echo "📚 Documentation:"
echo "  - PYTHON_VERSION_FIX.md - Python version解决方案"
echo "  - HEADLESS_QUICKSTART.md - 快速开始指南"
echo "  - ENVIRONMENT_SETUP.md - 详细配置说明"
echo ""