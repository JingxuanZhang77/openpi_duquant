#!/bin/bash
#
# Test OpenPI LIBERO Environment
# Quick verification that everything is installed correctly
#
# Usage: bash test_environment.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "OpenPI LIBERO Environment Test"
echo "========================================"
echo ""

# Test 1: Virtual environment exists
echo "[1/8] Checking virtual environment..."
if [ -d "examples/libero/.venv" ]; then
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
else
    echo -e "${RED}✗ Virtual environment not found${NC}"
    exit 1
fi

# Activate venv
source examples/libero/.venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$PWD/src:$PWD/third_party/libero:$PYTHONPATH"

# Test 2: Python version
echo "[2/8] Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1)
if echo "$PYTHON_VERSION" | grep -q "3.11"; then
    echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python version incorrect: $PYTHON_VERSION${NC}"
    exit 1
fi

# Test 3: PyTorch
echo "[3/8] Checking PyTorch..."
if python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch installed${NC}"
else
    echo -e "${RED}✗ PyTorch not found${NC}"
    exit 1
fi

# Test 4: CUDA
echo "[4/8] Checking CUDA..."
if python -c "import torch; assert torch.cuda.is_available(); print('CUDA device:', torch.cuda.get_device_name(0))" 2>/dev/null; then
    echo -e "${GREEN}✓ CUDA available${NC}"
else
    echo -e "${RED}✗ CUDA not available${NC}"
    exit 1
fi

# Test 5: JAX
echo "[5/8] Checking JAX..."
if python -c "import jax; print('JAX version:', jax.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ JAX installed${NC}"
else
    echo -e "${RED}✗ JAX not found${NC}"
    exit 1
fi

# Test 6: Transformers
echo "[6/8] Checking Transformers..."
if python -c "import transformers; print('Transformers version:', transformers.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ Transformers installed${NC}"
else
    echo -e "${RED}✗ Transformers not found${NC}"
    exit 1
fi

# Test 7: OpenPI
echo "[7/8] Checking OpenPI..."
if python -c "import openpi" 2>/dev/null; then
    echo -e "${GREEN}✓ OpenPI installed${NC}"
else
    echo -e "${RED}✗ OpenPI not found${NC}"
    exit 1
fi

# Test 8: LIBERO
echo "[8/8] Checking LIBERO..."
if python -c "import libero" 2>/dev/null; then
    echo -e "${GREEN}✓ LIBERO installed${NC}"
else
    echo -e "${RED}✗ LIBERO not found${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "========================================"
echo ""
echo "Environment is ready for OpenPI LIBERO experiments"
echo ""
echo "Next steps:"
echo "  source examples/libero/activate_env.sh"
echo "  bash examples/libero/run_optimized_duquant.sh"
