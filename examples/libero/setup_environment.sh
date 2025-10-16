#!/bin/bash
#
# OpenPI LIBERO Environment Setup Script
# This script automatically sets up the complete environment for OpenPI LIBERO DuQuant experiments
#
# Requirements:
#   - Ubuntu 20.04+ or similar Linux distribution
#   - NVIDIA GPU with CUDA support
#   - Sudo access for system package installation
#
# Usage:
#   bash setup_environment.sh
#
# The script will:
#   1. Install system dependencies
#   2. Install Python 3.11
#   3. Install uv (fast Python package manager)
#   4. Create virtual environment
#   5. Install all Python dependencies
#   6. Install third-party packages (libero, BitBLAS)
#   7. Verify installation

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
VENV_DIR="examples/libero/.venv"
REPO_ROOT=$(pwd)

# Helper functions
print_header() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_warning "$1 is not installed"
        return 1
    fi
}

# Check if running in OpenPI repository root
if [ ! -f "pyproject.toml" ] || [ ! -d "examples/libero" ]; then
    print_error "This script must be run from the OpenPI repository root directory"
    exit 1
fi

print_header "OpenPI LIBERO Environment Setup"
echo "This script will set up the complete environment for OpenPI LIBERO experiments"
echo ""
echo "Installation path: $REPO_ROOT"
echo "Python version: $PYTHON_VERSION"
echo "Virtual environment: $VENV_DIR"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# ============================================
# Step 1: Install System Dependencies
# ============================================
print_header "Step 1: Installing System Dependencies"

echo "Checking for sudo access..."
if ! sudo -v; then
    print_error "Sudo access required for system package installation"
    exit 1
fi

print_success "Updating package lists..."
sudo apt-get update -qq

print_success "Installing system dependencies..."
sudo apt-get install -y -qq \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    || true  # Don't fail if some packages are already installed

print_success "System dependencies installed"

# ============================================
# Step 2: Check/Install Python 3.11
# ============================================
print_header "Step 2: Checking Python Installation"

if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    print_success "Python 3.11 is already installed"
elif command -v python3 &> /dev/null && python3 --version | grep -q "3.11"; then
    PYTHON_CMD="python3"
    print_success "Python 3.11 is already installed (as python3)"
else
    print_warning "Python 3.11 not found, installing..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    PYTHON_CMD="python3.11"
    print_success "Python 3.11 installed"
fi

$PYTHON_CMD --version

# ============================================
# Step 3: Install uv (Fast Python Package Manager)
# ============================================
print_header "Step 3: Installing uv Package Manager"

if command -v uv &> /dev/null; then
    print_success "uv is already installed: $(uv --version)"
else
    print_warning "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "uv installed: $(uv --version)"
fi

# ============================================
# Step 4: Check CUDA Installation
# ============================================
print_header "Step 4: Checking CUDA Installation"

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    print_success "CUDA is installed: version $CUDA_VERSION"
else
    print_warning "CUDA not found in PATH"
    print_warning "PyTorch will be installed with bundled CUDA runtime"
fi

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
        print_success "GPU: $line"
    done
else
    print_error "nvidia-smi not found. NVIDIA drivers may not be installed correctly"
    print_error "Please install NVIDIA drivers before continuing"
    exit 1
fi

# ============================================
# Step 5: Create Virtual Environment
# ============================================
print_header "Step 5: Creating Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        print_success "Removed existing virtual environment"
    else
        print_warning "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_success "Creating virtual environment with uv..."
    uv venv "$VENV_DIR" --python $PYTHON_CMD
    print_success "Virtual environment created"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated: $(which python)"
python --version

# ============================================
# Step 6: Install PyTorch and JAX (CUDA 12.x)
# ============================================
print_header "Step 6: Installing PyTorch and JAX with CUDA 12.x"

print_success "Installing PyTorch 2.7.1 with CUDA 12..."
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu124

print_success "Installing JAX 0.5.3 with CUDA 12..."
uv pip install "jax[cuda12]==0.5.3"

print_success "PyTorch and JAX installed"

# ============================================
# Step 7: Install Core OpenPI Dependencies
# ============================================
print_header "Step 7: Installing Core OpenPI Dependencies"

print_success "Installing transformers and related packages..."
uv pip install transformers==4.53.2 sentencepiece accelerate

print_success "Installing other core dependencies..."
uv pip install \
    einops \
    equinox \
    flax==0.10.2 \
    ml_collections \
    orbax-checkpoint==0.11.13 \
    wandb \
    tyro \
    rich \
    tqdm

# ============================================
# Step 8: Install LIBERO-Specific Dependencies
# ============================================
print_header "Step 8: Installing LIBERO Dependencies"

cd "$REPO_ROOT/examples/libero"

print_success "Installing LIBERO requirements..."
uv pip install -r requirements.txt

# Install additional dependencies
uv pip install \
    datasets \
    lerobot \
    opencv-python \
    imageio \
    imageio-ffmpeg

cd "$REPO_ROOT"

# ============================================
# Step 9: Install Third-Party Packages
# ============================================
print_header "Step 9: Installing Third-Party Packages"

# Install LIBERO
print_success "Installing LIBERO (editable)..."
cd "$REPO_ROOT/third_party/libero"
uv pip install -e .
cd "$REPO_ROOT"

# Install OpenPI itself
print_success "Installing OpenPI (editable)..."
uv pip install -e .

# Optional: Install BitBLAS if needed
if [ -d "$REPO_ROOT/third_party/BitBLAS" ]; then
    print_success "BitBLAS found, installing..."
    cd "$REPO_ROOT/third_party/BitBLAS"
    uv pip install -e .
    cd "$REPO_ROOT"
fi

# ============================================
# Step 10: Verify Installation
# ============================================
print_header "Step 10: Verifying Installation"

echo "Running verification tests..."

# Test Python version
PYTHON_VER=$(python --version 2>&1)
if echo "$PYTHON_VER" | grep -q "3.11"; then
    print_success "Python version: $PYTHON_VER"
else
    print_error "Python version mismatch: $PYTHON_VER"
fi

# Test PyTorch
if python -c "import torch; print(f'PyTorch {torch.__version__}'); assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    print_success "PyTorch $TORCH_VER installed, CUDA device: $CUDA_AVAIL"
else
    print_error "PyTorch installation failed or CUDA not available"
fi

# Test JAX
if python -c "import jax; import jax.numpy as jnp; x = jnp.array([1, 2, 3]); print(f'JAX {jax.__version__}')" 2>/dev/null; then
    JAX_VER=$(python -c "import jax; print(jax.__version__)")
    print_success "JAX $JAX_VER installed"
else
    print_error "JAX installation failed"
fi

# Test transformers
if python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null; then
    TRANS_VER=$(python -c "import transformers; print(transformers.__version__)")
    print_success "Transformers $TRANS_VER installed"
else
    print_error "Transformers installation failed"
fi

# Test OpenPI
if python -c "import openpi" 2>/dev/null; then
    print_success "OpenPI package installed successfully"
else
    print_error "OpenPI package not found"
fi

# Test LIBERO
if python -c "import libero" 2>/dev/null; then
    print_success "LIBERO package installed successfully"
else
    print_error "LIBERO package not found"
fi

# ============================================
# Step 11: Create Environment Activation Script
# ============================================
print_header "Step 11: Creating Environment Activation Script"

cat > "$REPO_ROOT/examples/libero/activate_env.sh" << 'EOF'
#!/bin/bash
# Activate OpenPI LIBERO environment
# Usage: source examples/libero/activate_env.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/examples/libero/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run setup_environment.sh first"
    return 1
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Set environment variables
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/third_party/libero:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: Enable DuQuant optimizations
export OPENPI_DUQUANT_BATCH_ROT=1

echo "OpenPI LIBERO environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x "$REPO_ROOT/examples/libero/activate_env.sh"
print_success "Activation script created: examples/libero/activate_env.sh"

# ============================================
# Completion
# ============================================
print_header "Installation Complete!"

echo ""
echo "Environment setup successful!"
echo ""
echo "To activate the environment in the future, run:"
echo "  ${GREEN}source examples/libero/activate_env.sh${NC}"
echo ""
echo "To run LIBERO experiments:"
echo "  ${GREEN}cd examples/libero${NC}"
echo "  ${GREEN}source activate_env.sh${NC}"
echo "  ${GREEN}bash run_optimized_duquant.sh${NC}"
echo ""
echo "Virtual environment location: $VENV_DIR"
echo ""
print_success "Setup complete!"
