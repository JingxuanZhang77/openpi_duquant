#!/bin/bash
# Quick test of fresh venv installation
set -e

VENV_NAME=.venv_final_test
cd ~/VLM_REPO/openpi

# Clean up if exists
if [ -d "$VENV_NAME" ]; then
    rm -rf "$VENV_NAME"
fi

echo "Creating fresh UV environment..."
uv venv "$VENV_NAME" --python 3.11

echo "Activating..."
source "$VENV_NAME/bin/activate"

echo "Installing OpenPI..."
uv pip install -e . --no-cache 2>&1 | tail -5

echo "Installing LIBERO dependencies..."
uv pip install robosuite dm-control pyyaml --no-cache 2>&1 | tail -3

echo "Installing LIBERO..."
cd third_party/libero
uv pip install -e . --no-cache 2>&1 | tail -3
cd ../..

echo ""
echo "Checking installation..."
uv pip list | grep -E "(libero|openpi|robosuite|pyyaml)" || echo "No matches"

echo ""
echo "Testing imports..."
python << 'PYTEST'
import sys
print(f"Python: {sys.executable}")
print(f"sys.path[:3]: {sys.path[:3]}")

try:
    import yaml
    print(f"✓ yaml: {yaml.__version__}")
except Exception as e:
    print(f"✗ yaml: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ torch: {torch.__version__}")
except Exception as e:
    print(f"✗ torch: {e}")
    sys.exit(1)

try:
    import libero
    print(f"✓ libero imported")
except Exception as e:
    print(f"✗ libero: {e}")
    import os
    libero_path = "/home/jz97/VLM_REPO/openpi/third_party/libero"
    if os.path.exists(libero_path):
        print(f"  libero path exists: {libero_path}")
        print(f"  Contents: {os.listdir(libero_path)[:5]}")
    sys.exit(1)

try:
    import openpi
    print(f"✓ openpi imported")
except Exception as e:
    print(f"✗ openpi: {e}")
    sys.exit(1)

try:
    import robosuite
    print(f"✓ robosuite imported")
except Exception as e:
    print(f"✗ robosuite: {e}")

try:
    import dm_control
    print(f"✓ dm_control imported")
except Exception as e:
    print(f"✗ dm_control: {e}")

print("\n✓ All critical imports successful!")
PYTEST

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Fresh environment setup successful!"
else
    echo ""
    echo "✗ Import test failed"
    deactivate
    exit 1
fi

echo ""
echo "Cleaning up test environment..."
deactivate
cd ~/VLM_REPO/openpi
rm -rf "$VENV_NAME"
echo "✓ Test complete!"
