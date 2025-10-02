#!/bin/bash
# Setup LLM quantization by reusing DiT pack files
# DiT and LLM have identical structure (18 layers, same dimensions)

set -e

cd ~/VLM_REPO/openpi

DIT_PACK="/home/jz97/VLM_REPO/openpi/duquant_packed_b16_p1_rrestore_a999"
LLM_PACK="/home/jz97/VLM_REPO/openpi/duquant_packed_llm_w4a8"

echo "========================================"
echo "Setup LLM Pack from DiT"
echo "========================================"
echo ""

# Check DiT pack
if [ ! -d "$DIT_PACK" ]; then
    echo "❌ Error: DiT pack directory not found: $DIT_PACK"
    echo ""
    echo "Please run DiT quantization first:"
    echo "  bash examples/libero/run_optimized_duquant.sh"
    exit 1
fi

DIT_COUNT=$(find "$DIT_PACK" -name "*.npz" 2>/dev/null | wc -l)
echo "DiT pack files: $DIT_COUNT"

if [ "$DIT_COUNT" -lt 90 ]; then
    echo "⚠️  Warning: DiT pack incomplete ($DIT_COUNT < 126)"
    echo "Expected 126 files (18 layers × 7 linear)"
    echo ""
    read -p "Complete DiT packing first? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running DiT quantization..."
        bash examples/libero/run_optimized_duquant.sh
        exit 0
    else
        echo "Continuing with partial pack..."
    fi
fi

# Copy or link DiT pack to LLM
echo ""
echo "Setting up LLM pack directory..."

if [ -d "$LLM_PACK" ]; then
    echo "⚠️  LLM pack directory already exists"
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$LLM_PACK"
    else
        echo "Aborted."
        exit 1
    fi
fi

# Create symbolic link (faster than copy, saves disk space)
echo "Creating symbolic link: $LLM_PACK -> $DIT_PACK"
ln -s "$DIT_PACK" "$LLM_PACK"

echo ""
echo "✅ Done!"
echo ""
echo "========================================"
echo "Ready to run LLM quantization"
echo "========================================"
echo ""
echo "The LLM will use DiT's pack files (same structure)"
echo ""
echo "Run:"
echo "  bash examples/libero/run_llm_w4a8.sh"
echo ""
echo "Note: Pack files only depend on weight matrix shape,"
echo "      not actual values, so DiT and LLM can share."
echo "========================================"
