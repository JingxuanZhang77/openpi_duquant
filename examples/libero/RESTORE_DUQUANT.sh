#!/bin/bash
# Restore original DuQuant scripts (disable torch.compile)
# This reverts changes made by SPEED_UP_DUQUANT.sh

set -e

cd ~/VLM_REPO/openpi

echo "=========================================="
echo "DuQuant Script Restoration"
echo "=========================================="
echo ""
echo "This will restore your scripts from .backup files"
echo ""

# Check if backups exist
if [ ! -f "examples/libero/run_optimized_duquant.sh.backup" ]; then
    echo "❌ No backup files found!"
    echo "Nothing to restore."
    exit 1
fi

echo "📦 Restoring scripts from backups..."
cp examples/libero/run_optimized_duquant.sh.backup examples/libero/run_optimized_duquant.sh
cp examples/libero/run_llm_w4a8.sh.backup examples/libero/run_llm_w4a8.sh
cp examples/libero/run_dit_qkvo_w4a8.sh.backup examples/libero/run_dit_qkvo_w4a8.sh
cp examples/libero/run_full_llm_dit_w4a8.sh.backup examples/libero/run_full_llm_dit_w4a8.sh

echo "✅ Scripts restored to original state"
echo ""
echo "Torch.compile is now DISABLED (faster startup, slower inference)"
echo "To re-enable speedup, run: bash examples/libero/SPEED_UP_DUQUANT.sh"
echo ""
echo "=========================================="
