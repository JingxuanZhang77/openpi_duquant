#!/usr/bin/env python3
"""Test script to verify torch.compile fix for DuQuant."""

import torch
import sys
sys.path.insert(0, 'src')

from openpi.models_pytorch.duquant_layers import DuQuantLinear, DuQuantConfig

def test_torch_compile_compatibility():
    """Test that DuQuantLinear works with torch.compile."""
    print("=" * 80)
    print("Testing Torch.Compile Compatibility")
    print("=" * 80)
    print()

    # Create a simple Linear layer
    print("1. Creating base Linear layer...")
    base_linear = torch.nn.Linear(128, 256, bias=True)
    base_linear = base_linear.to(device='cuda', dtype=torch.bfloat16)
    print("   ✅ Base layer created")

    # Create DuQuant config
    print()
    print("2. Creating DuQuant config...")
    cfg = DuQuantConfig(
        weight_bits=4,
        act_bits=8,
        block_size=16,
        enable_permute=True,
        calib_batches=0,  # Disable calibration for quick test
    )
    print("   ✅ Config created")

    # Wrap with DuQuantLinear
    print()
    print("3. Wrapping with DuQuantLinear...")
    duquant_layer = DuQuantLinear(base_linear, name="test_layer", cfg=cfg)
    duquant_layer = duquant_layer.to(device='cuda')
    print("   ✅ DuQuantLinear created")

    # Test forward pass (non-compiled)
    print()
    print("4. Testing forward pass (non-compiled)...")
    x = torch.randn(2, 128, device='cuda', dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            y = duquant_layer(x)
        print(f"   ✅ Forward pass succeeded: {x.shape} -> {y.shape}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

    # Test with torch.compile
    print()
    print("5. Compiling with torch.compile...")
    try:
        compiled_layer = torch.compile(duquant_layer, mode="reduce-overhead")
        print("   ✅ Compilation succeeded")
    except Exception as e:
        print(f"   ❌ Compilation failed: {e}")
        return False

    # Test compiled forward pass
    print()
    print("6. Testing compiled forward pass...")
    try:
        with torch.no_grad():
            y_compiled = compiled_layer(x)
        print(f"   ✅ Compiled forward pass succeeded: {x.shape} -> {y_compiled.shape}")
    except Exception as e:
        print(f"   ❌ Compiled forward pass failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        return False

    # Test multiple forward passes (check for recompilation issues)
    print()
    print("7. Testing multiple compiled forward passes...")
    try:
        for i in range(3):
            with torch.no_grad():
                y = compiled_layer(x)
            print(f"   ✅ Pass {i+1}/3 succeeded")
    except Exception as e:
        print(f"   ❌ Multiple passes failed: {e}")
        return False

    # Compare outputs
    print()
    print("8. Comparing compiled vs non-compiled outputs...")
    with torch.no_grad():
        y_normal = duquant_layer(x)
        y_compiled = compiled_layer(x)

    diff = torch.abs(y_normal - y_compiled).max().item()
    print(f"   Max difference: {diff:.6e}")
    if diff < 1e-3:
        print("   ✅ Outputs match!")
    else:
        print(f"   ⚠️  Outputs differ by {diff:.6e}")

    print()
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Torch.compile is now compatible with DuQuant!")
    print("You can safely enable torch.compile in your scripts.")
    return True


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        sys.exit(1)

    try:
        success = test_torch_compile_compatibility()
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED WITH EXCEPTION")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
