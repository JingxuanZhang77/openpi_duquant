#!/usr/bin/env python3
"""Test W8A8 BitBLAS tuning cache persistence."""

import os
import sys
import shutil

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn


def test_cache_persistence():
    """Test that tuning cache is properly saved and loaded."""
    print("=" * 60)
    print("W8A8 Tuning Cache Persistence Test")
    print("=" * 60)

    from bitblas.cache import get_database_path, global_operator_cache
    from openpi.models_pytorch.bitblas_w8a8_layers import (
        BitBLASW8A8Linear,
        save_tuning_cache,
        load_tuning_cache,
        _MATMUL_CACHE,
    )

    # Get cache path
    cache_path = get_database_path()
    print(f"\nBitBLAS cache path: {cache_path}")
    print(f"Cache exists before test: {os.path.exists(cache_path)}")

    # Clear local matmul cache
    _MATMUL_CACHE.clear()

    # Test dimensions
    in_features = 2048
    out_features = 4096

    device = torch.device("cuda")

    # Create reference FP16 linear
    torch.manual_seed(42)
    linear_fp16 = nn.Linear(in_features, out_features, bias=False).half().to(device)

    print(f"\n--- Step 1: Create W8A8 layer with tuning ---")
    print(f"Dimensions: {in_features} -> {out_features}")

    # Create W8A8 layer WITH tuning
    w8a8_linear = BitBLASW8A8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        name="test_layer",
        enable_tuning=True,  # Enable tuning
        opt_M=[1, 16, 32],
    )
    w8a8_linear.load_from_linear(linear_fp16)
    w8a8_linear = w8a8_linear.to(device)

    # Do a forward pass
    x = torch.randn(4, in_features, dtype=torch.float16, device=device)
    with torch.no_grad():
        y = w8a8_linear(x)
    print(f"Forward pass output shape: {y.shape}")

    # Check global cache
    print(f"\nglobal_operator_cache size: {global_operator_cache.size()}")

    print(f"\n--- Step 2: Save tuning cache ---")
    save_tuning_cache()

    # Check if cache was created
    print(f"Cache exists after save: {os.path.exists(cache_path)}")
    if os.path.exists(cache_path):
        # List cache contents
        print(f"Cache contents:")
        for item in os.listdir(cache_path):
            item_path = os.path.join(cache_path, item)
            if os.path.isdir(item_path):
                sub_items = os.listdir(item_path)
                print(f"  {item}/ ({len(sub_items)} entries)")
                for sub in sub_items[:3]:
                    sub_path = os.path.join(item_path, sub)
                    if os.path.isdir(sub_path):
                        files = os.listdir(sub_path)
                        print(f"    {sub}/ ({len(files)} files)")
            else:
                print(f"  {item}")

    print(f"\n--- Step 3: Clear local cache and reload ---")
    _MATMUL_CACHE.clear()
    global_operator_cache.clear()
    print(f"global_operator_cache size after clear: {global_operator_cache.size()}")

    # Load from database
    loaded_count = load_tuning_cache()
    print(f"Loaded {loaded_count} operators from cache")

    print(f"\n--- Step 4: Create new layer (should use cached kernel) ---")

    # Create another W8A8 layer with same dimensions
    w8a8_linear2 = BitBLASW8A8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        name="test_layer_2",
        enable_tuning=False,  # Don't need tuning if cached
        opt_M=[1, 16, 32],
    )
    w8a8_linear2.load_from_linear(linear_fp16)
    w8a8_linear2 = w8a8_linear2.to(device)

    # Test it works
    with torch.no_grad():
        y2 = w8a8_linear2(x)
    print(f"Second forward pass output shape: {y2.shape}")

    # Compare outputs
    diff = (y - y2).abs().max().item()
    print(f"Max diff between outputs: {diff}")

    success = os.path.exists(cache_path) and loaded_count > 0
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: Cache persistence is working!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("FAILED: Cache was not properly saved/loaded")
        print("=" * 60)

    return success


if __name__ == "__main__":
    success = test_cache_persistence()
    sys.exit(0 if success else 1)
