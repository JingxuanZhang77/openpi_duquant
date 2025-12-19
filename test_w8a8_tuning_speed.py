#!/usr/bin/env python3
"""Test W8A8 tuning impact on speed."""

import os
import sys
import shutil
import time

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn


def benchmark_layer(layer, x, n_warmup=20, n_iter=100):
    """Benchmark a layer."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = layer(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            _ = layer(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / n_iter) * 1000  # ms


def test_tuning_speed():
    """Compare tuned vs untuned W8A8 speed."""
    print("=" * 60)
    print("W8A8 Tuning Speed Comparison")
    print("=" * 60)

    # Clear cache to start fresh
    cache_path = os.path.expanduser("~/.cache/bitblas")
    if os.path.exists(cache_path):
        print(f"Clearing existing cache at {cache_path}")
        shutil.rmtree(cache_path)

    from openpi.models_pytorch.bitblas_w8a8_layers import (
        BitBLASW8A8Linear,
        save_tuning_cache,
        _MATMUL_CACHE,
    )

    # Clear local cache
    _MATMUL_CACHE.clear()

    # Test dimensions (typical MLP sizes)
    test_cases = [
        (2048, 8192, "Small MLP (2048 -> 8192)"),
        (8192, 2048, "Small MLP (8192 -> 2048)"),
    ]

    device = torch.device("cuda")
    batch_sizes = [1, 4, 16, 32]

    for in_features, out_features, desc in test_cases:
        print(f"\n{'=' * 60}")
        print(f"{desc}")
        print(f"{'=' * 60}")

        # Create reference FP16 linear
        torch.manual_seed(42)
        linear_fp16 = nn.Linear(in_features, out_features, bias=False).half().to(device)

        # Create UNTUNED W8A8 layer
        _MATMUL_CACHE.clear()
        print(f"\nCreating UNTUNED W8A8 layer...")
        untuned_layer = BitBLASW8A8Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            name="untuned",
            enable_tuning=False,  # No tuning
            opt_M=[1, 16, 32, 64],
        )
        untuned_layer.load_from_linear(linear_fp16)
        untuned_layer = untuned_layer.to(device)

        # Create TUNED W8A8 layer
        _MATMUL_CACHE.clear()
        print(f"Creating TUNED W8A8 layer (this will take ~30-60 seconds)...")
        tuned_layer = BitBLASW8A8Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            name="tuned",
            enable_tuning=True,  # With tuning
            opt_M=[1, 16, 32, 64],
        )
        tuned_layer.load_from_linear(linear_fp16)
        tuned_layer = tuned_layer.to(device)

        # Benchmark for different batch sizes
        print(f"\n{'Batch':<8} {'FP16':<12} {'Untuned':<12} {'Tuned':<12} {'Tuned Speedup':<15}")
        print("-" * 60)

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

            fp16_time = benchmark_layer(linear_fp16, x)
            untuned_time = benchmark_layer(untuned_layer, x)
            tuned_time = benchmark_layer(tuned_layer, x)

            speedup_vs_untuned = untuned_time / tuned_time
            speedup_vs_fp16 = fp16_time / tuned_time

            print(f"{batch_size:<8} {fp16_time:.3f} ms    {untuned_time:.3f} ms    {tuned_time:.3f} ms    "
                  f"{speedup_vs_untuned:.2f}x vs untuned, {speedup_vs_fp16:.2f}x vs FP16")

    # Save tuned operators
    print(f"\nSaving tuned operators to cache...")
    save_tuning_cache()


if __name__ == "__main__":
    test_tuning_speed()
