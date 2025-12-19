#!/usr/bin/env python3
"""Test W8A8 BitBLAS implementation - correctness and speed."""

import os
import sys
import time

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn


def test_w8a8_correctness():
    """Test W8A8 layer correctness against FP16 reference."""
    print("=" * 60)
    print("W8A8 Correctness Test")
    print("=" * 60)

    from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear

    # Test dimensions (typical MLP layer)
    in_features = 2048
    out_features = 8192
    batch_sizes = [1, 4, 16, 32]

    device = torch.device("cuda")

    # Create reference FP16 linear
    torch.manual_seed(42)
    linear_fp16 = nn.Linear(in_features, out_features, bias=False).half().to(device)

    # Create W8A8 layer
    w8a8_linear = BitBLASW8A8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        name="test_layer",
        enable_tuning=False,
        opt_M=[1, 16, 32, 64],
    )
    w8a8_linear.load_from_linear(linear_fp16)
    w8a8_linear = w8a8_linear.to(device)

    print(f"\nDimensions: {in_features} -> {out_features}")
    print(f"qweight shape: {w8a8_linear.qweight.shape}")
    print(f"weight_scale shape: {w8a8_linear.weight_scale.shape}")

    all_passed = True

    for batch_size in batch_sizes:
        torch.manual_seed(123 + batch_size)
        x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

        # FP16 reference
        with torch.no_grad():
            y_fp16 = linear_fp16(x)

        # W8A8
        with torch.no_grad():
            y_w8a8 = w8a8_linear(x)

        # Compare
        error = (y_w8a8 - y_fp16).abs()
        rel_error = error / (y_fp16.abs() + 1e-6)

        max_abs_error = error.max().item()
        mean_abs_error = error.mean().item()
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()

        # INT8 quantization: use absolute error threshold
        # Relative error can be large when values are close to 0
        # For typical MLP outputs, max absolute error < 0.1 is acceptable
        passed = max_abs_error < 0.1 and mean_abs_error < 0.01

        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"\nBatch size: {batch_size}")
        print(f"  Absolute Error - Max: {max_abs_error:.6f}, Mean: {mean_abs_error:.6f}")
        print(f"  Relative Error - Max: {max_rel_error:.4f}, Mean: {mean_rel_error:.4f}")
        print(f"  Status: {status}")

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)

    return all_passed


def test_w8a8_speed():
    """Benchmark W8A8 vs FP16 speed."""
    print("\n" + "=" * 60)
    print("W8A8 Speed Benchmark")
    print("=" * 60)

    from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear

    # Test dimensions
    in_features = 2048
    out_features = 8192

    device = torch.device("cuda")

    # Create layers
    torch.manual_seed(42)
    linear_fp16 = nn.Linear(in_features, out_features, bias=False).half().to(device)

    w8a8_linear = BitBLASW8A8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        name="bench_layer",
        enable_tuning=False,
        opt_M=[1, 16, 32, 64],
    )
    w8a8_linear.load_from_linear(linear_fp16)
    w8a8_linear = w8a8_linear.to(device)

    print(f"\nDimensions: {in_features} -> {out_features}")

    for batch_size in [1, 4, 16, 32]:
        x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(10):
            _ = w8a8_linear(x)
            _ = linear_fp16(x)
        torch.cuda.synchronize()

        n_iter = 100

        # W8A8 benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            _ = w8a8_linear(x)
        torch.cuda.synchronize()
        w8a8_time = (time.time() - start) / n_iter * 1000

        # FP16 benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            _ = linear_fp16(x)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / n_iter * 1000

        speedup = fp16_time / w8a8_time

        print(f"\nBatch size: {batch_size}")
        print(f"  W8A8:  {w8a8_time:.3f} ms")
        print(f"  FP16:  {fp16_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x", "(FASTER)" if speedup > 1 else "(SLOWER)")


def test_memory_savings():
    """Test memory savings of W8A8 vs FP16."""
    print("\n" + "=" * 60)
    print("W8A8 Memory Comparison")
    print("=" * 60)

    from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear

    in_features = 2048
    out_features = 8192

    # FP16: 2 bytes per element
    fp16_bytes = in_features * out_features * 2
    # INT8: 1 byte per element + scale (negligible)
    int8_bytes = in_features * out_features * 1

    print(f"\nDimensions: {in_features} x {out_features}")
    print(f"FP16 weight size: {fp16_bytes / 1024 / 1024:.2f} MB")
    print(f"INT8 weight size: {int8_bytes / 1024 / 1024:.2f} MB")
    print(f"Memory savings: {(1 - int8_bytes / fp16_bytes) * 100:.1f}%")


if __name__ == "__main__":
    print("Testing W8A8 BitBLAS Implementation")
    print("=" * 60)

    # Test correctness
    passed = test_w8a8_correctness()

    if passed:
        # Test speed
        test_w8a8_speed()

        # Show memory savings
        test_memory_savings()
