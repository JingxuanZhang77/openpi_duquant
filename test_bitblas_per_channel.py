#!/usr/bin/env python3
"""Test BitBLAS with per-channel scale (group_size = K)."""

import os
import sys
import torch
import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_per_channel():
    """Test BitBLAS with per-channel scale (group_size = in_features)."""
    print("=" * 60)
    print("Test: BitBLAS INT4 with Per-Channel Scale")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    batch_size = 1
    in_features = 2048
    out_features = 16384
    # Use per-channel scale (group_size = in_features)
    group_size = in_features  # This means 1 scale per output channel
    device = torch.device("cuda")

    target = auto_detect_nvidia_target()

    # Create random FP16 weight
    torch.manual_seed(42)
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale (better accuracy)
    w_absmax = W_fp16.abs().max(dim=1, keepdim=True)[0]
    scale = w_absmax / 7.0
    scale = torch.clamp(scale, min=1e-6)

    # Quantize
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    W_dequant = W_int.half() * scale
    W_uint = (W_int + 8).to(torch.uint8)

    # Input
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # Reference
    y_ref = torch.nn.functional.linear(x, W_dequant)
    print(f"Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # Setup with group_size = in_features (1 group, per-channel)
    n_groups = 1  # Per-channel = 1 group
    scales = scale.contiguous().clone()  # (out_features, 1)
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    print(f"\nWeight config:")
    print(f"  group_size: {group_size} (per-channel)")
    print(f"  scales shape: {scales.shape}")
    print(f"  zeros shape: {zeros.shape}")

    matmul_config = MatmulConfig(
        M=[1, 16, 32, 64],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="uint4",
        out_dtype="float16",
        accum_dtype="float16",
        with_scaling=True,
        with_zeros=True,
        zeros_mode="original",
        group_size=group_size,  # Per-channel
        with_bias=False,
        layout="nt",
    )

    matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)

    # Transform weight
    W_uint_int8 = W_uint.to(torch.int8)
    qweight = matmul.transform_weight(W_uint_int8, scale=scales, zeros=zeros)
    if isinstance(qweight, list):
        qweight = qweight[0]

    print(f"  qweight shape: {qweight.shape}")

    # Run
    output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    matmul(x, qweight, scales, zeros, output=output)

    print(f"\nBitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")

    # Compare
    error = (output - y_ref).abs()
    rel_error = error / (y_ref.abs() + 1e-6)

    print(f"\nError vs Reference:")
    print(f"  Absolute - Max: {error.max():.6f}, Mean: {error.mean():.6f}")
    print(f"  Relative - Max: {rel_error.max():.4f}, Mean: {rel_error.mean():.4f}")

    # Speed test
    import time

    for _ in range(5):
        matmul(x, qweight, scales, zeros, output=output)
    torch.cuda.synchronize()

    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        matmul(x, qweight, scales, zeros, output=output)
    torch.cuda.synchronize()
    bitblas_time = (time.time() - start) / n_iter * 1000

    start = time.time()
    for _ in range(n_iter):
        y_fp16 = torch.nn.functional.linear(x, W_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / n_iter * 1000

    print(f"\nSpeed:")
    print(f"  BitBLAS INT4: {bitblas_time:.3f} ms")
    print(f"  FP16 matmul:  {fp16_time:.3f} ms")
    print(f"  Speedup: {fp16_time / bitblas_time:.2f}x")

    if error.max() < 0.1:
        print("\n✅ SUCCESS!")
        return True
    else:
        print("\n❌ FAIL: Error too large")
        return False


def test_with_int4_dtype():
    """Test BitBLAS with int4 dtype instead of uint4."""
    print("\n" + "=" * 60)
    print("Test: BitBLAS with INT4 (signed) dtype")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    batch_size = 1
    in_features = 2048
    out_features = 16384
    group_size = 128
    device = torch.device("cuda")

    target = auto_detect_nvidia_target()

    torch.manual_seed(42)
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale
    w_absmax = W_fp16.abs().max(dim=1, keepdim=True)[0]
    scale = w_absmax / 7.0
    scale = torch.clamp(scale, min=1e-6)

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    W_dequant = W_int.half() * scale

    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    y_ref = torch.nn.functional.linear(x, W_dequant)
    print(f"Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # Setup for SIGNED int4
    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous().clone()

    print(f"\nWeight config:")
    print(f"  W_dtype: int4 (signed)")
    print(f"  group_size: {group_size}")
    print(f"  scales shape: {scales.shape}")

    # Try int4 (signed) without zeros
    matmul_config = MatmulConfig(
        M=[1, 16, 32, 64],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="int4",  # Signed INT4
        out_dtype="float16",
        accum_dtype="float16",
        with_scaling=True,
        with_zeros=False,  # No zeros for signed
        group_size=group_size,
        with_bias=False,
        layout="nt",
    )

    matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)

    # Transform weight (keep as int8 container for int4)
    qweight = matmul.transform_weight(W_int.cuda(), scale=scales)
    if isinstance(qweight, list):
        qweight = qweight[0]

    print(f"  qweight shape: {qweight.shape}")

    # Run - note: no zeros argument for int4
    output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    matmul(x, qweight, scales, output=output)

    print(f"\nBitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")

    error = (output - y_ref).abs()
    print(f"Error - Max: {error.max():.6f}, Mean: {error.mean():.6f}")

    if error.max() < 0.1:
        print("\n✅ SUCCESS!")
        return True
    else:
        print("\n❌ FAIL!")
        return False


if __name__ == "__main__":
    test_per_channel()
    test_with_int4_dtype()
