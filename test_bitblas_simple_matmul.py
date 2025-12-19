#!/usr/bin/env python3
"""Test BitBLAS INT4 matmul correctness - simple version without DuQuant transforms."""

import os
import sys
import torch
import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_bitblas_basic():
    """Test BitBLAS INT4 matmul produces correct output."""
    print("=" * 60)
    print("Test: BitBLAS INT4 Matmul Basic Correctness")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    # Small test dimensions
    batch_size = 4
    in_features = 256
    out_features = 512
    group_size = 128
    device = torch.device("cuda")

    target = auto_detect_nvidia_target()
    print(f"Target: {target}")

    # Create random FP16 weight
    torch.manual_seed(42)
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Compute per-channel scale for symmetric quantization
    # scale = max(abs(W)) / 7  (for INT4 range [-8, 7])
    w_absmax = W_fp16.abs().max(dim=1, keepdim=True)[0]
    scale = w_absmax / 7.0
    scale = torch.clamp(scale, min=1e-6)

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)

    # Dequantized weight for reference
    W_dequant = W_int.half() * scale

    # Create input
    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # Reference output (FP16 matmul with dequantized weight)
    y_ref = torch.nn.functional.linear(x, W_dequant)
    print(f"Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # Create BitBLAS kernel with UINT4 + zeros
    # Signed INT4 [-8, 7] -> Unsigned [0, 15] by adding 8
    W_uint = (W_int + 8).to(torch.uint8)

    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous().clone()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    print(f"\nWeight shapes:")
    print(f"  W_uint: {W_uint.shape}, range: [{W_uint.min()}, {W_uint.max()}]")
    print(f"  scales: {scales.shape}, range: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"  zeros: {zeros.shape}, value: {zeros[0,0]:.1f}")

    # Create BitBLAS matmul
    matmul_config = MatmulConfig(
        M=[batch_size],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="uint4",
        out_dtype="float16",
        accum_dtype="float16",
        with_scaling=True,
        with_zeros=True,
        zeros_mode="original",
        group_size=group_size,
        with_bias=False,
        layout="nt",
    )

    matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)
    print(f"\nBitBLAS Matmul created")
    print(f"  weight_transform: {matmul.weight_transform}")

    # Use BitBLAS transform_weight
    W_uint_int8 = W_uint.to(torch.int8)  # BitBLAS expects int8 container
    qweight = matmul.transform_weight(W_uint_int8, scale=scales, zeros=zeros)

    if isinstance(qweight, list):
        qweight = qweight[0]

    print(f"  qweight shape: {qweight.shape}, dtype: {qweight.dtype}")

    # Run BitBLAS kernel
    output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    matmul(x, qweight, scales, zeros, output=output)

    print(f"\nBitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")

    # Compare
    error = (output - y_ref).abs()
    rel_error = error / (y_ref.abs() + 1e-6)

    print(f"\nError vs Reference:")
    print(f"  Absolute - Max: {error.max():.6f}, Mean: {error.mean():.6f}")
    print(f"  Relative - Max: {rel_error.max():.4f}, Mean: {rel_error.mean():.4f}")

    if error.max() < 0.1:
        print("\n✅ SUCCESS: BitBLAS INT4 matmul is correct!")
        return True
    else:
        print("\n❌ FAIL: Large error in BitBLAS output!")
        return False


def test_bitblas_larger():
    """Test with dimensions similar to actual model."""
    print("\n" + "=" * 60)
    print("Test: BitBLAS INT4 Matmul (Model-like Dimensions)")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    # Dimensions like gate_proj: 2048 -> 16384
    batch_size = 1  # Common for inference
    in_features = 2048
    out_features = 16384
    group_size = 128
    device = torch.device("cuda")

    target = auto_detect_nvidia_target()

    # Create random FP16 weight
    torch.manual_seed(42)
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale
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

    # BitBLAS setup
    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous().clone()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    matmul_config = MatmulConfig(
        M=[1, 16, 32, 64],  # Multiple batch sizes
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="uint4",
        out_dtype="float16",
        accum_dtype="float16",
        with_scaling=True,
        with_zeros=True,
        zeros_mode="original",
        group_size=group_size,
        with_bias=False,
        layout="nt",
    )

    matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)

    # Transform weight
    W_uint_int8 = W_uint.to(torch.int8)
    qweight = matmul.transform_weight(W_uint_int8, scale=scales, zeros=zeros)
    if isinstance(qweight, list):
        qweight = qweight[0]

    print(f"qweight shape: {qweight.shape}")

    # Run
    output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    matmul(x, qweight, scales, zeros, output=output)

    print(f"BitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")

    # Compare
    error = (output - y_ref).abs()
    print(f"Error - Max: {error.max():.6f}, Mean: {error.mean():.6f}")

    # Measure speed
    import time

    # Warmup
    for _ in range(5):
        matmul(x, qweight, scales, zeros, output=output)
    torch.cuda.synchronize()

    # Benchmark BitBLAS
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        matmul(x, qweight, scales, zeros, output=output)
    torch.cuda.synchronize()
    bitblas_time = (time.time() - start) / n_iter * 1000

    # Benchmark FP16 matmul
    start = time.time()
    for _ in range(n_iter):
        y_fp16 = torch.nn.functional.linear(x, W_dequant)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / n_iter * 1000

    print(f"\nSpeed comparison (batch_size={batch_size}):")
    print(f"  BitBLAS INT4: {bitblas_time:.3f} ms")
    print(f"  FP16 matmul:  {fp16_time:.3f} ms")
    print(f"  Speedup: {fp16_time / bitblas_time:.2f}x")

    # Memory comparison
    fp16_mem = W_fp16.numel() * 2  # 2 bytes per FP16
    int4_mem = (out_features * in_features) // 2 + scales.numel() * 2 + zeros.numel() * 2  # INT4 + scales + zeros

    print(f"\nMemory comparison:")
    print(f"  FP16 weight: {fp16_mem / 1024 / 1024:.2f} MB")
    print(f"  INT4 packed: {int4_mem / 1024 / 1024:.2f} MB")
    print(f"  Reduction: {fp16_mem / int4_mem:.2f}x")

    if error.max() < 0.1:
        print("\n✅ SUCCESS!")
        return True
    else:
        print("\n❌ FAIL!")
        return False


if __name__ == "__main__":
    test_bitblas_basic()
    test_bitblas_larger()
