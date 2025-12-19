#!/usr/bin/env python3
"""Diagnose BitBLAS kernel output issue.

This script compares:
1. BitBLAS kernel output (what we're using)
2. Dequant fallback output (what should be correct)
3. FP16 baseline (ground truth)

The goal is to find why 0% accuracy is happening.
"""

import os
import sys
import torch
import numpy as np

# Set environment for testing
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_bitblas_kernel_vs_dequant():
    """Compare BitBLAS kernel output with dequant fallback."""
    print("=" * 60)
    print("BitBLAS Kernel vs Dequant Fallback Comparison")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.quantization.utils import general_compress
    from bitblas.utils import auto_detect_nvidia_target

    # Test dimensions
    batch_size = 4
    in_features = 256
    out_features = 512
    group_size = 128
    bits = 4

    device = torch.device("cuda")
    target = auto_detect_nvidia_target()
    print(f"GPU Target: {target}")

    # Create random FP16 weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale (like DuQuant)
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0
    print(f"Scale range: [{scale.min():.6f}, {scale.max():.6f}]")

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    print(f"W_int range: [{W_int.min()}, {W_int.max()}]")

    # Convert to unsigned [0, 15]
    W_uint = (W_int + 8).to(torch.uint8)
    print(f"W_uint range: [{W_uint.min()}, {W_uint.max()}]")

    # Pack with general_compress
    W_uint_np = W_uint.cpu().numpy()
    qweight_np = general_compress(W_uint_np, source_bits=bits, storage_dtype=np.int8)
    qweight = torch.from_numpy(qweight_np).to(device).contiguous()
    print(f"qweight shape: {qweight.shape}")

    # Prepare scales
    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous().clone()
    print(f"scales shape: {scales.shape}")

    # Create input
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # Method 1: BitBLAS kernel (FIXED - with_zeros=True, W_dtype=uint4)
    print("\n" + "=" * 40)
    print("Method 1: BitBLAS Kernel (FIXED: W_dtype=uint4, with_zeros=True)")
    print("=" * 40)

    # Prepare zeros tensor (zeros=8 to convert unsigned [0,15] to signed [-8,7])
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    matmul_config = MatmulConfig(
        M=[batch_size],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="uint4",       # UNSIGNED INT4 [0, 15]
        out_dtype="float16",
        accum_dtype="float16",
        with_scaling=True,
        with_zeros=True,       # Use zeros for offset correction
        zeros_mode="original", # W = (W_uint - zeros) * scale
        group_size=group_size,
        with_bias=False,
        layout="nt",
    )

    try:
        matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)
        output_kernel = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)

        # FIXED: pass zeros as 4th argument
        matmul(x, qweight, scales, zeros, output_kernel)

        print(f"Kernel output range: [{output_kernel.min():.4f}, {output_kernel.max():.4f}]")
        print(f"Kernel output mean: {output_kernel.mean():.4f}")
    except Exception as e:
        print(f"BitBLAS kernel failed: {e}")
        output_kernel = None

    # Method 2: Dequant fallback
    print("\n" + "=" * 40)
    print("Method 2: Dequant Fallback")
    print("=" * 40)

    # Unpack unsigned INT4
    qweight_uint = qweight.cpu().numpy().view(np.uint8)
    packed_in = qweight.shape[1]
    W_unpacked = np.zeros((out_features, in_features), dtype=np.uint8)
    for i in range(packed_in):
        W_unpacked[:, 2*i] = qweight_uint[:, i] & 0x0F
        W_unpacked[:, 2*i + 1] = (qweight_uint[:, i] >> 4) & 0x0F

    W_unpacked_t = torch.from_numpy(W_unpacked).to(device)

    # Dequant: W_fp16 = (W_uint - 8) * scale
    W_dequant = torch.zeros((out_features, in_features), dtype=torch.float16, device=device)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        W_dequant[:, start:end] = (W_unpacked_t[:, start:end].half() - 8) * scales[:, g:g+1]

    output_dequant = torch.nn.functional.linear(x, W_dequant)
    print(f"Dequant output range: [{output_dequant.min():.4f}, {output_dequant.max():.4f}]")
    print(f"Dequant output mean: {output_dequant.mean():.4f}")

    # Method 3: FP16 baseline (no quantization)
    print("\n" + "=" * 40)
    print("Method 3: FP16 Baseline (ground truth)")
    print("=" * 40)

    output_fp16 = torch.nn.functional.linear(x, W_fp16)
    print(f"FP16 output range: [{output_fp16.min():.4f}, {output_fp16.max():.4f}]")
    print(f"FP16 output mean: {output_fp16.mean():.4f}")

    # Comparisons
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)

    # Dequant vs FP16 (should be close - quantization error only)
    error_dequant_fp16 = (output_dequant - output_fp16).abs()
    print(f"\nDequant vs FP16 (quantization error):")
    print(f"  Max abs error: {error_dequant_fp16.max():.4f}")
    print(f"  Mean abs error: {error_dequant_fp16.mean():.4f}")

    if output_kernel is not None:
        # Kernel vs Dequant (should be close if kernel is correct)
        error_kernel_dequant = (output_kernel - output_dequant).abs()
        print(f"\nBitBLAS Kernel vs Dequant:")
        print(f"  Max abs error: {error_kernel_dequant.max():.4f}")
        print(f"  Mean abs error: {error_kernel_dequant.mean():.4f}")

        # Kernel vs FP16
        error_kernel_fp16 = (output_kernel - output_fp16).abs()
        print(f"\nBitBLAS Kernel vs FP16:")
        print(f"  Max abs error: {error_kernel_fp16.max():.4f}")
        print(f"  Mean abs error: {error_kernel_fp16.mean():.4f}")

        # Check if kernel is way off
        if error_kernel_dequant.max() > 1.0:
            print("\n❌ PROBLEM DETECTED: BitBLAS kernel output differs significantly from dequant!")
            print("   This explains 0% accuracy - kernel is computing wrong values.")

            # Analyze the difference pattern
            print("\n   Analyzing error pattern...")
            print(f"   Output range ratio (kernel/dequant): {output_kernel.abs().mean():.4f} / {output_dequant.abs().mean():.4f} = {output_kernel.abs().mean() / output_dequant.abs().mean():.4f}")

            # Check if it's a zeros offset issue
            # If kernel treats unsigned [0,15] as signed [-8,7], output would be shifted
            W_wrong_sign = (W_unpacked_t.half() - 8).to(device)  # If kernel uses signed without offset
            output_wrong = torch.nn.functional.linear(x, W_wrong_sign)

            error_kernel_wrong = (output_kernel - output_wrong).abs()
            print(f"\n   Testing hypothesis: kernel uses signed INT4 without offset...")
            print(f"   Error if kernel output = (W_int - 8) * scale: {error_kernel_wrong.mean():.4f}")

        else:
            print("\n✅ BitBLAS kernel output matches dequant fallback")

    return output_kernel is not None


def test_signed_vs_unsigned():
    """Test how BitBLAS kernel interprets INT4 values."""
    print("\n" + "=" * 60)
    print("Test: Signed vs Unsigned INT4 Interpretation")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.quantization.utils import general_compress
    from bitblas.utils import auto_detect_nvidia_target

    # Simple test case
    batch_size = 1
    in_features = 8
    out_features = 4

    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # Create known weight values
    # We'll use scale = 1.0 to see raw quantized values
    W_int = torch.tensor([
        [-8, -7, -6, -5, -4, -3, -2, -1],  # Negative values
        [0, 1, 2, 3, 4, 5, 6, 7],          # Positive values
        [-8, 0, 7, -8, 0, 7, -8, 0],       # Mixed
        [7, 7, 7, 7, 7, 7, 7, 7],          # All max
    ], dtype=torch.int8)

    print(f"W_int (signed):\n{W_int}")

    # Convert to unsigned
    W_uint = (W_int + 8).to(torch.uint8)
    print(f"\nW_uint (unsigned, = W_int + 8):\n{W_uint}")

    # Pack
    W_uint_np = W_uint.numpy()
    qweight_np = general_compress(W_uint_np, source_bits=4, storage_dtype=np.int8)
    qweight = torch.from_numpy(qweight_np).to(device).contiguous()

    print(f"\nPacked qweight shape: {qweight.shape}")

    # Scale = 1.0 for simple math
    scales = torch.ones((out_features, 1), dtype=torch.float16, device=device)

    # Input = [1, 1, 1, 1, 1, 1, 1, 1] to sum all columns
    x = torch.ones((batch_size, in_features), dtype=torch.float16, device=device)

    # Expected outputs:
    # Row 0: -8-7-6-5-4-3-2-1 = -36
    # Row 1: 0+1+2+3+4+5+6+7 = 28
    # Row 2: -8+0+7-8+0+7-8+0 = -10
    # Row 3: 7*8 = 56
    print(f"\nExpected outputs (sum of signed INT4):")
    print(f"  Row 0: {W_int[0].sum().item()}")
    print(f"  Row 1: {W_int[1].sum().item()}")
    print(f"  Row 2: {W_int[2].sum().item()}")
    print(f"  Row 3: {W_int[3].sum().item()}")

    # What if kernel uses unsigned without offset?
    # Row 0: 0+1+2+3+4+5+6+7 = 28
    # Row 1: 8+9+10+11+12+13+14+15 = 92
    # Row 2: 0+8+15+0+8+15+0+8 = 54
    # Row 3: 15*8 = 120
    print(f"\nIf kernel uses unsigned [0,15] directly (wrong):")
    print(f"  Row 0: {W_uint[0].sum().item()}")
    print(f"  Row 1: {W_uint[1].sum().item()}")
    print(f"  Row 2: {W_uint[2].sum().item()}")
    print(f"  Row 3: {W_uint[3].sum().item()}")

    # Create kernel with group_size = 8 (whole row)
    matmul_config = MatmulConfig(
        M=[batch_size],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="int4",
        out_dtype="float16",
        accum_dtype="float16",
        with_scaling=True,
        with_zeros=False,
        group_size=8,
        with_bias=False,
        layout="nt",
    )

    try:
        matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)
        output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)

        matmul(x, qweight, scales, output)

        print(f"\nActual BitBLAS kernel output:")
        print(f"  {output.squeeze()}")

        # Analyze what the kernel is doing
        signed_expected = torch.tensor([[-36.0, 28.0, -10.0, 56.0]], dtype=torch.float16, device=device)
        unsigned_expected = torch.tensor([[28.0, 92.0, 54.0, 120.0]], dtype=torch.float16, device=device)

        error_signed = (output - signed_expected).abs().mean()
        error_unsigned = (output - unsigned_expected).abs().mean()

        print(f"\n  Error if kernel uses signed [-8,7]: {error_signed:.4f}")
        print(f"  Error if kernel uses unsigned [0,15]: {error_unsigned:.4f}")

        if error_signed < error_unsigned:
            print("\n  ✅ Kernel interprets INT4 as SIGNED [-8,7]")
            print("  But we're storing UNSIGNED [0,15] -> This is the bug!")
        else:
            print("\n  Kernel interprets INT4 as UNSIGNED [0,15]")

    except Exception as e:
        print(f"Kernel test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("BitBLAS Kernel Diagnostic Test")
    print("=" * 60)
    print("Purpose: Find why 0% accuracy with BitBLAS W4FP16")
    print()

    # Test 1: Compare kernel vs dequant vs FP16
    test_bitblas_kernel_vs_dequant()

    # Test 2: Understand how kernel interprets INT4
    test_signed_vs_unsigned()

    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
