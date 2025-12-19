#!/usr/bin/env python3
"""Compare BitBLAS true INT8 computation vs Fake W8A8 (FP16 simulation).

This test verifies that BitBLAS INT8 kernel produces results equivalent to
FP16 simulation with the same quantization strategy.

Quantization strategy (same for both):
- Weight: per-channel absmax, scale = max(|W[i,:]|) / 127
- Activation: per-tensor dynamic absmax, scale = max(|x|) / 127
"""

import os
import sys

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_w8a8_forward(x, W_fp16, weight_scale):
    """Fake W8A8: FP16 simulation with same quantization strategy as BitBLAS.

    Args:
        x: Input tensor (batch, in_features), FP16
        W_fp16: Original FP16 weight (out_features, in_features)
        weight_scale: Per-channel weight scale (out_features,)

    Returns:
        Output tensor (batch, out_features), FP16
    """
    # Step 1: Quantize activation (per-tensor absmax)
    act_absmax = x.abs().max()
    act_scale = act_absmax / 127.0
    x_q = (x / act_scale).round().clamp(-127, 127)  # Still FP16, but quantized values

    # Step 2: Quantize weight (per-channel absmax)
    W_q = (W_fp16 / weight_scale[:, None]).round().clamp(-127, 127)  # Still FP16

    # Step 3: FP16 matmul (simulating INT8 computation)
    # Note: This uses FP16 accumulator, while BitBLAS uses INT32
    y = F.linear(x_q, W_q)

    # Step 4: Dequantize
    y = y * (act_scale * weight_scale)

    return y


def fake_w8a8_forward_int32_accum(x, W_fp16, weight_scale):
    """Fake W8A8 with INT32 accumulator simulation.

    This more closely matches BitBLAS behavior by using float32 for accumulation.
    """
    # Step 1: Quantize activation
    act_absmax = x.abs().max()
    act_scale = act_absmax / 127.0
    x_q = (x / act_scale).round().clamp(-127, 127)

    # Step 2: Quantize weight
    W_q = (W_fp16 / weight_scale[:, None]).round().clamp(-127, 127)

    # Step 3: Matmul with FP32 accumulator (simulating INT32)
    y = F.linear(x_q.float(), W_q.float())

    # Step 4: Dequantize
    y = y * (act_scale.float() * weight_scale.float())

    return y.half()


def test_bitblas_vs_fake():
    """Test MSE between BitBLAS and Fake W8A8."""
    print("=" * 70)
    print("BitBLAS W8A8 vs Fake W8A8 MSE Comparison")
    print("=" * 70)

    from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear

    device = torch.device("cuda")

    # Test dimensions (typical sizes)
    test_cases = [
        (2048, 8192, "MLP up_proj (2048 -> 8192)"),
        (8192, 2048, "MLP down_proj (8192 -> 2048)"),
        (2048, 2048, "Attention proj (2048 -> 2048)"),
        (4096, 4096, "Large square (4096 -> 4096)"),
    ]

    batch_sizes = [1, 4, 16, 32]

    for in_features, out_features, desc in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test: {desc}")
        print(f"Dimensions: {in_features} -> {out_features}")
        print(f"{'=' * 70}")

        # Create reference FP16 linear
        torch.manual_seed(42)
        linear_fp16 = nn.Linear(in_features, out_features, bias=False).half().to(device)
        W_fp16 = linear_fp16.weight.data.clone()

        # Compute weight scale (per-channel absmax)
        weight_absmax = W_fp16.float().abs().max(dim=1)[0]
        weight_scale = (weight_absmax / 127.0).clamp(min=1e-8).half()

        # Create BitBLAS W8A8 layer
        bitblas_layer = BitBLASW8A8Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            name="test",
            enable_tuning=False,
            opt_M=[1, 4, 16, 32],
        )
        bitblas_layer.load_from_linear(linear_fp16)
        bitblas_layer = bitblas_layer.to(device)

        print(f"\n{'Batch':<8} {'MSE (FP16 accum)':<18} {'MSE (INT32 accum)':<18} {'Max Abs Diff':<15}")
        print("-" * 60)

        for batch_size in batch_sizes:
            # Generate random input
            torch.manual_seed(123 + batch_size)
            x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

            with torch.no_grad():
                # BitBLAS output (true INT8)
                y_bitblas = bitblas_layer(x)

                # Fake W8A8 with FP16 accumulator
                y_fake_fp16 = fake_w8a8_forward(x, W_fp16, weight_scale)

                # Fake W8A8 with INT32 accumulator simulation
                y_fake_int32 = fake_w8a8_forward_int32_accum(x, W_fp16, weight_scale)

            # Compute MSE
            mse_fp16 = ((y_bitblas - y_fake_fp16) ** 2).mean().item()
            mse_int32 = ((y_bitblas - y_fake_int32) ** 2).mean().item()
            max_diff = (y_bitblas - y_fake_int32).abs().max().item()

            print(f"{batch_size:<8} {mse_fp16:<18.2e} {mse_int32:<18.2e} {max_diff:<15.6f}")

        # Detailed analysis for batch=16
        print(f"\nDetailed analysis (batch=16):")
        torch.manual_seed(123 + 16)
        x = torch.randn(16, in_features, dtype=torch.float16, device=device)

        with torch.no_grad():
            y_bitblas = bitblas_layer(x)
            y_fake_int32 = fake_w8a8_forward_int32_accum(x, W_fp16, weight_scale)
            y_fp16_baseline = linear_fp16(x)  # FP16 baseline (no quantization)

        # Compare to FP16 baseline
        mse_bitblas_vs_fp16 = ((y_bitblas - y_fp16_baseline) ** 2).mean().item()
        mse_fake_vs_fp16 = ((y_fake_int32 - y_fp16_baseline) ** 2).mean().item()

        print(f"  BitBLAS vs FP16 baseline MSE: {mse_bitblas_vs_fp16:.2e}")
        print(f"  Fake W8A8 vs FP16 baseline MSE: {mse_fake_vs_fp16:.2e}")
        print(f"  BitBLAS vs Fake W8A8 MSE: {mse_int32:.2e}")

        # Relative error
        rel_error = (y_bitblas - y_fake_int32).abs() / (y_fake_int32.abs() + 1e-8)
        print(f"  Relative error (mean): {rel_error.mean().item():.2e}")
        print(f"  Relative error (max): {rel_error.max().item():.2e}")

    print(f"\n{'=' * 70}")
    print("Summary:")
    print("- MSE between BitBLAS and Fake W8A8 (INT32 accum) should be ~0")
    print("- Any difference is due to:")
    print("  1. BitBLAS weight transform (packing)")
    print("  2. Numerical precision differences")
    print("=" * 70)


if __name__ == "__main__":
    test_bitblas_vs_fake()
