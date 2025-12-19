#!/usr/bin/env python3
"""Test BitBLAS with small weight values like pretrained model."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_small_weights():
    """Test BitBLAS with small weight values."""
    print("=" * 60)
    print("Testing BitBLAS with small weight values")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target
    from openpi.models_pytorch.duquant_preprocess import compute_mse_scales

    batch_size = 4
    in_features = 2048
    out_features = 2048
    group_size = 128
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # Create small weights (like pretrained model)
    # Typical pretrained models have weights in range [-0.02, 0.02]
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device) * 0.02
    print(f"W_fp16 range: [{W_fp16.min():.6f}, {W_fp16.max():.6f}]")

    # Use MSE scales
    with torch.no_grad():
        scale = compute_mse_scales(W_fp16, bits=4)[:, None]
    print(f"Scale range: [{scale.min():.8f}, {scale.max():.8f}]")

    # Quantize
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)
    print(f"W_int range: [{W_int.min()}, {W_int.max()}]")
    print(f"W_uint range: [{W_uint.min()}, {W_uint.max()}]")

    # Expected
    W_dequant = (W_int.half() * scale)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    expected = torch.nn.functional.linear(x, W_dequant)
    print(f"\nInput x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Expected output range: [{expected.min():.4f}, {expected.max():.4f}]")

    # BitBLAS kernel
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
    print(f"\nMatmul created")

    n_groups = in_features // group_size
    scales_full = scale.expand(-1, n_groups).contiguous().clone()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    print(f"Scales shape: {scales_full.shape}")
    print(f"Scales sample: {scales_full[0, :4]}")

    # Transform
    transformed = matmul.transform_weight(W_uint.to(torch.int8), scale=scales_full, zeros=zeros)
    if isinstance(transformed, list):
        qweight = transformed[0]
    else:
        qweight = transformed
    print(f"qweight shape: {qweight.shape}")

    # Forward
    output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    matmul(x, qweight, scales_full, zeros, output=output)
    print(f"\nBitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")

    error = (output - expected).abs()
    print(f"Error - Max: {error.max():.6f}, Mean: {error.mean():.6f}")

    if error.max() < 0.1:
        print("\n✅ SUCCESS")
        return True
    else:
        print("\n❌ FAIL - checking if issue is scale related...")

        # Test with larger weights
        print("\n--- Test with larger weights (x100) ---")
        W_fp16_large = W_fp16 * 100
        scale_large = compute_mse_scales(W_fp16_large, bits=4)[:, None]
        print(f"Large scale range: [{scale_large.min():.6f}, {scale_large.max():.6f}]")

        W_int_large = torch.clamp(torch.round(W_fp16_large / scale_large), -8, 7).to(torch.int8)
        W_uint_large = (W_int_large + 8).to(torch.uint8)

        W_dequant_large = (W_int_large.half() * scale_large)
        expected_large = torch.nn.functional.linear(x, W_dequant_large)
        print(f"Expected large output range: [{expected_large.min():.4f}, {expected_large.max():.4f}]")

        scales_large = scale_large.expand(-1, n_groups).contiguous().clone()
        transformed_large = matmul.transform_weight(W_uint_large.to(torch.int8), scale=scales_large, zeros=zeros)
        if isinstance(transformed_large, list):
            qweight_large = transformed_large[0]
        else:
            qweight_large = transformed_large

        output_large = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
        matmul(x, qweight_large, scales_large, zeros, output=output_large)
        print(f"BitBLAS large output range: [{output_large.min():.4f}, {output_large.max():.4f}]")

        error_large = (output_large - expected_large).abs()
        print(f"Error large - Max: {error_large.max():.4f}, Mean: {error_large.mean():.4f}")

        return False


if __name__ == "__main__":
    test_small_weights()
