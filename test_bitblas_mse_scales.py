#!/usr/bin/env python3
"""Test BitBLAS with MSE scales like DuQuant."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_with_mse_scales():
    """Test BitBLAS with MSE scales."""
    print("=" * 60)
    print("Testing BitBLAS with MSE scales")
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

    # Create random weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    print(f"W_fp16 range: [{W_fp16.min():.4f}, {W_fp16.max():.4f}]")

    # Use MSE scales like DuQuant
    with torch.no_grad():
        scale = compute_mse_scales(W_fp16, bits=4)  # (out_features,)
    scale = scale[:, None]  # (out_features, 1)
    print(f"MSE Scale range: [{scale.min():.6f}, {scale.max():.6f}]")

    # Compare with simple max-abs scale
    scale_simple = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0
    print(f"Simple Scale range: [{scale_simple.min():.6f}, {scale_simple.max():.6f}]")

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)
    print(f"W_int range: [{W_int.min()}, {W_int.max()}]")
    print(f"W_uint range: [{W_uint.min()}, {W_uint.max()}]")

    # Expected output (dequantized matmul)
    W_dequant = (W_int.half() * scale).to(device)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    expected = torch.nn.functional.linear(x, W_dequant)
    print(f"\nExpected output range: [{expected.min():.4f}, {expected.max():.4f}]")

    # Create BitBLAS kernel
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

    try:
        matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)
        print(f"\nMatmul created")

        n_groups = in_features // group_size
        scales_full = scale.expand(-1, n_groups).contiguous().clone()
        zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

        print(f"Scales shape: {scales_full.shape}")

        # Transform using BitBLAS
        transformed = matmul.transform_weight(W_uint.cuda().to(torch.int8), scale=scales_full, zeros=zeros)

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
        print(f"Error - Max: {error.max():.4f}, Mean: {error.mean():.4f}")

        if error.max() < 1.0:
            print("\n✅ SUCCESS")
            return True
        else:
            print("\n❌ FAIL")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_with_mse_scales()
