#!/usr/bin/env python3
"""Test BitBLAS with larger dimensions like the real model."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_large_matmul():
    """Test BitBLAS with larger dimensions."""
    print("=" * 60)
    print("Testing BitBLAS with 2048x2048 matrix")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    batch_size = 4
    in_features = 2048
    out_features = 2048
    group_size = 128
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    print(f"Target: {target}")
    print(f"Dimensions: {in_features} -> {out_features}")
    print(f"Group size: {group_size}")

    # Create random weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    print(f"W_fp16 range: [{W_fp16.min():.4f}, {W_fp16.max():.4f}]")

    # Per-channel scale (like DuQuant's MSE scales)
    # Use max-abs / 7 for simplicity
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0
    print(f"Scale range: [{scale.min():.6f}, {scale.max():.6f}]")

    # Quantize to signed INT4 [-8, 7] then to unsigned [0, 15]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)
    print(f"W_int range: [{W_int.min()}, {W_int.max()}]")
    print(f"W_uint range: [{W_uint.min()}, {W_uint.max()}]")

    # Expected output (dequantized matmul)
    W_dequant = (W_int.half() * scale).to(device)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    expected = torch.nn.functional.linear(x, W_dequant)
    print(f"\nInput x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Expected output range: [{expected.min():.4f}, {expected.max():.4f}]")

    # Create BitBLAS kernel with uint4
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
        print(f"  source_format: {matmul.source_format}")
        print(f"  weight_transform: {type(matmul.weight_transform)}")

        n_groups = in_features // group_size
        scales_full = scale.expand(-1, n_groups).contiguous().clone()
        zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

        print(f"\nScales shape: {scales_full.shape}")
        print(f"Zeros shape: {zeros.shape}")
        print(f"Zeros value: {zeros[0, 0]}")

        # Transform using BitBLAS
        transformed = matmul.transform_weight(W_uint.cuda().to(torch.int8), scale=scales_full, zeros=zeros)

        if isinstance(transformed, list):
            qweight = transformed[0]
            print(f"Transformed returns list of length {len(transformed)}")
        else:
            qweight = transformed

        print(f"qweight shape: {qweight.shape}")
        print(f"qweight dtype: {qweight.dtype}")

        # Forward
        output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
        matmul(x, qweight, scales_full, zeros, output=output)

        print(f"\nBitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")

        error = (output - expected).abs()
        print(f"Error - Max: {error.max():.4f}, Mean: {error.mean():.4f}")

        rel_error = error / (expected.abs() + 1e-6)
        print(f"Rel error - Max: {rel_error.max():.4f}, Mean: {rel_error.mean():.4f}")

        if error.max() < 1.0:
            print("\n✅ SUCCESS: BitBLAS output matches expected!")
            return True
        else:
            print("\n❌ FAIL: Large error!")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_large_matmul()
