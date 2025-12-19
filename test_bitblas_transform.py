#!/usr/bin/env python3
"""Test BitBLAS with its own transform_weight function."""

import os
import sys
import torch
import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_with_transform_weight():
    """Test using BitBLAS transform_weight function."""
    print("=" * 60)
    print("Testing BitBLAS with transform_weight")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    batch_size = 4
    in_features = 256
    out_features = 512
    group_size = 128
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    print(f"Target: {target}")

    # Create random weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)

    # Expected output
    W_dequant = (W_int.half() * scale).to(device)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    expected = torch.nn.functional.linear(x, W_dequant)
    print(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")

    # Create BitBLAS kernel with int4
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
        group_size=group_size,
        with_bias=False,
        layout="nt",
    )

    try:
        matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)
        print(f"Matmul created successfully")
        print(f"  source_format: {matmul.source_format}")
        print(f"  bit: {matmul.bit}")
        print(f"  weight_transform: {matmul.weight_transform}")

        # Use BitBLAS's own transform_weight
        print("\nUsing BitBLAS transform_weight...")
        n_groups = in_features // group_size
        scales_full = scale.expand(-1, n_groups).contiguous().clone()

        # BitBLAS expects weights in signed format for int4
        # transform_weight will handle packing
        transformed = matmul.transform_weight(W_int.cuda(), scale=scales_full)

        if isinstance(transformed, list):
            qweight = transformed[0]
            print(f"Transformed returns list, qweight shape: {qweight.shape}")
        else:
            qweight = transformed
            print(f"Transformed returns tensor, shape: {qweight.shape}")

        # Forward
        output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
        matmul(x, qweight, scales_full, output=output)

        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        error = (output - expected).abs()
        print(f"Error - Max: {error.max():.4f}, Mean: {error.mean():.4f}")

        if error.max() < 1.0:
            print("✅ SUCCESS")
            return True
        else:
            print("❌ FAIL")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uint4_with_transform():
    """Test UINT4 with transform_weight."""
    print("\n" + "=" * 60)
    print("Testing UINT4 with transform_weight")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    batch_size = 4
    in_features = 256
    out_features = 512
    group_size = 128
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # Create random weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0

    # Quantize to signed INT4 [-8, 7] then to unsigned [0, 15]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)

    # Expected output
    W_dequant = (W_int.half() * scale).to(device)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    expected = torch.nn.functional.linear(x, W_dequant)
    print(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")

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
        print(f"Matmul created")
        print(f"  source_format: {matmul.source_format}")
        print(f"  weight_transform: {matmul.weight_transform}")

        n_groups = in_features // group_size
        scales_full = scale.expand(-1, n_groups).contiguous().clone()
        zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

        # Transform using BitBLAS
        # For uint4, pass unsigned values
        transformed = matmul.transform_weight(W_uint.cuda().to(torch.int8), scale=scales_full, zeros=zeros)

        if isinstance(transformed, list):
            qweight = transformed[0]
            if len(transformed) > 1:
                scales_t = transformed[1]
            if len(transformed) > 2:
                zeros_t = transformed[2]
            print(f"Transformed list length: {len(transformed)}")
        else:
            qweight = transformed

        print(f"qweight shape: {qweight.shape}")

        # Forward
        output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
        matmul(x, qweight, scales_full, zeros, output=output)

        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        error = (output - expected).abs()
        print(f"Error - Max: {error.max():.4f}, Mean: {error.mean():.4f}")

        if error.max() < 1.0:
            print("✅ SUCCESS")
            return True
        else:
            print("❌ FAIL")
            return False

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_with_transform_weight()
    test_uint4_with_transform()
