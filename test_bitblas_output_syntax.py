#!/usr/bin/env python3
"""Test different ways to call BitBLAS kernel."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_bitblas_call_variants():
    """Test different ways to call BitBLAS."""
    print("=" * 60)
    print("Testing BitBLAS call syntax variants")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    in_features = 2048
    out_features = 2048
    group_size = 128
    batch_size = 4
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # Create weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device) * 0.02
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)

    # Expected
    W_dequant = (W_int.half() * scale)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    expected = torch.nn.functional.linear(x, W_dequant)
    print(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")

    # Create kernel
    matmul_config = MatmulConfig(
        M=[1, 4, 16, 32],  # Include batch_size=4
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

    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous().clone()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    # Transform weight
    transformed = matmul.transform_weight(W_uint.to(torch.int8), scale=scales, zeros=zeros)
    if isinstance(transformed, list):
        qweight = transformed[0]
    else:
        qweight = transformed

    print(f"qweight shape: {qweight.shape}")

    # ===== Test variant 1: output= keyword argument =====
    print("\n--- Variant 1: output= keyword argument ---")
    output1 = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    print(f"output1 pre-call range: [{output1.min():.4f}, {output1.max():.4f}]")

    matmul(x, qweight, scales, zeros, output=output1)

    print(f"output1 post-call range: [{output1.min():.4f}, {output1.max():.4f}]")
    error1 = (output1 - expected).abs()
    print(f"error1: max={error1.max():.4f}, mean={error1.mean():.4f}")

    # ===== Test variant 2: positional argument =====
    print("\n--- Variant 2: positional argument ---")
    output2 = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    print(f"output2 pre-call range: [{output2.min():.4f}, {output2.max():.4f}]")

    matmul(x, qweight, scales, zeros, output2)

    print(f"output2 post-call range: [{output2.min():.4f}, {output2.max():.4f}]")
    error2 = (output2 - expected).abs()
    print(f"error2: max={error2.max():.4f}, mean={error2.mean():.4f}")

    # ===== Test variant 3: args list =====
    print("\n--- Variant 3: args list ---")
    output3 = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    print(f"output3 pre-call range: [{output3.min():.4f}, {output3.max():.4f}]")

    args = [x, qweight, scales, zeros, output3]
    matmul(*args)

    print(f"output3 post-call range: [{output3.min():.4f}, {output3.max():.4f}]")
    error3 = (output3 - expected).abs()
    print(f"error3: max={error3.max():.4f}, mean={error3.mean():.4f}")

    # ===== Test variant 4: return value (if any) =====
    print("\n--- Variant 4: return value ---")
    output4 = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    result = matmul(x, qweight, scales, zeros, output4)
    print(f"result type: {type(result)}")
    if result is not None:
        print(f"result range: [{result.min():.4f}, {result.max():.4f}]")

    print(f"output4 range: [{output4.min():.4f}, {output4.max():.4f}]")


if __name__ == "__main__":
    test_bitblas_call_variants()
