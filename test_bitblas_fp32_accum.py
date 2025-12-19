#!/usr/bin/env python3
"""Test BitBLAS with FP32 accumulator for better precision."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_fp32_accum():
    """Test BitBLAS with FP32 accumulator."""
    print("=" * 60)
    print("Test: BitBLAS INT4 with FP32 Accumulator")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target
    from openpi.models_pytorch.duquant_preprocess import fake_quantize_sym, compute_mse_scales

    batch_size = 1
    in_features = 2048
    out_features = 16384
    group_size = in_features  # Per-channel
    device = torch.device("cuda")

    target = auto_detect_nvidia_target()

    torch.manual_seed(42)
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    scale = compute_mse_scales(W_fp16, bits=4)
    W_fake = fake_quantize_sym(W_fp16, scale[:, None], bits=4, label="test")

    torch.manual_seed(123)
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    y_ref = torch.nn.functional.linear(x, W_fake)
    print(f"Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # Quantize for BitBLAS
    max_q, min_q = 7, -8
    W_int = torch.clamp(torch.round(W_fp16 / scale[:, None]), min_q, max_q).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)

    n_groups = 1
    scales_bitblas = scale[:, None].contiguous()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    # Test with FP32 accumulator
    print("\n--- FP32 Accumulator ---")
    matmul_config_fp32 = MatmulConfig(
        M=[1, 16, 32, 64],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="uint4",
        out_dtype="float16",
        accum_dtype="float32",  # FP32 accumulator
        with_scaling=True,
        with_zeros=True,
        zeros_mode="original",
        group_size=group_size,
        with_bias=False,
        layout="nt",
    )

    matmul_fp32 = Matmul(matmul_config_fp32, target=target, backend="tir", enable_tuning=False)

    W_uint_int8 = W_uint.to(torch.int8)
    qweight = matmul_fp32.transform_weight(W_uint_int8, scale=scales_bitblas, zeros=zeros)
    if isinstance(qweight, list):
        qweight = qweight[0]

    output_fp32 = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    matmul_fp32(x, qweight, scales_bitblas, zeros, output=output_fp32)

    print(f"BitBLAS FP32 accum output: [{output_fp32.min():.4f}, {output_fp32.max():.4f}]")

    error_fp32 = (output_fp32 - y_ref).abs()
    print(f"Error (FP32 accum) - Max: {error_fp32.max():.6f}, Mean: {error_fp32.mean():.6f}")

    # Test with FP16 accumulator for comparison
    print("\n--- FP16 Accumulator ---")
    matmul_config_fp16 = MatmulConfig(
        M=[1, 16, 32, 64],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="uint4",
        out_dtype="float16",
        accum_dtype="float16",  # FP16 accumulator
        with_scaling=True,
        with_zeros=True,
        zeros_mode="original",
        group_size=group_size,
        with_bias=False,
        layout="nt",
    )

    matmul_fp16 = Matmul(matmul_config_fp16, target=target, backend="tir", enable_tuning=False)

    qweight_fp16 = matmul_fp16.transform_weight(W_uint_int8, scale=scales_bitblas, zeros=zeros)
    if isinstance(qweight_fp16, list):
        qweight_fp16 = qweight_fp16[0]

    output_fp16 = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
    matmul_fp16(x, qweight_fp16, scales_bitblas, zeros, output=output_fp16)

    print(f"BitBLAS FP16 accum output: [{output_fp16.min():.4f}, {output_fp16.max():.4f}]")

    error_fp16 = (output_fp16 - y_ref).abs()
    print(f"Error (FP16 accum) - Max: {error_fp16.max():.6f}, Mean: {error_fp16.mean():.6f}")

    # Speed comparison
    print("\n--- Speed Comparison ---")
    import time

    for _ in range(5):
        matmul_fp32(x, qweight, scales_bitblas, zeros, output=output_fp32)
        matmul_fp16(x, qweight_fp16, scales_bitblas, zeros, output=output_fp16)
    torch.cuda.synchronize()

    n_iter = 100

    start = time.time()
    for _ in range(n_iter):
        matmul_fp32(x, qweight, scales_bitblas, zeros, output=output_fp32)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / n_iter * 1000

    start = time.time()
    for _ in range(n_iter):
        matmul_fp16(x, qweight_fp16, scales_bitblas, zeros, output=output_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / n_iter * 1000

    start = time.time()
    for _ in range(n_iter):
        _ = torch.nn.functional.linear(x, W_fake)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / n_iter * 1000

    print(f"  BitBLAS FP32 accum: {fp32_time:.3f} ms")
    print(f"  BitBLAS FP16 accum: {fp16_time:.3f} ms")
    print(f"  FP16 matmul (ref):  {ref_time:.3f} ms")

    print("\n--- Summary ---")
    print(f"  FP32 accum error: max={error_fp32.max():.4f}, mean={error_fp32.mean():.6f}")
    print(f"  FP16 accum error: max={error_fp16.max():.4f}, mean={error_fp16.mean():.6f}")

    if error_fp32.max() < error_fp16.max():
        print("  => FP32 accumulator has better precision")
    else:
        print("  => FP16 and FP32 similar precision")


if __name__ == "__main__":
    test_fp32_accum()
