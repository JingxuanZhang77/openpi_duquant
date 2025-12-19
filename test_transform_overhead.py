#!/usr/bin/env python3
"""Analyze the transform overhead in BitBLAS + DuQuant."""

import os
import sys
import torch
import time

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def analyze_transform_overhead():
    """Break down the time spent in each step."""
    print("=" * 60)
    print("Transform Overhead Analysis")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        print(f"Pack dir not found: {pack_dir}")
        return

    from openpi.models_pytorch.duquant_preprocess import (
        load_pack,
        apply_input_transform_optimized,
        apply_output_restore_optimized,
    )
    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target

    # Load a pack
    pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".npz")]
    layer_name = pack_files[0].replace(".npz", "")
    pack = load_pack(layer_name, pack_dir)

    out_features = len(pack.weight_scale)
    block_size = pack.meta.get("block_size", 16)
    block_out_size = pack.meta.get("block_out_size", block_size)
    if pack.R_in_blocks:
        in_features = len(pack.R_in_blocks) * block_size
    else:
        in_features = 2048

    print(f"Layer: {layer_name}")
    print(f"Dimensions: {in_features} -> {out_features}")
    print(f"R_in blocks: {len(pack.R_in_blocks) if pack.R_in_blocks else 0}")
    print(f"R_out blocks: {len(pack.R_out_blocks) if pack.R_out_blocks else 0}")

    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # Create test data
    torch.manual_seed(42)
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    x = torch.randn(1, in_features, dtype=torch.float16, device=device)

    # Setup BitBLAS kernel
    from openpi.models_pytorch.duquant_preprocess import compute_mse_scales

    scale = compute_mse_scales(W_fp16, bits=4)
    W_int = torch.clamp(torch.round(W_fp16 / scale[:, None]), -8, 7).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)

    matmul_config = MatmulConfig(
        M=[1, 16, 32, 64],
        N=out_features,
        K=in_features,
        A_dtype="float16",
        W_dtype="uint4",
        out_dtype="float16",
        accum_dtype="float32",
        with_scaling=True,
        with_zeros=True,
        zeros_mode="original",
        group_size=in_features,
        with_bias=False,
        layout="nt",
    )

    matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)

    scales = scale[:, None].contiguous()
    zeros = torch.full((out_features, 1), 8.0, dtype=torch.float16, device=device)

    qweight = matmul.transform_weight(W_uint.to(torch.int8), scale=scales, zeros=zeros)
    if isinstance(qweight, list):
        qweight = qweight[0]

    # Warmup
    for _ in range(5):
        x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)
        output = torch.empty(1, out_features, dtype=torch.float16, device=device)
        matmul(x_t, qweight, scales, zeros, output=output)
        y = apply_output_restore_optimized(output, pack, R_out_cache=None, block_out_size=block_out_size)
    torch.cuda.synchronize()

    n_iter = 100

    # Benchmark: Input transform
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)
    torch.cuda.synchronize()
    input_transform_time = (time.time() - start) / n_iter * 1000

    # Benchmark: BitBLAS kernel only
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        matmul(x_t, qweight, scales, zeros, output=output)
    torch.cuda.synchronize()
    kernel_time = (time.time() - start) / n_iter * 1000

    # Benchmark: Output restore
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        y = apply_output_restore_optimized(output, pack, R_out_cache=None, block_out_size=block_out_size)
    torch.cuda.synchronize()
    output_restore_time = (time.time() - start) / n_iter * 1000

    # Benchmark: Pure FP16 matmul
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        _ = torch.nn.functional.linear(x, W_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / n_iter * 1000

    total_time = input_transform_time + kernel_time + output_restore_time

    print("\n--- Time Breakdown ---")
    print(f"Input transform:   {input_transform_time:.3f} ms ({input_transform_time/total_time*100:.1f}%)")
    print(f"BitBLAS kernel:    {kernel_time:.3f} ms ({kernel_time/total_time*100:.1f}%)")
    print(f"Output restore:    {output_restore_time:.3f} ms ({output_restore_time/total_time*100:.1f}%)")
    print(f"---")
    print(f"Total BitBLAS:     {total_time:.3f} ms")
    print(f"FP16 matmul:       {fp16_time:.3f} ms")
    print(f"Slowdown:          {total_time/fp16_time:.1f}x")

    print("\n--- Analysis ---")
    if input_transform_time + output_restore_time > kernel_time:
        print("Transform overhead dominates! The transforms need to be optimized.")
        print("Options:")
        print("  1. Fuse transforms into CUDA kernels")
        print("  2. Remove transforms by pre-transforming weights (if mathematically equivalent)")
        print("  3. Use PyTorch JIT/compile to optimize transform code")
    else:
        print("BitBLAS kernel is the bottleneck.")


if __name__ == "__main__":
    analyze_transform_overhead()
