#!/usr/bin/env python3
"""Detailed speed test for BitBLAS."""

import os
import sys
import torch
import time

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_speed_detailed():
    """Detailed speed measurement."""
    print("=" * 60)
    print("Detailed Speed Test")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        print(f"Pack dir not found: {pack_dir}")
        return

    from openpi.models_pytorch.duquant_preprocess import load_pack
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

    # Load a pack
    pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".npz")]
    layer_name = pack_files[0].replace(".npz", "")
    pack = load_pack(layer_name, pack_dir)

    out_features = len(pack.weight_scale)
    block_size = pack.meta.get("block_size", 16)
    if pack.R_in_blocks:
        in_features = len(pack.R_in_blocks) * block_size
    else:
        in_features = 2048

    device = torch.device("cuda")

    print(f"Layer: {layer_name}")
    print(f"Dimensions: {in_features} -> {out_features}")

    # Create weight
    torch.manual_seed(42)
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    class FakeLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.bias = None
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]

    fake_linear = FakeLinear(W_fp16)

    # Create BitBLAS layer
    print("\nCreating BitBLAS layer...")
    bitblas_linear = BitBLASQuantLinear(
        in_features=in_features,
        out_features=out_features,
        name=layer_name,
        bits=4,
        group_size=128,
        bias=False,
        enable_tuning=False,
        opt_M=[1, 16, 32, 64],
        duquant_packdir=pack_dir,
    )
    bitblas_linear.load_from_linear(fake_linear, duquant_pack=pack)
    bitblas_linear = bitblas_linear.cuda()

    # Test different batch sizes
    for batch_size in [1, 4, 16, 32]:
        print(f"\n--- Batch size: {batch_size} ---")

        x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(10):
            _ = bitblas_linear(x)
            _ = torch.nn.functional.linear(x, W_fp16)
        torch.cuda.synchronize()

        n_iter = 100

        # BitBLAS with everything
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            _ = bitblas_linear(x)
        torch.cuda.synchronize()
        bitblas_time = (time.time() - start) / n_iter * 1000

        # Pure FP16 matmul
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_iter):
            _ = torch.nn.functional.linear(x, W_fp16)
        torch.cuda.synchronize()
        fp16_time = (time.time() - start) / n_iter * 1000

        speedup = fp16_time / bitblas_time

        print(f"  BitBLAS INT4 (full layer): {bitblas_time:.3f} ms")
        print(f"  FP16 matmul:               {fp16_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x", "(FASTER)" if speedup > 1 else "(SLOWER)")


if __name__ == "__main__":
    test_speed_detailed()
