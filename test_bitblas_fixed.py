#!/usr/bin/env python3
"""Test the fixed BitBLAS implementation (FP32 accum + per-channel scale)."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_bitblas_vs_duquant():
    """Test fixed BitBLAS layer vs DuQuant fake_quantize."""
    print("=" * 60)
    print("Test: Fixed BitBLAS (FP32 accum + per-channel) vs DuQuant")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        print(f"Pack dir not found: {pack_dir}")
        return None

    from openpi.models_pytorch.duquant_preprocess import (
        load_pack,
        apply_input_transform_optimized,
        apply_output_restore_optimized,
        fake_quantize_sym,
        compute_mse_scales,
    )
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

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

    device = torch.device("cuda")

    # Create random weight
    torch.manual_seed(42)
    W_orig = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # ========== DuQuant Reference Path ==========
    print("\n--- DuQuant fake_quantize Reference ---")

    W_t = W_orig.clone()

    if pack.perm is not None:
        perm = torch.from_numpy(pack.perm).long().to(device)
        W_t = W_t[:, perm]

    if pack.R_in_blocks:
        for b in range(in_features // block_size):
            if b in pack.R_in_blocks:
                R = torch.from_numpy(pack.R_in_blocks[b]).to(device, W_t.dtype)
                start = b * block_size
                end = start + block_size
                W_t[:, start:end] = W_t[:, start:end] @ R

    if pack.R_out_blocks:
        for b in range(out_features // block_out_size):
            if b in pack.R_out_blocks:
                R = torch.from_numpy(pack.R_out_blocks[b]).to(device, W_t.dtype)
                start = b * block_out_size
                end = start + block_out_size
                W_t[start:end, :] = R @ W_t[start:end, :]

    scale = compute_mse_scales(W_t, bits=4)
    W_t_fake = fake_quantize_sym(W_t, scale[:, None], bits=4, label="test")

    torch.manual_seed(123)
    x = torch.randn(1, in_features, dtype=torch.float16, device=device)

    x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)
    y_t_fake = torch.nn.functional.linear(x_t, W_t_fake)
    y_fake = apply_output_restore_optimized(y_t_fake, pack, R_out_cache=None, block_out_size=block_out_size)

    print(f"DuQuant output range: [{y_fake.min():.4f}, {y_fake.max():.4f}]")

    # ========== BitBLAS Path ==========
    print("\n--- BitBLAS (Fixed: FP32 accum + per-channel) ---")

    # Create Linear wrapper
    class FakeLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.bias = None
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]

    fake_linear = FakeLinear(W_orig)

    # Create BitBLAS layer
    bitblas_linear = BitBLASQuantLinear(
        in_features=in_features,
        out_features=out_features,
        name=layer_name,
        bits=4,
        group_size=128,  # Will be overridden to in_features internally
        bias=False,
        enable_tuning=False,
        opt_M=[1, 16, 32, 64],
        duquant_packdir=pack_dir,
    )

    print(f"BitBLAS effective_group_size: {bitblas_linear.effective_group_size}")

    bitblas_linear.load_from_linear(fake_linear, duquant_pack=pack)
    bitblas_linear = bitblas_linear.cuda()

    print(f"scales shape: {bitblas_linear.scales.shape}")
    print(f"zeros shape: {bitblas_linear.zeros.shape}")

    # Forward
    torch.manual_seed(123)
    x = torch.randn(1, in_features, dtype=torch.float16, device=device)

    with torch.no_grad():
        y_bitblas = bitblas_linear(x)

    print(f"BitBLAS output range: [{y_bitblas.min():.4f}, {y_bitblas.max():.4f}]")

    # Compare
    error = (y_bitblas - y_fake).abs()
    rel_error = error / (y_fake.abs() + 1e-6)

    print(f"\nError vs DuQuant:")
    print(f"  Absolute - Max: {error.max():.6f}, Mean: {error.mean():.6f}")
    print(f"  Relative - Max: {rel_error.max():.4f}, Mean: {rel_error.mean():.4f}")

    if error.max() < 0.1:
        print("\n✅ SUCCESS: BitBLAS matches DuQuant fake_quantize!")
        return True
    else:
        print("\n❌ FAIL: Error too large!")
        return False


def test_speed():
    """Benchmark speed."""
    print("\n" + "=" * 60)
    print("Speed Benchmark")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        return

    from openpi.models_pytorch.duquant_preprocess import load_pack
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

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

    torch.manual_seed(42)
    W_orig = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    class FakeLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)
            self.bias = None
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]

    fake_linear = FakeLinear(W_orig)

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

    x = torch.randn(1, in_features, dtype=torch.float16, device=device)

    import time

    # Warmup
    for _ in range(5):
        _ = bitblas_linear(x)
        _ = torch.nn.functional.linear(x, W_orig)
    torch.cuda.synchronize()

    n_iter = 100

    # BitBLAS
    start = time.time()
    for _ in range(n_iter):
        _ = bitblas_linear(x)
    torch.cuda.synchronize()
    bitblas_time = (time.time() - start) / n_iter * 1000

    # FP16 matmul
    start = time.time()
    for _ in range(n_iter):
        _ = torch.nn.functional.linear(x, W_orig)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / n_iter * 1000

    print(f"Dimensions: {in_features} -> {out_features}")
    print(f"BitBLAS INT4 (with transforms): {bitblas_time:.3f} ms")
    print(f"FP16 matmul (no transforms):    {fp16_time:.3f} ms")

    # Note: BitBLAS includes input/output transforms overhead
    print("\nNote: BitBLAS time includes input/output transform overhead")


if __name__ == "__main__":
    result = test_bitblas_vs_duquant()
    if result:
        test_speed()
