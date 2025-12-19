#!/usr/bin/env python3
"""Debug test BitBLAS vs DuQuant layer differences."""

import os
import sys
import torch
import numpy as np

# Set environment for testing
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
os.environ["OPENPI_DUQUANT_ROW_ROT"] = "restore"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_step_by_step():
    """Step by step comparison."""
    print("=" * 60)
    print("Step by Step Debug Test")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        print(f"WARNING: No pack dir: {pack_dir}")
        return

    from openpi.models_pytorch.duquant_preprocess import (
        load_pack,
        apply_input_transform_optimized,
        apply_output_restore_optimized,
        compute_mse_scales,
        fake_quantize_sym,
    )
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

    # Find a pack file
    pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".npz")]
    layer_name = pack_files[0].replace(".npz", "")
    print(f"Testing layer: {layer_name}")

    pack = load_pack(layer_name, pack_dir)

    # Get dimensions
    out_features = len(pack.weight_scale)
    block_size = pack.meta.get("block_size", 16)
    block_out_size = pack.meta.get("block_out_size", block_size)
    if pack.R_in_blocks:
        n_blocks = len(pack.R_in_blocks)
        in_features = n_blocks * block_size
    else:
        in_features = 2048

    print(f"Dimensions: {in_features} -> {out_features}")

    # Create original linear with fixed weights
    torch.manual_seed(42)
    W_orig = torch.randn(out_features, in_features, dtype=torch.float16, device="cuda")

    # ========== DuQuant reference path ==========
    print("\n--- DuQuant Reference Path ---")

    # Step 1: Transform weight (perm + R_in + R_out)
    W_t = W_orig.clone()

    # Apply permutation
    if pack.perm is not None:
        perm_t = torch.from_numpy(pack.perm).long().cuda()
        W_t = W_t.index_select(dim=1, index=perm_t)
        print(f"After perm: W_t range [{W_t.min():.4f}, {W_t.max():.4f}]")

    # Apply R_in
    if pack.R_in_blocks:
        for b in range(in_features // block_size):
            if b in pack.R_in_blocks:
                R = torch.from_numpy(pack.R_in_blocks[b]).cuda().half()
                start = b * block_size
                end = start + block_size
                W_t[:, start:end] = W_t[:, start:end] @ R
        print(f"After R_in: W_t range [{W_t.min():.4f}, {W_t.max():.4f}]")

    # Apply R_out
    if pack.R_out_blocks:
        for b in range(out_features // block_out_size):
            if b in pack.R_out_blocks:
                R = torch.from_numpy(pack.R_out_blocks[b]).cuda().half()
                start = b * block_out_size
                end = start + block_out_size
                W_t[start:end, :] = R @ W_t[start:end, :]
        print(f"After R_out: W_t range [{W_t.min():.4f}, {W_t.max():.4f}]")

    # Step 2: Compute scales and fake quantize
    w_scales = compute_mse_scales(W_t, bits=4)
    print(f"Scales range: [{w_scales.min():.6f}, {w_scales.max():.6f}]")

    W_t_quant = fake_quantize_sym(W_t, w_scales[:, None], bits=4, label="weight_ref")
    print(f"After quantize: W_t_quant range [{W_t_quant.min():.4f}, {W_t_quant.max():.4f}]")

    # Step 3: Input transform
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
    print(f"\nInput x range: [{x.min():.4f}, {x.max():.4f}]")

    x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)
    print(f"After input transform: x_t range [{x_t.min():.4f}, {x_t.max():.4f}]")

    # Step 4: Matmul
    y_t = torch.nn.functional.linear(x_t, W_t_quant, None)
    print(f"After matmul: y_t range [{y_t.min():.4f}, {y_t.max():.4f}]")

    # Step 5: Output restore
    y = apply_output_restore_optimized(y_t, pack, R_out_cache=None, block_out_size=block_out_size)
    print(f"DuQuant final output: y range [{y.min():.4f}, {y.max():.4f}]")

    # ========== BitBLAS path ==========
    print("\n--- BitBLAS Path ---")

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
        group_size=128,
        bias=False,
        enable_tuning=False,
        opt_M=[1, 16, 32],
        duquant_packdir=pack_dir,
    )

    bitblas_linear.load_from_linear(fake_linear, duquant_pack=pack)
    bitblas_linear = bitblas_linear.cuda()

    print(f"BitBLAS qweight shape: {bitblas_linear.qweight.shape}")
    print(f"BitBLAS scales range: [{bitblas_linear.scales.min():.6f}, {bitblas_linear.scales.max():.6f}]")

    # Step 3: Input transform (same as DuQuant)
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")

    # Check if BitBLAS forward applies input transform
    print("\nManual BitBLAS forward steps:")

    x_t_bitblas = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)
    print(f"After input transform: x_t_bitblas range [{x_t_bitblas.min():.4f}, {x_t_bitblas.max():.4f}]")

    # Test BitBLAS kernel directly
    output = torch.empty(4, out_features, dtype=torch.float16, device="cuda")

    if bitblas_linear.bitblas_matmul is not None:
        try:
            args = [x_t_bitblas, bitblas_linear.qweight, bitblas_linear.scales, bitblas_linear.zeros, output]
            bitblas_linear.bitblas_matmul(*args)
            print(f"BitBLAS kernel output: [{output.min():.4f}, {output.max():.4f}]")
        except Exception as e:
            print(f"BitBLAS kernel error: {e}")
    else:
        print("No BitBLAS kernel available")

    # Apply output restore
    y_bitblas = apply_output_restore_optimized(output, pack, R_out_cache=None, block_out_size=block_out_size)
    print(f"BitBLAS after restore: [{y_bitblas.min():.4f}, {y_bitblas.max():.4f}]")

    # Compare
    print("\n--- Comparison ---")
    error = (y - y_bitblas).abs()
    print(f"Error - Max: {error.max():.6f}, Mean: {error.mean():.6f}")

    # Test fallback path
    print("\n--- Test Fallback Dequant ---")
    W_dequant = bitblas_linear._dequantize_weight()
    print(f"Dequant W range: [{W_dequant.min():.4f}, {W_dequant.max():.4f}]")
    print(f"Reference W_t_quant range: [{W_t_quant.min():.4f}, {W_t_quant.max():.4f}]")

    w_error = (W_dequant - W_t_quant).abs()
    print(f"Weight error - Max: {w_error.max():.6f}, Mean: {w_error.mean():.6f}")

    # Test with fallback forward
    y_fallback = torch.nn.functional.linear(x_t_bitblas, W_dequant, None)
    y_fallback = apply_output_restore_optimized(y_fallback, pack, R_out_cache=None, block_out_size=block_out_size)
    print(f"Fallback output: [{y_fallback.min():.4f}, {y_fallback.max():.4f}]")

    fallback_error = (y - y_fallback).abs()
    print(f"Fallback error vs DuQuant - Max: {fallback_error.max():.6f}, Mean: {fallback_error.mean():.6f}")


if __name__ == "__main__":
    test_step_by_step()
