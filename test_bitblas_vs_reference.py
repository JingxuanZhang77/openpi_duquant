#!/usr/bin/env python3
"""Test BitBLAS output against a true reference (manual dequant matmul)."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_vs_reference():
    """Test BitBLAS output against true reference."""
    print("=" * 60)
    print("Test: BitBLAS vs True Reference (manual dequant)")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"

    from openpi.models_pytorch.duquant_preprocess import (
        load_pack,
        apply_input_transform_optimized,
        apply_output_restore_optimized,
        compute_mse_scales,
    )
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

    # Find a pack file
    pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".npz")]
    layer_name = pack_files[0].replace(".npz", "")
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

    print(f"Layer: {layer_name}")
    print(f"Dimensions: {in_features} -> {out_features}")

    # Create original linear
    torch.manual_seed(42)
    original_linear = torch.nn.Linear(in_features, out_features, bias=False)
    original_linear = original_linear.cuda().half()

    # ===== True Reference Path =====
    print("\n--- True Reference (manual transform + dequant matmul) ---")

    W = original_linear.weight.clone()

    # Apply transforms
    if pack.perm is not None:
        perm_t = torch.from_numpy(pack.perm).long().cuda()
        W = W[:, perm_t]
    if pack.R_in_blocks:
        for b in range(in_features // block_size):
            if b in pack.R_in_blocks:
                R = torch.from_numpy(pack.R_in_blocks[b]).cuda().half()
                start = b * block_size
                end = start + block_size
                W[:, start:end] = W[:, start:end] @ R
    if pack.R_out_blocks:
        for b in range(out_features // block_out_size):
            if b in pack.R_out_blocks:
                R = torch.from_numpy(pack.R_out_blocks[b]).cuda().half()
                start = b * block_out_size
                end = start + block_out_size
                W[start:end, :] = R @ W[start:end, :]

    # Quantize
    scale_ref = compute_mse_scales(W, 4)
    W_int = torch.clamp(torch.round(W / scale_ref[:, None]), -8, 7).to(torch.int8)

    # Dequant (true reference)
    W_dequant_ref = (W_int.half() * scale_ref[:, None])

    # Input
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")

    # Input transform
    x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)

    # Matmul
    y_t_ref = torch.nn.functional.linear(x_t, W_dequant_ref)

    # Output restore
    y_ref = apply_output_restore_optimized(y_t_ref, pack, R_out_cache=None, block_out_size=block_out_size)
    print(f"Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # ===== BitBLAS Path =====
    print("\n--- BitBLAS (true INT4 kernel) ---")

    bitblas_layer = BitBLASQuantLinear(
        in_features=in_features,
        out_features=out_features,
        name=layer_name,
        bits=4,
        group_size=128,
        bias=False,
        enable_tuning=False,
        opt_M=[1, 4, 16, 32],
        duquant_packdir=pack_dir,
    )

    bitblas_layer.load_from_linear(original_linear, duquant_pack=pack)
    bitblas_layer = bitblas_layer.cuda()

    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        y_bitblas = bitblas_layer(x)

    print(f"BitBLAS output range: [{y_bitblas.min():.4f}, {y_bitblas.max():.4f}]")

    # Compare
    error = (y_bitblas - y_ref).abs()
    print(f"\nError - Max: {error.max():.6f}, Mean: {error.mean():.6f}")

    rel_error = error / (y_ref.abs() + 1e-6)
    print(f"Rel error - Max: {rel_error.max():.4f}, Mean: {rel_error.mean():.4f}")

    if error.max() < 1.0:
        print("\n✅ SUCCESS: BitBLAS output matches true reference!")
        return True
    else:
        print("\n❌ FAIL: Large error vs true reference!")
        return False


if __name__ == "__main__":
    test_vs_reference()
