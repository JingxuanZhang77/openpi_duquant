#!/usr/bin/env python3
"""Test the exact flow used in bitblas_layers.py."""

import os
import sys
import torch
import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_full_flow():
    """Test the full flow matching bitblas_layers.py."""
    print("=" * 60)
    print("Testing Full BitBLAS Flow")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        print(f"WARNING: No pack dir: {pack_dir}")
        return

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target
    from openpi.models_pytorch.duquant_preprocess import (
        load_pack,
        compute_mse_scales,
        apply_input_transform_optimized,
        apply_output_restore_optimized,
    )

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

    group_size = 128
    bits = 4
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # ========== Step 1: Create original weight ==========
    torch.manual_seed(42)
    W = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    print(f"\nOriginal W range: [{W.min():.4f}, {W.max():.4f}]")

    # ========== Step 2: Apply DuQuant transforms (like load_from_linear) ==========
    print("\n--- Applying DuQuant transforms ---")

    # Apply permutation
    if pack.perm is not None:
        perm_t = torch.from_numpy(pack.perm).long().to(device)
        W = W[:, perm_t]
        print(f"After perm: W range [{W.min():.4f}, {W.max():.4f}]")

    # Apply R_in rotation (per-block)
    if pack.R_in_blocks:
        for b in range(in_features // block_size):
            if b in pack.R_in_blocks:
                R = torch.from_numpy(pack.R_in_blocks[b]).to(device, W.dtype)
                start = b * block_size
                end = start + block_size
                W[:, start:end] = W[:, start:end] @ R
        print(f"After R_in: W range [{W.min():.4f}, {W.max():.4f}]")

    # Apply R_out rotation (per-block)
    if pack.R_out_blocks:
        for b in range(out_features // block_out_size):
            if b in pack.R_out_blocks:
                R = torch.from_numpy(pack.R_out_blocks[b]).to(device, W.dtype)
                start = b * block_out_size
                end = start + block_out_size
                W[start:end, :] = R @ W[start:end, :]  # NOTE: R @ W, not R.T @ W
        print(f"After R_out: W range [{W.min():.4f}, {W.max():.4f}]")

    # ========== Step 3: Compute MSE scales and quantize ==========
    print("\n--- Quantizing ---")

    with torch.no_grad():
        weight_scale = compute_mse_scales(W, bits)  # (out_features,)
    print(f"Scale range: [{weight_scale.min():.6f}, {weight_scale.max():.6f}]")

    # Quantize: q = clamp(round(W / scale), -8, 7)
    W_scaled = W / weight_scale[:, None]
    W_int = torch.clamp(torch.round(W_scaled), -8, 7).to(torch.int8)
    W_uint = (W_int + 8).to(torch.uint8)  # [-8,7] -> [0,15]
    print(f"W_int range: [{W_int.min()}, {W_int.max()}]")
    print(f"W_uint range: [{W_uint.min()}, {W_uint.max()}]")

    # ========== Step 4: Create BitBLAS kernel and transform weights ==========
    print("\n--- Creating BitBLAS kernel ---")

    matmul_config = MatmulConfig(
        M=[1, 4, 16, 32],  # Multiple batch sizes
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
    print(f"Matmul created: source_format={matmul.source_format}")

    # Prepare scales and zeros
    n_groups = in_features // group_size
    scales = weight_scale[:, None].expand(-1, n_groups).contiguous().clone()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)
    print(f"Scales shape: {scales.shape}, range: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"Zeros shape: {zeros.shape}, value: {zeros[0,0]}")

    # Transform weight
    W_uint_int8 = W_uint.to(torch.int8)  # BitBLAS expects int8 container
    transformed = matmul.transform_weight(W_uint_int8, scale=scales, zeros=zeros)

    if isinstance(transformed, list):
        qweight = transformed[0]
        print(f"Transformed list length: {len(transformed)}")
    else:
        qweight = transformed
    print(f"qweight shape: {qweight.shape}, dtype: {qweight.dtype}")

    # ========== Step 5: Test forward pass ==========
    print("\n--- Testing forward pass ---")

    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device=device)
    print(f"Input x range: [{x.min():.4f}, {x.max():.4f}]")

    # Apply input transform
    x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)
    print(f"x_t (after input transform) range: [{x_t.min():.4f}, {x_t.max():.4f}]")

    # BitBLAS forward
    output = torch.empty(4, out_features, dtype=torch.float16, device=device)
    matmul(x_t, qweight, scales, zeros, output=output)
    print(f"BitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")

    # Apply output restore
    y = apply_output_restore_optimized(output, pack, R_out_cache=None, block_out_size=block_out_size)
    print(f"Final output range: [{y.min():.4f}, {y.max():.4f}]")

    # ========== Step 6: Compare with reference ==========
    print("\n--- Reference comparison ---")

    # Reference: dequant + matmul
    W_dequant = (W_int.half() * weight_scale[:, None])
    expected = torch.nn.functional.linear(x_t, W_dequant)
    expected = apply_output_restore_optimized(expected, pack, R_out_cache=None, block_out_size=block_out_size)
    print(f"Reference output range: [{expected.min():.4f}, {expected.max():.4f}]")

    error = (y - expected).abs()
    print(f"Error - Max: {error.max():.4f}, Mean: {error.mean():.4f}")

    if error.max() < 1.0:
        print("\n✅ SUCCESS")
        return True
    else:
        print("\n❌ FAIL")
        return False


if __name__ == "__main__":
    test_full_flow()
