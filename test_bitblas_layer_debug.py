#!/usr/bin/env python3
"""Debug the BitBLASQuantLinear class directly."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_layer_directly():
    """Test BitBLASQuantLinear directly."""
    print("=" * 60)
    print("Testing BitBLASQuantLinear directly")
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

    # Create original linear
    torch.manual_seed(42)
    original_linear = torch.nn.Linear(in_features, out_features, bias=False)
    original_linear = original_linear.cuda().half()
    print(f"Original weight range: [{original_linear.weight.min():.4f}, {original_linear.weight.max():.4f}]")

    # Create BitBLAS layer
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

    # Load weights
    bitblas_layer.load_from_linear(original_linear, duquant_pack=pack)
    bitblas_layer = bitblas_layer.cuda()

    print(f"BitBLAS qweight shape: {bitblas_layer.qweight.shape}")
    print(f"BitBLAS scales shape: {bitblas_layer.scales.shape}")
    print(f"BitBLAS scales range: [{bitblas_layer.scales.min():.6f}, {bitblas_layer.scales.max():.6f}]")
    print(f"BitBLAS zeros value: {bitblas_layer.zeros[0, 0]}")

    # Test forward
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
    print(f"\nInput x range: [{x.min():.4f}, {x.max():.4f}]")

    # Get output using BitBLAS layer's forward
    with torch.no_grad():
        y_bitblas = bitblas_layer(x)
    print(f"BitBLAS layer output range: [{y_bitblas.min():.4f}, {y_bitblas.max():.4f}]")

    # Manual test: apply input transform and call kernel directly
    print("\n--- Manual test ---")
    x_t = apply_input_transform_optimized(x, pack, perm_cache=None, R_in_cache=None, block_size=block_size)
    print(f"x_t range: [{x_t.min():.4f}, {x_t.max():.4f}]")

    # Call kernel directly
    output_direct = torch.empty(4, out_features, dtype=torch.float16, device="cuda")
    if bitblas_layer.bitblas_matmul is not None:
        try:
            args = [x_t, bitblas_layer.qweight, bitblas_layer.scales, bitblas_layer.zeros, output_direct]
            bitblas_layer.bitblas_matmul(*args)
            print(f"Direct kernel output range: [{output_direct.min():.4f}, {output_direct.max():.4f}]")
        except Exception as e:
            print(f"Kernel error: {e}")
    else:
        print("No BitBLAS kernel!")

    # Compare with reference
    print("\n--- Reference (fresh computation) ---")
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
    W_dequant = (W_int.half() * scale_ref[:, None])

    # Forward
    y_ref = torch.nn.functional.linear(x_t, W_dequant)
    y_ref = apply_output_restore_optimized(y_ref, pack, R_out_cache=None, block_out_size=block_out_size)
    print(f"Reference output range: [{y_ref.min():.4f}, {y_ref.max():.4f}]")

    # Error
    error = (y_bitblas - y_ref).abs()
    print(f"\nError - Max: {error.max():.4f}, Mean: {error.mean():.4f}")


if __name__ == "__main__":
    test_layer_directly()
