#!/usr/bin/env python3
"""Debug the forward() function step by step."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_forward_step_by_step():
    """Test forward() step by step."""
    print("=" * 60)
    print("Debug forward() step by step")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"

    from openpi.models_pytorch.duquant_preprocess import (
        load_pack,
        apply_input_transform_optimized,
        apply_output_restore_optimized,
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

    bitblas_layer.load_from_linear(original_linear, duquant_pack=pack)
    bitblas_layer = bitblas_layer.cuda()

    print(f"\nBitBLAS layer created:")
    print(f"  qweight shape: {bitblas_layer.qweight.shape}")
    print(f"  qweight device: {bitblas_layer.qweight.device}")
    print(f"  bitblas_matmul: {bitblas_layer.bitblas_matmul}")
    print(f"  duquant_pack: {bitblas_layer.duquant_pack is not None}")

    # Create input
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
    print(f"\nInput x:")
    print(f"  shape: {x.shape}")
    print(f"  device: {x.device}")
    print(f"  range: [{x.min():.4f}, {x.max():.4f}]")

    # ===== Manual forward mimicking bitblas_layer.forward() =====
    print("\n--- Manual forward() mimicking ---")

    # Step 1: Input transform
    x_transformed = apply_input_transform_optimized(
        x, bitblas_layer.duquant_pack,
        perm_cache=None,
        R_in_cache=None,
        block_size=bitblas_layer.duquant_pack.meta.get("block_size", 16)
    )
    print(f"\nStep 1 - After input transform:")
    print(f"  x_transformed range: [{x_transformed.min():.4f}, {x_transformed.max():.4f}]")

    # Step 2: Matmul
    print(f"\nStep 2 - Calling BitBLAS kernel:")
    print(f"  x_transformed device: {x_transformed.device}")
    print(f"  qweight device: {bitblas_layer.qweight.device}")
    print(f"  scales device: {bitblas_layer.scales.device}")
    print(f"  zeros device: {bitblas_layer.zeros.device}")

    output_shape = x_transformed.shape[:-1] + (bitblas_layer.out_features,)
    output = torch.empty(output_shape, dtype=x_transformed.dtype, device=x_transformed.device)
    print(f"  output pre-fill range: [{output.min():.4f}, {output.max():.4f}]")

    # Call kernel
    try:
        args = [x_transformed, bitblas_layer.qweight, bitblas_layer.scales, bitblas_layer.zeros, output]
        bitblas_layer.bitblas_matmul(*args)
        print(f"  BitBLAS kernel SUCCESS")
        print(f"  output post-kernel range: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"  BitBLAS kernel FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Step 3: Output restore
    y = apply_output_restore_optimized(
        output, bitblas_layer.duquant_pack,
        R_out_cache=None,
        block_out_size=bitblas_layer.duquant_pack.meta.get("block_out_size", 16)
    )
    print(f"\nStep 3 - After output restore:")
    print(f"  final output range: [{y.min():.4f}, {y.max():.4f}]")

    # ===== Compare with layer.forward() =====
    print("\n--- Calling bitblas_layer.forward() ---")
    torch.manual_seed(123)
    x2 = torch.randn(4, in_features, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        y_layer = bitblas_layer(x2)

    print(f"layer.forward() output range: [{y_layer.min():.4f}, {y_layer.max():.4f}]")

    # Compare
    error = (y - y_layer).abs()
    print(f"\nError between manual and layer.forward(): max={error.max():.6f}")


if __name__ == "__main__":
    test_forward_step_by_step()
