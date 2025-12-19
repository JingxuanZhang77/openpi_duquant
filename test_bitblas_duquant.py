#!/usr/bin/env python3
"""Test BitBLAS layer with DuQuant transforms.

This test verifies that BitBLASQuantLinear produces the same output as
DuQuantLinear when using the same DuQuant pack and transforms.
"""

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


def test_bitblas_vs_duquant():
    """Compare BitBLASQuantLinear vs DuQuantLinear output."""
    print("=" * 60)
    print("Test: BitBLAS vs DuQuant Layer Comparison")
    print("=" * 60)

    # Check if DuQuant pack exists
    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        print(f"WARNING: No pack dir: {pack_dir}")
        return None

    from openpi.models_pytorch.duquant_preprocess import load_pack
    from openpi.models_pytorch.duquant_layers import DuQuantLinear, DuQuantConfig
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

    # Find a pack file
    pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".npz")]
    if not pack_files:
        print(f"WARNING: No pack files in {pack_dir}")
        return None

    # Use first pack file
    layer_name = pack_files[0].replace(".npz", "")
    print(f"Testing layer: {layer_name}")

    # Load pack
    pack = load_pack(layer_name, pack_dir)
    if pack is None:
        print(f"WARNING: Failed to load pack for {layer_name}")
        return None

    # Get dimensions from pack
    out_features = len(pack.weight_scale)
    block_size = pack.meta.get("block_size", 16)
    if pack.R_in_blocks:
        n_blocks = len(pack.R_in_blocks)
        in_features = n_blocks * block_size
    else:
        in_features = 2048

    print(f"Dimensions: {in_features} -> {out_features}")
    print(f"Block size: {block_size}")
    print(f"R_in blocks: {len(pack.R_in_blocks) if pack.R_in_blocks else 0}")
    print(f"R_out blocks: {len(pack.R_out_blocks) if pack.R_out_blocks else 0}")
    print(f"Perm: {'Yes' if pack.perm is not None else 'No'}")

    # Create original linear with fixed weights
    torch.manual_seed(42)
    original_linear = torch.nn.Linear(in_features, out_features, bias=False)
    original_linear = original_linear.cuda().half()

    # Create DuQuant config
    duquant_cfg = DuQuantConfig(
        weight_bits=4,
        act_bits=0,  # No activation quantization for this test
        block_size=block_size,
        pack_dir=pack_dir,
        row_rot_mode="restore",
    )

    # Create DuQuantLinear
    duquant_linear = DuQuantLinear(
        original_linear,
        name=layer_name,
        cfg=duquant_cfg,
    ).cuda()

    print("\nDuQuantLinear created:")
    print(f"  Weight shape: {duquant_linear._weight.shape}")
    print(f"  Weight range: [{duquant_linear._weight.min():.4f}, {duquant_linear._weight.max():.4f}]")

    # Create BitBLAS layer
    bitblas_linear = BitBLASQuantLinear(
        in_features=in_features,
        out_features=out_features,
        name=layer_name,
        bits=4,
        group_size=128,
        bias=False,
        enable_tuning=False,
        opt_M=[1, 4, 16, 32],  # Include batch size 4 for this test
        duquant_packdir=pack_dir,
    )

    # Load weights into BitBLAS layer
    bitblas_linear.load_from_linear(original_linear, duquant_pack=pack)
    bitblas_linear = bitblas_linear.cuda()

    print("\nBitBLASQuantLinear created:")
    print(f"  qweight shape: {bitblas_linear.qweight.shape}")
    print(f"  scales shape: {bitblas_linear.scales.shape}")
    print(f"  zeros shape: {bitblas_linear.zeros.shape}")

    # Test forward pass
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")

    print(f"\nInput x shape: {x.shape}")
    print(f"Input x range: [{x.min():.4f}, {x.max():.4f}]")

    # DuQuant forward
    with torch.no_grad():
        y_duquant = duquant_linear(x)

    print(f"\nDuQuant output shape: {y_duquant.shape}")
    print(f"DuQuant output range: [{y_duquant.min():.4f}, {y_duquant.max():.4f}]")
    print(f"DuQuant output mean: {y_duquant.mean():.4f}")

    # BitBLAS forward
    with torch.no_grad():
        y_bitblas = bitblas_linear(x)

    print(f"\nBitBLAS output shape: {y_bitblas.shape}")
    print(f"BitBLAS output range: [{y_bitblas.min():.4f}, {y_bitblas.max():.4f}]")
    print(f"BitBLAS output mean: {y_bitblas.mean():.4f}")

    # Compare
    error = (y_duquant - y_bitblas).abs()
    print(f"\nError - Max: {error.max():.6f}, Mean: {error.mean():.6f}")

    # Relative error
    rel_error = error / (y_duquant.abs() + 1e-6)
    print(f"Rel error - Max: {rel_error.max():.4f}, Mean: {rel_error.mean():.4f}")

    # Check if outputs are close
    if error.max() < 1.0:
        print("\nSUCCESS: BitBLAS output matches DuQuant!")
        return True
    else:
        print("\nFAIL: Large error between BitBLAS and DuQuant!")

        # Debug: compare intermediate values
        print("\nDEBUG: Checking intermediate values...")

        # Check scales
        duquant_linear._maybe_update_weight_cache()
        print(f"DuQuant scales range: [{duquant_linear._w_scales.min():.6f}, {duquant_linear._w_scales.max():.6f}]")
        print(f"BitBLAS scales range: [{bitblas_linear.scales.min():.6f}, {bitblas_linear.scales.max():.6f}]")

        return False


if __name__ == "__main__":
    result = test_bitblas_vs_duquant()
    if result is None:
        print("\nTest skipped (no pack files)")
        sys.exit(0)
    elif result:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
