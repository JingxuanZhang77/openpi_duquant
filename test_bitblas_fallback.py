#!/usr/bin/env python3
"""Test BitBLAS INT4 storage with dequant fallback.

This script tests:
1. INT4 weight packing with general_compress
2. Dequantization accuracy
3. Forward pass with DuQuant transforms
"""

import os
import sys
import torch
import numpy as np

# Set environment for testing
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
os.environ["OPENPI_BITBLAS_FORCE_FALLBACK"] = "1"  # Force fallback mode
os.environ["OPENPI_DUQUANT_ROW_ROT"] = "restore"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_general_compress():
    """Test general_compress packing and unpacking with UNSIGNED values."""
    print("=" * 60)
    print("Test 1: general_compress Packing/Unpacking (Unsigned)")
    print("=" * 60)

    from bitblas.quantization.utils import general_compress

    # IMPORTANT: general_compress expects UNSIGNED values [0, 15]
    # Create test data: unsigned INT4 [0, 15]
    test_data = np.array([
        [0, 15, 7, 8, 1, 9, 3, 12],
        [4, 11, 2, 13, 5, 10, 6, 14],
    ], dtype=np.uint8)

    print(f"Input shape: {test_data.shape}")
    print(f"Input data:\n{test_data}")
    print(f"Input range: [{test_data.min()}, {test_data.max()}]")

    # Pack
    packed = general_compress(test_data, source_bits=4, storage_dtype=np.int8)
    print(f"\nPacked shape: {packed.shape}")
    print(f"Packed data (hex):")
    for i, row in enumerate(packed):
        hex_str = " ".join(f"{b & 0xFF:02X}" for b in row)
        print(f"  Row {i}: {hex_str}")

    # Unpack unsigned values
    packed_uint = packed.view(np.uint8)
    unpacked = np.zeros_like(test_data)
    for i in range(packed.shape[1]):
        unpacked[:, 2*i] = packed_uint[:, i] & 0x0F
        unpacked[:, 2*i + 1] = (packed_uint[:, i] >> 4) & 0x0F

    print(f"\nUnpacked data:\n{unpacked}")

    if np.array_equal(test_data, unpacked):
        print("✅ Pack/unpack roundtrip successful")
        return True
    else:
        print("❌ Pack/unpack mismatch!")
        print(f"Diff:\n{test_data.astype(np.int16) - unpacked.astype(np.int16)}")
        return False


def test_quantize_dequantize():
    """Test quantization and dequantization accuracy with unsigned storage."""
    print("\n" + "=" * 60)
    print("Test 2: Quantize/Dequantize Accuracy (Unsigned Storage)")
    print("=" * 60)

    from bitblas.quantization.utils import general_compress

    # Create random FP16 weights
    out_features, in_features = 64, 128
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16)

    print(f"Original W shape: {W_fp16.shape}")
    print(f"Original W range: [{W_fp16.min():.4f}, {W_fp16.max():.4f}]")

    # Per-channel scale (like DuQuant)
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0
    print(f"Scale range: [{scale.min():.6f}, {scale.max():.6f}]")

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)
    print(f"Quantized W (signed) range: [{W_int.min()}, {W_int.max()}]")

    # Convert to unsigned [0, 15] for general_compress
    W_uint = (W_int + 8).to(torch.uint8)
    print(f"Quantized W (unsigned) range: [{W_uint.min()}, {W_uint.max()}]")

    # Pack
    W_uint_np = W_uint.numpy()
    packed = general_compress(W_uint_np, source_bits=4, storage_dtype=np.int8)
    print(f"Packed shape: {packed.shape}")

    # Unpack unsigned values
    packed_uint = packed.view(np.uint8)
    packed_in = packed.shape[1]
    unpacked = np.zeros((out_features, in_features), dtype=np.uint8)
    for i in range(packed_in):
        unpacked[:, 2*i] = packed_uint[:, i] & 0x0F
        unpacked[:, 2*i + 1] = (packed_uint[:, i] >> 4) & 0x0F

    W_unpacked = torch.from_numpy(unpacked)

    # Verify pack/unpack
    if not torch.equal(W_uint, W_unpacked):
        print("❌ Pack/unpack mismatch!")
        return False
    print("✅ Pack/unpack verified")

    # Dequantize: W_fp16 = (W_uint - 8) * scale
    zeros = 8.0
    W_dequant = (W_unpacked.half() - zeros) * scale
    print(f"Dequantized W range: [{W_dequant.min():.4f}, {W_dequant.max():.4f}]")

    # Calculate error
    error = (W_fp16 - W_dequant).abs()
    print(f"Max absolute error: {error.max():.6f}")
    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Relative error: {(error / (W_fp16.abs() + 1e-6)).max():.4f}")

    # Forward pass comparison
    x = torch.randn(16, in_features, dtype=torch.float16)
    y_orig = torch.nn.functional.linear(x, W_fp16)
    y_quant = torch.nn.functional.linear(x, W_dequant)

    output_error = (y_orig - y_quant).abs()
    print(f"\nForward pass output error:")
    print(f"  Max: {output_error.max():.4f}")
    print(f"  Mean: {output_error.mean():.4f}")

    if output_error.max() < 1.0:
        print("✅ Quantization error acceptable")
        return True
    else:
        print("❌ Quantization error too large!")
        return False


def test_bitblas_layer_fallback():
    """Test BitBLASQuantLinear with fallback mode."""
    print("\n" + "=" * 60)
    print("Test 3: BitBLASQuantLinear Fallback Mode")
    print("=" * 60)

    # Check if DuQuant pack exists
    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
    if not os.path.exists(pack_dir):
        print(f"⚠️ Skipping (no pack dir: {pack_dir})")
        return None

    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear
    from openpi.models_pytorch.duquant_preprocess import load_pack

    # Find a pack file
    pack_files = [f for f in os.listdir(pack_dir) if f.endswith(".npz")]
    if not pack_files:
        print(f"⚠️ Skipping (no pack files in {pack_dir})")
        return None

    # Use first pack file
    layer_name = pack_files[0].replace(".npz", "")
    print(f"Testing layer: {layer_name}")

    # Load pack
    pack = load_pack(layer_name, pack_dir)
    if pack is None:
        print(f"⚠️ Failed to load pack for {layer_name}")
        return None

    # Get dimensions
    out_features = len(pack.weight_scale)
    # Infer in_features from R_in_blocks
    if pack.R_in_blocks:
        block_size = pack.meta.get("block_size", 16)
        n_blocks = len(pack.R_in_blocks)
        in_features = n_blocks * block_size
    else:
        in_features = 2048  # Default

    print(f"Dimensions: {in_features} -> {out_features}")

    # Create original linear
    original_linear = torch.nn.Linear(in_features, out_features, bias=False).cuda().half()

    # Create BitBLAS layer
    try:
        bitblas_layer = BitBLASQuantLinear(
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

        # Load weights
        bitblas_layer.load_from_linear(original_linear, duquant_pack=pack)
        bitblas_layer = bitblas_layer.cuda()

        print(f"✅ BitBLASQuantLinear created")
        print(f"   qweight shape: {bitblas_layer.qweight.shape}")
        print(f"   scales shape: {bitblas_layer.scales.shape}")

        # Memory savings
        orig_mem = in_features * out_features * 2  # FP16
        quant_mem = bitblas_layer.qweight.numel() + bitblas_layer.scales.numel() * 2
        print(f"   Memory: {orig_mem / 1024:.1f}KB -> {quant_mem / 1024:.1f}KB ({quant_mem/orig_mem*100:.1f}%)")

        # Test forward (fallback mode)
        x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
        output = bitblas_layer(x)

        print(f"\nForward pass (fallback):")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

        if output.isnan().any():
            print("❌ Output contains NaN!")
            return False
        if output.isinf().any():
            print("❌ Output contains Inf!")
            return False

        print("✅ Forward pass successful")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("BitBLAS INT4 Fallback Mode Test Suite")
    print("=" * 60)
    print(f"OPENPI_BITBLAS_FORCE_FALLBACK={os.environ.get('OPENPI_BITBLAS_FORCE_FALLBACK')}")
    print("")

    results = {}

    # Test 1: Pack/unpack
    results["pack_unpack"] = test_general_compress()

    # Test 2: Quantize/dequantize
    results["quant_dequant"] = test_quantize_dequantize()

    # Test 3: Layer fallback
    results["layer_fallback"] = test_bitblas_layer_fallback()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else ("⚠️ SKIP" if passed is None else "❌ FAIL")
        print(f"  {name}: {status}")

    all_passed = all(v is not False for v in results.values())
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
