#!/usr/bin/env python3
"""Test BitBLAS INT4 kernel implementation.

This script tests:
1. BitBLAS kernel creation with MatmulWeightOnlyDequantize
2. Weight packing with general_compress
3. Forward pass with DuQuant transforms
4. Comparison with dequant fallback
"""

import os
import sys
import torch
import numpy as np

# Set environment for testing
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
os.environ["OPENPI_DUQUANT_ROW_ROT"] = "restore"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_bitblas_imports():
    """Test BitBLAS imports."""
    print("=" * 60)
    print("Test 1: BitBLAS Imports")
    print("=" * 60)

    try:
        from bitblas.ops.matmul_dequantize import (
            MatmulWeightOnlyDequantize,
            MatmulWeightOnlyDequantizeConfig,
        )
        from bitblas.quantization.utils import general_compress, interleave_weight
        from bitblas.utils import auto_detect_nvidia_target
        print("✅ All BitBLAS imports successful")

        target = auto_detect_nvidia_target()
        print(f"   Detected GPU target: {target}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_general_compress():
    """Test general_compress packing."""
    print("\n" + "=" * 60)
    print("Test 2: general_compress Packing")
    print("=" * 60)

    from bitblas.quantization.utils import general_compress

    # Create test data: unsigned INT4 [0, 15]
    test_data = np.array([
        [0, 15, 7, 8, 1, 14, 3, 12],  # Row 0
        [4, 11, 2, 13, 5, 10, 6, 9],  # Row 1
    ], dtype=np.int8)

    print(f"Input shape: {test_data.shape}")
    print(f"Input data:\n{test_data}")

    # Pack
    packed = general_compress(test_data, source_bits=4, storage_dtype=np.int8)
    print(f"\nPacked shape: {packed.shape}")
    print(f"Packed data (hex):")
    for i, row in enumerate(packed):
        hex_str = " ".join(f"{b & 0xFF:02X}" for b in row)
        print(f"  Row {i}: {hex_str}")

    # Verify: unpack and compare
    unpacked = np.zeros_like(test_data)
    for i in range(packed.shape[1]):
        unpacked[:, 2*i] = packed[:, i] & 0x0F
        unpacked[:, 2*i + 1] = (packed[:, i] >> 4) & 0x0F

    print(f"\nUnpacked data:\n{unpacked}")

    if np.array_equal(test_data, unpacked):
        print("✅ Pack/unpack roundtrip successful")
        return True
    else:
        print("❌ Pack/unpack mismatch!")
        return False


def test_bitblas_kernel_creation():
    """Test BitBLAS kernel creation."""
    print("\n" + "=" * 60)
    print("Test 3: BitBLAS Kernel Creation")
    print("=" * 60)

    from bitblas.ops.matmul_dequantize import (
        MatmulWeightOnlyDequantize,
        MatmulWeightOnlyDequantizeConfig,
    )
    from bitblas.utils import auto_detect_nvidia_target

    # Test dimensions (small for quick test)
    in_features = 256
    out_features = 512
    group_size = 128

    target = auto_detect_nvidia_target()

    config = MatmulWeightOnlyDequantizeConfig(
        M=[1, 16, 32],
        N=out_features,
        K=in_features,
        in_dtype="float16",
        out_dtype="float16",
        accum_dtype="float16",
        bit=4,
        storage_dtype="int8",
        source_format="uint",
        with_scaling=True,
        with_zeros=True,
        group_size=group_size,
        fast_decoding=False,
        with_bias=False,
        layout="nt",
        zeros_mode="original",
    )

    print(f"Creating kernel for: {in_features} -> {out_features}")
    print(f"Target: {target}")

    try:
        matmul = MatmulWeightOnlyDequantize(config, target=target)
        print("✅ Kernel created successfully")
        return matmul, in_features, out_features, group_size
    except Exception as e:
        print(f"❌ Kernel creation failed: {e}")
        return None, None, None, None


def test_forward_pass(matmul, in_features, out_features, group_size):
    """Test forward pass with BitBLAS kernel."""
    print("\n" + "=" * 60)
    print("Test 4: Forward Pass")
    print("=" * 60)

    if matmul is None:
        print("⚠️ Skipping (no kernel)")
        return False

    from bitblas.quantization.utils import general_compress

    device = torch.device("cuda")

    # Create random FP16 weights
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Quantize to signed INT4 [-8, 7]
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0  # Per-channel scale
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)

    # Convert to unsigned [0, 15]
    W_uint = (W_int + 8).to(torch.int8)

    # Pack with general_compress
    W_uint_np = W_uint.cpu().numpy()
    qweight_np = general_compress(W_uint_np, source_bits=4, storage_dtype=np.int8)
    qweight = torch.from_numpy(qweight_np).to(device).contiguous()

    # Prepare scales and zeros
    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    print(f"qweight shape: {qweight.shape}")
    print(f"scales shape: {scales.shape}")
    print(f"zeros shape: {zeros.shape}")

    # Create input
    batch_size = 16
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # Prepare output
    output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)

    # Run BitBLAS kernel
    try:
        args = [x, qweight, scales, zeros, output]
        matmul(*args)
        print(f"BitBLAS output shape: {output.shape}")
        print(f"BitBLAS output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"BitBLAS output mean: {output.mean():.4f}")

        # Compare with FP16 baseline
        W_dequant = (W_uint.half() - 8) * scale  # Dequantize
        expected = torch.nn.functional.linear(x, W_dequant)

        # Calculate error
        abs_error = (output - expected).abs()
        rel_error = abs_error / (expected.abs() + 1e-6)

        print(f"\nComparison with FP16 dequant:")
        print(f"  Expected range: [{expected.min():.4f}, {expected.max():.4f}]")
        print(f"  Max abs error: {abs_error.max():.6f}")
        print(f"  Mean abs error: {abs_error.mean():.6f}")
        print(f"  Max rel error: {rel_error.max():.4f}")

        if abs_error.max() < 1.0:  # Reasonable threshold
            print("✅ Forward pass successful")
            return True
        else:
            print("❌ Forward pass has large errors")
            return False

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bitblas_quant_linear():
    """Test BitBLASQuantLinear layer."""
    print("\n" + "=" * 60)
    print("Test 5: BitBLASQuantLinear Layer")
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

    # Load pack to get dimensions
    pack = load_pack(layer_name, pack_dir)
    if pack is None:
        print(f"⚠️ Failed to load pack for {layer_name}")
        return None

    # Get dimensions from pack metadata
    meta = pack.meta
    out_features = len(pack.weight_scale)
    in_features = meta.get("in_features", 2048)  # Default guess

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

        print(f"✅ BitBLASQuantLinear created successfully")
        print(f"   qweight shape: {bitblas_layer.qweight.shape}")
        print(f"   scales shape: {bitblas_layer.scales.shape}")
        print(f"   zeros shape: {bitblas_layer.zeros.shape}")

        # Test forward
        x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")
        output = bitblas_layer(x)

        print(f"\nForward pass:")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

        return True

    except Exception as e:
        print(f"❌ BitBLASQuantLinear failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("BitBLAS INT4 Kernel Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Imports
    results["imports"] = test_bitblas_imports()

    # Test 2: general_compress
    results["compress"] = test_general_compress()

    # Test 3: Kernel creation
    matmul, in_features, out_features, group_size = test_bitblas_kernel_creation()
    results["kernel"] = matmul is not None

    # Test 4: Forward pass
    results["forward"] = test_forward_pass(matmul, in_features, out_features, group_size)

    # Test 5: BitBLASQuantLinear
    results["layer"] = test_bitblas_quant_linear()

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
