#!/usr/bin/env python3
"""Test different BitBLAS configurations to find the correct one."""

import os
import sys
import torch
import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_variant(name, W_dtype, with_zeros, zeros_mode, store_unsigned):
    """Test a specific configuration variant."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  W_dtype={W_dtype}, with_zeros={with_zeros}, zeros_mode={zeros_mode}")
    print(f"  store_unsigned={store_unsigned}")
    print("=" * 60)

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.quantization.utils import general_compress
    from bitblas.utils import auto_detect_nvidia_target

    batch_size = 4
    in_features = 256
    out_features = 512
    group_size = 128
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # Create random weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)

    # Store format
    if store_unsigned:
        W_store = (W_int + 8).to(torch.uint8)  # [0, 15]
    else:
        W_store = W_int  # [-8, 7]

    # Pack
    W_store_np = W_store.cpu().numpy().astype(np.int8)
    qweight_np = general_compress(W_store_np, source_bits=4, storage_dtype=np.int8)
    qweight = torch.from_numpy(qweight_np).to(device).contiguous()

    # Scales and zeros
    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous().clone()

    if with_zeros:
        zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)
    else:
        zeros = None

    # Expected output (ground truth via dequant)
    W_dequant = (W_int.half() * scale).to(device)  # Correct dequant
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    expected = torch.nn.functional.linear(x, W_dequant)

    # Create kernel
    config_kwargs = {
        "M": [batch_size],
        "N": out_features,
        "K": in_features,
        "A_dtype": "float16",
        "W_dtype": W_dtype,
        "out_dtype": "float16",
        "accum_dtype": "float16",
        "with_scaling": True,
        "with_zeros": with_zeros,
        "group_size": group_size,
        "with_bias": False,
        "layout": "nt",
    }
    if zeros_mode:
        config_kwargs["zeros_mode"] = zeros_mode

    try:
        matmul_config = MatmulConfig(**config_kwargs)
        matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)

        output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)

        # Call kernel
        if with_zeros:
            matmul(x, qweight, scales, zeros, output=output)
        else:
            matmul(x, qweight, scales, output=output)

        # Compare
        error = (output - expected).abs()
        print(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"Error - Max: {error.max():.4f}, Mean: {error.mean():.4f}")

        if error.max() < 1.0:
            print("✅ SUCCESS - Output matches expected!")
            return True
        else:
            print("❌ FAIL - Output differs from expected")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("BitBLAS Configuration Variant Test")
    print("=" * 60)
    print("Goal: Find which configuration produces correct output")
    print()

    results = {}

    # Variant 1: int4 + no zeros (symmetric, signed storage)
    results["int4_no_zeros_signed"] = test_variant(
        name="INT4 Symmetric (no zeros, signed storage)",
        W_dtype="int4",
        with_zeros=False,
        zeros_mode=None,
        store_unsigned=False,
    )

    # Variant 2: int4 + no zeros (symmetric, but store unsigned)
    results["int4_no_zeros_unsigned"] = test_variant(
        name="INT4 Symmetric (no zeros, unsigned storage)",
        W_dtype="int4",
        with_zeros=False,
        zeros_mode=None,
        store_unsigned=True,
    )

    # Variant 3: uint4 + with zeros (asymmetric)
    results["uint4_with_zeros"] = test_variant(
        name="UINT4 with zeros (asymmetric)",
        W_dtype="uint4",
        with_zeros=True,
        zeros_mode="original",
        store_unsigned=True,
    )

    # Variant 4: uint4 + no zeros
    results["uint4_no_zeros"] = test_variant(
        name="UINT4 no zeros",
        W_dtype="uint4",
        with_zeros=False,
        zeros_mode=None,
        store_unsigned=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAIL"
        print(f"  {name}: {status}")

    successful = [k for k, v in results.items() if v]
    if successful:
        print(f"\n✅ Working configurations: {successful}")
    else:
        print("\n❌ No working configuration found!")


if __name__ == "__main__":
    main()
