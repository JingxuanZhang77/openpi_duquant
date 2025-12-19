#!/usr/bin/env python3
"""Test BitBLAS UINT4 kernel configuration."""

import os
import sys
import torch
import numpy as np

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_uint4_kernel():
    """Test UINT4 kernel with zeros offset."""
    print("=" * 60)
    print("Testing BitBLAS UINT4 Kernel")
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

    print(f"Target: {target}")

    # Create random weight
    W_fp16 = torch.randn(out_features, in_features, dtype=torch.float16, device=device)

    # Per-channel scale
    scale = W_fp16.abs().max(dim=1, keepdim=True)[0] / 7.0

    # Quantize to signed INT4 [-8, 7]
    W_int = torch.clamp(torch.round(W_fp16 / scale), -8, 7).to(torch.int8)

    # Convert to unsigned [0, 15]
    W_uint = (W_int + 8).to(torch.uint8)
    print(f"W_uint range: [{W_uint.min()}, {W_uint.max()}]")

    # Pack
    W_uint_np = W_uint.cpu().numpy()
    qweight_np = general_compress(W_uint_np, source_bits=4, storage_dtype=np.int8)
    qweight = torch.from_numpy(qweight_np).to(device).contiguous()

    # Scales and zeros
    n_groups = in_features // group_size
    scales = scale.expand(-1, n_groups).contiguous().clone()
    zeros = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    # Input
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)

    # Expected output via dequant
    # Unpack
    qweight_uint = qweight.cpu().numpy().view(np.uint8)
    packed_in = qweight.shape[1]
    W_unpacked = np.zeros((out_features, in_features), dtype=np.uint8)
    for i in range(packed_in):
        W_unpacked[:, 2*i] = qweight_uint[:, i] & 0x0F
        W_unpacked[:, 2*i + 1] = (qweight_uint[:, i] >> 4) & 0x0F

    W_unpacked_t = torch.from_numpy(W_unpacked).to(device)

    # Dequant
    W_dequant = torch.zeros((out_features, in_features), dtype=torch.float16, device=device)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        W_dequant[:, start:end] = (W_unpacked_t[:, start:end].half() - 8) * scales[:, g:g+1]

    expected = torch.nn.functional.linear(x, W_dequant)
    print(f"Expected output range: [{expected.min():.4f}, {expected.max():.4f}]")

    # Create UINT4 kernel with zeros
    print("\nCreating UINT4 kernel...")
    matmul_config = MatmulConfig(
        M=[batch_size],
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

    try:
        matmul = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)
        print("Kernel created successfully")

        output = torch.empty(batch_size, out_features, dtype=torch.float16, device=device)
        matmul(x, qweight, scales, zeros, output)

        print(f"Kernel output range: [{output.min():.4f}, {output.max():.4f}]")

        # Compare
        error = (output - expected).abs()
        print(f"\nKernel vs Dequant Error:")
        print(f"  Max: {error.max():.6f}")
        print(f"  Mean: {error.mean():.6f}")

        if error.max() < 1.0:
            print("\n✅ UINT4 kernel output matches dequant!")
            return True
        else:
            print("\n❌ UINT4 kernel output differs from dequant")
            return False

    except Exception as e:
        print(f"Kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_uint4_kernel()
