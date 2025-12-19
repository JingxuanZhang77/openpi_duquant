#!/usr/bin/env python3
"""Debug BitBLAS kernel arguments."""

import os
import sys
import torch

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_BITBLAS_DEBUG"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_args_comparison():
    """Compare args between working test and BitBLASQuantLinear."""
    print("=" * 60)
    print("Comparing BitBLAS kernel arguments")
    print("=" * 60)

    pack_dir = "duquant_packed_full_llm_dit_mlp_w4a8_atm"

    from bitblas.ops import Matmul, MatmulConfig
    from bitblas.utils import auto_detect_nvidia_target
    from openpi.models_pytorch.duquant_preprocess import load_pack, compute_mse_scales
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

    group_size = 128
    device = torch.device("cuda")
    target = auto_detect_nvidia_target()

    # Create original linear
    torch.manual_seed(42)
    original_linear = torch.nn.Linear(in_features, out_features, bias=False)
    original_linear = original_linear.cuda().half()

    # ===== Path 1: Direct BitBLAS creation (working) =====
    print("\n--- Path 1: Direct BitBLAS (working) ---")

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
    scale_direct = compute_mse_scales(W, 4)
    W_int_direct = torch.clamp(torch.round(W / scale_direct[:, None]), -8, 7).to(torch.int8)
    W_uint_direct = (W_int_direct + 8).to(torch.uint8)

    # Create kernel
    matmul_config = MatmulConfig(
        M=[1, 4, 16, 32],
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

    matmul_direct = Matmul(matmul_config, target=target, backend="tir", enable_tuning=False)

    n_groups = in_features // group_size
    scales_direct = scale_direct[:, None].expand(-1, n_groups).contiguous().clone()
    zeros_direct = torch.full((out_features, n_groups), 8.0, dtype=torch.float16, device=device)

    # Transform
    transformed_direct = matmul_direct.transform_weight(W_uint_direct.to(torch.int8), scale=scales_direct, zeros=zeros_direct)
    if isinstance(transformed_direct, list):
        qweight_direct = transformed_direct[0]
    else:
        qweight_direct = transformed_direct

    print(f"Direct qweight: shape={qweight_direct.shape}, dtype={qweight_direct.dtype}")
    print(f"Direct scales: shape={scales_direct.shape}, range=[{scales_direct.min():.8f}, {scales_direct.max():.8f}]")
    print(f"Direct zeros: shape={zeros_direct.shape}, value={zeros_direct[0,0]}")

    # Test forward
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")

    output_direct = torch.empty(4, out_features, dtype=torch.float16, device=device)
    matmul_direct(x, qweight_direct, scales_direct, zeros_direct, output=output_direct)
    print(f"Direct output range: [{output_direct.min():.4f}, {output_direct.max():.4f}]")

    # ===== Path 2: BitBLASQuantLinear =====
    print("\n--- Path 2: BitBLASQuantLinear ---")

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

    print(f"Layer qweight: shape={bitblas_layer.qweight.shape}, dtype={bitblas_layer.qweight.dtype}")
    print(f"Layer scales: shape={bitblas_layer.scales.shape}, range=[{bitblas_layer.scales.min():.8f}, {bitblas_layer.scales.max():.8f}]")
    print(f"Layer zeros: shape={bitblas_layer.zeros.shape}, value={bitblas_layer.zeros[0,0]}")

    # Test forward
    torch.manual_seed(123)
    x = torch.randn(4, in_features, dtype=torch.float16, device="cuda")

    output_layer = torch.empty(4, out_features, dtype=torch.float16, device=device)
    if bitblas_layer.bitblas_matmul is not None:
        bitblas_layer.bitblas_matmul(x, bitblas_layer.qweight, bitblas_layer.scales, bitblas_layer.zeros, output=output_layer)
        print(f"Layer output range: [{output_layer.min():.4f}, {output_layer.max():.4f}]")
    else:
        print("No matmul!")

    # ===== Compare =====
    print("\n--- Comparison ---")

    # Compare qweight
    qw_match = torch.allclose(qweight_direct, bitblas_layer.qweight)
    print(f"qweight match: {qw_match}")
    if not qw_match:
        diff = (qweight_direct - bitblas_layer.qweight).abs()
        print(f"qweight diff: max={diff.max()}, mean={diff.float().mean():.4f}")
        print(f"qweight direct sample: {qweight_direct[0, :8].tolist()}")
        print(f"qweight layer sample: {bitblas_layer.qweight[0, :8].tolist()}")

    # Compare scales
    sc_match = torch.allclose(scales_direct, bitblas_layer.scales)
    print(f"scales match: {sc_match}")
    if not sc_match:
        diff = (scales_direct - bitblas_layer.scales).abs()
        print(f"scales diff: max={diff.max():.8f}, mean={diff.mean():.8f}")

    # Compare zeros
    z_match = torch.allclose(zeros_direct, bitblas_layer.zeros)
    print(f"zeros match: {z_match}")


if __name__ == "__main__":
    test_args_comparison()
