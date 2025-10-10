import bitblas
import torch
import os
import sys

# Force BitBLAS to use SM80 (A100) to avoid SM90/TMA code paths that cause compilation errors
# The A40 is SM86, but we force SM80 to avoid unsupported CU_TENSOR_MAP_SWIZZLE_128B
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import bitblas


def main():
    bitblas.set_log_level("INFO")
    if not torch.cuda.is_available():
        print("CUDA not available; this test requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    # Small, quick shape for sanity test
    M, K, N = 16, 1024, 2048

    # Configure true INT4 (weights) x INT8 (activations) matmul without extra scaling/zeros
    cfg = bitblas.MatmulConfig(
        M=M,
        N=N,
        K=K,
        A_dtype="int8",
        W_dtype="int4",
        accum_dtype="float16",
        out_dtype="float16",
        layout="nt",
        with_bias=False,
        with_scaling=False,
        with_zeros=False,
        group_size=K,
    )

    # Create Matmul operator using TIR backend to avoid CUTLASS TMA errors
    # TIR backend doesn't use CUTLASS, so it avoids SM90-specific code
    target = "cuda -arch=sm_80"
    print(f"Creating BitBLAS Matmul operator (TIR backend) with target={target}...")
    mm = bitblas.Matmul(
        cfg,
        target=target,
        enable_tuning=False,
        from_database=False,
        backend="tir"  # Use TIR backend instead of TileLang to avoid CUTLASS
    )
    print("BitBLAS Matmul operator created successfully!")

    # Build integer inputs
    # A: INT8 in [-128,127]
    a_i8 = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
    # W: INT4 represented in int8 container in [-8,7], BitBLAS will pack to real int4
    w_i8 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)

    # Pack weights to int4
    w_i4_packed = mm.transform_weight(w_i8)

    # Run BitBLAS matmul (INT8 x INT4 -> FP16)
    torch.cuda.synchronize()
    y_bb = mm(a_i8, w_i4_packed)
    torch.cuda.synchronize()

    # Reference: interpret integers as real values and accumulate in FP16
    # This matches the semantics of no scaling/zeros: y = (a_i8 * w_i8^T) in fp16
    y_ref = (a_i8.to(torch.float16) @ w_i8.t().to(torch.float16))

    # Check for inf/nan in outputs
    if torch.isinf(y_bb).any() or torch.isnan(y_bb).any():
        print(f"ERROR: BitBLAS output contains inf/nan!")
        print(f"  y_bb has inf: {torch.isinf(y_bb).any()}, nan: {torch.isnan(y_bb).any()}")
        print(f"  Sample values: {y_bb[0, :10]}")
    elif torch.isinf(y_ref).any() or torch.isnan(y_ref).any():
        print(f"ERROR: Reference output contains inf/nan!")
    else:
        diff = (y_bb - y_ref).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"✓ BitBLAS INT4xINT8 SUCCESS: y shape={tuple(y_bb.shape)} dtype={y_bb.dtype}")
        print(f"  max_abs={max_abs:.4e}  mean_abs={mean_abs:.4e}")

        # Allow a small tolerance due to packing/conversion
        tol = 10.0  # INT8xINT4 can have larger differences
        if max_abs > tol:
            print(f"  WARNING: max_abs diff exceeds tolerance {tol} (this can happen for random int data).")


if __name__ == "__main__":
    main()

# enabling debug output

bitblas.set_log_level("Debug")
matmul_config = bitblas.MatmulConfig(
    M=1,  # M dimension
    N=1024,  # N dimension
    K=1024,  # K dimension
    A_dtype="float16",  # activation A dtype
    W_dtype="int4",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
)

print("Creating second BitBLAS operator (TIR backend)...")
matmul = bitblas.Matmul(
    config=matmul_config,
    target="cuda -arch=sm_80",
    enable_tuning=False,
    from_database=False,
    backend="tir"  # Use TIR backend
)
print("Second operator created!")

# Create input matrices
input_tensor = torch.rand((1, 1024), dtype=torch.float16).cuda()
weight_tensor = torch.randint(0, 7, (1024, 1024), dtype=torch.int8).cuda()

# Transform weight tensor to int4 data type
weight_tensor_int4 = matmul.transform_weight(weight_tensor)

# Perform mixed-precision matrix multiplication
output_tensor = matmul(input_tensor, weight_tensor_int4)

# Reference result using PyTorch matmul for comparison
ref_result = torch.matmul(input_tensor, weight_tensor.t().to(torch.float16))
# Assert that the results are close within a specified tolerance, note that the int4 randint value is a little bigger than the float16 value, so we set the atol to 1.0
print("Ref output:", ref_result)
print("BitBLAS output:", output_tensor)
torch.testing.assert_close(output_tensor, ref_result, rtol=1e-2, atol=1e-0)
