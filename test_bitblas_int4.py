#!/usr/bin/env python3
"""Test BitBLAS INT4 matmul directly."""
import torch
import sys
import os

# Add BitBLAS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party/BitBLAS'))

try:
    from bitblas import Matmul, MatmulConfig
    print("✓ BitBLAS imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BitBLAS: {e}")
    sys.exit(1)

# Test configuration: simple W4A16 matmul
M, N, K = 1, 256, 256
print(f"\nTesting W4A16 matmul: [{M}, {K}] @ [{K}, {N}]")

# Create BitBLAS matmul operator
config = MatmulConfig(
    M=M,
    N=N,
    K=K,
    A_dtype="float16",      # Input dtype
    W_dtype="int4",          # Weight dtype (4-bit)
    out_dtype="float16",     # Output dtype
    accum_dtype="int32",     # Accumulation dtype
    layout="nt",             # Non-transposed A, transposed B
    with_bias=False,
    with_scaling=True,       # Enable per-channel scaling
    group_size=-1,           # Per-channel quantization
)

print(f"Config: A_dtype={config.A_dtype}, W_dtype={config.W_dtype}, layout={config.layout}")

try:
    print("\nCreating BitBLAS Matmul operator...")
    matmul_op = Matmul(config=config, enable_tuning=False)
    print("✓ BitBLAS Matmul operator created")
except Exception as e:
    print(f"✗ Failed to create operator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create test tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

if device.type != "cuda":
    print("✗ CUDA not available, BitBLAS requires GPU")
    sys.exit(1)

# Create random input activation
A = torch.randn(M, K, dtype=torch.float16, device=device)
print(f"Input A: {A.shape} {A.dtype}")

# Create random INT4 weights (stored as int8)
# BitBLAS expects INT4 values in range [-8, 7]
W_int4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
print(f"Weight W (INT4): {W_int4.shape} {W_int4.dtype}")

# Create per-channel weight scales
scales = torch.randn(N, dtype=torch.float16, device=device).abs() + 0.1
print(f"Scales: {scales.shape} {scales.dtype}")

# Try to run the matmul
try:
    print("\nRunning BitBLAS INT4 matmul...")
    # Note: Need to check BitBLAS API for exact input format
    # This is a test to see what format it expects
    output = matmul_op(A, W_int4, scales)
    print(f"✓ Output: {output.shape} {output.dtype}")
    print(f"  Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")

except Exception as e:
    print(f"✗ Failed to run matmul: {e}")
    import traceback
    traceback.print_exc()

    # Try alternative API
    print("\nTrying alternative API with transform...")
    try:
        # BitBLAS might need weights in a specific packed format
        # Let's check if there's a transform function
        if hasattr(matmul_op, 'transform_weight'):
            print("Found transform_weight method")
            W_transformed = matmul_op.transform_weight(W_int4)
            print(f"Transformed weight: {W_transformed.shape}")
            output = matmul_op(A, W_transformed, scales)
            print(f"✓ Output: {output.shape} {output.dtype}")
        else:
            print("No transform_weight method found")
            print(f"Available methods: {[m for m in dir(matmul_op) if not m.startswith('_')]}")
    except Exception as e2:
        print(f"Alternative API also failed: {e2}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("BitBLAS INT4 test completed")
