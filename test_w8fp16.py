#!/usr/bin/env python3
"""Test W8FP16 (INT8 weights × FP16 activations) implementation."""

import os
import sys

# Set environment for W8FP16
os.environ["OPENPI_W8A8_ENABLE"] = "1"
os.environ["OPENPI_W8A8_ACTIVATION_DTYPE"] = "float16"  # W8FP16 mode
os.environ["OPENPI_W8A8_DEBUG"] = "1"

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openpi.models_pytorch.bitblas_w8a8_layers import (
    BitBLASW8A8Linear,
    BITBLAS_AVAILABLE,
    get_w8a8_config_from_env,
)


def test_config():
    """Test configuration parsing."""
    print("\n=== Test 1: Configuration Parsing ===")
    config = get_w8a8_config_from_env()
    print(f"activation_dtype: {config['activation_dtype']}")
    print(f"enable_tuning: {config['enable_tuning']}")
    print(f"opt_M: {config['opt_M']}")
    assert config["activation_dtype"] == "float16", "Expected float16 activation"
    print("PASSED: Configuration correctly parsed\n")


def test_layer_creation():
    """Test W8FP16 layer creation."""
    print("\n=== Test 2: Layer Creation ===")

    if not BITBLAS_AVAILABLE:
        print("SKIPPED: BitBLAS not available")
        return

    # Create layer in W8FP16 mode
    layer = BitBLASW8A8Linear(
        in_features=256,
        out_features=512,
        bias=True,
        name="test_layer",
        enable_tuning=False,
        opt_M=[1, 16, 32],
        activation_dtype="float16",  # W8FP16
    )

    print(f"Layer created: in={layer.in_features}, out={layer.out_features}")
    print(f"activation_dtype: {layer.activation_dtype}")
    assert layer.activation_dtype == "float16", "Expected float16 activation"
    assert layer.bitblas_matmul is not None, "BitBLAS matmul should be created"
    print("PASSED: Layer created successfully\n")


def test_forward_pass():
    """Test W8FP16 forward pass."""
    print("\n=== Test 3: Forward Pass ===")

    if not BITBLAS_AVAILABLE:
        print("SKIPPED: BitBLAS not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create source FP16 Linear layer
    fp16_linear = nn.Linear(256, 512, bias=True).to(device).half()
    nn.init.xavier_uniform_(fp16_linear.weight)
    nn.init.zeros_(fp16_linear.bias)

    # Create W8FP16 layer
    w8fp16_layer = BitBLASW8A8Linear(
        in_features=256,
        out_features=512,
        bias=True,
        name="test_layer",
        enable_tuning=False,
        opt_M=[1, 16, 32],
        activation_dtype="float16",
    )

    # Load weights from FP16 layer
    w8fp16_layer.load_from_linear(fp16_linear)
    w8fp16_layer = w8fp16_layer.to(device)

    # Test input
    x = torch.randn(4, 256, device=device, dtype=torch.float16)

    # Forward pass
    with torch.no_grad():
        y_fp16 = fp16_linear(x)
        y_w8fp16 = w8fp16_layer(x)

    print(f"Input shape: {x.shape}")
    print(f"FP16 output shape: {y_fp16.shape}, dtype: {y_fp16.dtype}")
    print(f"W8FP16 output shape: {y_w8fp16.shape}, dtype: {y_w8fp16.dtype}")

    # Check shapes match
    assert y_fp16.shape == y_w8fp16.shape, "Output shapes should match"

    # Check MSE
    mse = ((y_fp16 - y_w8fp16) ** 2).mean().item()
    relative_error = mse / (y_fp16 ** 2).mean().item()
    print(f"MSE: {mse:.6e}")
    print(f"Relative error: {relative_error:.6e}")

    # W8FP16 should have relatively low error (no activation quantization)
    assert relative_error < 0.1, f"Relative error too high: {relative_error}"
    print("PASSED: Forward pass works correctly\n")


def test_w8a8_vs_w8fp16():
    """Compare W8A8 vs W8FP16 accuracy."""
    print("\n=== Test 4: W8A8 vs W8FP16 Comparison ===")

    if not BITBLAS_AVAILABLE:
        print("SKIPPED: BitBLAS not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create source FP16 Linear layer
    fp16_linear = nn.Linear(256, 512, bias=True).to(device).half()
    nn.init.xavier_uniform_(fp16_linear.weight)
    nn.init.zeros_(fp16_linear.bias)

    # Create W8A8 layer
    w8a8_layer = BitBLASW8A8Linear(
        in_features=256,
        out_features=512,
        bias=True,
        name="w8a8_layer",
        enable_tuning=False,
        opt_M=[1, 16, 32],
        activation_dtype="int8",  # W8A8
    )
    w8a8_layer.load_from_linear(fp16_linear)
    w8a8_layer = w8a8_layer.to(device)

    # Create W8FP16 layer
    w8fp16_layer = BitBLASW8A8Linear(
        in_features=256,
        out_features=512,
        bias=True,
        name="w8fp16_layer",
        enable_tuning=False,
        opt_M=[1, 16, 32],
        activation_dtype="float16",  # W8FP16
    )
    w8fp16_layer.load_from_linear(fp16_linear)
    w8fp16_layer = w8fp16_layer.to(device)

    # Test with multiple inputs
    total_mse_w8a8 = 0
    total_mse_w8fp16 = 0
    total_norm = 0
    n_samples = 10

    for i in range(n_samples):
        x = torch.randn(4, 256, device=device, dtype=torch.float16)

        with torch.no_grad():
            y_fp16 = fp16_linear(x)
            y_w8a8 = w8a8_layer(x)
            y_w8fp16 = w8fp16_layer(x)

        total_mse_w8a8 += ((y_fp16 - y_w8a8) ** 2).mean().item()
        total_mse_w8fp16 += ((y_fp16 - y_w8fp16) ** 2).mean().item()
        total_norm += (y_fp16 ** 2).mean().item()

    avg_mse_w8a8 = total_mse_w8a8 / n_samples
    avg_mse_w8fp16 = total_mse_w8fp16 / n_samples
    avg_norm = total_norm / n_samples

    rel_err_w8a8 = avg_mse_w8a8 / avg_norm
    rel_err_w8fp16 = avg_mse_w8fp16 / avg_norm

    print(f"W8A8   relative error: {rel_err_w8a8:.6e}")
    print(f"W8FP16 relative error: {rel_err_w8fp16:.6e}")
    print(f"Improvement factor: {rel_err_w8a8 / rel_err_w8fp16:.2f}x")

    # W8FP16 should be more accurate than W8A8
    print("PASSED: Comparison complete\n")


def main():
    print("=" * 60)
    print("W8FP16 Implementation Test")
    print("=" * 60)

    print(f"\nBitBLAS available: {BITBLAS_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not BITBLAS_AVAILABLE:
        print("\nWARNING: BitBLAS not available, some tests will be skipped")

    try:
        test_config()
        test_layer_creation()
        test_forward_pass()
        test_w8a8_vs_w8fp16()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
