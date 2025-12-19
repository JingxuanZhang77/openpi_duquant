"""Test BitBLAS output correctness vs DuQuant."""

import os
import sys
import torch
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, "/home/jz97/VLM_REPO/openpi/src")

def test_single_layer():
    """Test a single BitBLAS layer against DuQuant output."""

    # Create a simple linear layer
    in_features = 2048
    out_features = 16384
    batch_size = 16

    # Create test input
    x = torch.randn(batch_size, in_features, dtype=torch.half, device='cuda')

    # Create original FP16 linear layer
    linear_fp16 = torch.nn.Linear(in_features, out_features, bias=False).cuda().half()

    # Get FP16 output (ground truth)
    with torch.no_grad():
        y_fp16 = linear_fp16(x)

    logger.info(f"FP16 output shape: {y_fp16.shape}")
    logger.info(f"FP16 output range: [{y_fp16.min():.6f}, {y_fp16.max():.6f}]")
    logger.info(f"FP16 output mean: {y_fp16.mean():.6f}, std: {y_fp16.std():.6f}")

    # Create BitBLAS layer
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

    bitblas_layer = BitBLASQuantLinear(
        in_features=in_features,
        out_features=out_features,
        name="test_layer",
        bits=4,
        group_size=128,
        bias=False,
        enable_tuning=False,
        opt_M=[1, 16, 32, 64],
        duquant_packdir=None,
    ).cuda()

    # Load weights from FP16 layer
    bitblas_layer.load_from_linear(linear_fp16)

    # Get BitBLAS output
    with torch.no_grad():
        y_bitblas = bitblas_layer(x)

    logger.info(f"BitBLAS output shape: {y_bitblas.shape}")
    logger.info(f"BitBLAS output range: [{y_bitblas.min():.6f}, {y_bitblas.max():.6f}]")
    logger.info(f"BitBLAS output mean: {y_bitblas.mean():.6f}, std: {y_bitblas.std():.6f}")

    # Compute error
    abs_error = (y_bitblas - y_fp16).abs()
    rel_error = abs_error / (y_fp16.abs() + 1e-6)

    logger.info(f"\n=== Error Analysis ===")
    logger.info(f"Absolute error - mean: {abs_error.mean():.6f}, max: {abs_error.max():.6f}")
    logger.info(f"Relative error - mean: {rel_error.mean():.6f}, max: {rel_error.max():.6f}")

    # Check if BitBLAS kernel was used
    if bitblas_layer.bitblas_matmul is not None:
        logger.info(f"✅ BitBLAS kernel is available")
    else:
        logger.warning(f"❌ BitBLAS kernel is NOT available, using dequant fallback")

    # Check if error is acceptable
    if abs_error.mean() > 0.1:
        logger.error(f"❌ ERROR TOO LARGE! Mean abs error: {abs_error.mean():.6f}")
        logger.error(f"This could explain the 0% accuracy!")
    else:
        logger.info(f"✅ Error is acceptable")

    # Check for NaN or Inf
    if torch.isnan(y_bitblas).any() or torch.isinf(y_bitblas).any():
        logger.error(f"❌ BitBLAS output contains NaN or Inf!")
        logger.error(f"NaN count: {torch.isnan(y_bitblas).sum()}")
        logger.error(f"Inf count: {torch.isinf(y_bitblas).sum()}")
    else:
        logger.info(f"✅ No NaN or Inf in output")

    return abs_error.mean().item()

if __name__ == "__main__":
    logger.info("Testing BitBLAS layer output correctness...")
    error = test_single_layer()

    if error > 0.1:
        logger.error(f"\n{'='*60}")
        logger.error(f"CRITICAL: BitBLAS output has large error: {error:.6f}")
        logger.error(f"This is likely causing the 0% accuracy!")
        logger.error(f"{'='*60}")
        sys.exit(1)
    else:
        logger.info(f"\n✅ BitBLAS output is correct (error: {error:.6f})")
