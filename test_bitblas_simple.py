"""Simple test: Check if BitBLAS forward produces reasonable output."""

import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/jz97/VLM_REPO/openpi/src")

def test_bitblas_forward():
    """Test BitBLAS forward pass sanity."""

    # Create a simple linear layer
    in_features = 2048
    out_features = 16384
    batch_size = 1
    seq_len = 10

    # Create test input - IMPORTANT: Non-zero input!
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.half, device='cuda')

    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Input range: [{x.min():.6f}, {x.max():.6f}]")
    logger.info(f"Input mean: {x.mean():.6f}, std: {x.std():.6f}")

    # Create original FP16 linear layer
    linear_fp16 = torch.nn.Linear(in_features, out_features, bias=False).cuda().half()

    # Get FP16 baseline
    with torch.no_grad():
        y_fp16 = linear_fp16(x)

    logger.info(f"\nFP16 baseline:")
    logger.info(f"  Output shape: {y_fp16.shape}")
    logger.info(f"  Output range: [{y_fp16.min():.6f}, {y_fp16.max():.6f}]")
    logger.info(f"  Output mean: {y_fp16.mean():.6f}, std: {y_fp16.std():.6f}")

    # Create BitBLAS layer
    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

    layer_name = "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj"

    bitblas_layer = BitBLASQuantLinear(
        in_features=in_features,
        out_features=out_features,
        name=layer_name,
        bits=4,
        group_size=128,
        bias=False,
        enable_tuning=False,
        opt_M=[1, 16, 32, 64],
        duquant_packdir="duquant_packed_full_llm_dit_mlp_w4a8_atm",
    ).cuda()

    # Load same weights
    try:
        bitblas_layer.load_from_linear(linear_fp16)
        logger.info(f"\n✅ Successfully loaded weights")
        logger.info(f"  DuQuant pack: {bitblas_layer.duquant_pack is not None}")
        if bitblas_layer.duquant_pack:
            logger.info(f"  Has perm: {bitblas_layer.duquant_pack.perm is not None}")
            logger.info(f"  Has R_in: {len(bitblas_layer.duquant_pack.R_in_blocks)} blocks")
            logger.info(f"  Has R_out: {len(bitblas_layer.duquant_pack.R_out_blocks)} blocks")
            logger.info(f"  row_rot_mode: {bitblas_layer.duquant_pack.meta.get('row_rot_mode', 'unknown')}")
    except Exception as e:
        logger.error(f"❌ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get BitBLAS output
    logger.info(f"\nRunning BitBLAS forward pass...")
    try:
        with torch.no_grad():
            y_bitblas = bitblas_layer(x)

        logger.info(f"BitBLAS output:")
        logger.info(f"  Output shape: {y_bitblas.shape}")
        logger.info(f"  Output range: [{y_bitblas.min():.6f}, {y_bitblas.max():.6f}]")
        logger.info(f"  Output mean: {y_bitblas.mean():.6f}, std: {y_bitblas.std():.6f}")

        # Check for NaN or Inf
        if torch.isnan(y_bitblas).any():
            logger.error(f"\n❌ BitBLAS output contains NaN!")
            logger.error(f"   NaN count: {torch.isnan(y_bitblas).sum()} / {y_bitblas.numel()}")
            return False

        if torch.isinf(y_bitblas).any():
            logger.error(f"\n❌ BitBLAS output contains Inf!")
            logger.error(f"   Inf count: {torch.isinf(y_bitblas).sum()} / {y_bitblas.numel()}")
            return False

        # Check if output is all zeros (wrong!)
        if y_bitblas.abs().max() < 1e-6:
            logger.error(f"\n❌ BitBLAS output is all zeros!")
            return False

        # Compare with FP16 baseline
        abs_error = (y_bitblas - y_fp16).abs()
        logger.info(f"\nComparison with FP16 baseline:")
        logger.info(f"  Abs error - mean: {abs_error.mean():.6f}, max: {abs_error.max():.6f}")
        logger.info(f"  Rel error - mean: {(abs_error / (y_fp16.abs() + 1e-6)).mean():.6f}")

        # Show sample values
        logger.info(f"\nSample values (first 5):")
        for i in range(min(5, y_fp16.shape[-1])):
            logger.info(f"  [{i}] FP16: {y_fp16[0, 0, i]:.6f}, BitBLAS: {y_bitblas[0, 0, i]:.6f}, diff: {abs_error[0, 0, i]:.6f}")

        logger.info(f"\n✅ BitBLAS forward pass completed successfully")
        return True

    except Exception as e:
        logger.error(f"\n❌ BitBLAS forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Testing BitBLAS forward pass...")
    success = test_bitblas_forward()

    if not success:
        logger.error("\n" + "=" * 60)
        logger.error("FAILED: BitBLAS forward pass has issues!")
        logger.error("=" * 60)
        sys.exit(1)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS: BitBLAS forward pass works!")
        logger.info("=" * 60)
        sys.exit(0)
