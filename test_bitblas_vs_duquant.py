"""Test BitBLAS output vs DuQuant output on a real layer."""

import os
import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/jz97/VLM_REPO/openpi/src")

os.environ["OPENPI_BITBLAS_ENABLE"] = "0"  # Disable for now
os.environ["OPENPI_DUQUANT_WBITS_DEFAULT"] = "4"
os.environ["OPENPI_DUQUANT_ABITS"] = "0"  # FP16 activations
os.environ["OPENPI_DUQUANT_BLOCK"] = "64"
os.environ["OPENPI_DUQUANT_PERMUTE"] = "1"
os.environ["OPENPI_DUQUANT_ROW_ROT"] = "restore"
os.environ["OPENPI_DUQUANT_PACKDIR"] = "duquant_packed_full_llm_dit_mlp_w4a8_atm"
os.environ["OPENPI_DUQUANT_INCLUDE"] = "(.*language_model.*(gate_proj|up_proj|down_proj).*)"
os.environ["OPENPI_DUQUANT_EXCLUDE"] = "(?:^|\\.)(norm|ln|layernorm|emb|embed|vision_tower)(?:\\.|$)"

def test_layer_equivalence():
    """Test that BitBLAS and DuQuant produce same output."""

    # Create a simple linear layer
    in_features = 2048
    out_features = 16384
    batch_size = 1
    seq_len = 10

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.half, device='cuda')

    # Create original FP16 linear layer
    linear_fp16 = torch.nn.Linear(in_features, out_features, bias=False).cuda().half()

    logger.info("=" * 60)
    logger.info("Step 1: Create DuQuant layer")
    logger.info("=" * 60)

    from openpi.models_pytorch.duquant_layers import DuQuantLinear
    from openpi.models_pytorch.duquant_preprocess import load_pack, DuQuantConfig

    # Load pack
    layer_name = "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj"
    pack = load_pack(layer_name, "duquant_packed_full_llm_dit_mlp_w4a8_atm")

    if pack is None:
        logger.error(f"Failed to load pack for {layer_name}")
        return False

    cfg = DuQuantConfig(
        weight_bits=4,
        act_bits=0,  # FP16 activations
        block_size=64,
        row_rot_mode="restore",
        act_percentile=99.9,
    )

    duquant_layer = DuQuantLinear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        name=layer_name,
        pack=pack,
        cfg=cfg,
    ).cuda()

    # Load weights
    duquant_layer._weight.copy_(linear_fp16.weight.data)

    # Get DuQuant output
    with torch.no_grad():
        y_duquant = duquant_layer(x)

    logger.info(f"DuQuant output shape: {y_duquant.shape}")
    logger.info(f"DuQuant output range: [{y_duquant.min():.6f}, {y_duquant.max():.6f}]")
    logger.info(f"DuQuant output mean: {y_duquant.mean():.6f}, std: {y_duquant.std():.6f}")

    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Create BitBLAS layer")
    logger.info("=" * 60)

    from openpi.models_pytorch.bitblas_layers import BitBLASQuantLinear

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
    bitblas_layer.load_from_linear(linear_fp16)

    # Get BitBLAS output
    with torch.no_grad():
        y_bitblas = bitblas_layer(x)

    logger.info(f"BitBLAS output shape: {y_bitblas.shape}")
    logger.info(f"BitBLAS output range: [{y_bitblas.min():.6f}, {y_bitblas.max():.6f}]")
    logger.info(f"BitBLAS output mean: {y_bitblas.mean():.6f}, std: {y_bitblas.std():.6f}")

    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Compare outputs")
    logger.info("=" * 60)

    # Compute error
    abs_error = (y_bitblas - y_duquant).abs()
    rel_error = abs_error / (y_duquant.abs() + 1e-6)

    logger.info(f"Absolute error - mean: {abs_error.mean():.6f}, max: {abs_error.max():.6f}")
    logger.info(f"Relative error - mean: {rel_error.mean():.6f}, max: {rel_error.max():.6f}")

    # Check for NaN or Inf
    if torch.isnan(y_bitblas).any():
        logger.error(f"❌ BitBLAS output contains NaN!")
        logger.error(f"   NaN locations: {torch.isnan(y_bitblas).sum()} / {y_bitblas.numel()}")
        return False

    if torch.isinf(y_bitblas).any():
        logger.error(f"❌ BitBLAS output contains Inf!")
        logger.error(f"   Inf locations: {torch.isinf(y_bitblas).sum()} / {y_bitblas.numel()}")
        return False

    # Check if error is acceptable
    if abs_error.mean() > 0.5:  # Allow some quantization error
        logger.error(f"❌ ERROR TOO LARGE! Mean abs error: {abs_error.mean():.6f}")
        logger.error(f"This explains the 0% accuracy!")

        # Show sample values
        logger.error("\nSample comparison (first 5 values):")
        for i in range(min(5, y_duquant.shape[-1])):
            logger.error(f"  [{i}] DuQuant: {y_duquant[0, 0, i]:.6f}, BitBLAS: {y_bitblas[0, 0, i]:.6f}, diff: {abs_error[0, 0, i]:.6f}")

        return False
    else:
        logger.info(f"✅ Error is acceptable (mean: {abs_error.mean():.6f})")
        return True

if __name__ == "__main__":
    logger.info("Testing BitBLAS vs DuQuant layer equivalence...")
    success = test_layer_equivalence()

    if not success:
        logger.error("\n" + "=" * 60)
        logger.error("CRITICAL: BitBLAS output does NOT match DuQuant!")
        logger.error("This is why accuracy is 0%!")
        logger.error("=" * 60)
        sys.exit(1)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS: BitBLAS matches DuQuant!")
        logger.info("=" * 60)
        sys.exit(0)
